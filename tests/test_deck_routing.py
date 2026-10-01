"""C04: resolve_role behind routing_mode (specs/001-agent-capability-routing).

Two states, and both must be working states. legacy: byte-for-byte the walk
that existed before this component. deck: the role's band resolves to the
leftmost ranked card that clears the headroom gate, then skip-self and
skip-exhausted apply exactly as they do to a [roles] entry, and anything the
deck cannot answer falls through to legacy.
"""
from __future__ import annotations

import json

import pytest

from mcp_brain_router import deck as deckmod
from mcp_brain_router.config import Config, DEFAULT_ROLE_BANDS
from mcp_brain_router.router import Provider, Role, provider_for_model, resolve_role


def _cfg(mode: str, **kw) -> Config:
    # Kimi and GLM are discontinued (Mohan, 2026-09-30); the legacy lists hold the
    # three providers the rig still uses, Codex first so the reason stays plain.
    return Config(
        roles={"worker": ["gpt-5.6-terra", "grok-4.5", "claude-sonnet-5"],
               "simple": ["grok-4.5", "claude-haiku-4-5-20251001"],
               "adversary": ["grok-4.5", "claude-opus-5-5"], "thinker": ["claude-fable-5-1", "grok-4.5"]},
        routing_mode=mode, role_bands=dict(DEFAULT_ROLE_BANDS), **kw,
    )


def _card(router_model: str, provider: str, pass_rate: float = 0.9, cap: float = 60.0) -> dict:
    """A card benched 3 times with this pass rate (ROUTER: the deck ranks on measured quality)."""
    return {"slug": router_model.replace(".", "-"), "router_model": router_model,
            "provider": provider, "model": router_model, "effort": "base",
            "capability": cap, "price_blended": 1.0, "total_cost_usd": None,
            "size_class": "small", "p0_pass": True, "pass_rate": pass_rate, "trials": 3}


_THREE = [("gpt-5.6-terra", "openai", 0.90), ("grok-4.5", "xai", 0.92), ("claude-sonnet-5", "anthropic", 0.95)]


def _deck_file(tmp_path, ranked_by_band: dict[str, list[dict]]):
    doc = {"bands": [{"band": b, "floor": 0.0, "floor_status": "ESTIMATE",
                      "ranked": ranked_by_band.get(b, []), "unranked": []}
                     for b in deckmod.BANDS]}
    p = tmp_path / "deck.json"
    p.write_text(json.dumps(doc))
    return p


_REAL_LOAD = deckmod.load_deck  # captured before any monkeypatch


def _load(p):
    return _REAL_LOAD(p)


# ---------------------------------------------------------------- legacy is untouched

def test_legacy_mode_ignores_the_deck_entirely(tmp_path, monkeypatch):
    p = _deck_file(tmp_path, {"B3": [_card("grok-4.5", "xai"), _card("claude-sonnet-5", "anthropic")]})
    monkeypatch.setattr(deckmod, "load_deck", lambda path=None: _load(p))
    monkeypatch.setattr(deckmod, "read_quota", lambda path=None: {})
    a = resolve_role(Role.WORKER, "claude", _cfg("legacy"), mode="agentic")
    # [roles] worker is gpt-5.6-terra first; the deck's head card (grok) must NOT win.
    assert a.model == "gpt-5.6-terra"
    assert a.reason == "first eligible configured candidate"


def test_default_mode_is_legacy():
    assert Config(roles={"worker": ["grok-4.5"]}).routing_mode == "legacy"


# ---------------------------------------------------------------- deck path

def test_deck_mode_takes_leftmost_ranked_card(tmp_path, monkeypatch):
    p = _deck_file(tmp_path, {"B3": [_card(*c) for c in _THREE]})
    monkeypatch.setattr(deckmod, "load_deck", lambda path=None: _load(p))
    monkeypatch.setattr(deckmod, "read_quota", lambda path=None: {})
    a = resolve_role(Role.WORKER, "claude", _cfg("deck"), mode="agentic")
    assert a.model == "gpt-5.6-terra"
    assert a.provider is Provider.CODEX
    assert a.backend == "codex"
    assert a.reason.startswith("deck B3")


def test_deck_mode_applies_skip_self_to_adversary_only(tmp_path, monkeypatch):
    # rigor: small — skip-self narrowed to adversary (2026-09-07)
    cards = [_card(*c) for c in _THREE]
    p = _deck_file(tmp_path, {"B3": cards, "B4": cards})
    monkeypatch.setattr(deckmod, "load_deck", lambda path=None: _load(p))
    monkeypatch.setattr(deckmod, "read_quota", lambda path=None: {})
    # Orchestrator is the head card's own provider. Worker keeps the leftmost card;
    # adversary skips it.
    a = resolve_role(Role.WORKER, "codex", _cfg("deck"), mode="agentic")
    assert a.model == "gpt-5.6-terra"
    a = resolve_role(Role.ADVERSARY, "codex", _cfg("deck"), mode="agentic")
    assert a.model == "grok-4.5"


def test_deck_mode_applies_skip_exhausted(tmp_path, monkeypatch):
    p = _deck_file(tmp_path, {"B3": [_card(*c) for c in _THREE]})
    monkeypatch.setattr(deckmod, "load_deck", lambda path=None: _load(p))
    monkeypatch.setattr(deckmod, "read_quota", lambda path=None: {})
    a = resolve_role(Role.WORKER, "claude", _cfg("deck"), mode="agentic",
                     exhausted_providers=[Provider.CODEX])
    assert a.model == "grok-4.5"


def test_deck_mode_headroom_gates_by_DECK_provider_name(tmp_path, monkeypatch):
    """The gate compares Card.provider against headroom keys -- both deck names.
    openai at 95% must remove gpt-5.6-terra even though the router calls it 'codex'."""
    p = _deck_file(tmp_path, {"B3": [_card(*c) for c in _THREE]})
    monkeypatch.setattr(deckmod, "load_deck", lambda path=None: _load(p))
    monkeypatch.setattr(deckmod, "read_quota", lambda path=None: {"openai": 0.95, "xai": 0.10})
    a = resolve_role(Role.WORKER, "claude", _cfg("deck"), mode="agentic")
    assert a.model == "grok-4.5"


def test_deck_mode_falls_through_to_legacy_when_band_is_empty(tmp_path, monkeypatch):
    p = _deck_file(tmp_path, {})  # week 0: every band unranked
    monkeypatch.setattr(deckmod, "load_deck", lambda path=None: _load(p))
    monkeypatch.setattr(deckmod, "read_quota", lambda path=None: {})
    a = resolve_role(Role.WORKER, "claude", _cfg("deck"), mode="agentic")
    assert a.model == "gpt-5.6-terra"
    assert a.reason.startswith("reference-thin-evidence")


def test_deck_mode_falls_through_when_deck_unreadable(monkeypatch):
    monkeypatch.setattr(deckmod, "load_deck", lambda path=None: {})
    a = resolve_role(Role.WORKER, "claude", _cfg("deck"), mode="agentic")
    assert a.model == "gpt-5.6-terra"


def test_deck_anthropic_card_uses_agentic_rule(tmp_path, monkeypatch):
    """The three-way Anthropic rule is shared with the legacy walk, not re-implemented."""
    p = _deck_file(tmp_path, {"B5": [_card("claude-fable-5-1", "anthropic", 0.95, cap=72),
                                     _card("claude-opus-5-5", "anthropic", 0.93, cap=71),
                                     _card("claude-sonnet-5", "anthropic", 0.90, cap=68)]})
    monkeypatch.setattr(deckmod, "load_deck", lambda path=None: _load(p))
    monkeypatch.setattr(deckmod, "read_quota", lambda path=None: {})
    a = resolve_role(Role.THINKER, "codex", _cfg("deck"), mode="agentic")
    assert a.provider is Provider.ANTHROPIC and a.backend == "anthropic-cli" and not a.execute_natively
    b = resolve_role(Role.THINKER, "codex", _cfg("deck"), mode="chat")
    assert b.execute_natively and b.backend is None


# ---------------------------------------------------------------- the namespace crossing

def test_deck_and_router_provider_names_agree_on_every_live_card():
    """provider_for_model(router_model) must land on the Provider that
    DECK_TO_ROUTER says Card.provider maps to -- for every card in the live
    deck. A card where they disagree would be gated under one name and
    skipped under another."""
    import pathlib
    # deck.json, not cards.json: router_model is derived by the builder, and the
    # deck is what resolve_role actually reads.
    p = pathlib.Path.home() / ".local/state/brain-router/deck.json"
    if not p.exists():
        pytest.skip("no live deck.json")
    bands = json.loads(p.read_text())["bands"]
    cards = [c for b in bands for c in (b.get("ranked") or []) + (b.get("unranked") or [])]
    misrouted, unroutable = [], []
    for c in cards:
        want = deckmod.router_provider_name(c["provider"])
        try:
            got = provider_for_model(c["router_model"]).value
        except ValueError:
            unroutable.append(c["slug"])   # router skips these; see resolve_role
            continue
        if got != want:
            misrouted.append((c["slug"], c["provider"], want, got))
    # A MISROUTE is fatal: gated under one provider, skipped under another, work
    # sent to the wrong CLI. An UNROUTABLE card is skipped by the deck path and
    # can never win; it is reported so the prefix table can grow, not failed on.
    assert not misrouted, f"{len(misrouted)} cards misroute, e.g. {misrouted[:3]}"
    print(f"unroutable (skipped by deck path): {sorted(unroutable)}")


# ---------------------------------------------------------------- refuter findings, 2026-09-05

def test_deck_mode_under_two_measured_falls_to_legacy_with_warning(tmp_path, monkeypatch, caplog):
    """ROUTER at run time: a band with fewer than 2 measured cards has no comparison
    behind "best" and cannot route by deck (replaces R26's 3-ranked rule)."""
    p = _deck_file(tmp_path, {"B3": [_card("grok-4.5", "xai")]})
    monkeypatch.setattr(deckmod, "load_deck", lambda path=None: _load(p))
    monkeypatch.setattr(deckmod, "read_quota", lambda path=None: {})
    import logging
    with caplog.at_level(logging.WARNING):
        a = resolve_role(Role.WORKER, "claude", _cfg("deck"), mode="agentic")
    assert a.model == "gpt-5.6-terra"  # legacy first candidate, not the deck's lone grok
    assert any("reference-thin-evidence" in r.message for r in caplog.records)


def test_routing_hole_never_hands_work_to_the_gated_provider(tmp_path, monkeypatch, caplog):
    """C03 STOP: when the gate empties a band, the legacy walk must not select the
    very provider the gate removed. Codex is legacy-first for worker; gate it out."""
    p = _deck_file(tmp_path, {"B3": [_card("gpt-5.6-terra", "openai"),
                                     _card("gpt-5.5", "openai"),
                                     _card("gpt-6-astra", "openai")]})
    monkeypatch.setattr(deckmod, "load_deck", lambda path=None: _load(p))
    monkeypatch.setattr(deckmod, "read_quota", lambda path=None: {"openai": 0.99})
    import logging
    with caplog.at_level(logging.WARNING):
        a = resolve_role(Role.WORKER, "claude", _cfg("deck"), mode="agentic")
    assert a.provider is not Provider.CODEX, "legacy walk handed work to the gated-out provider"
    assert a.model == "grok-4.5"
    assert a.reason.startswith("fallback-quota")
    assert any("routing hole" in r.message for r in caplog.records)


def test_config_save_round_trips_routing_mode_and_role_bands(tmp_path, monkeypatch):
    from mcp_brain_router import config as cfgmod
    monkeypatch.setattr(cfgmod, "CONFIG_DIR", tmp_path)
    monkeypatch.setattr(cfgmod, "CONFIG_FILE", tmp_path / "config.toml")
    c = _cfg("deck"); c.role_bands = {"worker": "B2"}
    c.save()
    (tmp_path / "config.toml").chmod(0o600)
    back = Config.load()
    assert back.routing_mode == "deck"
    assert back.role_bands["worker"] == "B2"


def test_deck_min_measured_one_routes_a_single_measured_card(tmp_path, monkeypatch):
    """deck_min_measured (default 2) can be lowered on purpose; the old ranked-count
    setting no longer gates anything (ROUTER, 2026-10-01)."""
    p = _deck_file(tmp_path, {"B3": [_card("grok-4.5", "xai")]})
    monkeypatch.setattr(deckmod, "load_deck", lambda path=None: _load(p))
    monkeypatch.setattr(deckmod, "read_quota", lambda path=None: {})
    monkeypatch.setattr(deckmod, "read_daily_calls", lambda *a, **k: {})
    cfg = _cfg("deck")
    cfg.deck_min_measured = 1
    a = resolve_role(Role.WORKER, "claude", cfg, mode="agentic")
    assert a.model == "grok-4.5"
