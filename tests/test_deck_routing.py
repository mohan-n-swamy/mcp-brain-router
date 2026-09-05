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
    return Config(
        roles={"worker": ["kimi", "glm-5.3", "grok-4.5"], "simple": ["kimi", "glm-5.3-flash"],
               "adversary": ["grok-4.5", "claude-opus-4-8"], "thinker": ["claude-fable-5", "kimi"]},
        routing_mode=mode, role_bands=dict(DEFAULT_ROLE_BANDS), **kw,
    )


def _card(router_model: str, provider: str, cost: float, cap: float = 60.0) -> dict:
    return {"slug": router_model.replace(".", "-"), "router_model": router_model,
            "provider": provider, "model": router_model, "effort": "base",
            "capability": cap, "price_blended": 1.0, "total_cost_usd": cost,
            "size_class": "small", "p0_pass": True}


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
    p = _deck_file(tmp_path, {"B3": [_card("grok-4.5", "xai", 0.01)]})
    monkeypatch.setattr(deckmod, "load_deck", lambda path=None: _load(p))
    monkeypatch.setattr(deckmod, "read_quota", lambda path=None: {})
    a = resolve_role(Role.WORKER, "claude", _cfg("legacy"), mode="agentic")
    # [roles] worker is kimi first; the deck's cheapest card (grok) must NOT win.
    assert a.model == "kimi"
    assert a.reason == "first eligible configured candidate"


def test_default_mode_is_legacy():
    assert Config(roles={"worker": ["kimi"]}).routing_mode == "legacy"


# ---------------------------------------------------------------- deck path

def test_deck_mode_takes_leftmost_ranked_card(tmp_path, monkeypatch):
    p = _deck_file(tmp_path, {"B3": [_card("glm-5.3", "zhipu", 0.10),
                                     _card("kimi-k3", "moonshot", 0.13)]})
    monkeypatch.setattr(deckmod, "load_deck", lambda path=None: _load(p))
    monkeypatch.setattr(deckmod, "read_quota", lambda path=None: {})
    a = resolve_role(Role.WORKER, "claude", _cfg("deck"), mode="agentic")
    assert a.model == "glm-5.3"
    assert a.provider is Provider.ZHIPU
    assert a.backend == "glm"
    assert a.reason.startswith("deck B3")


def test_deck_mode_applies_skip_self(tmp_path, monkeypatch):
    p = _deck_file(tmp_path, {"B3": [_card("glm-5.3", "zhipu", 0.10),
                                     _card("kimi-k3", "moonshot", 0.13)]})
    monkeypatch.setattr(deckmod, "load_deck", lambda path=None: _load(p))
    monkeypatch.setattr(deckmod, "read_quota", lambda path=None: {})
    # Orchestrator is glm's own provider -> the leftmost card is skipped.
    a = resolve_role(Role.WORKER, "glm", _cfg("deck"), mode="agentic")
    assert a.model == "kimi-k3"


def test_deck_mode_applies_skip_exhausted(tmp_path, monkeypatch):
    p = _deck_file(tmp_path, {"B3": [_card("glm-5.3", "zhipu", 0.10),
                                     _card("kimi-k3", "moonshot", 0.13)]})
    monkeypatch.setattr(deckmod, "load_deck", lambda path=None: _load(p))
    monkeypatch.setattr(deckmod, "read_quota", lambda path=None: {})
    a = resolve_role(Role.WORKER, "claude", _cfg("deck"), mode="agentic",
                     exhausted_providers=[Provider.ZHIPU])
    assert a.model == "kimi-k3"


def test_deck_mode_headroom_gates_by_DECK_provider_name(tmp_path, monkeypatch):
    """The gate compares Card.provider against headroom keys -- both deck names.
    moonshot at 95% must remove kimi-k3 even though the router calls it 'kimi'."""
    p = _deck_file(tmp_path, {"B3": [_card("kimi-k3", "moonshot", 0.05),
                                     _card("glm-5.3", "zhipu", 0.10)]})
    monkeypatch.setattr(deckmod, "load_deck", lambda path=None: _load(p))
    monkeypatch.setattr(deckmod, "read_quota", lambda path=None: {"moonshot": 0.95, "zhipu": 0.10})
    a = resolve_role(Role.WORKER, "claude", _cfg("deck"), mode="agentic")
    assert a.model == "glm-5.3"


def test_deck_mode_falls_through_to_legacy_when_band_is_empty(tmp_path, monkeypatch):
    p = _deck_file(tmp_path, {})  # week 0: every band unranked
    monkeypatch.setattr(deckmod, "load_deck", lambda path=None: _load(p))
    monkeypatch.setattr(deckmod, "read_quota", lambda path=None: {})
    a = resolve_role(Role.WORKER, "claude", _cfg("deck"), mode="agentic")
    assert a.model == "kimi"
    assert a.reason == "first eligible configured candidate"


def test_deck_mode_falls_through_when_deck_unreadable(monkeypatch):
    monkeypatch.setattr(deckmod, "load_deck", lambda path=None: {})
    a = resolve_role(Role.WORKER, "claude", _cfg("deck"), mode="agentic")
    assert a.model == "kimi"


def test_deck_anthropic_card_uses_agentic_rule(tmp_path, monkeypatch):
    """The three-way Anthropic rule is shared with the legacy walk, not re-implemented."""
    p = _deck_file(tmp_path, {"B5": [_card("claude-fable-5.1", "anthropic", 0.50, cap=72)]})
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
