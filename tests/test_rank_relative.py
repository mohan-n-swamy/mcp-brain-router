"""ROUTER (2026-10-01): best result first, lowest cost second (Mohan).

A card ranks in a band only if it was measured there (>= 3 trials, no P0 failure) and its
pass rate is within 10 points of the band's best measured card ("10 points ok"). Among those,
Mohan's cost order decides: Codex, then Grok, then Claude ("yes codex grok claude"). A band
with fewer than two measured cards has no real comparison and routes by the [roles] list.
"""
from __future__ import annotations

import json

from mcp_brain_router import deck as deckmod
from mcp_brain_router import server
from mcp_brain_router.config import Config, DEFAULT_ROLE_BANDS
from mcp_brain_router.deck import Card, build_bands
from mcp_brain_router.router import Provider, Role, resolve_role

FLOORS = {b: 0.0 for b in deckmod.BANDS}


def _m(pass_rate, trials=3, p0=True):
    return {"total_cost_usd": None, "p0_pass": p0, "size_class": "small", "cost_basis": "bench",
            "pass_rate": pass_rate, "trials": trials}


def _card(router_model, provider, band="B3", **m) -> dict:
    return {"slug": router_model.replace(".", "-"), "router_model": router_model, "provider": provider,
            "model": router_model, "effort": "base", "capability": 60.0, "price_blended": 1.0,
            "total_cost_usd": None, "size_class": "small", "p0_pass": None,
            "by_band": {band: _m(**m)} if m else {}}


CODEX, GROK, CLAUDE = ("gpt-5.6-terra", "openai"), ("grok-4.5", "xai"), ("claude-sonnet-5", "anthropic")


def _b3(rows):
    return {r.band: r for r in build_bands([Card(**c) for c in rows], floors=FLOORS)}["B3"]


def test_worse_loses_to_the_reference():
    row = _b3([_card(*CODEX, pass_rate=0.80), _card(*GROK, pass_rate=0.90), _card(*CLAUDE, pass_rate=0.95)])
    assert [c.router_model for c in row.ranked] == ["grok-4.5", "claude-sonnet-5"]
    assert "gpt-5.6-terra" in [c.router_model for c in row.unranked]


def test_worse_loses_even_when_it_is_the_only_cheap_card():
    row = _b3([_card(*CODEX, pass_rate=0.60), _card(*CLAUDE, pass_rate=0.95)])
    assert [c.router_model for c in row.ranked] == ["claude-sonnet-5"]


def test_cost_order_decides_among_eligible_cards():
    row = _b3([_card(*CLAUDE, pass_rate=0.95), _card(*GROK, pass_rate=0.92), _card(*CODEX, pass_rate=0.90)])
    assert [c.router_model for c in row.ranked] == ["gpt-5.6-terra", "grok-4.5", "claude-sonnet-5"]


def test_exactly_ten_points_is_eligible():
    row = _b3([_card(*CODEX, pass_rate=0.85), _card(*CLAUDE, pass_rate=0.95)])
    assert [c.router_model for c in row.ranked] == ["gpt-5.6-terra", "claude-sonnet-5"]


def test_unlisted_provider_cannot_rank_or_raise_the_bar():
    row = _b3([_card("glm-5.3", "zhipu", pass_rate=1.0), _card(*CODEX, pass_rate=0.88),
               _card(*CLAUDE, pass_rate=0.95)])
    assert [c.router_model for c in row.ranked] == ["gpt-5.6-terra", "claude-sonnet-5"]
    assert "glm-5.3" in [c.router_model for c in row.unranked]


def test_thin_or_failed_measurement_cannot_rank():
    row = _b3([_card(*CODEX, pass_rate=0.99, trials=2), _card(*GROK, pass_rate=0.99, p0=False),
               _card(*CLAUDE, pass_rate=0.95), _card("claude-haiku-4-5-20251001", "anthropic")])
    assert [c.router_model for c in row.ranked] == ["claude-sonnet-5"]


# ---------------------------------------------------------------- run time

def _cfg():
    return Config(roles={"worker": ["claude-sonnet-5", "grok-4.5", "gpt-5.6-terra"],
                         "simple": ["claude-haiku-4-5-20251001", "grok-4.5"],
                         "adversary": ["gpt-6-astra", "claude-opus-5-5", "grok-4.5"],
                         "thinker": ["gpt-6-astra", "claude-opus-5-5", "grok-4.5"]},
                  routing_mode="deck", role_bands=dict(DEFAULT_ROLE_BANDS), codex_enabled=True,
                  grok_enabled=True)


def _deck(tmp_path, monkeypatch, ranked, unranked=(), quota=None):
    doc = {"bands": [{"band": b, "floor": 0.0, "floor_status": "ESTIMATE",
                      "ranked": list(ranked) if b == "B3" else [],
                      "unranked": list(unranked) if b == "B3" else []} for b in deckmod.BANDS]}
    p = tmp_path / "deck.json"
    p.write_text(json.dumps(doc))
    real = deckmod.load_deck.__wrapped__ if hasattr(deckmod.load_deck, "__wrapped__") else deckmod.load_deck
    monkeypatch.setattr(deckmod, "load_deck", lambda path=None: real(p))
    monkeypatch.setattr(deckmod, "read_quota", lambda path=None: quota or {})
    monkeypatch.setattr(deckmod, "read_daily_calls", lambda path=None, today=None: {})


def test_deck_routes_the_eligible_cheapest(tmp_path, monkeypatch):
    _deck(tmp_path, monkeypatch, [_card(*CODEX, pass_rate=0.90), _card(*CLAUDE, pass_rate=0.95)])
    a = resolve_role(Role.WORKER, "claude", _cfg(), mode="agentic")
    assert a.model == "gpt-5.6-terra"
    assert a.reason.startswith("deck B3: eligible-cheapest")


def test_one_card_band_uses_the_roles_list(tmp_path, monkeypatch):
    _deck(tmp_path, monkeypatch, [_card(*GROK, pass_rate=0.90)])
    a = resolve_role(Role.WORKER, "claude", _cfg(), mode="agentic")
    assert a.model == "claude-sonnet-5"
    assert "reference-thin-evidence" in a.reason


def test_stale_week0_deck_is_inert(tmp_path, monkeypatch):
    week0 = [dict(_card("glm-5.3", "zhipu"), total_cost_usd=0.10, p0_pass=True),
             dict(_card("kimi-k3", "moonshot"), total_cost_usd=0.13, p0_pass=True)]
    _deck(tmp_path, monkeypatch, week0)
    a = resolve_role(Role.WORKER, "claude", _cfg(), mode="agentic")
    assert a.model == "claude-sonnet-5"
    assert "reference-thin-evidence" in a.reason


def test_quota_fallback_logged_in_the_reason(tmp_path, monkeypatch):
    _deck(tmp_path, monkeypatch, [_card(*CODEX, pass_rate=0.90), _card(*CLAUDE, pass_rate=0.95)],
          quota={"openai": 0.95, "anthropic": 0.95})
    a = resolve_role(Role.WORKER, "claude", _cfg(), mode="agentic")
    assert a.provider is Provider.XAI
    assert "fallback-quota" in a.reason


def test_quota_fallback_logged_in_the_audit_line(tmp_path, monkeypatch):
    log = tmp_path / "audit.jsonl"
    monkeypatch.setattr(server, "_DELEGATION_LOG", log)
    server._log_delegation({"backend": "grok", "reason": "fallback-quota (band B3 gated)"}, 1)
    assert json.loads(log.read_text())["route_reason"] == "fallback-quota (band B3 gated)"
