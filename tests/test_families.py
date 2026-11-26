"""004: one model, many routes. Floor by job and tier; same-family routes ordered by live
headroom, then cost (Antigravity, Grok, Claude). Mohan, 2026-11-26."""
from __future__ import annotations

from mcp_brain_router import quota
from mcp_brain_router.config import COST_RANK, DEFAULT_ROLES, FAMILIES, ROLES, Slot, _expand

NOW = 1_000_000.0
DAY = 86400


def Q(weekly, left_days):
    at = NOW + left_days * DAY
    return {"five_hour": None, "weekly": weekly, "age_s": 5.0, "weekly_reset_at": at,
            "weekly_reset_s": at - NOW}


def test_family_expands_to_every_route_in_cost_order():
    out = _expand(("claude-sonnet-5-5", "high"))
    assert [s.id for s in out] == ["agy-claude-sonnet-5-5-high", "claude-sonnet-5-5"]
    assert {s.family for s in out} == {"claude-sonnet-5-5"}
    assert [COST_RANK[s.provider] for s in out] == sorted(COST_RANK[s.provider] for s in out)


def test_floor_never_bends_to_a_capped_route():
    for level in ("xhigh", "max"):
        assert [s.id for s in _expand(("claude-opus-5-5", level))] == ["claude-opus-5-5"]


def test_no_role_slot_runs_below_its_floor_on_a_capped_route():
    caps = {m: cap for routes in FAMILIES.values() for m, cap in routes}
    for r in ROLES.values():
        for s in r.slots:
            cap = caps.get(s.model)
            assert cap is None or s.level in ("low", "medium", "high"), (r.name, s)


def test_floors_follow_mohans_decision():
    for role in ("coder", "processor", "qa"):
        assert any(m.startswith("agy-claude-") for m in DEFAULT_ROLES[role]), role
    for role in ("writer", "thinker", "adversary"):
        assert not any(m.startswith("agy-") for m in DEFAULT_ROLES[role]), role


def run(model="claude-sonnet-5-5"):
    return _expand((model, "high"))


def test_ultra_first_while_everyone_is_on_pace():
    q = {"antigravity": Q(50, 3.5), "anthropic": Q(50, 3.5)}
    out, notes = quota.order_families(run(), q, NOW, COST_RANK)
    assert out[0].provider == "antigravity" and notes == []


def test_more_headroom_wins_over_cost():
    # Antigravity well ahead of pace, Claude well under: Claude goes first.
    q = {"antigravity": Q(90 - 1, 6), "anthropic": Q(10, 1)}
    out, notes = quota.order_families(run(), q, NOW, COST_RANK)
    assert out[0].provider == "anthropic" and notes and "anthropic before antigravity" in notes[0]


def test_missing_reading_keeps_cost_order():
    out, notes = quota.order_families(run(), {"anthropic": Q(5, 1)}, NOW, COST_RANK)
    # anthropic has room (band -1), antigravity has no reading (band 0): room wins
    assert out[0].provider == "anthropic"
    out, notes = quota.order_families(run(), {}, NOW, COST_RANK)
    assert out[0].provider == "antigravity" and notes == []


def test_reordering_never_crosses_a_family():
    slots = [*run("claude-sonnet-5-5"), Slot("gemini-4-neon", "high"), *run("claude-opus-5-5")]
    q = {"antigravity": Q(95 - 6, 6), "anthropic": Q(5, 1)}
    out, _ = quota.order_families(slots, q, NOW, COST_RANK)
    assert [s.family or s.model for s in out] == [s.family or s.model for s in slots]


def test_reorder_reports_the_family_note():
    q = {"antigravity": Q(80, 6), "anthropic": Q(10, 1)}
    new, why = quota.reorder(ROLES["coder"].slots, q, "coder", NOW)
    assert new[0].id == "claude-sonnet-5-5" and why and "claude-sonnet-5-5:" in why
