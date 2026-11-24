"""Quota-aware order of the normal walk: a pure reorder, a never-raising reader, the wiring.

Fixture quota dicts feed reorder(); the walk tests fake the backends at the CLI edge
(same seam as test_model_failover) and patch quota.read.
"""
from __future__ import annotations

import json
import os
import time
from unittest.mock import patch

import pytest

from mcp_brain_router import backends, quota, server
from mcp_brain_router.config import DEFAULT_ROLES, ROLES, Config

NOW = 1_000_000.0
DAY = 86400


def Q(weekly=None, five=None, left_days=None, now=NOW):
    at = None if left_days is None else now + left_days * DAY
    return {"five_hour": five, "weekly": weekly, "age_s": 5.0, "weekly_reset_at": at,
            "weekly_reset_s": None if at is None else at - now}


def models(role, quota_dict, name=None):
    new, why = quota.reorder(ROLES[name or role].slots, quota_dict, role, NOW)
    return [s.model for s in new], why


def test_ninety_moves_to_end():
    for q in (Q(weekly=93), Q(five=90, weekly=10)):
        order, why = models("thinker", {"anthropic": q})
        assert order == ["gemini-4-neon", "gemini-4-carbon", "grok-4.7", "claude-fable-5-2", "claude-opus-5-5"]
        assert why.startswith("anthropic moved last:")
    # 89 is not enough
    assert models("thinker", {"anthropic": Q(weekly=89, five=89)}) == ([s.model for s in ROLES["thinker"].slots], None)


def test_pace():
    # anthropic: 80% used with 4 of 7 days left (43% elapsed) -> ahead; antigravity on pace
    q = {"anthropic": Q(weekly=80, left_days=4), "antigravity": Q(weekly=10, left_days=6)}
    order, why = models("coder", q)
    assert order == ["gemini-4-neon", "claude-sonnet-5-5", "claude-opus-5-5"]
    assert "anthropic yields first place" in why
    assert models("worker", q, name="processor")[0][0] == "gemini-4-hydrogen"  # first already antigravity: no change
    for role in ("thinker", "adversary", "writer", "qa"):
        assert models(role, q)[1] is None  # below 90: capability order stands
    # within 10 points of pace: no move; unknown reset: rule does not apply
    assert models("coder", {"anthropic": Q(weekly=50, left_days=4), "antigravity": Q(weekly=1, left_days=6)})[1] is None
    assert models("coder", {"anthropic": Q(weekly=80), "antigravity": Q(weekly=1, left_days=6)})[1] is None
    # nothing to yield to: every other provider is also ahead, or has no reading
    assert models("coder", {"anthropic": Q(weekly=80, left_days=4)})[1] is None
    # local entries are never moved to the front
    simple, _ = models("simple", {"antigravity": Q(weekly=80, left_days=4), "anthropic": Q(weekly=5, left_days=6)})
    assert simple[0] == "apple-fm"  # apple-fm stays first; no quota, no move


def test_pace_simple_moves_a_remote_when_it_leads():
    order, _ = models("explorer", {"anthropic": Q(weekly=80, left_days=4), "antigravity": Q(weekly=1, left_days=6)})
    assert order == ["claude-haiku-5-5", "claude-sonnet-5-5"]  # explorer has no other provider: unchanged
    order, why = models("processor", {"antigravity": Q(weekly=80, left_days=4), "anthropic": Q(weekly=5, left_days=6)})
    assert order[0] == "claude-sonnet-5-5" and "antigravity yields" in why


def test_never_removes():
    everything_hot = {p: Q(weekly=99, five=99, left_days=3) for p in ("anthropic", "antigravity", "grok")}
    for name, role in ROLES.items():
        for q in (everything_hot, {"anthropic": Q(weekly=95)}, {"grok": Q(five=91)}):
            new, _ = quota.reorder(role.slots, q, name, NOW)
            assert sorted(map(id, new)) == sorted(map(id, role.slots))
    assert quota.reorder([], {"anthropic": Q(weekly=99)}, "coder", NOW) == ([], None)


def test_stale_or_missing(tmp_path):
    assert models("thinker", {}) == ([s.model for s in ROLES["thinker"].slots], None)
    # read(): missing dir, stale file, fresh file
    assert quota.read(tmp_path / "nope", now=NOW) == {}
    f = tmp_path / ".anthropic-quota.cache"
    f.write_text(json.dumps({"buckets": [{"key": "wk", "pct": 97, "resets_at": NOW + DAY}]}))
    os.utime(f, (NOW - 3600, NOW - 3600))
    assert quota.read(tmp_path, now=NOW) == {}  # 1 h old > 30 min
    assert "anthropic" in quota.read(tmp_path, now=NOW, stale_s=7200)
    os.utime(f, (NOW - 60, NOW - 60))
    assert quota.read(tmp_path, now=NOW)["anthropic"]["weekly"] == 97
    # a bucket whose reset already passed is dropped
    f.write_text(json.dumps({"buckets": [{"key": "wk", "pct": 97, "resets_at": NOW - 5}]}))
    os.utime(f, (NOW - 60, NOW - 60))
    assert quota.read(tmp_path, now=NOW) == {}


def test_reader_never_raises(tmp_path):
    for body in ("", "not json", "[]", "{}", '{"buckets": 3}', '{"buckets": [{"key": "wk"}]}',
                 '{"weekly": null}', '[{"w": "wk", "used": "x", "reset": 5}]', "\x00\xff"):
        for name in (".anthropic-quota.cache", ".antigravity-quota.cache", ".grok-quota.cache"):
            (tmp_path / name).write_text(body, errors="ignore")
        assert quota.read(tmp_path, now=NOW) == {}
    (tmp_path / ".grok-quota.cache").unlink()
    (tmp_path / ".grok-quota.cache").mkdir()  # a directory where a file should be
    assert quota.read(tmp_path, now=NOW) == {}
    assert quota.read(object(), now=NOW) == {}  # even a nonsense path


def test_reader_parses_real_shaped_files(tmp_path):
    now = 1795561259.0  # an instant inside the sample's windows
    (tmp_path / ".anthropic-quota.cache").write_text(json.dumps(
        {"plan": "max_20x", "source": "stdin", "updated_at": 1795561256, "buckets": [
            {"key": "5h", "pct": 84, "resets_at": 1795570200}, {"key": "wk", "pct": 55, "resets_at": 1795928400}]}))
    (tmp_path / ".antigravity-quota.cache").write_text(
        '[{"w":"5h","used":13,"reset":"2026-11-25T02:08:01Z"},{"w":"wk","used":3,"reset":"2026-12-01T21:08:01Z"}]\n')
    (tmp_path / ".grok-quota.cache").write_text(json.dumps({"weekly": {"config": {
        "currentPeriod": {"type": "USAGE_PERIOD_TYPE_WEEKLY", "start": "2026-11-20T05:28:04.461+00:00",
                          "end": "2026-11-27T05:28:04.461+00:00"},
        "creditUsagePercent": 58.0, "productUsage": [{"product": "GrokBot", "usagePercent": 44.0}]}},
        "monthly": {"config": {}}}))
    for p in tmp_path.iterdir():
        os.utime(p, (now - 2, now - 2))
    got = quota.read(tmp_path, now=now)
    assert got["anthropic"]["five_hour"] == 84 and got["anthropic"]["weekly"] == 55
    assert got["anthropic"]["weekly_reset_s"] == 1795928400 - now
    assert got["antigravity"]["five_hour"] == 13 and got["antigravity"]["weekly"] == 3
    assert got["grok"]["weekly"] == 58 and got["grok"]["five_hour"] is None
    assert 0 < got["grok"]["weekly_reset_s"] < 7 * DAY and got["grok"]["age_s"] == 2


# --- the walk -----------------------------------------------------------------------------------

def _cfg() -> Config:
    return Config(grok_enabled=True, fm_host="fm-host", ollama_host="http://127.0.0.1:11435",
                  roles={r: list(o) for r, o in DEFAULT_ROLES.items()})


class Calls:
    def __init__(self):
        self.models = []

    def _go(self, prompt, model):
        self.models.append(model)
        return {"content": "answer", "usage": None}

    async def walk(self, role, quota_dict, orchestrator="grok", data_class=None):
        patches = (patch.object(backends, "call_antigravity", lambda p, m, effort=None: self._go(p, m)),
                   patch.object(backends, "call_anthropic_agentic", lambda p, m, cwd=None, effort=None: self._go(p, m)),
                   patch.object(backends, "call_grok", lambda p, m, effort=None: self._go(p, m)),
                   patch.object(quota, "read", lambda: quota_dict),
                   patch.object(server, "_load_config", return_value=_cfg()))
        for p in patches:
            p.start()
        try:
            return await server._delegate_role_impl(role, "the task", orchestrator, cwd="/tmp", data_class=data_class)
        finally:
            for p in patches:
                p.stop()


@pytest.mark.asyncio
async def test_reports_reorder():
    c = Calls()
    out = await c.walk("thinker", {"anthropic": Q(weekly=93)})
    assert out["model"].startswith("gemini-4-neon")
    assert out["quota_note"] == "anthropic moved last: weekly 93% used"
    out = await Calls().walk("thinker", {"anthropic": Q(weekly=40)})  # nothing changed: no note
    assert "quota_note" not in out and out["model"] == "claude-fable-5-2"
    # a failed walk carries the note too
    from mcp_brain_router.backends import BackendError

    def boom(*a, **kw):
        raise BackendError("down", backend="x", failure_kind="process_error")
    with patch.object(backends, "call_antigravity", boom), patch.object(backends, "call_anthropic_agentic", boom), \
            patch.object(backends, "call_grok", boom), patch.object(quota, "read", lambda: {"anthropic": Q(weekly=95)}), \
            patch.object(server, "_load_config", return_value=_cfg()):
        out = await server._delegate_role_impl("thinker", "x", "grok", cwd="/tmp")
    assert out["exhausted"] and "anthropic moved last" in out["quota_note"]


@pytest.mark.asyncio
async def test_walk_pace_picks_the_other_provider_for_volume():
    c = Calls()
    t = time.time()  # the walk reads the real clock
    out = await c.walk("coder", {"anthropic": Q(weekly=80, left_days=4, now=t), "antigravity": Q(weekly=10, left_days=6, now=t)})
    assert c.models[0].startswith("gemini-4-neon") and "anthropic yields first place" in out["quota_note"]


@pytest.mark.asyncio
async def test_adversary_still_skips_caller():
    c = Calls()
    # grok is the only provider under 90, so it would lead the list; the caller is grok, so it must be skipped
    hot = {"anthropic": Q(weekly=95), "antigravity": Q(five=92)}
    out = await c.walk("adversary", hot, orchestrator="grok")
    assert out["provider"] != "grok" and all(not m.startswith("grok") for m in c.models)
    # class orange still reaches Claude only after a reorder
    c = Calls()
    out = await c.walk("thinker", {"anthropic": Q(weekly=95)}, data_class="orange")
    assert out["provider"] == "anthropic" and all(m.startswith("claude") for m in c.models)
