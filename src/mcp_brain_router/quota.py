"""Quota-aware order for the normal role walk.

read() parses the status line's own cache files (~/.claude/state/.*-quota.cache):
plain file reads, no subprocess, no network, never raises. reorder() is pure.
The deck path has its own gate (deck.py) and does not use this module.
"""
from __future__ import annotations

import json
import math
import os
import time
from datetime import datetime, timezone
from pathlib import Path

STATE_DIR = Path(os.environ.get("BRAIN_ROUTER_QUOTA_DIR") or Path.home() / ".claude" / "state")
# A reading older than this (file mtime) is treated as missing: a dead poller must not steer routing.
STALE_S = float(os.environ.get("BRAIN_ROUTER_QUOTA_STALE_S") or 30 * 60)
HOT_PCT = 90.0       # at or above this used %, a provider goes last
PACE_SLACK = 10.0    # points ahead of the week's elapsed share before a provider yields first place
WEEK_S = 604800
VOLUME_ROLES = {"coder", "processor", "explorer", "simple", "worker"}


_BAD = object()


def _num(v):
    return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) else None


def _iso(v) -> float | None:
    if not isinstance(v, str):
        return None
    try:
        dt = datetime.fromisoformat(v.replace("Z", "+00:00"))
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    ts = dt.timestamp()
    return ts if math.isfinite(ts) else None


def _reset(raw, parse):
    if raw is None:
        return None
    val = parse(raw)
    return _BAD if val is None else val


def _anthropic(d):  # {"buckets": [{"key": "5h"|"wk", "pct": 84, "resets_at": epoch}]}
    return {b["key"]: (_num(b.get("pct")), _reset(b.get("resets_at"), _num)) for b in d["buckets"]}


def _antigravity(d):  # [{"w": "5h"|"wk", "used": 13, "reset": "2026-11-25T02:08:01Z"}]
    return {b["w"]: (_num(b.get("used")), _reset(b.get("reset"), _iso)) for b in d}


def _grok(d):  # {"weekly": {"config": {"creditUsagePercent": 58.0, "currentPeriod": {"end": iso}}}}
    c = d["weekly"]["config"]
    return {"wk": (_num(c.get("creditUsagePercent")), _reset(c["currentPeriod"].get("end"), _iso))}


_FILES = {"anthropic": (".anthropic-quota.cache", _anthropic),
          "antigravity": (".antigravity-quota.cache", _antigravity),
          "grok": (".grok-quota.cache", _grok)}


def read(state_dir=None, now: float | None = None, stale_s: float | None = None) -> dict:
    """{provider: {five_hour, weekly, weekly_reset_s, weekly_reset_at, age_s}}; a provider whose
    file is missing, unreadable, malformed or stale has no entry. A bucket whose reset time has
    passed is dropped (its percent is from before the reset)."""
    now = time.time() if now is None else now
    stale_s = STALE_S if stale_s is None else stale_s
    out = {}
    for provider, (name, parse) in _FILES.items():
        try:
            path = Path(state_dir or STATE_DIR) / name
            age = now - path.stat().st_mtime
            if age > stale_s:
                continue
            buckets = {k: v for k, v in parse(json.loads(path.read_text())).items()
                       if v[0] is not None and (v[1] is None or (v[1] is not _BAD and v[1] > now))}
        except Exception:  # noqa: BLE001 — a quota reading must never break a call
            continue
        wk, h5 = buckets.get("wk", (None, None)), buckets.get("5h", (None, None))
        if wk[0] is None and h5[0] is None:
            continue
        out[provider] = {"five_hour": h5[0], "weekly": wk[0], "age_s": age,
                         "weekly_reset_at": wk[1],
                         "weekly_reset_s": None if wk[1] is None else wk[1] - now}
    return out


def _hot(q) -> str | None:
    if not isinstance(q, dict):
        return None
    for label, key in (("weekly", "weekly"), ("5h", "five_hour")):
        v = _num(q.get(key))
        if v is not None and v >= HOT_PCT:
            return f"{label} {v:.0f}% used"
    return None


def _elapsed(q, now) -> float | None:
    if not isinstance(q, dict):
        return None
    at, now_f = _num(q.get("weekly_reset_at")), _num(now)
    left = None if at is None or now_f is None else at - now_f
    return None if left is None or not 0 < left <= WEEK_S else 100 * (1 - left / WEEK_S)


def _ahead(q, now) -> bool:
    el = _elapsed(q, now)
    wk = _num(q.get("weekly")) if isinstance(q, dict) else None
    return el is not None and wk is not None and wk > el + PACE_SLACK


def reorder(slots, quota: dict, role: str, now: float):
    """Pure. slots: items with a .provider. Returns (new_slots, reason | None). Never removes a slot."""
    saved = []
    try:
        saved = list(slots)
        if not saved or not isinstance(quota, dict) or not quota:
            return saved, None
        hot = {p: r for p, q in quota.items() if p in _FILES and (r := _hot(q))}
        pinned = {i: s for i, s in enumerate(saved) if s.provider not in _FILES}
        remote = [s for s in saved if s.provider in _FILES]
        out_remote = [s for s in remote if s.provider not in hot] + [s for s in remote if s.provider in hot]
        notes = []
        for p, why in hot.items():
            # moved only if some other non-local provider's slot sat after one of p's
            seen = False
            for s in remote:
                seen = seen or s.provider == p
                if seen and s.provider not in hot:
                    notes.append(f"{p} moved last: {why}")
                    break
        if role in VOLUME_ROLES and out_remote and out_remote[0].provider in quota and _ahead(quota[out_remote[0].provider], now):
            # shortcut: target needs a finite weekly reading and known elapsed week, not hot and not ahead.
            for i, s in enumerate(out_remote[1:], 1):
                q = quota.get(s.provider)
                wk = _num(q.get("weekly")) if isinstance(q, dict) else None
                el = _elapsed(q, now)
                if wk is not None and el is not None and s.provider not in hot and wk <= el + PACE_SLACK:
                    p0, q0 = out_remote[0].provider, quota[out_remote[0].provider]
                    notes.append(f"{p0} yields first place: weekly {_num(q0.get('weekly')):.0f}% used, "
                                 f"{_elapsed(q0, now):.0f}% of week gone")
                    out_remote.insert(0, out_remote.pop(i))
                    break
        it = iter(out_remote)
        out = [pinned[i] if i in pinned else next(it) for i in range(len(saved))]
        return out, "; ".join(notes) or None
    except Exception:  # noqa: BLE001 — reorder must never raise
        return saved, None
