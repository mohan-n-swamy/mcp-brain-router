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


def _read_one(path: Path, parse, now: float, stale_s: float) -> dict | None:
    try:
        age = now - path.stat().st_mtime
        if age > stale_s:
            return None
        buckets = {k: v for k, v in parse(json.loads(path.read_text())).items()
                   if v[0] is not None and (v[1] is None or (v[1] is not _BAD and v[1] > now))}
    except Exception:  # noqa: BLE001 — a quota reading must never break a call
        return None
    wk, h5 = buckets.get("wk", (None, None)), buckets.get("5h", (None, None))
    if wk[0] is None and h5[0] is None:
        return None
    return {"five_hour": h5[0], "weekly": wk[0], "age_s": age,
            "weekly_reset_at": wk[1],
            "weekly_reset_s": None if wk[1] is None else wk[1] - now}


def _pool_rank(q: dict, now: float) -> tuple:
    # Prefer non-hot pool first, then lower pace band, then lower 5h/weekly used%.
    band = _pace_band(q, now)
    hot = 1 if _hot(q) else 0
    return (hot, 99 if band is None else band, q.get("five_hour") or 0.0, q.get("weekly") or 0.0)


def read(state_dir=None, now: float | None = None, stale_s: float | None = None) -> dict:
    """{provider: {five_hour, weekly, weekly_reset_s, weekly_reset_at, age_s}}; a provider whose
    file is missing, unreadable, malformed or stale has no entry. A bucket whose reset time has
    passed is dropped (its percent is from before the reset).
    For anthropic, considers both Personal (.anthropic-quota.cache) and Orange (.anthropic-quota.oh.cache)
    when fresh, returning the healthier pool so anthropic is only marked hot when BOTH pools are hot."""
    now = time.time() if now is None else now
    stale_s = STALE_S if stale_s is None else stale_s
    try:
        base = Path(state_dir or STATE_DIR)
    except Exception:
        return {}
    out = {}
    for provider, (name, parse) in _FILES.items():
        candidates = []
        primary = _read_one(base / name, parse, now, stale_s)
        if primary is not None:
            candidates.append(primary)
        if provider == "anthropic":
            oh = _read_one(base / ".anthropic-quota.oh.cache", parse, now, stale_s)
            if oh is not None:
                candidates.append(oh)
        if not candidates:
            continue
        out[provider] = min(candidates, key=lambda q: _pool_rank(q, now))
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


def _pace_band(q, now):
    """-1 under pace by more than PACE_SLACK (room to spend), 0 on pace, 1 ahead of pace,
    2 hot (>= HOT_PCT weekly or 5h). None when there is no usable reading."""
    if _hot(q):
        return 2
    el = _elapsed(q, now)
    wk = _num(q.get("weekly")) if isinstance(q, dict) else None
    if el is None or wk is None:
        return None
    room = el - wk
    return -1 if room > PACE_SLACK else 1 if room < -PACE_SLACK else 0


def order_families(slots, quota: dict, now: float, cost_rank: dict):
    """004: inside each run of routes of one model family, put the route with the most
    live headroom first; cost_rank (Antigravity, Grok, Claude) breaks ties. Slots never
    cross a family boundary, so the role's model order (its quality) is untouched."""
    out, notes, i = list(slots), [], 0
    key_of = lambda s: (getattr(s, "family", None), s.level)
    while i < len(out):
        fam = getattr(out[i], "family", None)
        j = i + 1
        while fam and j < len(out) and key_of(out[j]) == key_of(out[i]):
            j += 1
        run = out[i:j]
        bands = [_pace_band(quota.get(s.provider), now) for s in run]
        # A route with no usable reading never moves past one that has a reading:
        # the run reorders only when every route in it is measured.
        if len(run) > 1 and None not in bands:
            def _room(s):
                q = quota.get(s.provider)
                if _hot(q):
                    return float("-inf")
                el = _elapsed(q, now)
                wk = _num(q.get("weekly")) if isinstance(q, dict) else None
                return float("-inf") if el is None or wk is None else el - wk
            rooms = {s.id: _room(s) for s in run}
            finite = [r for r in rooms.values() if math.isfinite(r)]
            wide_gap = len(finite) > 1 and (max(finite) - min(finite)) >= PACE_SLACK
            band_of = dict(zip((s.id for s in run), bands))
            if wide_gap:
                best = sorted(run, key=lambda s: (1 if band_of[s.id] == 2 else 0, -rooms[s.id], cost_rank.get(s.provider, 99)))
            else:
                best = sorted(run, key=lambda s: (band_of[s.id], cost_rank.get(s.provider, 99)))
            if best != run:
                k = next(n for n, (a, b) in enumerate(zip(best, run)) if a != b)
                notes.append(f"{fam}: {best[k].provider} before {run[k].provider} (live headroom)")
                out[i:j] = best
        i = j
    return out, notes


def reorder(slots, quota: dict, role: str, now: float):
    """Pure. slots: items with a .provider. Returns (new_slots, reason | None). Never removes a slot."""
    saved = []
    try:
        saved = list(slots)
        if not saved or not isinstance(quota, dict) or not quota:
            return saved, None
        from .config import COST_RANK
        base, fam_notes = order_families(saved, quota, now, COST_RANK)
        hot = {p: r for p, q in quota.items() if p in _FILES and (r := _hot(q))}
        pinned = {i: s for i, s in enumerate(base) if s.provider not in _FILES}
        remote = [s for s in base if s.provider in _FILES]
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
        out = [pinned[i] if i in pinned else next(it) for i in range(len(base))]
        if out == saved:
            return saved, None  # nothing moved: no note
        return out, "; ".join(fam_notes + notes) or None
    except Exception:  # noqa: BLE001 — reorder must never raise
        return saved, None
