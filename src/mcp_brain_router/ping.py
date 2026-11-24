"""Per-model liveness pings for the role walk.

A model that failed with quota or a timeout gets its sibling pinged (one word,
short cap) before the real call. Results sit in a small JSON file, per model id,
for a short lifetime, so a dead model is not re-tried on every call.
"""
import json
import os
import tempfile
import time
from pathlib import Path

PROMPT = "Reply with the single word OK."
CACHE = Path.home() / ".local" / "state" / "brain-router-ping.json"  # same dir as the delegation log
TTL_S = float(os.getenv("BRAIN_ROUTER_PING_TTL_S", 300))
CAP_S = float(os.getenv("BRAIN_ROUTER_PING_CAP_S", 30))
SLOW_CAP_S = 45  # Carbon High needs about 32 s on a good day
RESERVE_S = float(os.getenv("BRAIN_ROUTER_RESERVE_S", 45))  # kept for the entries after a slow one
LOCAL_ENTRY_CAP_S = 15  # apple-fm and gpt-oss are single-host services that fail by hanging
# Failure kinds after which a sibling is pinged before it runs.
SUSPECT_KINDS = frozenset({"quota_exhausted", "transient_error", "timeout"})


def now() -> float:
    return time.time()


def cap_for(model: str) -> float:
    return SLOW_CAP_S if "carbon" in model else CAP_S


def _read() -> dict:
    try:
        data = json.loads(CACHE.read_text())
        return data if isinstance(data, dict) else {}
    except Exception:  # noqa: BLE001 — a missing or corrupt cache is an empty cache
        return {}


def fresh(model: str):
    """'ok' or 'down' if this model has an unexpired entry, else None."""
    entry = _read().get(model)
    try:
        if entry and now() - float(entry["ts"]) < TTL_S:
            return entry["state"]
    except (KeyError, TypeError, ValueError):
        pass
    return None


def record(model: str, ok: bool) -> None:
    """Never raises: the cache must not break a delegation."""
    try:
        data = _read()
        data[model] = {"state": "ok" if ok else "down", "ts": now()}
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        # A unique temp file: two threads or two servers must not share one.
        fd, tmp = tempfile.mkstemp(dir=CACHE.parent, prefix=CACHE.name + ".")
        with os.fdopen(fd, "w") as fh:
            fh.write(json.dumps(data))
        os.replace(tmp, CACHE)
    except Exception:  # noqa: BLE001
        pass
