"""Key pool for Jev `decide` calls (JEV-LOAD slice, 2026-09-27).

DECIDE-SWITCH (2026-09-28): a slot is (name, url, provider, key file), so the same
pool can list self-hosted laya-serve and TypeSafe Jev in any order with failover.
Plain key-file lists still work and become Jev slots named jev-0, jev-1, ...

Slots are the key files in `jev_key_files`, tried in list order. A slot that is
rate-limited, rejected, unavailable, timed out or unreachable hands the same call
to the next slot, and a `retry_after` puts that slot in cooldown for the life of
this process. Validation and response errors are raised at once: they would fail
the same way on every key. Each slot is tried at most once per call. Keys are read
afresh per call and never returned, logged or put into an error.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

from . import jev

FAILOVER_KINDS = frozenset(
    {"rate_limited", "authentication_error", "unavailable", "timeout", "network_error"}
)

# "<url>|<key file>" -> time.monotonic() before which the slot is skipped
_cooldown_until: dict[str, float] = {}


class SlotConfigError(Exception):
    """Every slot's key file is missing, unreadable, empty or not owner-only."""


@dataclass(frozen=True)
class Slot:
    name: str
    url: str
    provider: str
    key_file: str
    model: Optional[str] = None  # checkpoint this slot asks for; None = the caller's model


def slots_from_key_files(paths: list[str]) -> list[Slot]:
    return [Slot("jev-%d" % i, jev.JEV_ENDPOINT, "jev", p) for i, p in enumerate(paths)]


def normalize_backends(value: Any) -> list[Slot]:
    """`decide_backends`: a list of {name, url, provider, key_file}, names unique."""
    if value is None:
        return []
    if not isinstance(value, list):
        raise SlotConfigError("decide_backends must be a list")
    slots = []
    for item in value:
        if not isinstance(item, dict) or not all(
                isinstance(item.get(k), str) and item[k].strip() for k in ("name", "url", "provider", "key_file")):
            raise SlotConfigError("each decide_backends entry needs name, url, provider and key_file")
        if item["provider"] not in jev.PROVIDERS:
            raise SlotConfigError("decide_backends provider must be jev or laya")
        try:
            jev.validate_endpoint(item["url"])
        except jev.JevError:
            raise SlotConfigError("decide_backends url must be https, or http to loopback or a tailnet address") from None
        if "model" in item and not (isinstance(item["model"], str) and item["model"].strip()):
            raise SlotConfigError("decide_backends model must be a non-empty string when given")
        slots.append(Slot(item["name"], item["url"], item["provider"], item["key_file"], item.get("model")))
    if len({s.name for s in slots}) != len(slots):
        raise SlotConfigError("decide_backends names must be unique")
    return slots


def normalize_key_files(value: Any) -> list[str]:
    """`jev_key_files` must be a list of non-empty path strings."""
    if value is None:
        return []
    if not isinstance(value, list) or not all(isinstance(p, str) and p.strip() for p in value):
        raise SlotConfigError("jev_key_files must be a list of key-file paths")
    return list(value)


def read_key(path: str) -> str:
    p = Path(path).expanduser()
    try:
        if p.stat().st_mode & 0o077:
            raise SlotConfigError("key file must have owner-only permissions (chmod 600)")
        key = p.read_text().strip()
    except (OSError, UnicodeError):
        raise SlotConfigError("key file unreadable") from None
    if not key:
        raise SlotConfigError("key file is empty")
    return key


async def evaluate(
    state: Any,
    questions: dict[str, dict[str, Any]],
    model: str,
    slots: list,
    *,
    evaluate_fn: Optional[Callable] = None,
    clock: Callable[[], float] = time.monotonic,
) -> tuple[dict[str, Any], int, list[dict[str, Any]]]:
    """Return (result, serving slot index, attempts before it).

    Raises the slot's own JevError for a non-failover failure (with `key_slot` set),
    the single slot's error unchanged when there is only one slot, SlotConfigError when
    no slot has a usable key file, and JevError(failure_kind="all_slots_exhausted")
    otherwise.
    """
    call = evaluate_fn or jev.evaluate
    slots = [s if isinstance(s, Slot) else Slot("jev-%d" % i, jev.JEV_ENDPOINT, "jev", s) for i, s in enumerate(slots)]
    attempts: list[dict[str, Any]] = []
    last: Optional[jev.JevError] = None
    for slot, spec in enumerate(slots):
        path = "%s|%s" % (spec.url, spec.key_file)  # cooldown key
        if _cooldown_until.get(path, 0.0) > clock():
            attempts.append({"slot": slot, "outcome": "cooldown"})
            continue
        try:
            key = read_key(spec.key_file)
        except SlotConfigError:
            attempts.append({"slot": slot, "outcome": "configuration_error"})
            continue
        try:
            default_jev = spec.provider == "jev" and spec.url == jev.JEV_ENDPOINT
            extra = {} if default_jev else {"endpoint": spec.url, "provider": spec.provider}
            result = await call(state, questions, key, spec.model or model, **extra)
        except jev.JevError as exc:
            exc.key_slot = slot
            if exc.failure_kind not in FAILOVER_KINDS:
                raise
            if exc.retry_after:
                _cooldown_until[path] = clock() + exc.retry_after
            attempts.append({"slot": slot, "outcome": exc.failure_kind})
            last = exc
            continue
        _cooldown_until.pop(path, None)
        return result, slot, attempts

    if all(a["outcome"] == "configuration_error" for a in attempts):
        raise SlotConfigError("no usable Jev key file")
    if len(slots) == 1 and last is not None:
        raise last
    keys = ["%s|%s" % (s.url, s.key_file) for s in slots]
    waits = [_cooldown_until[p] - clock() for p in keys if _cooldown_until.get(p, 0.0) > clock()]
    summary = ", ".join("slot %d %s" % (a["slot"], a["outcome"]) for a in attempts)
    raise jev.JevError(
        "all %d Jev key slots failed: %s" % (len(slots), summary),
        failure_kind="all_slots_exhausted",
        retry_after=round(min(waits), 3) if waits else None,
    )
