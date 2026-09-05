#!/usr/bin/env python3
"""probe-band-winners.py — C12/SC13: probe every band winner through the live router.

Proves each band's leftmost ranked card is a route that actually works, in agentic
mode, BEFORE deck routing is enabled (C08 acts on this report; the probe itself
only reports). Every deck winner is an untested route until something calls it —
the design session only ever exercised kimi through this router, and the deck
routes away from kimi.

Exact-card routing, never a role: while routing_mode is legacy a role resolves
through [roles] and would test the legacy list, then be recorded as proof the
DECK's winners are alive. zhipu/openai cards reach their exact model via
route()+model_override on the tier that owns the backend; xai/moonshot/anthropic
cards have no owning tier and go straight to _route_agentic. deepseek has no
agentic adapter at all (backends.py has no call_deepseek_agentic), so a deepseek
card is reported as a failed probe rather than silently skipped.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from mcp_brain_router.config import Config  # noqa: E402
from mcp_brain_router.deck import BANDS, DECK_TO_ROUTER, load_deck  # noqa: E402
from mcp_brain_router.router import (  # noqa: E402
    Complexity,
    RouteResult,
    _route_agentic,
    route,
)

PROMPT = "Reply with exactly: ALIVE"
TIMEOUT_S = 120  # a hang is a failure, not a wait; one design-session call took 534 s
RUN_ID = "probe-band-winners"

# Deck provider name -> tier whose route() call lands on the card's backend with
# model_override pinned to the exact card (zhipu -> glm, openai -> codex).
ROUTE_BY_TIER = {"zhipu": Complexity.CODE, "openai": Complexity.ADVERSARIAL}
# Deck providers with no owning tier: dispatch straight to the agentic router by
# backend name. moonshot/openai here are DECK names; the backend labels are the
# router's (DECK_TO_ROUTER exists for exactly this translation).
AGENTIC_DIRECT = {"xai": "grok", "moonshot": "kimi", "anthropic": "anthropic-cli"}


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def say(args: argparse.Namespace, msg: str) -> None:
    # --json keeps stdout pure (the spec pipes it to jq); human lines go to stderr
    print(msg, file=sys.stderr if args.json else sys.stdout)


async def _fake_call(provider: str, card) -> RouteResult:
    """--fake stub: replaces ONLY the live provider call. The wait_for wrapper,
    timing, ok-check, probe.json write and exit-code logic all still run, so the
    synthetic proof exercises the whole loop with zero provider calls."""
    await asyncio.sleep(0.01)  # real coroutine, so wait_for/time.monotonic see it
    backend = AGENTIC_DIRECT.get(provider) or {
        "zhipu": "glm", "openai": "codex",
    }[provider]
    return RouteResult(
        content="ALIVE", model=card.router_model, backend=backend,
        complexity=None, headroom_used=False,
    )


async def probe_one(band: str, card, config: Config, cwd: str,
                    fake: bool, args: argparse.Namespace) -> dict:
    provider = card.provider  # deck name (moonshot/openai), never the router's
    row = {
        "band": band, "slug": card.slug, "router_model": card.router_model,
        "provider": provider, "backend": "none", "model": card.router_model,
        "exhausted": False, "elapsed_ms": None, "ok": False, "error": None,
    }
    if provider not in ROUTE_BY_TIER and provider not in AGENTIC_DIRECT:
        # AGENTIC_PROVIDERS = {zhipu, openai, xai, moonshot, anthropic}; anything
        # else (deepseek) has no agentic adapter. The exclusion is printed, never
        # silent, and counts as a FAILED probe — the band's winner is not alive
        # in agentic mode regardless of why.
        row["error"] = "no agentic adapter"
        say(args, f"  {band}: {card.slug} provider {provider!r} excluded from "
                  f"agentic contention (no adapter) -- FAILED probe")
        return row

    t0 = time.monotonic()
    try:
        if fake:
            result = await asyncio.wait_for(
                _fake_call(provider, card), TIMEOUT_S)
        elif provider in ROUTE_BY_TIER:
            result = await asyncio.wait_for(
                route(ROUTE_BY_TIER[provider], PROMPT,
                      model_override=card.router_model,
                      config=config, mode="agentic", cwd=cwd),
                TIMEOUT_S)
        else:
            result = await asyncio.wait_for(
                _route_agentic(AGENTIC_DIRECT[provider], PROMPT,
                               card.router_model, config, cwd),
                TIMEOUT_S)
    except asyncio.TimeoutError:
        row["elapsed_ms"] = int((time.monotonic() - t0) * 1000)
        row["error"] = f"timeout after {TIMEOUT_S}s"
        say(args, f"  {band}: {card.slug} TIMED OUT after {TIMEOUT_S}s")
        return row
    except Exception as e:  # noqa: BLE001 -- any backend/credential failure is a failed probe
        row["elapsed_ms"] = int((time.monotonic() - t0) * 1000)
        row["error"] = f"{type(e).__name__}: {e}"
        say(args, f"  {band}: {card.slug} ERROR {row['error']}")
        return row

    row["elapsed_ms"] = int((time.monotonic() - t0) * 1000)
    row["backend"] = result.backend
    row["model"] = result.model or card.router_model
    row["exhausted"] = bool(result.exhausted)
    row["ok"] = "ALIVE" in (result.content or "")
    return row


async def probe_all(deck: dict, config: Config, fake: bool,
                    args: argparse.Namespace) -> list[dict]:
    rows: list[dict] = []
    # A fresh temp cwd per run: the delegate lane depends on the worker holding
    # Read,Edit,Write,Bash in the caller's directory, so a probe must run in a
    # real (throwaway) directory, not the server's or this script's.
    with tempfile.TemporaryDirectory(prefix="probe-band-") as tmp:
        for band in BANDS:
            card = deck[band].ranked[0]
            say(args, f"probing {band} winner {card.slug} "
                      f"(provider={card.provider} model={card.router_model})")
            rows.append(await probe_one(band, card, config, tmp, fake, args))
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--json", action="store_true",
                    help="print the rows array as JSON on stdout")
    ap.add_argument("--deck", metavar="PATH",
                    help="probe winners from this deck.json instead of the live one")
    ap.add_argument("--fake", action="store_true",
                    help="stub the provider call (returns ALIVE); zero provider calls")
    ap.add_argument("--out", metavar="PATH",
                    help="write probe.json here instead of the default state path")
    args = ap.parse_args()

    # Stamps every record the run emits so bench tooling can filter it out.
    os.environ["BRAIN_ROUTER_BENCH_RUN_ID"] = RUN_ID

    config = Config.load()
    mode = getattr(config, "routing_mode", "legacy")
    say(args, f"routing_mode = {mode} -- the probe bypasses it by design: each band "
              f"winner is called as an exact card (model_override / _route_agentic), "
              f"never through a role, so the legacy [roles] walk is not what gets tested")

    deck_path = (Path(args.deck).expanduser() if args.deck
                 else Path.home() / ".local" / "state" / "brain-router" / "deck.json")
    sha_before = sha256_file(deck_path)
    deck = load_deck(args.deck)

    missing = [b for b in BANDS if not (deck.get(b) and deck[b].ranked)]
    if missing:
        for b in missing:
            say(args, f"band {b}: no ranked card in {deck_path}")
        say(args, f"no ranked winner in: {', '.join(missing)} -- on week 0 this is "
                  f"the EXPECTED result (deck built, costs not yet measured); "
                  f"nothing was probed and probe.json was not written")
        return 2

    rows = asyncio.run(probe_all(deck, config, args.fake, args))

    # The probe reports; C08 acts. It must never be the thing that changed the deck.
    sha_after = sha256_file(deck_path)
    say(args, f"deck.json sha256 before: {sha_before}")
    say(args, f"deck.json sha256 after:  {sha_after}")
    assert sha_before == sha_after, "probe must not modify deck.json"

    out_path = (Path(args.out).expanduser() if args.out
                else Path.home() / ".local" / "state" / "brain-router" / "probe.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(
        {"ts_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
         "run_id": RUN_ID, "fake": args.fake, "rows": rows},
        indent=2) + "\n")
    say(args, f"wrote {out_path}")

    for r in rows:
        say(args, f"  {r['band']} {r['slug']} backend={r['backend']} "
                  f"exhausted={r['exhausted']} {r['elapsed_ms']}ms "
                  f"ok={r['ok']} error={r['error']}")

    if args.json:
        print(json.dumps(rows, indent=2))

    failed = [r["band"] for r in rows if not r["ok"]]
    if failed:
        say(args, f"SC13 FAIL: bands whose winner did not answer ALIVE: "
                  f"{', '.join(failed)} -- routing_mode stays legacy")
        return 1
    say(args, "SC13 PASS: every band winner answered ALIVE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
