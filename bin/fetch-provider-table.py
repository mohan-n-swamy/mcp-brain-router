#!/usr/bin/env python3
"""C01 — fetch the provider table into cards.json.

One writer, one file. Everything about how capability is obtained is sealed
behind this module: the router never sees Artificial Analysis, it sees the deck.

Two rules here are load-bearing and were each written after a real failure:

  R14  Reachability is DERIVED, never a hand-typed model list. A typed list
       produced 17 cards and silently lost Fable 5.1, the capability ceiling.
       We filter on `model_creator.slug`, a field the API supplies, so there is
       no pattern to maintain and a new model from a reachable creator appears
       on its own. The creator set itself is irreducibly a fact about which
       subscriptions exist -- that is a provider list, not a model list.

  R15  A null or zero price is NOT free access, it is a missing price. Such a
       card would sort leftmost in every band it clears, forever. Measured on
       the live feed: 227 of 643 models carry a zero blended price, so this
       exclusion covers a third of the corpus, not an edge case.

  R9   `artificial_analysis_agentic_index` DOES NOT EXIST in this feed --
       verified absent from all 643 models. Capability is composed locally from
       the agentic benchmarks that are present. If that field ever appears, STOP:
       the composition formula must be revisited rather than silently shadowed.
"""
from __future__ import annotations

import json
import os
import re
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

AA_URL = "https://artificialanalysis.ai/api/v2/data/llms/models"
AA_KEY_FILE = Path.home() / "secret/artificial-analysis"
OUT = Path.home() / ".local/state/brain-router/cards.json"

# Providers Mohan holds a subscription to, by the API's own creator slug.
# Not a model list: adding a model to any of these is picked up automatically.
#
# deepseek removed 2026-09-05 (Mohan, R15). Its cards led B1-B3 by list price and
# the rig cannot run them: no agentic adapter exists, the lane requires one, and
# the account is a $0.13 pay-as-you-go balance. Reachable-by-subscription is not
# executable-by-the-lane; this set now means the latter. The CREATOR_TO_PROVIDER
# entry below stays so headroom.json's deepseek key still resolves.
REACHABLE_CREATORS = {"anthropic", "zai", "kimi", "xai", "openai"}

# The router's own provider names, which `gated()` and headroom.json key on.
CREATOR_TO_PROVIDER = {
    "anthropic": "anthropic",
    "zai": "zhipu",
    "kimi": "moonshot",
    "xai": "xai",
    "openai": "openai",
    "deepseek": "deepseek",
}

# R9: composed locally because the agentic index is absent from the feed.
AGENTIC_BENCHMARKS = (
    "terminalbench_v2_1",
    "terminalbench_hard",
    "tau2",
    "tau_banking",
    "lcr",
)
MIN_BENCHMARKS = 3  # fewer than this and the composite is noise, so emit null.

EFFORT_SUFFIX = re.compile(r"-(low|medium|high|xhigh|non-reasoning)$")


def fetch() -> list[dict]:
    key = AA_KEY_FILE.read_text(encoding="utf-8").strip()
    req = urllib.request.Request(AA_URL, headers={"x-api-key": key})
    with urllib.request.urlopen(req, timeout=60) as resp:
        if resp.status != 200:
            raise SystemExit(f"AA returned HTTP {resp.status}")
        body = json.load(resp)
    return body.get("data", body)


def creator_of(row: dict) -> str | None:
    c = row.get("model_creator")
    if isinstance(c, dict):
        return c.get("slug")
    return c


def split_effort(slug: str) -> tuple[str, str]:
    """R10: per-effort rows ARE fetchable, as separate slugs. Parse the suffix
    into its own column so (provider, model, effort) can be a real row key."""
    m = EFFORT_SUFFIX.search(slug)
    if not m:
        return slug, "base"
    return slug[: m.start()], m.group(1)


def capability_of(row: dict) -> float | None:
    evals = row.get("evaluations") or {}
    present = [evals[b] for b in AGENTIC_BENCHMARKS if isinstance(evals.get(b), (int, float))]
    if len(present) < MIN_BENCHMARKS:
        return None
    # Scaled to 0-100. The source benchmarks are fractions, but every band floor
    # in the pack is stated on a 0-100 scale (35/48/55/59/62) because they were
    # first derived against artificial_analysis_intelligence_index, which uses
    # that range. Emitting the raw fraction here would clear no floor at all --
    # caught by the C01 fit-check, not by any test.
    return round(100 * sum(present) / len(present), 2)


def main() -> int:
    rows = fetch()
    if "artificial_analysis_agentic_index" in ((rows[0].get("evaluations") or {}) if rows else {}):
        print(
            "STOP: artificial_analysis_agentic_index now EXISTS in the feed. R9 assumed "
            "it does not and composes a local substitute. Revisit the formula before "
            "trusting either number.",
            file=sys.stderr,
        )
        return 2

    cards, excluded = [], []
    for row in rows:
        creator = creator_of(row)
        if creator not in REACHABLE_CREATORS:
            continue

        slug = row.get("slug") or ""
        pricing = row.get("pricing") or {}
        price = pricing.get("price_1m_blended_3_to_1")
        capability = capability_of(row)
        model, effort = split_effort(slug)

        # R15: missing price is not free access.
        if not isinstance(price, (int, float)) or price <= 0:
            excluded.append({"slug": slug, "reason": "null-or-zero-price", "price": price})
            continue
        if capability is None:
            excluded.append({"slug": slug, "reason": f"fewer than {MIN_BENCHMARKS} agentic benchmarks"})
            continue

        cards.append(
            {
                "slug": slug,
                "model": model,
                "effort": effort,
                "provider": CREATOR_TO_PROVIDER[creator],
                "creator": creator,
                "capability": capability,
                "price_blended": price,          # reference only, NEVER the sort key (R19/R25)
                "total_cost_usd": None,          # MEASURED by C07, or the card cannot rank (R21)
                "p0_pass": None,                 # scored by C07 (R28)
            }
        )

    cards.sort(key=lambda c: (-c["capability"], c["price_blended"]))
    out = {
        "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": AA_URL,
        "models_seen": len(rows),
        "cards": cards,
        "excluded": excluded,
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    tmp = OUT.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(out, indent=2), encoding="utf-8")
    os.replace(tmp, OUT)  # a half-written cards.json must never be readable

    print(f"cards={len(cards)} excluded={len(excluded)} of {len(rows)} models -> {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
