#!/usr/bin/env python3
"""C02 — build deck.json from cards.json + bench/results.json.

Two inputs, one output, and no network. C01 owns *what is reachable and how capable*;
C07 owns *what it actually cost and whether it was right*. This builder is the only
place those two meet, and it is deliberately incapable of producing either on its own:
it fetches nothing, calls no provider, and never writes cards.json.

The rules that shape this file, each written after a real failure mode:

  R21  Cost is MEASURED, never estimated. An unmeasured card is present in its band
       and permanently unranked. It is not dropped -- dropping it would hide that a
       cheap high-capability card exists and has simply never been run -- and it is
       not ranked on list price, which would let a newly-shipped model capture routing
       on marketing numbers alone. Present-but-unranked is the honest third state.

  R25  Cost ranks, quota gates, and the two never mix. This builder never reads
       headroom.json. `gated()` runs at lookup time, over an already-sorted list.

  R26  Week 0 -- every band ESTIMATE, ranked=0, unranked>0 -- is the CORRECT output,
       not an empty result to be papered over. The deck's job at week 0 is to say
       "these cards clear this floor and none of them has been measured yet".

  R27  A floor is ESTIMATE until a reference-job pass derives it. The floors in
       deck.py were calibrated against a different metric than the deck now uses
       (risks.md R3), so a DERIVED stamp without a supporting pass would manufacture
       exactly the false confidence the fetched provider table exists to prevent.

Cost and P0 are read ONLY from bench/results.json -- never from cards.json, whose
`total_cost_usd`/`p0_pass` are C01's null placeholders. Reading them through would
mean a future non-null value in the fetcher's output could rank a card, which is the
provider-table-as-cost defect R21 forbids outright. If cards.json ever carries a
non-null cost, this script STOPs rather than trusting it.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import re
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SPEC = Path.home() / "code workshop" / "specs" / "001-agent-capability-routing"
CARDS = Path.home() / ".local" / "state" / "brain-router" / "cards.json"
RESULTS = SPEC / "bench" / "results.json"
OUT = Path.home() / ".local" / "state" / "brain-router" / "deck.json"

sys.path.insert(0, str(REPO / "src"))

from mcp_brain_router.deck import BANDS, Card, build_bands  # noqa: E402


class Refused(Exception):
    """A STOP condition from C02. Raised before anything is written."""


# A hyphen between two digit groups is a decimal point in the router's naming:
# AA's slug `glm-5-3-flash` is the router's `glm-5.3-flash`. Confirmed against
# bench/baseline.json, whose measured rows carry `model: "glm-5.3-flash"`, and
# consistent with router.py's `grok-4.5` and install.py's `glm-5.3`.
#
# This is a NAME-SHAPE derivation, not a verified provider id. C01 does not emit
# router_model, so there is no authoritative slug -> router-id table anywhere in the
# rig; this reconstructs one. It is safe only because nothing acts on the field until
# C07 has actually called the model, at which point a wrong id fails loudly rather
# than silently mis-routing. The real fix belongs in C01, not here.
_VERSION_HYPHEN = re.compile(r"(?<=\d)-(?=\d)")


def router_model_for(slug: str) -> str:
    return _VERSION_HYPHEN.sub(".", slug)


def read_json(path: Path) -> dict | None:
    """None means ABSENT, which is a legitimate week-0 state for results.json.

    Distinguished from `{"results": []}` only in the summary line -- both produce
    the same deck, because neither carries a measurement.
    """
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def load_measurements(doc: dict | None) -> tuple[dict[str, dict[str, dict]], set[str], int]:
    """Index results rows by (card, band), and collect the bands a reference job passed.

    Returns (by_card, derived_bands, pegged_low). by_card maps key -> {band -> entry},
    where entry is that band's measurement {total_cost_usd, p0_pass, size_class,
    cost_basis} -- the per-band shape R30 requires. A row keys on `card` (C07's own
    field name), or on `model` for a row written before that field existed;
    `build_cards` then looks up under both the slug and the router_model, since either
    shape names the same card and a lookup miss would silently un-rank a measured one.

    C07 records one row per (card, band, size_class), so a card measured in two bands
    has two rows and BOTH survive here, keyed by band. That fixes the old hazard where
    a card passing P0 in B1 and failing it in B4 took its single verdict from file
    order. What still cannot be flattened is two rows for the SAME (card, band) --
    which in practice means the two size classes of B4/B5 -- with DIFFERENT verdicts:
    there is no third key below band to separate them, so the builder STOPs rather
    than pick one silently.

    A pegged row (peg-costs.py) carries a peg_confidence for how much of its cost
    rides on token counts borrowed from another model. A LOW-confidence peg is
    output-dominated -- the cost prices the donor's verbosity, not this card -- so
    ranking on it would rank the wrong model. It is demoted here to UNMEASURED
    (total_cost_usd None), which R21 already handles: present-but-unranked. Medium
    and high pegs rank normally; measured rows are untouched.
    """
    by_card: dict[str, dict[str, dict]] = {}
    derived: set[str] = set()
    pegged_low = 0
    for row in (doc or {}).get("results") or []:
        key = row.get("card") or row.get("model")
        if key and row.get("cost_basis") == "pegged" and row.get("peg_confidence") == "low":
            pegged_low += 1
            print(f"warning: pegged row {key!r} (band {row.get('band')}) has low "
                  "peg_confidence — output-dominated, cost demoted to UNMEASURED "
                  "and unrankable (R21)",
                  file=sys.stderr)
            # Demoted, not dropped: the entry carries total_cost_usd None, so the
            # card stays present in its band and permanently unranked (R21).
            row = {**row, "total_cost_usd": None}
        if key:
            bands = by_card.setdefault(key, {})
            band = row.get("band")
            if band in bands:
                prev = bands[band]
                if prev.get("p0_pass") != row.get("p0_pass"):
                    # STOP. Same card, same band -- the rows differ only by size
                    # class (R30) -- and the verdicts disagree. There is nothing
                    # below "band" to key them apart, so keeping either row is a
                    # silent pick, and the small/large choice would follow file
                    # order. That has to be settled in the bench, not here.
                    print(
                        f"STOP: {key!r} band {band} has results rows with DIFFERENT "
                        f"p0_pass ({prev.get('p0_pass')} vs {row.get('p0_pass')}). "
                        "Same (card, band) differing only by size class must agree, "
                        "or the builder cannot flatten them without following file "
                        "order. Re-run the bench cell (R30: median of n=3) before "
                        "building.",
                        file=sys.stderr,
                    )
                    raise SystemExit(3)
                # Same verdict either way, so file order cannot change the outcome.
                # Still announced: it means two size-class rows collapsed to one.
                print(f"warning: {key!r} band {band} has more than one results row; "
                      "verdicts agree, last row read wins",
                      file=sys.stderr)
            bands[band] = {
                "total_cost_usd": row.get("total_cost_usd"),
                "p0_pass": row.get("p0_pass"),
                "size_class": row.get("size_class") or "small",
                "cost_basis": row.get("cost_basis") or "measured",
            }
        # R27: a floor is DERIVED only from a reference-job PASS. A completed run that
        # failed a P0 item, or one that was never scored (p0_pass None), derives
        # nothing -- it is evidence about the card, not about where the floor sits.
        if row.get("band") in BANDS and row.get("p0_pass") is True:
            derived.add(row["band"])
    return by_card, derived, pegged_low


def build_cards(raw: list[dict], by_card: dict[str, dict[str, dict]]) -> tuple[list[Card], int, list[str]]:
    """Map cards.json rows onto the Card dataclass, dropping capability-less cards.

    A card with `capability is None` cleared fewer than C01's minimum agentic
    benchmarks. It has no place on any floor -- an unknown capability is not a low
    one, and admitting it to B1 would put an unmeasurable model in the cheapest,
    highest-volume band.
    """
    cards: list[Card] = []
    dropped = 0
    matched: list[str] = []
    for c in raw:
        if c.get("capability") is None:
            dropped += 1
            continue
        if c.get("total_cost_usd") is not None:
            raise Refused(
                f"STOP: cards.json carries a non-null total_cost_usd for {c.get('slug')!r} "
                f"({c['total_cost_usd']}). Cost may come only from a measured bench run "
                "(R21); a cost in the provider table is a list price wearing a cost's name."
            )
        slug = c["slug"]
        rm = router_model_for(slug)
        m = by_card.get(slug) or by_card.get(rm) or {}
        if m:
            matched.append(slug)
        # The scalars keep the LAST row read, exactly as the pre-R30 builder did, so
        # a consumer still holding a one-measurement-per-card card sees no change.
        # Ranking never reads them once by_band is populated -- deck.py routes every
        # band through its own entry -- so this is compatibility, not a selection
        # rule, and a file-order change cannot alter which bands the card ranks in.
        last = m[next(reversed(m))] if m else {}
        cards.append(
            Card(
                slug=slug,
                router_model=rm,
                provider=c["provider"],
                model=c["model"],
                effort=c["effort"],
                capability=c["capability"],
                price_blended=c["price_blended"],  # reference only, NEVER the sort key (R19/R25)
                total_cost_usd=last.get("total_cost_usd"),
                # size_class is a property of a MEASUREMENT, not of a card: C07 records
                # cost per (card, band, size_class). With no measurement there is no
                # size to report, so the field defaults to the smaller job shape
                # rather than claiming a large one was ever run.
                size_class=last.get("size_class") or "small",
                p0_pass=last.get("p0_pass"),
                by_band=dict(m),
            )
        )
    return cards, dropped, matched


def check_stops(rows, results_doc: dict | None, derived: set[str],
                results_path: Path = RESULTS) -> None:
    """Both STOPs, checked over the EMITTED rows rather than assumed from inputs.

    Asserting these against the data structures that were actually built is the whole
    point. A check written against the inputs restates how the code was meant to work;
    this one can still catch a build_bands that stopped working that way.
    """
    measured = bool((results_doc or {}).get("results"))
    if not measured:
        loaded = [r.band for r in rows if r.ranked]
        if loaded:
            raise Refused(
                "STOP: bands " + ", ".join(loaded) + " have ranked cards while "
                f"{results_path} carries no results. Ranking without measurement means "
                "cost was estimated, which R21 forbids outright."
            )
    for r in rows:
        if r.floor_status == "DERIVED" and r.band not in derived:
            raise Refused(
                f"STOP: band {r.band} floor emitted as DERIVED with no supporting "
                "reference-job pass in bench/results.json (R27). An undecorated guess is "
                "honest; a guess labelled DERIVED is not."
            )


def summary_lines(rows) -> list[str]:
    """Exactly the shape the contract's jq prints, so the two can be diffed by eye."""
    return [
        f"{r.band} floor={r.floor} {r.floor_status} "
        f"ranked={len(r.ranked)} unranked={len(r.unranked)}"
        for r in rows
    ]


def main() -> int:
    ap = argparse.ArgumentParser(description="C02: build deck.json from cards.json + bench results")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the band summary and write nothing at all")
    # Overrides exist for the tests: a builder whose STOP paths can only be
    # exercised against the live state files cannot be regression-tested without
    # writing into ~/.local/state, which a test must never do.
    ap.add_argument("--cards", default=None,
                    help="cards.json to read instead of the default state path")
    ap.add_argument("--out", default=None,
                    help="write the deck here instead of the state path (C08 builds to a "
                         "side file, diffs, then moves it into place)")
    ap.add_argument("--results", default=None,
                    help="results.json to read instead of the default bench path")
    args = ap.parse_args()

    cards_path = Path(args.cards).expanduser() if args.cards else CARDS
    results_path = Path(args.results).expanduser() if args.results else RESULTS
    cards_doc = read_json(cards_path)
    if cards_doc is None:
        print(f"STOP: {cards_path} is absent. Run bin/fetch-provider-table.py (C01) first.",
              file=sys.stderr)
        return 2
    results_doc = read_json(results_path)

    try:
        by_card, derived, pegged_low = load_measurements(results_doc)
        cards, dropped, matched = build_cards(cards_doc.get("cards") or [], by_card)
        rows = build_bands(cards, derived=derived)
        check_stops(rows, results_doc, derived, results_path=results_path)
    except Refused as e:
        print(e, file=sys.stderr)
        return 3

    # An orphan measurement is a signal, not noise: it means the bench ran a card the
    # provider table no longer lists (renamed, withdrawn, or priced out by R15), and
    # its cost is silently doing nothing.
    orphans = sorted(set(by_card) - set(matched) - {router_model_for(s) for s in matched})
    for o in orphans:
        print(f"warning: results row {o!r} matches no card in cards.json", file=sys.stderr)

    doc = {
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "cards_path": str(cards_path),
        "results_path": str(results_path),
        "results_present": "absent" if results_doc is None
        else ("empty" if not (results_doc.get("results") or []) else "present"),
        "cards_in": len(cards_doc.get("cards") or []),
        "cards_dropped_no_capability": dropped,
        "cards_measured": len(matched),
        "bands": [dataclasses.asdict(r) for r in rows],
    }

    for line in summary_lines(rows):
        print(line)
    print(f"cards={len(cards)} dropped={dropped} measured={len(matched)} "
          f"results={doc['results_present']}")
    if pegged_low:
        print(f"pegged_low_demoted={pegged_low} (low-confidence pegs treated as unmeasured)")

    out = Path(args.out) if args.out else OUT
    if args.dry_run:
        # Writes NOTHING -- not the file, not its parent directory. The STOPs above
        # still ran, so a dry-run reports what a real run would refuse.
        print(f"dry-run: {out} not written")
        return 0

    out.parent.mkdir(parents=True, exist_ok=True)
    # Temp file in the destination directory: os.replace is atomic only within one
    # filesystem, and a half-written deck.json must never be readable by the router.
    fd, tmp = tempfile.mkstemp(dir=str(out.parent), prefix=".deck.", suffix=".json")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(doc, fh, indent=2)
        os.replace(tmp, out)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise

    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
