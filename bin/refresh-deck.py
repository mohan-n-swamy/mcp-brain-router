#!/usr/bin/env python3
"""C08 — weekly refresh: fetch -> benchmark -> build -> diff -> probe.

Orchestrates C01, C07 and C02 and computes NOTHING itself -- no capability,
no cost, no ordering. Its entire value is the diff (R17): weekly was chosen
over monthly to match the quota reset, which means 4x the refreshes and 4x
the chances of a SILENT routing shift. So the refresh never overwrites
silently: the previous deck is kept as deck.prev.json and every change a
reader could act on is printed before the new deck is trusted.

C12's probe is not built yet. Skipping it silently would be the exact
failure mode this component exists to prevent, so the skip is printed.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
STATE = Path.home() / ".local" / "state" / "brain-router"
DECK = STATE / "deck.json"
PREV = STATE / "deck.prev.json"

FETCH = REPO / "bin" / "fetch-provider-table.py"
BENCH = REPO / "bin" / "run-bench.py"
BUILD = REPO / "bin" / "build-deck.py"

# R7: a card whose trials vary wildly has a cost, but not a trustworthy one.
# It is FLAGGED for a human, never silently ranked on the median.
SPREAD_FACTOR = 3

# A card row may carry per-trial costs under any of these keys, depending on
# which C07 shape wrote it. Absent keys mean absent data -- nothing is inferred.
TRIAL_KEYS = ("trials", "trial_costs", "costs")


def read_deck(path: Path) -> dict | None:
    """None means ABSENT -- a legitimate first-run state, not an error."""
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def card_label(card: dict | None) -> str:
    """'slug @ $cost' or 'none' -- the diff prints cards, not just counts."""
    if card is None:
        return "none"
    cost = card.get("total_cost_usd")
    cost_s = f"${cost}" if isinstance(cost, (int, float)) else "unmeasured"
    return f"{card.get('slug', '?')} @ {cost_s}"


def band_cards(band: dict) -> set[str]:
    """Enter/leave detection covers BOTH lists: a card demoted from ranked to
    unranked stays in the band (that is a winner change, reported separately),
    while a card gone from both lists has left the provider table entirely."""
    return {c.get("slug") for c in band.get("ranked") or []} | {
        c.get("slug") for c in band.get("unranked") or []
    }


def trial_spread(card: dict) -> str | None:
    """Return a flag line if the row carries trials AND they vary > 3x median.

    Returns None when there is no trials data -- 'no evidence' is not 'no
    spread', and treating it as either would invent the other.
    """
    for key in TRIAL_KEYS:
        vals = card.get(key)
        if isinstance(vals, list) and vals \
                and all(isinstance(v, (int, float)) for v in vals):
            vals = sorted(vals)
            n = len(vals)
            median = vals[n // 2] if n % 2 else (vals[n // 2 - 1] + vals[n // 2]) / 2
            if median > 0 and vals[-1] > SPREAD_FACTOR * median:
                return (f"  TRIAL SPREAD {card.get('slug')}: max {vals[-1]} exceeds "
                        f"{SPREAD_FACTOR}x median {median} (R7) -- flagged, not ranked")
    return None


def diff_decks(old: dict, new: dict) -> list[str]:
    """Every change a router consumer could act on, as printable lines.

    Metadata (built_at, cards_measured, ...) is deliberately ignored: it
    changes on every run and reporting it would train the reader that a
    'changed' line usually means nothing.
    """
    lines: list[str] = []
    old_bands = {b.get("band"): b for b in old.get("bands") or []}
    new_bands = {b.get("band"): b for b in new.get("bands") or []}

    for name in sorted(old_bands.keys() | new_bands.keys()):
        ob, nb = old_bands.get(name), new_bands.get(name)
        if ob is None or nb is None:
            lines.append(f"band {name}: {'added' if ob is None else 'removed'}")
            continue

        old_ranked = ob.get("ranked") or []
        new_ranked = nb.get("ranked") or []
        # The leftmost ranked card IS the route. Its change is the one line
        # that must never scroll by unnoticed, so both cards and both costs.
        old_first = old_ranked[0] if old_ranked else None
        new_first = new_ranked[0] if new_ranked else None
        if (old_first or {}).get("slug") != (new_first or {}).get("slug"):
            lines.append(
                f"band {name}: winner changed {card_label(old_first)} -> "
                f"{card_label(new_first)}"
            )

        old_slugs, new_slugs = band_cards(ob), band_cards(nb)
        for slug in sorted(old_slugs - new_slugs):
            lines.append(f"band {name}: card left: {slug}")
        for slug in sorted(new_slugs - old_slugs):
            lines.append(f"band {name}: card entered: {slug}")

        if ob.get("floor_status") != nb.get("floor_status"):
            lines.append(
                f"band {name}: floor_status {ob.get('floor_status')} -> "
                f"{nb.get('floor_status')}"
            )

        # R7 flags attach to measured cards wherever they sit.
        for card in new_ranked:
            flag = trial_spread(card)
            if flag:
                lines.append(flag)

    return lines


def diff_only() -> int:
    prev = read_deck(PREV)
    if prev is None:
        print("baseline: no previous deck")
        return 0
    cur = read_deck(DECK)
    if cur is None:
        print("STOP: deck.prev.json exists but deck.json is absent -- the current "
              "deck cannot be missing while a previous one survives.", file=sys.stderr)
        return 2
    lines = diff_decks(prev, cur)
    if not lines:
        print("diff: no changes")
        return 0
    for line in lines:
        print(line)
    print(f"diff: {len(lines)} change(s)")
    return 1


def bench_supports_seed_round() -> bool:
    """run-bench.py --help is the ONLY honest oracle for the flag's existence:
    reading its source would answer a different question (is the flag coded?)
    and hard-coding 'someday' would invent a flag that does not exist. If the
    flag is absent the whole sequence must STOP rather than run a benchmark
    shape nobody specified -- C07 spend is real spend."""
    try:
        out = subprocess.run(
            [sys.executable, str(BENCH), "--help"],
            capture_output=True, text=True, check=True,
        )
    except subprocess.CalledProcessError as e:
        print(f"STOP: {BENCH} --help failed:\n{e.stderr}", file=sys.stderr)
        raise SystemExit(3)
    return "--seed-round" in out.stdout


def run(step: str, cmd: list[str]) -> None:
    print(f"== {step}: {' '.join(str(c) for c in cmd[1:])}")
    # Output streams through untouched: each step's own STOPs and warnings
    # must reach the log verbatim, not be summarised by the orchestrator.
    rc = subprocess.run(cmd).returncode
    if rc != 0:
        print(f"STOP: {step} exited {rc}; nothing downstream was run.",
              file=sys.stderr)
        raise SystemExit(rc)


def full_refresh() -> int:
    if not bench_supports_seed_round():
        print(
            "STOP: run-bench.py does not accept --seed-round. The refresh's "
            "benchmark step is specified as `run-bench.py --seed-round` and "
            "running any other shape instead would benchmark something C08 "
            "never specified. Fix C07 first; nothing was fetched or written.",
            file=sys.stderr,
        )
        return 3

    run("fetch (C01)", [sys.executable, str(FETCH)])
    run("benchmark (C07)", [sys.executable, str(BENCH), "--seed-round"])

    # deck.prev.json is written BEFORE the new deck exists so there is no
    # window -- and no failure of build-deck -- in which the old deck is
    # overwritten without a copy surviving.
    old = read_deck(DECK)
    if old is not None:
        PREV.parent.mkdir(parents=True, exist_ok=True)
        PREV.write_text(json.dumps(old, indent=2) + "\n", encoding="utf-8")
        print(f"kept previous deck at {PREV}")

    run("build (C02)", [sys.executable, str(BUILD)])

    new = read_deck(DECK)
    if old is None:
        print("baseline: no previous deck to diff against")
    else:
        assert new is not None
        lines = diff_decks(old, new)
        for line in lines:
            print(line)
        print(f"diff: {len(lines)} change(s)" if lines else "diff: no changes")

    # C12 is not built. A silent skip is the failure mode this file exists
    # to prevent, so the skip is itself a printed line in the weekly log.
    print("probe: C12 not built, skipped")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description="C08: weekly refresh -- fetch, benchmark, build, diff, probe")
    ap.add_argument("--diff-only", action="store_true",
                    help="compare deck.prev.json against deck.json, write nothing, "
                         "exit 1 if anything changed")
    args = ap.parse_args()

    if args.diff_only:
        return diff_only()
    return full_refresh()


if __name__ == "__main__":
    sys.exit(main())
