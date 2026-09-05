#!/usr/bin/env python3
"""C08 — weekly refresh: fetch -> benchmark -> build -> diff -> probe.

Orchestrates C01, C07 and C02 and computes NOTHING itself -- no capability,
no cost, no ordering. Its entire value is the diff (R17): weekly was chosen
over monthly to match the quota reset, which means 4x the refreshes and 4x
the chances of a SILENT routing shift. So the refresh never overwrites
silently: the previous deck is kept as deck.prev.json and every change a
reader could act on is printed before the new deck is trusted.

The probe (C12) runs LAST, against the freshly written deck: a winner that
does not answer is a routing change the diff cannot see, so the refresh ends
by proving each band winner is alive rather than by trusting the build.
"""
from __future__ import annotations

import argparse
import re
import json
import os
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
PROBE = REPO / "bin" / "probe-band-winners.py"

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


RESULTS = Path.home() / "code workshop" / "specs" / "001-agent-capability-routing" / "bench" / "results.json"


def spread_flags() -> list[str]:
    """R7 detection: a ranked card whose 3 trials varied by more than 3x is flagged,
    not silently ranked. The trials live in bench/results.json (C07 writes them);
    deck.json carries only the median, so reading the deck for this was vacuous
    (refuter finding)."""
    import statistics
    out: list[str] = []
    try:
        rows = json.loads(RESULTS.read_text(encoding="utf-8")).get("results") or []
    except Exception:
        return out
    for r in rows:
        costs = [t.get("cost_usd") for t in (r.get("trials") or []) if isinstance(t.get("cost_usd"), (int, float))]
        if len(costs) >= 2:
            med = statistics.median(costs)
            if med > 0 and max(costs) > 3 * med:
                out.append(f"TRIAL SPREAD {r.get('card') or r.get('id')} {r.get('band')}: "
                           f"max {max(costs):.4f} exceeds 3x median {med:.4f} -- flagged, not ranked")
    return out


def run_probe() -> int:
    """Probe the new deck's band winners (C12) and map its exit code onto ours.

    Output is captured, not streamed, because exit 1 needs the --json rows to
    name the failing bands -- but the probe's human lines still reach the log
    verbatim, since a captured-but-swallowed report is a silent skip with a
    subprocess in the middle.
    """
    proc = subprocess.run(
        [sys.executable, str(PROBE), "--json"],
        capture_output=True, text=True,
    )
    if proc.stderr:
        sys.stderr.write(proc.stderr)

    if proc.returncode == 0:
        return 0
    if proc.returncode == 2:
        # Week 0: costs not yet measured, so no band has a ranked winner. The
        # deck was still fetched, benchmarked, built and diffed -- that is a
        # successful refresh of a young deck, not a launchd failure.
        print("probe: deck has no ranked winners (week 0) -- not a failure")
        return 0

    # Exit 1 (or anything unexpected): a winner that did not answer ALIVE. The
    # previous deck is deliberately NOT restored -- C08 reports, the human
    # acts -- and the nonzero exit makes launchd log the run either way.
    bands: list[str] = []
    try:
        bands = [r.get("band", "?") for r in json.loads(proc.stdout)
                 if not r.get("ok")]
    except json.JSONDecodeError:
        print(f"probe: exited {proc.returncode} and its --json output did not "
              f"parse; see its report above", file=sys.stderr)
    if bands:
        print(f"probe: FAILING BANDS: {', '.join(bands)}", file=sys.stderr)
    print(f"probe: exited {proc.returncode}; previous deck NOT restored "
          f"(deck.prev.json holds it) -- C08 reports, the human acts",
          file=sys.stderr)
    return proc.returncode or 1


def bench_plan(mode: str, remeasure: bool) -> tuple[int, str | None]:
    """Ask C07 for its plan without spending: (calls, approve-hash). C07 prints
    'plan hash: X' and either 'TOTAL PROVIDER CALLS  N' (seed) or 'To spend N
    tiny calls' (probe); zero calls prints no hash."""
    cmd = [sys.executable, str(BENCH), mode, "--dry-run"] + (["--remeasure"] if remeasure else [])
    out = subprocess.run(cmd, capture_output=True, text=True)
    if out.returncode != 0:
        print(f"STOP: {mode} --dry-run exited {out.returncode}:\n{out.stderr}", file=sys.stderr)
        raise SystemExit(out.returncode)
    text = out.stdout
    h = re.search(r"^plan hash: (\S+)$", text, re.M)
    n = re.search(r"^TOTAL PROVIDER CALLS\s+(\d+)$", text, re.M) \
        or re.search(r"^To spend (\d+) tiny calls", text, re.M)
    calls = int(n.group(1)) if n else 0
    return calls, (h.group(1) if h else None)


def spend_within_budget(step: str, mode: str, max_calls: int, remeasure: bool) -> None:
    """Unattended spend is allowed only up to a standing budget (must-not #11
    for a scheduled job). Over budget: STOP, print the plan, spend nothing."""
    calls, h = bench_plan(mode, remeasure)
    if calls == 0:
        print(f"== {step}: nothing to spend, skipped")
        return
    if calls > max_calls:
        print(f"STOP: {step} wants {calls} provider calls; standing budget is "
              f"--max-calls {max_calls}. Nothing was spent. To run it by hand:\n"
              f"  python3 {BENCH} {mode} --approve {h}"
              f"{' --remeasure' if remeasure else ''}", file=sys.stderr)
        raise SystemExit(4)
    print(f"== {step}: {calls} calls within budget {max_calls}, approving plan {h}")
    run(step, [sys.executable, str(BENCH), mode, "--approve", h]
        + (["--remeasure"] if remeasure else []))


def full_refresh(max_calls: int = 0, remeasure: bool = False) -> int:
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
    # New cards from the fetch may enter a contender set; their model ids are
    # probed (one tiny call each) before the seed can spend three trials on
    # an id the CLI refuses. Both steps draw on the same standing budget.
    spend_within_budget("probe new cards (C07)", "--probe-cards", max_calls, False)
    spend_within_budget("benchmark (C07)", "--seed-round", max_calls, remeasure)

    # deck.prev.json is written BEFORE the new deck exists so there is no
    # window -- and no failure of build-deck -- in which the old deck is
    # overwritten without a copy surviving.
    old = read_deck(DECK)
    if old is not None:
        PREV.parent.mkdir(parents=True, exist_ok=True)
        PREV.write_text(json.dumps(old, indent=2) + "\n", encoding="utf-8")
        print(f"kept previous deck at {PREV}")

    # Build to a SIDE file, emit the diff, and only then move it into place. The
    # first version wrote deck.json first and diffed second, so an interrupt
    # between the two left a new deck on disk with no diff ever shown (refuter).
    NEW = STATE / "deck.new.json"
    run("build (C02)", [sys.executable, str(BUILD), "--out", str(NEW)])

    new = read_deck(NEW)
    if new is None:
        print("STOP: build produced no deck; the previous deck stays in place", file=sys.stderr)
        return 1
    if old is None:
        print("baseline: no previous deck to diff against")
    else:
        lines = diff_decks(old, new) + spread_flags()
        for line in lines:
            print(line)
        print(f"diff: {len(lines)} change(s)" if lines else "diff: no changes")
    os.replace(NEW, DECK)
    print(f"wrote {DECK}")

    # LAST, and only on a full refresh (--diff-only writes nothing and must
    # never spend provider quota): prove the new deck's winners are alive.
    return run_probe()


def main() -> int:
    ap = argparse.ArgumentParser(
        description="C08: weekly refresh -- fetch, benchmark, build, diff, probe")
    ap.add_argument("--diff-only", action="store_true",
                    help="compare deck.prev.json against deck.json, write nothing, "
                         "exit 1 if anything changed")
    ap.add_argument("--max-calls", type=int, default=0,
                    help="standing budget of provider calls this unattended run may "
                         "spend on probe + benchmark; over budget STOPs with the plan "
                         "(default 0: never spend unattended)")
    ap.add_argument("--remeasure", action="store_true",
                    help="re-run cells already measured (monthly drift check)")
    args = ap.parse_args()

    if args.diff_only:
        return diff_only()
    return full_refresh(max_calls=args.max_calls, remeasure=args.remeasure)


if __name__ == "__main__":
    sys.exit(main())
