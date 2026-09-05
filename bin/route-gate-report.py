#!/usr/bin/env python3
"""route-gate-report.py — C13 adoption metric: observed vs delegated.

R13 had a numerator (brain-router-delegations.jsonl) and no denominator:
counting jobs the session did NOT delegate means detecting them, the same
unsolved problem as triggering. The route gate's observation log solves both at
once — every line is one observed B1-B3 job shape, whether or not it routed.
This report joins the two.

Usage: route-gate-report.py [--since 7d]
Missing files are zeros, not errors: on a fresh install delegated=0 is the
honest starting number.
"""

import argparse
import datetime
import json
import os
import pathlib
import sys

HOME = os.environ.get("HOME", str(pathlib.Path.home()))
OBSERVATIONS = pathlib.Path(HOME, ".local/state/brain-router/route-observations.jsonl")
DELEGATIONS = pathlib.Path(HOME, ".local/state/brain-router-delegations.jsonl")


def parse_since(spec):
    """'7d' / '24h' / '2w' -> cutoff datetime; anything unparsable -> None."""
    spec = (spec or "").strip().lower()
    if not spec:
        return None
    unit = spec[-1]
    if unit not in "dhw" or not spec[:-1].isdigit():
        return None
    n = int(spec[:-1])
    seconds = {"d": 86400, "h": 3600, "w": 604800}[unit] * n
    return datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(seconds=seconds)


def parse_ts(value):
    """Observation/delegation timestamps may be ISO strings or epoch floats."""
    try:
        return datetime.datetime.fromisoformat(str(value))
    except Exception:
        pass
    try:
        return datetime.datetime.fromtimestamp(float(value), tz=datetime.timezone.utc)
    except Exception:
        return None


def count_lines(path, cutoff, key="ts"):
    total = 0
    if not path.exists():
        return total
    try:
        with open(path, "r", errors="ignore") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except Exception:
                    continue  # a torn line must not zero the report
                if cutoff is not None:
                    ts = parse_ts(rec.get(key, ""))
                    if ts is None or ts < cutoff:
                        continue
                total += 1
    except Exception:
        pass
    return total


# per-shape tally for observations only; delegations have no shape field
SHAPE_COUNTS = {}


def tally_shape(rec):
    shape = rec.get("shape")
    if shape:
        SHAPE_COUNTS[shape] = SHAPE_COUNTS.get(shape, 0) + 1


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--since", default=None, help="window, e.g. 7d / 24h / 2w")
    args = ap.parse_args()

    cutoff = parse_since(args.since)

    observed = 0
    if OBSERVATIONS.exists():
        try:
            with open(OBSERVATIONS, "r", errors="ignore") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except Exception:
                        continue
                    if cutoff is not None:
                        ts = parse_ts(rec.get("ts", ""))
                        if ts is None or ts < cutoff:
                            continue
                    observed += 1
                    tally_shape(rec)
        except Exception:
            pass

    delegated = count_lines(DELEGATIONS, cutoff)

    window = f" (since {args.since})" if args.since else ""
    print(f"route-gate report{window}")
    for shape in sorted(SHAPE_COUNTS):
        print(f"  observed {shape}: {SHAPE_COUNTS[shape]}")
    print(f"  observed total: {observed}")
    print(f"  delegated (all bands -- log carries no band; upper bound): {delegated}")
    if observed > 0:
        print(f"  ratio delegated/observed: {delegated / observed:.2f}")
    else:
        print("  ratio delegated/observed: n/a (nothing observed yet)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
