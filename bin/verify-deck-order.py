#!/usr/bin/env python3
"""verify-deck-order.py — C02's own check, named in its COMMANDS and never built.

For every band: the ranked list must equal that list re-sorted by the ONE
ordering rule, sort_key_for(band); every ranked card must carry a measured cost
for that band and p0_pass True; and no unranked card may satisfy both (a card
that could rank but did not is a builder bug, not a deck state). Exit 0 or 1.
Reads deck.json only; writes nothing.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from mcp_brain_router.deck import load_deck, measurement_for, rankable, sort_key_for  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--deck", default=None)
    a = ap.parse_args()
    deck = load_deck(a.deck)
    if not deck:
        print("no deck to verify"); return 1
    bad = 0
    for band, row in sorted(deck.items()):
        want = sorted(row.ranked, key=sort_key_for(band))
        if [c.slug for c in want] != [c.slug for c in row.ranked]:
            print(f"{band}: ranked order is not (cost ASC, capability DESC)"); bad += 1
        for c in row.ranked:
            m = measurement_for(c, band)
            if m.get("total_cost_usd") is None or m.get("p0_pass") is not True:
                print(f"{band}: ranked card {c.slug} has no measured cost or a failed/unmeasured P0"); bad += 1
        stray = [c.slug for c in row.unranked if rankable(c, band)]
        if stray:
            print(f"{band}: rankable cards left unranked: {stray}"); bad += 1
        print(f"{band}: ranked={len(row.ranked)} unranked={len(row.unranked)} {'OK' if not bad else ''}")
    print(f"verify-deck-order: {'PASS' if not bad else f'{bad} violation(s)'}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
