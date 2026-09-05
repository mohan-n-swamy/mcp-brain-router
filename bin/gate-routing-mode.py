#!/usr/bin/env python3
"""gate-routing-mode.py — SC5: routing_mode stays legacy until the deck can rank.

Exit 0 when the configured mode is a state the deck can honestly serve:
  legacy  -> always fine; the [roles] walk needs nothing from the deck
  deck    -> only if EVERY band each role draws from has >= 3 ranked cards
             (R26: the top-3 must carry measured cost before the deck goes live)
Exit 1 otherwise, naming the band that is short. Enabling deck mode on an
unranked deck would route every role through an empty list and fall through to
legacy anyway -- which looks like it works and measures nothing.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from mcp_brain_router.config import Config, DEFAULT_ROLE_BANDS  # noqa: E402
from mcp_brain_router.deck import load_deck  # noqa: E402

NEED = 3


def main() -> int:
    cfg = Config.load()
    mode = getattr(cfg, "routing_mode", "legacy")
    print(f"routing_mode = {mode}")
    if mode == "legacy":
        return 0
    if mode != "deck":
        print(f"unknown routing_mode {mode!r}; only legacy|deck are valid")
        return 1
    deck = load_deck()
    if not deck:
        print("deck.json unreadable or empty -- deck mode cannot be honest")
        return 1
    NEED = getattr(cfg, "deck_min_ranked", 3)  # R26 default 3; week 0 may relax
    short = []
    for role, band in (cfg.role_bands or DEFAULT_ROLE_BANDS).items():
        row = deck.get(band)
        n = len(row.ranked) if row else 0
        print(f"  {role:10} -> {band}  ranked={n}  need>={NEED}")
        if n < NEED:
            short.append(f"{role}->{band} has {n}")
    if short:
        print("SC5 FAIL: deck mode with an unranked top-3:", "; ".join(short))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
