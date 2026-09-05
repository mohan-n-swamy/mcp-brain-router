"""SC12 (R33): every band still resolves to a valid card when the effort cards
are removed.

R19 redefined cost as measured total, under which an effort variant is never
leftmost -- dearer and, per the fetched index, less capable than its own base.
R7 kept effort anyway: the option is bought deliberately. The binding guard is
that the deck must never have come to DEPEND on that plumbing -- no band may go
empty without it. This is that guard, run against the live card table.
"""
from __future__ import annotations

import json
import pathlib

import pytest

from mcp_brain_router.deck import BANDS, Card, build_bands

CARDS = pathlib.Path.home() / ".local/state/brain-router/cards.json"


def _cards() -> list[Card]:
    raw = json.loads(CARDS.read_text())["cards"]
    out = []
    for c in raw:
        if c.get("capability") is None:
            continue
        out.append(Card(
            slug=c["slug"], router_model=c.get("router_model") or c["slug"],
            provider=c["provider"], model=c.get("model") or c["slug"],
            effort=c.get("effort") or "base", capability=float(c["capability"]),
            price_blended=float(c.get("price_blended") or 0),
            total_cost_usd=None, size_class="small", p0_pass=None,
        ))
    return out


@pytest.mark.skipif(not CARDS.exists(), reason="no live cards.json")
def test_every_band_has_cards_with_effort_variants_removed():
    all_cards = _cards()
    base_only = [c for c in all_cards if c.effort == "base"]
    removed = len(all_cards) - len(base_only)
    assert removed > 0, "the live table carries no effort variants; the guard proves nothing"
    rows = build_bands(base_only)
    assert [r.band for r in rows] == list(BANDS)
    empty = [r.band for r in rows if not (r.ranked or r.unranked)]
    assert not empty, (
        f"bands {empty} have NO eligible card once effort variants are removed "
        f"({removed} of {len(all_cards)}). R33: the deck has come to depend on "
        "plumbing it must not need."
    )


@pytest.mark.skipif(not CARDS.exists(), reason="no live cards.json")
def test_removing_effort_variants_never_changes_a_base_card_placement():
    """Effort plumbing must not move the base cards. Same card, same bands."""
    all_cards = _cards()
    with_eff = {b.band: {c.slug for c in b.ranked + b.unranked} for b in build_bands(all_cards)}
    base_only = [c for c in all_cards if c.effort == "base"]
    without = {b.band: {c.slug for c in b.ranked + b.unranked} for b in build_bands(base_only)}
    base_slugs = {c.slug for c in base_only}
    for band in BANDS:
        assert with_eff[band] & base_slugs == without[band]
