"""SC3/SC4: the deck's ordering rule and its two unrankable gates (C02).

Four rules are proved here:
  - `sort_key` is (cost ASC, capability DESC) — cost dominates, always (R18/R25).
  - An unmeasured card cannot rank (R21/SC4).
  - A card that failed a P0 acceptance item cannot rank, however cheap (R28).
  - Over-qualification is free: a card populates its band and every band below.
"""

import pytest

from mcp_brain_router.deck import (
    BAND_FLOORS,
    BANDS,
    Card,
    build_bands,
    rankable,
    sort_key,
)


def _card(slug: str, capability: float, cost: float | None = None,
          p0: bool | None = None, provider: str = "zhipu") -> Card:
    """Default is an UNMEASURED card — no cost, no P0 result. That is the week-0
    shape (R26), so the zero-extra-arg call is the common case."""
    return Card(
        slug=slug,
        router_model=slug,
        provider=provider,
        model=slug,
        effort="medium",
        capability=capability,
        price_blended=0.0,
        total_cost_usd=cost,
        size_class="small",
        p0_pass=p0,
    )


def _bands_containing(rows, card) -> list[str]:
    return [r.band for r in rows if card in r.ranked or card in r.unranked]


# --- 1. the sort key -------------------------------------------------------

def test_sort_key_breaks_cost_ties_by_capability_descending():
    # Equal cost: the MORE capable card wins the tie (capability DESC).
    weak = _card("tie-weak", 70.0, cost=0.002, p0=True)
    strong = _card("tie-strong", 80.0, cost=0.002, p0=True)

    assert sorted([weak, strong], key=sort_key) == [strong, weak]
    assert sorted([strong, weak], key=sort_key) == [strong, weak]


def test_cost_dominates_capability_even_when_cheaper_is_dumber():
    # The whole point of (cost ASC, capability DESC): the cheap, LESS capable
    # card ranks ahead of the expensive, more capable one. Capability is only a
    # tiebreaker; it never overturns a cost difference.
    cheap_dumb = _card("cheap-dumb", 40.0, cost=0.001, p0=True)
    dear_smart = _card("dear-smart", 95.0, cost=0.005, p0=True)
    tie_strong = _card("tie-strong", 80.0, cost=0.002, p0=True)
    tie_weak = _card("tie-weak", 70.0, cost=0.002, p0=True)

    ordered = sorted([dear_smart, tie_weak, tie_strong, cheap_dumb], key=sort_key)
    assert [c.slug for c in ordered] == [
        "cheap-dumb", "tie-strong", "tie-weak", "dear-smart",
    ]


def test_build_bands_uses_the_same_ordering_rule():
    # build_bands must not sort with a key of its own — B1's floor is 35.0, so
    # every card below clears it and the band order must equal sort_key order.
    cards = [
        _card("dear-smart", 95.0, cost=0.005, p0=True),
        _card("tie-weak", 70.0, cost=0.002, p0=True),
        _card("cheap-dumb", 40.0, cost=0.001, p0=True),
        _card("tie-strong", 80.0, cost=0.002, p0=True),
    ]

    rows = build_bands(cards)
    b1 = rows[0]

    assert b1.band == "B1"
    assert [c.slug for c in b1.ranked] == [
        "cheap-dumb", "tie-strong", "tie-weak", "dear-smart",
    ]
    assert b1.ranked == sorted(cards, key=sort_key)


# --- 2. unmeasured cannot rank (SC4 / R21) ---------------------------------

def test_unmeasured_card_never_ranks_in_any_band():
    unmeasured = _card("brand-new", 90.0, cost=None, p0=True)
    measured = _card("known", 90.0, cost=0.004, p0=True)

    rows = build_bands([unmeasured, measured])

    assert not rankable(unmeasured)
    # SC4: no ranked list anywhere in the deck holds a null-cost card.
    assert all(c.total_cost_usd is not None for r in rows for c in r.ranked)
    for r in rows:
        assert unmeasured not in r.ranked
        # It clears every floor on capability, so it is PRESENT — just unranked.
        assert unmeasured in r.unranked


def test_sort_key_refuses_an_unmeasured_card():
    unmeasured = _card("brand-new", 90.0, cost=None, p0=True)

    with pytest.raises(AssertionError):
        sort_key(unmeasured)


# --- 3/4. P0 failure and unmeasured P0 are unrankable (R28) ----------------

def test_cheapest_card_in_the_band_still_cannot_rank_if_p0_failed():
    # The failing card is the CHEAPEST by an order of magnitude. Without R28 it
    # would head every band — the cheapest wrong answer winning on price alone.
    cheapest_but_wrong = _card("fast-and-wrong", 90.0, cost=0.0001, p0=False)
    correct = _card("correct-cheap", 90.0, cost=0.002, p0=True)
    correct_dear = _card("correct-dear", 90.0, cost=0.009, p0=True)

    rows = build_bands([cheapest_but_wrong, correct, correct_dear])

    assert not rankable(cheapest_but_wrong)
    for r in rows:
        assert cheapest_but_wrong not in r.ranked
        assert cheapest_but_wrong in r.unranked
        # The next-cheapest P0-passing card takes the head of the band.
        assert r.ranked[0] is correct
        assert [c.slug for c in r.ranked] == ["correct-cheap", "correct-dear"]


def test_unmeasured_p0_is_also_unrankable_only_true_ranks():
    p0_unknown = _card("p0-unknown", 90.0, cost=0.001, p0=None)
    p0_failed = _card("p0-failed", 90.0, cost=0.001, p0=False)
    p0_passed = _card("p0-passed", 90.0, cost=0.001, p0=True)

    assert rankable(p0_unknown) is False
    assert rankable(p0_failed) is False
    assert rankable(p0_passed) is True

    rows = build_bands([p0_unknown, p0_failed, p0_passed])
    for r in rows:
        assert r.ranked == [p0_passed]
        assert set(r.unranked) == {p0_unknown, p0_failed}


# --- 5. over-qualification fill -------------------------------------------

def test_card_clearing_the_b5_floor_populates_every_band():
    over = _card("over-qualified", BAND_FLOORS["B5"], cost=0.003, p0=True)

    rows = build_bands([over])

    # Uses the module's own floors — no local copy to drift out of date.
    assert _bands_containing(rows, over) == list(BANDS)
    assert all(over in r.ranked for r in rows)


def test_mid_capability_card_fills_only_the_bands_it_clears():
    mid = _card("mid", 50.0, cost=0.003, p0=True)

    rows = build_bands([mid])

    expected = [b for b in BANDS if 50.0 >= BAND_FLOORS[b]]
    assert expected == ["B1", "B2"]           # 35.0 and 48.0
    assert _bands_containing(rows, mid) == expected
    # Absence proves the floor actually cuts: B3=55.0, B4=59.0, B5=62.0.
    for r in rows:
        if r.band not in expected:
            assert mid not in r.ranked and mid not in r.unranked


# --- 6. week 0 ------------------------------------------------------------

def test_week_zero_all_unmeasured_is_the_expected_state():
    # R26: on week 0 bench/results.json is empty, so EVERY card is unmeasured and
    # every band prints ranked=0, unranked>0. This is the CORRECT week-0 state,
    # not a failure — a non-empty `ranked` here would mean cost was estimated,
    # which R21 forbids outright.
    cards = [
        _card("b1-only", 36.0),
        _card("mid", 50.0),
        _card("clears-b5", 70.0),
    ]

    rows = build_bands(cards)

    assert len(rows) == 5
    for r in rows:
        assert r.ranked == []
        assert r.unranked, f"band {r.band} must still list its unmeasured cards"


# --- 7. floor status ------------------------------------------------------

def test_floors_are_estimate_unless_explicitly_derived():
    cards = [_card("any", 70.0)]

    assert all(r.floor_status == "ESTIMATE" for r in build_bands(cards))

    rows = build_bands(cards, derived={"B3"})
    assert {r.band: r.floor_status for r in rows} == {
        "B1": "ESTIMATE",
        "B2": "ESTIMATE",
        "B3": "DERIVED",
        "B4": "ESTIMATE",
        "B5": "ESTIMATE",
    }


# --- 8. five bands, no B6 -------------------------------------------------

def test_deck_has_exactly_five_bands_and_never_emits_b6():
    # A capability-95 card is included deliberately: B6 must be absent because
    # the deck emits five rows, not because no card happened to qualify.
    rows = build_bands([_card("top", 95.0, cost=0.001, p0=True)])

    assert len(rows) == 5
    assert [r.band for r in rows] == ["B1", "B2", "B3", "B4", "B5"]
    assert [r.band for r in rows] == list(BANDS)
    assert "B6" not in {r.band for r in rows}
    assert [r.floor for r in rows] == [BAND_FLOORS[b] for b in BANDS]
