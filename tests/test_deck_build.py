"""SC3/SC4: the deck's ordering rule and its two unrankable gates (C02).

Four rules are proved here:
  - `sort_key` is (Mohan's cost order ASC, pass rate DESC); it only ever orders cards
    already within 10 points of the band's best (ROUTER, 2026-10-01; replaces R18's
    cost-in-dollars key, which ranked on prices he does not pay).
  - An unmeasured card cannot rank (R21/SC4).
  - A card that failed a P0 acceptance item cannot rank, however cheap (R28).
  - Over-qualification is free: a card populates its band and every band below.

R30 adds the per-band shape: cost and verdict belong to a (card, band) pair, so
a card may rank in one band and be unrankable in another.
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from mcp_brain_router.deck import (
    BAND_FLOORS,
    BANDS,
    Card,
    build_bands,
    rankable,
    sort_key,
)

BIN = Path(__file__).resolve().parent.parent / "bin" / "build-deck.py"


def _card(slug: str, capability: float, cost: float | None = None,
          p0: bool | None = None, provider: str = "openai",
          by_band: dict | None = None, pass_rate: float | None = None,
          trials: int | None = None) -> Card:
    """Default is an UNMEASURED card — no pass rate, no P0 result. That is the week-0
    shape (R26), so the zero-extra-arg call is the common case. `cost` stays as a
    reference field only; it no longer ranks anything."""
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
        by_band=by_band or {},
        pass_rate=pass_rate,
        trials=trials,
    )


def _meas(slug, capability, pass_rate, provider="openai", p0=True):
    """A card benched 3 times in its band(s) with the given pass rate."""
    return _card(slug, capability, p0=p0, provider=provider, pass_rate=pass_rate, trials=3)


def _bands_containing(rows, card) -> list[str]:
    return [r.band for r in rows if card in r.ranked or card in r.unranked]


# --- 1. the sort key -------------------------------------------------------

def test_sort_key_breaks_cost_order_ties_by_pass_rate_descending():
    # Same provider (same place in the cost order): the BETTER card wins the tie.
    weak = _meas("tie-weak", 70.0, 0.80)
    strong = _meas("tie-strong", 80.0, 0.90)

    assert sorted([weak, strong], key=sort_key) == [strong, weak]
    assert sorted([strong, weak], key=sort_key) == [strong, weak]


def test_cost_order_dominates_pass_rate_among_eligible_cards():
    # Among cards within 10 points of the best, the cheaper capacity leads even when
    # a dearer one scored higher: Codex, then Grok, then Claude (Mohan, 2026-10-01).
    codex = _meas("codex", 90.0, 0.86, provider="openai")
    grok = _meas("grok", 90.0, 0.90, provider="xai")
    claude = _meas("claude", 90.0, 0.95, provider="anthropic")

    ordered = sorted([claude, grok, codex], key=sort_key)
    assert [c.slug for c in ordered] == ["codex", "grok", "claude"]


def test_build_bands_uses_the_same_ordering_rule():
    # build_bands must not sort with a key of its own: every card is within 10 points
    # of the best, so the band order must equal sort_key order.
    cards = [
        _meas("claude", 95.0, 0.95, provider="anthropic"),
        _meas("grok", 70.0, 0.90, provider="xai"),
        _meas("codex-b", 40.0, 0.88, provider="openai"),
        _meas("codex-a", 80.0, 0.92, provider="openai"),
    ]

    rows = build_bands(cards)
    b1 = rows[0]

    assert b1.band == "B1"
    assert [c.slug for c in b1.ranked] == ["codex-a", "codex-b", "grok", "claude"]
    assert b1.ranked == sorted(cards, key=sort_key)


# --- 2. unmeasured cannot rank (SC4 / R21) ---------------------------------

def test_unmeasured_card_never_ranks_in_any_band():
    unmeasured = _card("brand-new", 90.0, p0=True)
    measured = _meas("known", 90.0, 0.90)

    rows = build_bands([unmeasured, measured])

    assert not rankable(unmeasured)
    # SC4: no ranked list anywhere in the deck holds an unmeasured card.
    assert all(c.pass_rate is not None and c.trials >= 3 for r in rows for c in r.ranked)
    for r in rows:
        assert unmeasured not in r.ranked
        # It clears every floor on capability, so it is PRESENT — just unranked.
        assert unmeasured in r.unranked


def test_sort_key_refuses_an_unmeasured_card():
    unmeasured = _card("brand-new", 90.0, p0=True)

    with pytest.raises(AssertionError):
        sort_key(unmeasured)


# --- 3/4. P0 failure and unmeasured P0 are unrankable (R28) ----------------

def test_cheapest_card_in_the_band_still_cannot_rank_if_p0_failed():
    # The failing card is first in the cost order and scored highest. Without R28 it
    # would head every band: the cheapest wrong answer winning on cost alone.
    cheapest_but_wrong = _meas("fast-and-wrong", 90.0, 0.99, provider="openai", p0=False)
    correct = _meas("correct-grok", 90.0, 0.90, provider="xai")
    correct_dear = _meas("correct-claude", 90.0, 0.92, provider="anthropic")

    rows = build_bands([cheapest_but_wrong, correct, correct_dear])

    assert not rankable(cheapest_but_wrong)
    for r in rows:
        assert cheapest_but_wrong not in r.ranked
        assert cheapest_but_wrong in r.unranked
        # The next card in the cost order that passed P0 takes the head of the band.
        assert r.ranked[0] is correct
        assert [c.slug for c in r.ranked] == ["correct-grok", "correct-claude"]


def test_unmeasured_p0_is_also_unrankable_only_true_ranks():
    p0_unknown = _meas("p0-unknown", 90.0, 0.9, p0=None)
    p0_failed = _meas("p0-failed", 90.0, 0.9, p0=False)
    p0_passed = _meas("p0-passed", 90.0, 0.9, p0=True)

    assert rankable(p0_unknown) is False
    assert rankable(p0_failed) is False
    assert rankable(p0_passed) is True

    rows = build_bands([p0_unknown, p0_failed, p0_passed])
    for r in rows:
        assert r.ranked == [p0_passed]
        # by slug, not a set of Cards: the per-band dict (R30) makes Card unhashable.
        assert {c.slug for c in r.unranked} == {p0_unknown.slug, p0_failed.slug}


# --- 5. over-qualification fill -------------------------------------------

def test_card_clearing_the_b5_floor_populates_every_band():
    over = _meas("over-qualified", BAND_FLOORS["B5"], 0.9)

    rows = build_bands([over])

    # Uses the module's own floors — no local copy to drift out of date.
    assert _bands_containing(rows, over) == list(BANDS)
    assert all(over in r.ranked for r in rows)


def test_mid_capability_card_fills_only_the_bands_it_clears():
    mid = _meas("mid", 50.0, 0.9)

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
    rows = build_bands([_meas("top", 95.0, 0.9)])

    assert len(rows) == 5
    assert [r.band for r in rows] == ["B1", "B2", "B3", "B4", "B5"]
    assert [r.band for r in rows] == list(BANDS)
    assert "B6" not in {r.band for r in rows}
    assert [r.floor for r in rows] == [BAND_FLOORS[b] for b in BANDS]


# --- 9. per-band measurements (R30) ---------------------------------------

def _entry(cost: float | None, p0: bool | None, pass_rate: float = 0.9, trials: int = 3) -> dict:
    return {"total_cost_usd": cost, "p0_pass": p0, "size_class": "small",
            "cost_basis": "measured", "pass_rate": pass_rate, "trials": trials}


def test_card_ranks_where_it_passed_and_is_unranked_where_it_failed():
    """One card, two verdicts: pass B1 at $0.10, fail B4 at $0.20 (R30).

    Under the one-scalar-per-card Card this card's fate followed file order; the
    per-band shape makes B1 rank it on $0.10 while B4 keeps it present-but-
    unranked for the P0 failure (R28). B2/B3 carry no row, so the card is
    UNMEASURED there — a B1 pass must never rank the same card in a harder band.
    """
    # The scalars deliberately disagree with the B1 row (compatibility fields,
    # last row read): ranking must ignore them once by_band is populated.
    split = _card("split-verdict", 95.0, cost=0.20, p0=False, by_band={
        "B1": _entry(0.10, True),
        "B4": _entry(0.20, False),
    })

    assert rankable(split, "B1") is True
    assert rankable(split, "B4") is False
    assert rankable(split, "B2") is False   # no row -> unmeasured in band

    rows = {r.band: r for r in build_bands([split])}

    assert split in rows["B1"].ranked
    assert split in rows["B4"].unranked
    assert split not in rows["B4"].ranked
    for b in ("B2", "B3"):
        assert split in rows[b].unranked
        assert split not in rows[b].ranked


def _write_inputs(tmp_path, cards_rows, results_rows):
    tmp_path.mkdir(parents=True, exist_ok=True)  # callers may pass a fresh subdir
    cards_p = tmp_path / "cards.json"
    results_p = tmp_path / "results.json"
    cards_p.write_text(json.dumps({"cards": cards_rows}))
    results_p.write_text(json.dumps({"results": results_rows}))
    return cards_p, results_p


def _run_builder(tmp_path, cards_rows, results_rows):
    cards_p, results_p = _write_inputs(tmp_path, cards_rows, results_rows)
    return subprocess.run(
        [sys.executable, str(BIN), "--cards", str(cards_p),
         "--results", str(results_p), "--dry-run"],
        capture_output=True, text=True,
    )


def _cards_row(slug="split-verdict", capability=95.0):
    return {"slug": slug, "provider": "openai", "model": slug, "effort": "base",
            "capability": capability, "price_blended": 1.0, "total_cost_usd": None,
            "p0_pass": None}


def _results_row(card="split-verdict", band="B1", cost=0.10, p0=True,
                 size_class="small"):
    return {"card": card, "band": band, "size_class": size_class,
            "total_cost_usd": cost, "p0_pass": p0, "cost_basis": "measured",
            "pass_rate": 0.9, "trials": 3}


def test_same_card_and_band_with_conflicting_verdicts_fails_the_band(tmp_path):
    """The STOP, now correctly scoped (R30).

    Rows in DIFFERENT bands may disagree — that is the per-band shape working, and
    the build must succeed. Rows for the SAME (card, band) differing only by size
    class have nothing below "band" to separate them, so the builder refuses
    rather than follow file order.
    """
    # Cross-band disagreement: builds fine, exit 0.
    ok = _run_builder(
        tmp_path / "ok",
        [_cards_row()],
        [_results_row(band="B1", p0=True), _results_row(band="B4", p0=False)],
    )
    assert ok.returncode == 0, ok.stderr
    # B1 shows DERIVED, not ESTIMATE: its reference job PASSED (R27).
    assert "B1 floor=35.0 DERIVED ranked=1 unranked=0" in ok.stdout

    # Same (card, band), different verdicts: the band verdict is False (R28),
    # the build succeeds, and the card is unranked in that band.
    clash = _run_builder(
        tmp_path / "clash",
        [_cards_row()],
        [_results_row(band="B1", size_class="small", p0=True),
         _results_row(band="B1", size_class="large", p0=False)],
    )
    assert clash.returncode == 0, clash.stderr
    assert "different p0_pass" in clash.stderr
    assert "ranked=0 unranked=1" in clash.stdout.splitlines()[0]

def test_week_zero_builder_output_is_unchanged(tmp_path):
    """R26 via the builder itself: no results rows means every band ranked=0 and
    the eligible cards present-but-unranked — and the run succeeds."""
    r = _run_builder(
        tmp_path,
        [_cards_row(slug="clears-b5", capability=95.0),
         _cards_row(slug="b1-only", capability=36.0)],
        [],
    )

    assert r.returncode == 0, r.stderr
    for band in ("B1", "B2", "B3", "B4", "B5"):
        line = next(l for l in r.stdout.splitlines() if l.startswith(band + " "))
        # Token match, not substring: "unranked=2" must never satisfy a check
        # written for "ranked=...".
        tokens = line.split()
        assert "ranked=0" in tokens, line
        assert any(t.startswith("unranked=") and t != "unranked=0" for t in tokens), line
    assert "B1 floor=35.0 ESTIMATE ranked=0 unranked=2" in r.stdout
    assert r.stdout.rstrip().splitlines()[-1].startswith("dry-run:")
