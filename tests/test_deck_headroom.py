"""SC6 (C03): headroom GATES, it never SORTS.

The distinction this file defends: `gated()` is a filter over an already-sorted
list. It may remove cards; it may never change the relative order of the ones it
keeps, and it may never influence `sort_key`. An earlier design folded headroom
into the cost figure, so the deck silently reordered itself as load drifted with
no measurement having changed. These tests are what make that regression loud.

Contract: specs/001-agent-capability-routing/components/C03-headroom-gate.md
"""

import inspect
import logging

import pytest

from mcp_brain_router.deck import BandRow, Card, gated, sort_key


def _card(slug: str, provider: str, cost: float, capability: float = 80.0) -> Card:
    return Card(
        slug=slug,
        router_model=slug,
        provider=provider,
        model=slug,
        effort="medium",
        capability=capability,
        price_blended=99.0,  # deliberately misleading: never the sort key (R19/R25)
        total_cost_usd=cost,
        size_class="small",
        p0_pass=True,
    )


@pytest.fixture
def band_row() -> BandRow:
    """Six cards, cost-ascending, providers deliberately INTERLEAVED.

    zhipu sits at positions 0 and 4 with three other providers between them, so
    any per-provider regrouping or re-sort inside gated() shows up immediately as
    a changed order rather than hiding behind an accidentally-stable arrangement.

    deepseek is present on purpose: it is pay-as-you-go, so its live headroom
    entry carries a balance and no `used_week`. xai is present and deliberately
    ABSENT from every quota dict below — a provider the poller never reported.
    """
    return BandRow(
        band="B3",
        floor=55.0,
        floor_status="ESTIMATE",
        ranked=[
            _card("glm-5-3-flash", "zhipu", 0.001),
            _card("gpt-5-6-terra", "openai", 0.002),
            _card("deepseek-v4", "deepseek", 0.003),
            _card("kimi-k3", "moonshot", 0.004),
            _card("glm-5-3-air", "zhipu", 0.005),
            _card("grok-5", "xai", 0.006),
        ],
        unranked=[],
    )


def _quota(zhipu: float) -> dict[str, float | None]:
    """The live quota shape: numbers, a null (deepseek), and a missing key (xai)."""
    return {
        "zhipu": zhipu,
        "openai": 0.10,
        "moonshot": 0.10,
        "deepseek": None,
    }


# --- 1. ORDER PRESERVED -----------------------------------------------------
# The half that proves gating is a filter, not a re-rank.

def test_order_of_survivors_is_identical_at_20_percent_and_95_percent(band_row):
    rested = gated(band_row, _quota(zhipu=0.20))
    busy = gated(band_row, _quota(zhipu=0.95))

    # Both calls must be filters of the SAME source list, in source order.
    assert rested == [c for c in band_row.ranked if c in rested]
    assert busy == [c for c in band_row.ranked if c in busy]

    # And the survivors common to both appear in the same relative order.
    survived_both = [c for c in rested if c in busy]
    assert [c.slug for c in busy] == [c.slug for c in survived_both]

    # Stated once more as literal sequences, so a future reorder cannot pass by
    # satisfying a clever set comparison.
    assert [c.slug for c in rested] == [
        "glm-5-3-flash",
        "gpt-5-6-terra",
        "deepseek-v4",
        "kimi-k3",
        "glm-5-3-air",
        "grok-5",
    ]
    assert [c.slug for c in busy] == [
        "gpt-5-6-terra",
        "deepseek-v4",
        "kimi-k3",
        "grok-5",
    ]


def test_pairwise_precedence_is_never_inverted_by_quota(band_row):
    """Stronger than sequence equality: no surviving PAIR may swap.

    Sweeps zhipu across the whole range. For every pair of cards that survives
    two different quota states, the one that came first must still come first.
    """
    baseline = {c.slug: i for i, c in enumerate(band_row.ranked)}
    for level in (0.0, 0.20, 0.50, 0.89, 0.95, 1.0):
        survivors = gated(band_row, _quota(zhipu=level))
        positions = [baseline[c.slug] for c in survivors]
        assert positions == sorted(positions), (
            f"gated() reordered survivors at zhipu={level}: {positions}"
        )


# --- 2. GATED PROVIDER ABSENT ----------------------------------------------

def test_gated_provider_cards_are_absent_at_95_percent(band_row):
    survivors = gated(band_row, _quota(zhipu=0.95))

    assert all(c.provider != "zhipu" for c in survivors)
    assert "glm-5-3-flash" not in {c.slug for c in survivors}
    assert "glm-5-3-air" not in {c.slug for c in survivors}
    # Removed, not demoted: the count drops by exactly the gated provider's cards.
    assert len(survivors) == len(band_row.ranked) - 2


# --- 3. NULL IS NOT ZERO ----------------------------------------------------

def test_null_used_week_is_ungated_not_zero(band_row):
    """deepseek is pay-as-you-go: it carries a balance, not a weekly fraction.

    None means NO SIGNAL. C03's rule is that a missing signal must never empty a
    band, so a null-quota provider survives.

    Note what this test can and cannot separate. `used()` reaches "ungated" by
    mapping None to 0.0, so null-as-zero and null-as-ungated produce identical
    survivors for any positive threshold. The ONE observable difference from the
    pre-fix code is the crash — pinned by the next test, which is why both exist.
    """
    quota = {"zhipu": 0.95, "openai": 0.95, "moonshot": 0.95, "deepseek": None}

    survivors = gated(band_row, quota)

    assert "deepseek-v4" in {c.slug for c in survivors}


def test_pre_fix_comparison_on_none_raises_typeerror():
    """The regression this guards is a real crash, not a hypothetical one.

    The first version of the gate read `quota.get(p, 0.0) < threshold`. A .get()
    default only covers a MISSING key — deepseek HAS a key whose value is None,
    so the default never fires and the comparison raises. Asserting the crash
    here means the old expression cannot quietly return.
    """
    quota: dict[str, float | None] = {"deepseek": None}

    with pytest.raises(TypeError) as exc:
        _ = quota.get("deepseek", 0.0) < 0.90  # the pre-fix expression, verbatim

    assert "'<' not supported between instances of 'NoneType' and 'float'" in str(
        exc.value
    )

    # The shipped gate handles the same input without raising, and keeps the card.
    row = BandRow(
        band="B3",
        floor=55.0,
        floor_status="ESTIMATE",
        ranked=[_card("deepseek-v4", "deepseek", 0.003)],
        unranked=[],
    )
    assert [c.slug for c in gated(row, quota)] == ["deepseek-v4"]


# --- 4. MISSING KEY ---------------------------------------------------------

def test_provider_absent_from_quota_dict_is_ungated(band_row):
    """xai never appears in the quota dict. A dead poller must not empty a band."""
    survivors = gated(band_row, _quota(zhipu=0.95))
    assert "grok-5" in {c.slug for c in survivors}


def test_empty_quota_dict_gates_nothing(band_row):
    """An unreadable headroom source yields {} (see read_quota) — fail open."""
    assert gated(band_row, {}) == band_row.ranked


# --- 5. THRESHOLD BOUNDARY --------------------------------------------------

def test_exactly_at_threshold_is_gated_just_under_is_not(band_row):
    """The implementation keeps a card on `used < threshold`.

    So 0.90 is already past the gate and 0.8999... is not. Asserted from the
    source's comparison, not guessed.
    """
    at = gated(band_row, _quota(zhipu=0.90))
    assert all(c.provider != "zhipu" for c in at)

    under = gated(band_row, _quota(zhipu=0.8999999))
    assert [c.slug for c in under if c.provider == "zhipu"] == [
        "glm-5-3-flash",
        "glm-5-3-air",
    ]


def test_boundary_honours_a_custom_threshold(band_row):
    assert all(c.provider != "zhipu" for c in gated(band_row, _quota(0.50), threshold=0.50))
    assert any(c.provider == "zhipu" for c in gated(band_row, _quota(0.49), threshold=0.50))


# --- 6. EMPTY BAND WARNS ----------------------------------------------------

def test_fully_gated_band_returns_empty_and_logs_a_warning(band_row, caplog):
    """C03's STOP: gating must never SILENTLY empty a band."""
    quota = {
        "zhipu": 0.99,
        "openai": 0.99,
        "moonshot": 0.99,
        "deepseek": 0.99,
        "xai": 0.99,
    }

    with caplog.at_level(logging.WARNING, logger="mcp_brain_router.deck"):
        survivors = gated(band_row, quota)

    assert survivors == []

    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert warnings, "gating emptied a band without logging a WARNING"
    assert "B3" in warnings[0].getMessage()


def test_a_band_that_keeps_a_card_does_not_warn(band_row, caplog):
    with caplog.at_level(logging.WARNING, logger="mcp_brain_router.deck"):
        survivors = gated(band_row, _quota(zhipu=0.95))

    assert survivors
    assert [r for r in caplog.records if r.levelno == logging.WARNING] == []


def test_an_already_empty_band_does_not_warn(caplog):
    """No ranked cards is a measurement gap, not a gating hole — different alarm."""
    row = BandRow(band="B5", floor=62.0, floor_status="ESTIMATE", ranked=[], unranked=[])

    with caplog.at_level(logging.WARNING, logger="mcp_brain_router.deck"):
        assert gated(row, {"zhipu": 0.99}) == []

    assert [r for r in caplog.records if r.levelno == logging.WARNING] == []


# --- 7. SORT KEY UNTOUCHED --------------------------------------------------

def test_sort_key_output_is_independent_of_quota(band_row):
    """Same input, same key — whatever the quota state is."""
    before = [sort_key(c) for c in band_row.ranked]

    for level in (0.0, 0.20, 0.90, 0.99, 1.0):
        gated(band_row, _quota(zhipu=level))
        assert [sort_key(c) for c in band_row.ranked] == before

    # The key is cost ASC then capability DESC — nothing about provider load.
    assert before == sorted(before)
    assert sort_key(_card("x", "zhipu", 0.5, capability=90.0)) == (0.5, -90.0)


def test_sort_key_takes_no_quota_and_mentions_none(band_row):
    """C03's postcondition: no quota/headroom concept inside sort_key at all."""
    assert list(inspect.signature(sort_key).parameters) == ["c"]

    source = inspect.getsource(sort_key).lower()
    assert "quota" not in source
    assert "headroom" not in source


def test_gated_does_not_mutate_the_band_row(band_row):
    """A filter returns a new list; the source ranking is left exactly as built."""
    original = list(band_row.ranked)

    gated(band_row, _quota(zhipu=0.95))

    assert band_row.ranked == original
    assert [c.slug for c in band_row.ranked] == [
        "glm-5-3-flash",
        "gpt-5-6-terra",
        "deepseek-v4",
        "kimi-k3",
        "glm-5-3-air",
        "grok-5",
    ]
