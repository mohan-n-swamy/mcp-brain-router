"""deck.py — C02/C03: the band/card deck (specs/001-agent-capability-routing).

Types and both contracts are specified verbatim in that pack's lld.md. This module
holds the ONLY ordering rule in the system, and the gate that is deliberately not
part of it.

C02 owns Card, BandRow, sort_key and build_bands. C03 owns gated(). The split is
the point: cost ranks, quota gates, and the two never mix (R25). An earlier design
folded headroom into the cost figure, which made a busy provider look expensive and
a rested one look cheap -- the deck then reordered itself as quota drifted, with no
measurement having changed.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal, Optional

logger = logging.getLogger(__name__)

Band = Literal["B1", "B2", "B3", "B4", "B5"]
Effort = Literal["base", "low", "medium", "high", "xhigh"]

# R3: hand-chosen, and the deck says so. Every floor ships floor_status "ESTIMATE"
# until a reference-job pass derives it, because a typed floor that manufactures a
# conclusion is the same defect the fetched provider table exists to prevent.
BAND_FLOORS: dict[str, float] = {"B1": 35.0, "B2": 48.0, "B3": 55.0, "B4": 59.0, "B5": 62.0}
BANDS: tuple[str, ...] = ("B1", "B2", "B3", "B4", "B5")


@dataclass(frozen=True)
class Card:
    slug: str                        # AA slug, hyphenated (R11): "glm-5-3-flash"
    router_model: str                # the name the router already uses: "glm-5.3-flash"
    provider: str                    # anthropic|zhipu|moonshot|xai|openai|deepseek
    model: str
    effort: Effort
    capability: float                # composed index (R9), 0-100
    price_blended: float             # reference only; NEVER the sort key (R19/R25)
    total_cost_usd: Optional[float]  # MEASURED median of n=3 (R21/R30). None => unrankable
    size_class: Literal["small", "large"]   # <10k / >=10k input tokens (R30)
    p0_pass: Optional[bool]          # None => unmeasured; False => unrankable in band (R28)


@dataclass(frozen=True)
class BandRow:
    band: Band
    floor: float
    floor_status: Literal["ESTIMATE", "DERIVED"]
    ranked: list[Card]               # measured, P0-passing, sorted
    unranked: list[Card]             # clears the floor, no measurement yet (R26)


# C03's postcondition is a literal grep for quota/headroom inside sort_key, so those
# words are kept OUT of the function entirely -- including its docstring. The rule they
# state lives here instead: sort_key is the ONLY ordering rule in the system, it ranks
# on measured cost then capability, and nothing about provider load may enter it. C03's
# STOP says that adding it means the design drifted back to headroom-as-cost, where the
# deck silently reorders as load drifts and no measurement has changed.
def sort_key(c: Card) -> tuple[float, float]:
    """The single authoritative ordering rule (R25): cost ASC, capability DESC (R18)."""
    assert c.total_cost_usd is not None, "unmeasured card cannot rank (R21)"
    return (c.total_cost_usd, -c.capability)


def rankable(c: Card) -> bool:
    """A card ranks only if cost was MEASURED and no P0 acceptance item failed.

    Two independent disqualifiers, and both matter. No cost means R21's rule that an
    unmeasured card cannot rank -- otherwise a newly-shipped model captures routing on
    an attractive index alone, before anyone has seen it work. A failed P0 means R28:
    without it the cheapest wrong answer wins its band, because a polished, materially
    incorrect reply returned in four seconds for a cent beats every correct one on cost.
    """
    return c.total_cost_usd is not None and c.p0_pass is True


def build_bands(cards: list[Card], floors: dict[str, float] | None = None,
                derived: set[str] | None = None) -> list[BandRow]:
    """Place every card in its own band and every band below it.

    Over-qualification is free: a card that clears B5 can do B1 work, so it populates
    B1 too. Computed once here at build time and never at lookup, which is what keeps
    the router's hot path a list read rather than a scan.
    """
    floors = floors or BAND_FLOORS
    derived = derived or set()
    rows: list[BandRow] = []
    for b in BANDS:
        floor = floors[b]
        eligible = [c for c in cards if c.capability is not None and c.capability >= floor]
        ranked = sorted([c for c in eligible if rankable(c)], key=sort_key)
        unranked = [c for c in eligible if not rankable(c)]
        rows.append(BandRow(
            band=b, floor=floor,
            # R27/R3: DERIVED requires a supporting reference-job pass. Absent that,
            # the floor is a guess and the file says so to every reader.
            floor_status="DERIVED" if b in derived else "ESTIMATE",
            ranked=ranked, unranked=unranked,
        ))
    return rows


def gated(band_row: BandRow, quota: dict[str, float | None],
          threshold: float = 0.90) -> list[Card]:
    """C03: remove cards whose provider is past threshold. Order is NEVER changed.

    This is a filter over an already-sorted list, so every survivor keeps its relative
    position. That is SC6, and it is the whole distinction between gating and sorting.
    """
    def used(provider: str) -> float:
        u = quota.get(provider)
        # None means NO QUOTA SIGNAL, not "zero used". A pay-as-you-go provider
        # (deepseek) has a balance, not a weekly fraction; a dead poller gives the
        # same shape. Both mean "cannot gate this", and C03's rule is that a missing
        # signal must never empty a band -- so treat it as ungated.
        #
        # The isinstance guard fixes a real crash, not a hypothetical one. The first
        # version read quota.get(provider, 0.0) < threshold, which raises
        # TypeError: '<' not supported between instances of 'NoneType' and 'float'
        # the moment deepseek enters the deck. A .get() default only covers a MISSING
        # key; deepseek has a key whose value is a balance, so the default never fires.
        return u if isinstance(u, (int, float)) else 0.0

    survivors = [c for c in band_row.ranked if used(c.provider) < threshold]
    if band_row.ranked and not survivors:
        # C03's STOP: gating must never silently empty a band. Report the hole; do not
        # fall through to an over-quota provider, and do not quietly return nothing.
        logger.warning(
            "headroom gate emptied band %s: all %d ranked cards are past %.0f%% quota. "
            "This is a routing hole, not a routing decision.",
            band_row.band, len(band_row.ranked), threshold * 100,
        )
    return survivors


def read_quota(path=None) -> dict[str, float | None]:
    """Per-provider weekly usage from headroom.json (C14), keyed by Card.provider.

    Unreadable source returns {} and logs at WARNING -- gated() then treats every
    provider as ungated, because C03's rule is that a missing signal must never empty
    a band. Failing open here is deliberate: a dead poller must not stop the rig.
    """
    import json
    import pathlib
    p = pathlib.Path(path) if path else pathlib.Path.home() / ".local" / "state" / "headroom.json"
    try:
        doc = json.loads(p.read_text())
    except Exception as e:  # noqa: BLE001 -- a dead quota source must not stop routing
        logger.warning("headroom unreadable (%s); no provider will be gated: %s", p, e)
        return {}
    return {name: entry.get("used_week") for name, entry in (doc.get("providers") or {}).items()}


# The router and the deck name the same providers differently: the router's
# Provider enum says kimi and codex where cards.json and headroom.json say
# moonshot and openai. This is the ONE place that translation lives. A lookup
# that crosses the two without it does not fail -- it reads nothing, which is
# how a provider at 68% weekly once displayed as having no quota at all.
DECK_TO_ROUTER: dict[str, str] = {"moonshot": "kimi", "openai": "codex"}
ROUTER_TO_DECK: dict[str, str] = {v: k for k, v in DECK_TO_ROUTER.items()}


def router_provider_name(deck_provider: str) -> str:
    return DECK_TO_ROUTER.get(deck_provider, deck_provider)


def load_deck(path=None) -> dict[str, BandRow]:
    """deck.json -> {band: BandRow}. Unreadable deck returns {} so the caller can
    fall through to the legacy walk; a missing deck must never stop routing."""
    import json
    import pathlib
    p = pathlib.Path(path) if path else pathlib.Path.home() / ".local" / "state" / "brain-router" / "deck.json"
    try:
        doc = json.loads(p.read_text())
    except Exception as e:  # noqa: BLE001 -- fail open to legacy
        logger.warning("deck unreadable (%s); legacy routing only: %s", p, e)
        return {}
    out: dict[str, BandRow] = {}
    for row in doc.get("bands") or []:
        cards = lambda xs: [Card(**{k: x.get(k) for k in Card.__dataclass_fields__}) for x in xs]  # noqa: E731
        out[row["band"]] = BandRow(
            band=row["band"], floor=row["floor"], floor_status=row["floor_status"],
            ranked=cards(row.get("ranked") or []), unranked=cards(row.get("unranked") or []),
        )
    return out
