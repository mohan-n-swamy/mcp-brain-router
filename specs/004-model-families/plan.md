# 004 — One model, many routes

Mohan, 2026-11-26: "the brain router needs to know all that is available across providers
and use that, so claude direct sonnet and agy sonnet are different but equivalent; same
with gpt-oss on antigravity and on annapurna. The idea is to optimise performance with
cost. The floor is set by job and tier."

This extends his 2026-10-01 rule (deck.py): best result first, lowest cost second, where
cost is which flat subscription's week he would rather spend, not a list price.

## The problem today

A role's order names routes, not models. `claude-sonnet-5-5` (direct, Claude Max week)
and `agy-claude-sonnet-5-5` (Antigravity, Google AI Ultra) are two unrelated entries,
placed by hand (PR #30). Nothing in the router knows they are the same model, so:

- the quota pace logic can move a provider, but cannot say "same model, cheaper route";
- a new route (another provider starts serving Sonnet) means editing every role by hand;
- A28 compares providers, not model families (review finding 1 on PR #30).

## Design

1. **Families, one table** (`config.FAMILIES`). A family is one model at one quality:
   `claude-sonnet-5-5`, `claude-opus-5-5`, `gemini-4-neon`, `gpt-oss-120b`, ... Each lists
   its routes with the levels each route can serve:
   - `claude-sonnet-5-5`: `anthropic` (low..max), `antigravity` as `agy-claude-sonnet-5-5` (low..high)
   - `claude-opus-5-5`: `anthropic` (low..max), `antigravity` (low..high)
   - `gemini-4-*`: `antigravity` only
   - `gpt-oss-20b`: `ollama` (annapurna). `gpt-oss-120b`: `antigravity` until it retires
     2026-11-30. These are **not** one family: 20b and 120b are different models.
2. **Roles name families at a level: that is the floor.** The job (role) and tier pick
   the family and level; the router may serve it by any route that reaches that level.
   A route that cannot reach the level (agy tops out at `high`) is not eligible for an
   `xhigh`/`max` slot, so the floor is never quietly lowered.
3. **Route order inside a family = cost.** Among eligible routes, the router takes the
   one whose week Mohan would rather spend: `COST_ORDER` gains `antigravity` (proposed
   first: flat fee, 3% used), then the existing quota pace and the 90% "move last" rule
   reorder within the family. A failed route skips that route only; a quota stop skips
   that provider for every family, as today.
4. **A28 by family vendor.** Skip-self compares the family's maker (Claude, Gemini,
   Grok, GPT-OSS), not the billing provider. Fixes PR #30 finding 1 when the caller's
   model is known.
5. **Orange unchanged.** `claude_only` keeps only routes with provider `anthropic`.

## What changes for a caller

Nothing in the call. `render_roles()` shows families; the response names the route that
answered (`agy-claude-sonnet-5-5-high` vs `claude-sonnet-5-5`) and why (`cost: antigravity
before anthropic` / `anthropic hot: 93%`).

## Dynamic balance (Mohan, 2026-11-26: "balance the quota available at any point in time
## and make the routing dynamic")

Route order inside a family is decided per call, from the live quota reading
(`quota.read()`, the same source the status bar shows), not fixed in the table:

- **Headroom score** per provider = share of the week left minus share of the week's
  quota used (positive = under pace, room to spend). The 5-hour window counts too: a
  provider past 90% of its 5-hour window drops to last until it resets.
- Routes order by headroom score; COST_ORDER breaks ties within a small band (10 points),
  so Ultra is spent first while every provider is on pace, and the router shifts to
  whichever subscription has the most room as the week runs on.
- No reading for a provider = no signal, not zero use: it keeps its COST_ORDER place.
- The response carries `quota_note` saying which route won and why.

## Decided (Mohan, 2026-11-26)

- COST_ORDER: antigravity, then xai (Grok), then anthropic.
- The floor never bends: an `xhigh`/`max` slot never goes to a route capped at `high`;
  the walk moves on to the next family instead.
- Claude floors: coder, processor/worker and qa run Claude at `high`, so the Ultra route
  (agy) is eligible and goes first. writer and thinker keep `xhigh`/`max` and use direct
  Claude only. This replaces PR #30's hand-placed agy slots, which lowered floors silently.

## Checks

- Every role slot resolves to a family; every family route id is one its CLI lists
  (`agy models`, ollama tags).
- Same-family routes order by COST_ORDER, and quota hot moves only that provider.
- A Claude caller's adversary never resolves to any Claude-family route.
- Class orange resolves only to `anthropic` routes.
- Level floor: no `max`/`xhigh` slot resolves to a route capped at `high`.
