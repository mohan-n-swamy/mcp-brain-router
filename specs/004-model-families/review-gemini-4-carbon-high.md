SUCCESS 564.472354
7 real defects found (3 high, 2 med, 2 low).

### 1. [HIGH] Missing, corrupt, or unknown-reset quota readings promote expensive routes over measured ones
- **Where:** [quota.py](src/mcp_brain_router/quota.py#L120-L128), [quota.py](src/mcp_brain_router/quota.py#L143-L144)
- **Failure scenario:** `_pace_band()` returns `0` when `el is None or wk is None`, claiming *"a missing reading never steers routing"*. Because `0 < +1`, whenever Antigravity is ahead of pace (e.g. `weekly=25%` on day 1 → band `+1`, 75% Ultra quota still free) and Anthropic has **no valid pace reading** — missing/stale `.anthropic-quota.cache`, `weekly=NaN`, or `{"weekly": 89.0, "weekly_reset_at": None}` — Anthropic gets band `0`, beats Antigravity `(0, 2) < (1, 0)`, moves to slot 0 across `coder`/`processor`/`qa`, and logs the false reason `"claude-sonnet-5-5: anthropic before antigravity (live headroom)"`. This breaks the existing `test_unknown_pace_and_nan_are_never_promoted` guarantee for every family-expanded role.
- **Fix:** Keep routes lacking a valid pace reading (`el is None or wk is None`) pinned at their `COST_RANK` index within `run` (reorder only measured routes among themselves), or assign unmeasured routes band `+1` when comparing against cheaper routes so an unmeasured provider never overtakes a measured one.

### 2. [HIGH] Caller `effort` override bends the floor onto capped `agy-claude-*` routes
- **Where:** [config.py](src/mcp_brain_router/config.py#L113-L122), [server.py](src/mcp_brain_router/server.py#L766-L773)
- **Failure scenario:** `_expand()` bakes the role's default level (`high`) into `ROLES` at import time (`agy-claude-sonnet-5-5-high` first in `coder`). Calling `delegate(role="coder", orchestrator="claude", cwd="/tmp", effort="max", prompt="...")` (or `effort="xhigh"`) resolves to `agy-claude-sonnet-5-5-high`, drops `effort` (`level = None` for `Provider.ANTIGRAVITY`), and runs at `high` (`effort_applied: False`) instead of advancing to the uncapped `claude-sonnet-5-5` sibling in the same family — violating *"never lower the floor"*. Passing `effort="low"` or `"medium"` likewise runs `agy` at hardcoded `-high`.
- **Fix:** When caller `effort` is supplied in `_delegate_role_impl` / `resolve_role`, skip `FAMILIES` routes where `not _reaches(effort, cap)`, and for eligible Antigravity routes substitute `-<effort>` on the CLI model ID.

### 3. [HIGH] A28 skip-self fails in the reverse direction when the caller is `agy-claude-*`
- **Where:** [router.py](src/mcp_brain_router/router.py#L181), [router.py](src/mcp_brain_router/router.py#L277-L282) (vs [plan.md](specs/004-model-families/plan.md#L40-L42))
- **Failure scenario:** Spec 004 §4 states skip-self compares family maker rather than billing provider when the caller's model is known. `resolve_role(Role.ADVERSARY, "agy-claude-opus-5-5-high", config)` reduces `orchestrator` to `orchestrator_owner = Provider.ANTIGRAVITY` at line 181, discarding the model string. `_same_brain("claude-fable-5-2", Provider.ANTHROPIC, Provider.ANTIGRAVITY)` returns `False`, so an `agy-claude-*` caller gets Anthropic Claude (`claude-fable-5-2` / `claude-opus-5-5`) as its own adversary while skipping Gemini (`gemini-4-neon-xhigh`). (And passing `orchestrator="claude-opus-5-5"` from `BRAIN_ROUTER_CALLER=antigravity` raises `validation_error` at [server.py](src/mcp_brain_router/server.py#L534-L539).)
- **Fix:** Extract a `model_maker(model_or_provider)` helper (`agy-claude-*` → `"anthropic"`, `gemini-*` → `"google"`, etc.) and compare makers in `_same_brain` while keeping billing `Provider` in `server.py`'s caller-identity check.

### 4. [MED] `_pace_band()` ignores `_hot()` (`>=90%` 5h/weekly), promoting 5h-hot routes in `order_families()` and emitting contradictory/no-op `quota_note`s in `reorder()`
- **Where:** [quota.py](src/mcp_brain_router/quota.py#L120-L149), [quota.py](src/mcp_brain_router/quota.py#L160-L188)
- **Failure scenario:** With `anthropic` at `five_hour=95, weekly=20, left_days=1` (`_pace_band = -1`, `_hot = "5h 95% used"`) and `antigravity` at `five_hour=10, weekly=80, left_days=1` (`_pace_band = 0`):
  1. Direct `order_families()` puts 95%-5h-exhausted `anthropic` first (violating [plan.md](specs/004-model-families/plan.md#L58-L59)).
  2. `reorder(ROLES["coder"].slots, ...)` puts `antigravity` first and `anthropic` last via `hot`, yet returns the self-contradictory note `"claude-sonnet-5-5: anthropic before antigravity (live headroom); claude-opus-5-5: anthropic before antigravity (live headroom); anthropic moved last: 5h 95% used"`.
  3. On `reorder(_expand(("claude-sonnet-5-5", "high")), ...)` where `anthropic` was already last, `out == saved` (net no-op), yet `reorder()` returns that non-`None` note instead of `None`, causing `_quota_ordered` to attach a `quota_note` when the order did not change.
- **Fix:** Return band `2` from `_pace_band(q, now)` when `_hot(q)` is truthy (so `order_families` never promotes a hot route), and guard `reorder()` with `if out == saved: return saved, None`.

### 5. [MED] `propose.py` keys scores on `s.model` instead of `s.family or s.model`, splitting families across ranks
- **Where:** [config.py](src/mcp_brain_router/config.py#L118), [propose.py](src/mcp_brain_router/propose.py#L41-L52), [propose.py](src/mcp_brain_router/propose.py#L81-L91)
- **Failure scenario:** `python -m mcp_brain_router.propose` currently flags `agy-claude-opus-5-5` and `agy-claude-sonnet-5-5` in `models_without_any_row`, pins them in place as unscored, and sorts direct `claude-*` slots around them — ranking `coder` as `agy-claude-sonnet-5-5@high > claude-opus-5-5@high > claude-sonnet-5-5@high > agy-claude-opus-5-5@high > gemini-4-neon@high` (interleaving Sonnet and Opus across family boundaries).
- **Fix:** Use `fam = s.family or s.model` in `propose.py` for score lookups, `top`, `in_role`, and `missing` (or collapse consecutive same-`(family, level)` slots before ranking).

### 6. [LOW] `order_families()` groups runs by `s.family` without `s.level` and misformats notes for >=3-route families
- **Where:** [quota.py](src/mcp_brain_router/quota.py#L137-L147)
- **Failure scenario:**
  1. If a role lists consecutive fallback levels of the same model — e.g. `(("claude-sonnet-5-5", "xhigh"), ("claude-sonnet-5-5", "medium"))` — `order_families` merges both levels into one `family` run and sorts `agy-claude-sonnet-5-5-medium` (cost 0) ahead of `claude-sonnet-5-5@xhigh` (cost 2), lowering the `xhigh` floor to `medium` (and colliding on bare `.id` in [server.py](src/mcp_brain_router/server.py#L469) `by_id` and [config.py](src/mcp_brain_router/config.py#L189) `slot_level`).
  2. When a family has 3 routes (`[antigravity, grok, anthropic]` per `COST_RANK`) and `grok` is `+1` while `antigravity` and `anthropic` are `0`, `best` becomes `[antigravity, anthropic, grok]`; `best != run` is `True`, and line 146 emits `"claude-sonnet-5-5: antigravity before antigravity (live headroom)"`.
- **Fix:** Group consecutive runs by `(s.family, s.level)`, and format the note from the first index `k` where `best[k] != run[k]`.

### 7. [LOW] `_expand()` with `level=None` emits bare `agy-claude-*` IDs rejected by `agy` CLI
- **Where:** [config.py](src/mcp_brain_router/config.py#L75), [config.py](src/mcp_brain_router/config.py#L109-L118)
- **Failure scenario:** `_expand()` accepts bare model strings (`level=None`) and `_reaches(None, "high")` returns `True`. `_expand("claude-sonnet-5-5")` produces `Slot("agy-claude-sonnet-5-5", None, "claude-sonnet-5-5")` with `.id == "agy-claude-sonnet-5-5"`, which passes `_AGY_MODEL_RE` and invokes `agy --model claude-sonnet-5-5` — rejected at runtime because `agy models` only accepts `claude-sonnet-5-5-{low,medium,high}`. (`test_every_gemini_role_id_is_one_agy_lists` in [test_antigravity.py](tests/test_antigravity.py#L255) also checks only `gemini*`, skipping `agy-claude-*`.)
- **Fix:** In `_reaches(level, cap)`, return `cap is None` when `level is None` (or require an explicit `level` for `FAMILIES` entries in `_expand`), and check `m.removeprefix("agy-")` against `agy models` in `test_antigravity.py`.


[exited with code 0]
