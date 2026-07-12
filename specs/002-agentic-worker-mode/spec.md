# 002 — Agentic Worker Mode (delegate DOES the work)

**WHAT / WHY** (no stack). Today `delegate` is TEXT-ONLY: every backend (`call_deepseek`/`call_glm` HTTP, `call_codex` subprocess in a `/private/tmp` sandbox) returns a string. The orchestrator (Claude or codex) must then apply every diff + run every command itself — burning orchestrator tokens on mechanical work the rig's HARD RULE says a worker should do. Mohan 2026-07-12: *"I want no burn on Claude tokens except for orchestration"* and *"the brain-router delegate should be changed … make delegate DO the work (write files)."*

The router must gain an **agentic execution mode**: a worker task shells to the per-provider **CLI harness** in the REAL working directory, so the worker reads the spec, writes files, and runs checks itself. Orchestrator-agnostic: the launcher may be the **codex CLI OR Claude Code** — same router worker path either way.

## End-State (the gold)

`delegate(complexity|role, prompt, mode='agentic'|'chat')` where:

- **`mode='agentic'`** (the DEFAULT for the `worker` role) — the router spawns a CLI agent with real file+shell tools in the caller's cwd, keyed by resolved provider:

  | provider | agentic CLI worker | note |
  |---|---|---|
  | zhipu    | `cc-glm -p <prompt>`      | **first port of call for BOTH worker AND cheap** (both tiers default to GLM) |
  | codex    | `codex exec <prompt>` in **real cwd** (NOT `-C /private/tmp`, NOT `--ignore-rules`) | adversary tier (when CC orchestrates); worker/cheap fallback when GLM exhausted |
  | anthropic | `cc-brain claude -p <prompt>` (Opus CLI) | adversary + worker/cheap fallback **when codex orchestrates** (codex can't be its own adversary — router.py resolve_role rule) OR when codex is also exhausted. Agentic mode shells the resolved Anthropic candidate to the `claude` CLI instead of returning `execute_natively=True` |

  **DeepSeek is OUT** — dropped from the routing entirely (Mohan 2026-07-12). The `cheap` tier no longer maps to deepseek; it defaults to GLM like `worker`. `cc-deepseek` harness is not used.

## Model roster per tier (Mohan 2026-07-12) — candidate lists, first-eligible-non-orchestrator wins

GLM is the FIRST port of call for worker AND cheap. The other candidates are the fallbacks (used on GLM exhaustion / when GLM's provider is the orchestrator), picked by the existing adversary-differs + exhausted-providers rules in `resolve_role`.

| Tier | Candidates (ordered) | Canonical model IDs |
|---|---|---|
| **worker** | GLM 5.2 → Claude Sonnet → Codex 5.6 Terra | `glm-5.2` · `claude-sonnet-5` · `gpt-5.6-terra` |
| **cheap** | GLM 4.7 → Claude Haiku → Codex 5.6 Luna | `glm-4.7` · `claude-haiku-4-5` · `gpt-5.6-luna` |
| **adversary** | Claude Opus 4.8 OR Codex 5.6 Sol (always one; differs from orchestrator) | `claude-opus-4-8` · `gpt-5.6-sol` |

**Model-ID provenance (truth pillar):**
- Anthropic IDs VERIFIED against the built-in `claude-api` skill catalog (cached 2026-06-24): `claude-sonnet-5`, `claude-haiku-4-5`, `claude-opus-4-8` — exact strings, no date suffixes. These map to `--model` on the `claude` / `cc-brain claude` CLI.
- GLM IDs VERIFIED LIVE 2026-07-12 via `MOHAN_CC_GLM_MODEL=<id> cc-glm -p` against z.ai: `glm-5.2`→GLM52-OK, `glm-4.7`→GLM47-OK (both exit 0). The roster originally said "GLM 4.5" but no `glm-4.5` exists anywhere in the stack AND was never a valid z.ai model; corrected to `glm-4.7` per Mohan 2026-07-12 (the real cheap/fast GLM, now live-confirmed).
- Codex IDs VERIFIED LIVE 2026-07-12 via `codex exec -m <model>` (codex-cli 0.144.1): `gpt-5.6-sol`→CODEX-SOL-OK, `gpt-5.6-terra`→TERRA-OK, `gpt-5.6-luna`→LUNA-OK. All three accepted + returned. Probe flags: `--skip-git-repo-check --ignore-user-config -c mcp_servers={} -c model_reasoning_effort="low"` (per feedback_codex_full_config_mcp_stall).

These land in `[roles]` config candidate lists (the mechanism already exists in `resolve_role`) — worker/cheap/adversary each get their ordered model list; provider-diversity + orchestrator-carveout fall out for free.

- **`mode='chat'`** — the existing TEXT-ONLY path, unchanged. Kept for pure refutation / summarize / classify where a worker must NOT touch the filesystem (adversarial-verify wants read-only refuters). DEFAULT for `adversary`/`thinker`/`simple` roles.

- **Persistent, not per-session.** The per-role default mode lives in config (or a code default table), so no orchestrator ever re-asks "should the worker do the work" — the answer is permanently agentic-for-workers. Feedback memory: `feedback_router_agentic_worker_mode`.

## Success Criteria (measurable, derived backward from the gold)

1. **SC-1 (agentic writes files):** `delegate(complexity='code', mode='agentic', prompt='create /tmp/rt_probe_<pid>.txt containing OK')` results in that file existing with content `OK` — proven by a filesystem read in the test, not the returned text. ≥1 automated test.
2. **SC-2 (chat unchanged):** `delegate(complexity='code', mode='chat', prompt=...)` returns text and writes NO file — byte-identical behavior to pre-change `delegate` (regression test: existing test_smoke/test_roles stay green, count moves by exactly the tests added). `cheap` now routes to GLM too (deepseek removed) — assert `cheap` resolves to the glm backend.
3. **SC-3 (default policy):** `delegate(role='worker', prompt=...)` with no explicit mode resolves to agentic; `delegate(role='adversary', ...)` resolves to chat. Asserted in a unit test on the default-resolution function (no backend call).
4. **SC-4 (codex-orchestrator adversary → Opus CLI):** with `orchestrator='codex'`, resolving the `adversary` role in agentic mode targets the `claude` CLI harness (provider=anthropic), NOT native-execute, NOT codex. Unit test on resolve_role + mode.
5. **SC-5 (3 harnesses reachable):** a smoke test (marked slow/manual, real subprocess) confirms each of `cc-glm`/`codex exec`/`cc-brain claude` launches headless and can create a scratch file. ≥1 harness (`cc-glm`) verified live in the assemble transcript; others documented with the exact command. (No `cc-deepseek` — DeepSeek dropped.)
6. **SC-6 (no secret/PII leak in shell):** the agentic worker prompt is passed via argv/stdin the same secure way `call_codex` does today; no API key echoed; `codex exec` runs with the lean flags (mcp_servers={}, low effort) EXCEPT the sandbox-cwd and ignore-rules flags which are dropped so it can edit the repo. Verified by grep of the constructed argv in a test.
7. **SC-7 (deploy-vgate):** served==built — after merge, `mcp-brain-router` MCP restart serves the new `mode` param; a live `delegate(mode='agentic')` call writes a file. Orchestrator gates the merge+restart (never auto-deployed).

8. **SC-8 (exhaustion fallback → Opus CLI as agentic worker):** the orchestrator is carved out of its own worker/adversary pool. Starting point determines the cascade:
   - **Claude Code orchestrates:** worker cascade = GLM (code) → codex (adversarial, on GLM `exhausted=true`) → Opus native (both exhausted). [today's behavior, unchanged]
   - **codex orchestrates:** `cc ⟺ codex CLI` provides worker/adversary/cheap tiers; codex may not be its OWN adversary (resolve_role rule) → adversary resolves to **Opus** (`cc-brain claude` CLI). When **GLM + codex are BOTH exhausted**, the final fallback worker/adversary/cheap is **cc = Claude Code (Opus)** shelled as an agentic CLI — NOT a dead-end.
   In agentic mode, that Opus fallback is `cc-brain claude -p` (writes files), NOT `execute_natively` (there may be no native Anthropic loop when codex is the launcher). Unit test: with `orchestrator='codex'` and `exhausted_providers={zhipu, codex}`, resolving `worker`/`adversary` in agentic mode targets the anthropic CLI harness.

## Non-goals

- Tier→provider mapping IS changing in one way: **DeepSeek removed**; `cheap` now maps to GLM (was deepseek). `code`=glm, `adversarial`=codex unchanged. This is a deliberate net-change, not a non-goal.
- Not touching the 7 stray uncommitted gpt-5.6-migration files on main (leave them; not part of this change).
- Not building a sandbox/permission layer for the agentic worker beyond what the CLI harness itself enforces (future work if needed).
