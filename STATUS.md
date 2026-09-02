# mcp-brain-router — STATUS

**Updated:** 2026-08-25
**Branch:** main
**HEAD:** f1fda8b — feat: budget role shards, universal skip-self, knowledge rebase
**Tree:** DIRTY (CLI-worker contract + glm-5.3 shard ids, uncommitted)
**Knowledge:** wrap 2026-08-25 (CLI contract) · prior rebase 2026-07-30
**Safety tag:** rebase-pre-20260730-1016
**Deploy:** not deployed — `ssh kamakshi` this wrap: hostname vultr; no brain-router git dir; no brain-router container. Open Brain compose is unrelated.

## Current goal

Ship the live-CLI worker contract (Grok/GLM/Kimi/Codex/Anthropic) and respawn the host MCP so `/delegate` uses it.

## What is now true

Agentic CLI workers match the live CLIs, not the 2026-07 mock contract:

- Grok: no `--max-turns 10` (that cancelled real work — 114 audit `process_error` with `max_turns_reached`); JSON `stopReason` accepts `end_turn` and `EndTurn`; no Claude-ID `--tools` allowlist; lean `GROK_HOME` at `~/.local/state/brain-router-grok-home` + `GROK_MEMORY=0`.
- GLM + Anthropic: `--permission-mode bypassPermissions` (was `acceptEdits` — file edits passed, shell verify could die).
- All CLI workers: `process_error` keeps a sanitized stderr/JSON snippet. Quota-only cascade unchanged.

Live config `~/.config/mcp-brain-router/config.toml` roles (also `DEFAULT_ROLES` in WT `config.py`): worker `glm-5.3 → kimi → grok-4.5 → sonnet → terra` · simple same GLM-first · thinker/adversary `grok-4.5` lead, Claude, Codex.

## Latest verified evidence

- **pytest -q** this wrap: **158 passed, 1 skipped** (ruff clean on `backends.py` + `test_agentic.py`).
- **Live V-gate** this wrap: `call_grok_agentic` wrote `OK` to `/tmp/grok-agentic-vgate-20260825/vgate-ok.txt` in 11s; returned content (not `no_tool_effect`).
- **VPS this wrap:** vultr reachable; brain-router **not** installed.
- Editable install: `mcp-brain-router` 0.1.0 → repo `src/`. Host Grok MCP process still holds pre-fix code until respawn.

## Blocker

This Grok session's `brain-router` stdio process is still the old import. `/mcps` disable+enable (or full Grok restart) required. Code is on disk.

## Next action

1. Respawn `brain-router` MCP in the Grok TUI (`/mcps` → Space off/on).
2. Commit WT (don't `git add -A` — `DOCTRINE.md` is untracked extra). Suggested:
   `git add src/mcp_brain_router/backends.py tests/test_agentic.py src/mcp_brain_router/config.py src/mcp_brain_router/router.py src/mcp_brain_router/server.py STATUS.md && git commit`
3. One live `delegate(role="worker", orchestrator="grok", mode="agentic", cwd=…)` and confirm audit JSONL.

---

_Refresh with `bin/gen-status.rb mcp-brain-router` before /save, /park, /wrap-up when that helper is used._
