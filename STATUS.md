# mcp-brain-router — STATUS

**Updated:** 2026-07-20
**Branch:** main
**HEAD:** 2fb4e2c — feat: promote kimi (K3) into default role shards — first in thinker/adversary, second in worker
**Tree:** DIRTY

## Current goal

Kimi K3 promoted into role shards (thinker/adversary first, worker second) — live config + repo defaults synced.

## Latest verified evidence

- Commit 2fb4e2c: `DEFAULT_ROLES` (config.py) + README config example updated.
- Live config verified: `~/.config/mcp-brain-router/config.toml` has kimi first-tier in `thinker` + `adversary`, second-tier in `worker`.
- `resolve_role` V-gate: kimi surfaces on first-candidate exhaustion in all 3 roles.
- **V-gate:** `python3 -m pytest -q` → **151 passed, 1 skipped**.
- Role priority: kimi K3 judged Codex Sol / Claude Fable tier; quota-driven placement (kimi 1%, Codex 91%) preserves Claude + Codex for orchestration.

## Blocker

- **2fb4e2c not pushed** to public GitHub (`mohan-n-swamy/mcp-brain-router`)
- Stray untracked tarballs in repo root: `httpx-0.28.1.tar.gz`, `mcp-1.28.1.tar.gz`
- No live kimi delegation yet — `·Kimi` dot unproven end-to-end through MCP path

## Next action

Push origin main, fire one `delegate(role='thinker', orchestrator='claude', cwd=...)` and confirm `backend=kimi` in `~/.local/state/brain-router-delegations.jsonl`.

---

_Refresh with `bin/gen-status.rb mcp-brain-router` before /save, /park, /wrap-up. Machine header (Updated/Branch/HEAD/Tree) is auto-filled; the prose is yours._
