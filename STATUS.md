# mcp-brain-router — STATUS

**Updated:** 2026-07-14
**Branch:** feat/grok-backend-tier
**HEAD:** 3ca6fae — fix(security): grok prompt via temp --prompt-file, not -p (injection-safe)
**Tree:** dirty — provider-shard migration in progress

## Current goal

Replace caller-owned complexity cascades with one role-owned agentic provider
shard: GLM → Grok → Codex → Claude, quota-only advancement.

## Latest verified evidence

- `pytest`: 126 passed, 1 live-only skipped. `ruff check .`: PASS. `git diff --check`: PASS.
- Three-lens adversarial review found and repaired quota-cascade, full-exhaustion,
  caller-spoofing, installer-state, schema, env-leak, and compliance drift.
- **Live write-proofs GREEN (2026-07-14, disk-verified not prose):**
  - GLM agentic: `RUN_AGENTIC_LIVE=1 pytest …test_glm_agentic_writes_file` → 1 passed (prior `process_error` was GLM quota, cleared on reset).
  - Grok agentic: direct `call_grok_agentic` probe → file on disk `content='OK'`.
  - Worker-role via LIVE MCP (`delegate role='worker'` after server restart) → `WORKER_ROLE_OK` on disk, backend glm-5.2, 29.7s.
- Install is **editable** (`Editable project location` → this repo) → restart serves the checked-out ref; post-merge `git checkout main` + MCP restart serves main, no reinstall.

## Blocker

None. All V-gates green.

## Next action

- [ ] Land: commit shard migration → PR → merge to main → `git checkout main` → restart MCP → V-gate served==built (`delegate role='worker'` from a fresh session).

---
_Refresh with `bin/gen-status.rb mcp-brain-router` before /save, /park, /wrap-up. Machine header (Updated/Branch/HEAD/Tree) is auto-filled; the prose is yours._
