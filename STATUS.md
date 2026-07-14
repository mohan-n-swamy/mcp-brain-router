# mcp-brain-router — STATUS

**Updated:** 2026-07-14
**Branch:** main
**HEAD:** 9f81c41 — status: park — §9.6 confirmed crash-on-transient bug on main, fix designed not applied
**Tree:** clean

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

## Fix landed (branch fix/transient-cascade — 2026-07-15)

The §9.6 cascade-abort bug is FIXED on branch `fix/transient-cascade` (pending PR+merge).
Refined diagnosis (better than the refuters'): NOT an unhandled crash — server.py:440/:276
catch the `BackendError` base gracefully. The real defect was a **cascade-abort**: a transient
5xx/timeout escaped `route_assignment`/`route`'s `except BackendQuotaError`, broke the whole
`while` cascade loop, and returned `exhausted=False` → the shard aborted instead of failing over
GLM→Grok→Codex→Claude. Fix: catch `BackendTransientError` at router.py (anthropic-cli, grok,
route() tier) → `RouteResult(exhausted=True, failure_kind="transient_error")` (advance cascade,
honest label — user chose "advance on transient"). G-guards: `test_codex_transient_advances_cascade_but_is_not_quota`,
`test_anthropic_cli_transient_advances_cascade_not_quota`, `test_worker_transient_on_glm_fails_over_to_next_provider`.
V-gate: 128 pass (was 126, +2 net), ruff clean, diff-check clean, no drift.
NEXT: PR → merge → restart MCP → V-gate served==built.

## Blocker (pre-fix, historical)

**main (`16fc2b8`) had a CONFIRMED crash-on-transient bug — now fixed on branch (above).** §9.6
adversarial-verify (workflow `wf_2e970209-1ff`, 5 lenses, 23 agents) ran AFTER merge and
CONFIRMED 6 findings (the manufacturing-gate was right to block the "shipped-clean" claim):

- **CRITICAL ×2 (same root cause) — the load-bearing bug:** `route()` (router.py:395) and
  `route_assignment()` (router.py:221, +grok path :250) catch ONLY `BackendQuotaError`.
  Backends raise `BackendTransientError` (backends.py:384, a SIBLING class not subclass) on
  5xx/timeout. → transient error escapes UNHANDLED → GLM 503 or codex 360s-timeout CRASHES
  the delegate instead of returning a clean failure. 126 green unit tests missed it (mock the boundary).
- **CRITICAL ×2 (security, lower real-world sev):** `BRAIN_ROUTER_CALLER` read from env
  (server.py:115, :313) with NO whitelist validation → spoofed caller bypasses
  adversary-excludes-orchestrator (server.py:328 only gates known callers). Attacker needs
  process-env compromise, so defer-able but confirmed.
- **HIGH ×1:** same transient-escape at the classify layer (backends.py:445 classifies
  correctly, caller drops it). **MEDIUM ×1:** BRAIN_ROUTER_CALLER single-point-of-failure, no G-guard.
- 8 findings were adversarially triaged FALSE_POSITIVE; 1 UNVERIFIABLE_FROM_CODE (anthropic-cli
  quota-marker match depends on runtime CLI output — flag, don't assert).

Deploy also incomplete: MCP not restarted to serve main; served==built V-gate not run.

## Next action

- [ ] **FIX the transient-escape bug (user approved "fix exception-handler now", interrupted before Edit).**
  DESIGN (already traced): do NOT copy the refuter's "advance cascade same as quota" — backends.py:406-408
  comment is explicit that transient ≠ exhausted (a 503 blip must NOT burn the whole cascade). Correct fix:
  catch `BackendTransientError` at router.py:395 AND :221 AND :250 → return a structured `RouteResult`
  with `exhausted=False`, `failure_kind="transient_error"`, `backend="none"` (clean failure, no false-advance).
  Pattern: `except (BackendQuotaError, BackendTransientError) as e:` then branch on type for exhausted flag,
  OR a second `except BackendTransientError` block. G-guard: `test_transient_does_not_crash_and_not_exhausted`
  (unit test asserting a raised BackendTransientError → RouteResult exhausted=False, no exception).
  WAS MID-TRACE: needed to read server.py cascade loop (how it reads `exhausted` vs an exception) to place fix — do that first.
- [ ] Re-run suite (126+G-guard) + ruff + diff-check. Commit on a `fix/transient-cascade` branch → PR → merge.
- [ ] Defer (or same session): caller-identity whitelist validation (server.py:115/313) + G-guard.
- [ ] THEN deploy: restart brain-router MCP (editable install → serves main), V-gate `delegate role='worker'` → WORKER_ROLE_OK (served==built).
- [ ] Optional: delete branch `feat/grok-backend-tier`; update [[mcp-brain-router]] wiki page.

Full verify result: `/private/tmp/claude-502/-Users-mohannarayanswamy-code-workshop/2018fb71-702f-48b0-a4e8-d414a28ce886/tasks/wisewq394.output` (610 lines).

---
_Refresh with `bin/gen-status.rb mcp-brain-router` before /save, /park, /wrap-up. Machine header (Updated/Branch/HEAD/Tree) is auto-filled; the prose is yours._
