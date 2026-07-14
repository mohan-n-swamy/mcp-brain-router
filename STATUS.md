# mcp-brain-router — STATUS

**Updated:** 2026-07-14
**Branch:** main
**HEAD:** 9f81c41 — status: park — §9.6 confirmed crash-on-transient bug on main, fix designed not applied
**Tree:** clean

## Current goal

Replace caller-owned complexity cascades with one role-owned agentic provider
shard: GLM → Grok → Codex → Claude, quota-only advancement.

## Latest verified evidence (2026-07-15 — shard SHIPPED + both §9.6 bugs FIXED)

- **Suite: 130 pass, 1 live-skip · ruff clean · diff-check clean** (on main after both fixes).
- Role-owned provider shard (worker=GLM→Grok→Codex→Claude) merged to main via PR #13 (`16fc2b8`).
- §9.6 adversarial-verify (workflow `wf_2e970209-1ff`, 23 agents, 5 lenses) ran AFTER merge → 6 CONFIRMED,
  8 false-positive, 1 unverifiable. The manufacturing-gate correctly BLOCKED the premature "deployed" claim
  → that block forced the pass that caught the bugs. Both confirmed bug-classes now fixed:
  - **Transient cascade-abort (PR #14, `536f7f9`):** `route()`/`route_assignment()` caught only
    `BackendQuotaError`; sibling `BackendTransientError` (5xx/timeout) escaped → broke the server `while`
    cascade loop → shard aborted instead of failing over. Fix: catch it at all 3 sites →
    `RouteResult(exhausted=True, failure_kind="transient_error")` (advance cascade, honest label).
    3 G-guards incl. end-to-end GLM-transient→Grok-failover.
  - **Caller-identity spoof (PR #15, `9b45191`):** `BRAIN_ROUTER_CALLER` read from env unvalidated →
    spoofed caller could bypass adversary-excludes-orchestrator. Fix: `_read_caller()` normalizes any
    non-{claude,codex,grok} value (incl `\f`/`\v` control-char bypasses — hardened past a security-review P2)
    → "unknown". 2 G-guards. Security-reviewed: no P0/P1.
- **Deploy V-gate GREEN for the transient fix:** after MCP restart (serving `536f7f9`), live
  `delegate role='worker'` → `DEPLOY_OK_536f7f9` on disk (glm-5.2, 33s, disk-verified not prose).
- Install is **editable** → restart serves the checked-out ref, no reinstall.

## Blocker

None. Both §9.6 bugs fixed + merged. Suite green.

## Next action

- [ ] **DEPLOY the security fix:** MCP currently serves `536f7f9` (transient fix) but NOT yet `9b45191`
  (security fix). Restart the brain-router MCP once more → then `delegate role='worker'` V-gate = served==built for `9b45191`.
- [ ] Optional: update `wiki/mcp-brain-router.md` (shard shipped); OB-capture the milestone.
- DONE this session: PR #13 (shard), #14 (transient fix), #15 (security fix) all merged; branches
  feat/grok-backend-tier + fix/transient-cascade + fix/caller-identity-whitelist deleted.

Full §9.6 result: `/private/tmp/claude-502/-Users-mohannarayanswamy-code-workshop/2018fb71-702f-48b0-a4e8-d414a28ce886/tasks/wisewq394.output` (610 lines).

---
_Refresh with `bin/gen-status.rb mcp-brain-router` before /save, /park, /wrap-up. Machine header (Updated/Branch/HEAD/Tree) is auto-filled; the prose is yours._
