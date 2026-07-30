> **Snapshot:** local e5524eb (working tree DIRTY: budget shards + universal skip-self) · vps kamakshi checked / brain-router NOT deployed · package 0.1.1 · generated 2026-07-30T04:46:58Z · first-hand verified this run.

# PR-FAQ — mcp-brain-router

**BLUF:** Local Model Context Protocol (MCP) server that lets Claude Code, Codex, or Grok **delegate agentic sub-tasks** to cheaper or specialized LLM workers, with **role-owned provider cascades** that advance only on confirmed quota (or transient) exhaustion. Orchestration stays on the seat you chose; grunt work runs on external providers' own credentials.

## Problem

Frontier seats (Claude / Codex / Grok subscriptions) are expensive and quota-capped. Spending them on every file-write or simple transform burns the orchestrator pool. Prior options were: do everything natively (costly), or ad-hoc shell out to another CLI (no shared policy, no audit, no cascade).

## Who it's for

- Mohan's multi-model rig (primary user).
- Anyone running Claude Code / Codex / Grok who wants a **single MCP tool** (`delegate`) with explicit roles and skip-self discipline.
- Not a hosted multi-tenant SaaS — **local stdio process** registered into host CLIs.

## FAQs

### 1. What does this replace?

Ad-hoc `claude -p` / `codex exec` / `grok` shells from orchestrator prompts, and earlier complexity-only DeepSeek/GLM tier routing. One MCP tool + config-owned shards replace both.

### 2. How accurate / reliable is routing?

Role path: ordered candidates from `DEFAULT_ROLES` / config `[roles]`. Advances on **quota exhaustion** and **transient backend errors** (`BackendTransientError` → `failure_kind=transient_error`). Auth, validation, process, and empty-output failures **stop loud** — no silent fall-through. Unit suite exercises resolve_role matrices and cascade labels (this run: see CHANGELOG / CAPABILITIES evidence).

### 3. What does a call cost?

Depends on which provider wins. Volume roles lead with **Kimi** (subscription pool). Claude Fable/Opus only lead thinker/adversary on the largest seat. GLM is **off role cascades** (quality regression 2026-07-21) but still available via explicit `complexity=cheap|code`. No per-call dollar meter in the package — operators watch provider dashboards + `brain-router-delegations.jsonl`.

### 4. What's NOT done?

- Working-tree budget shards + universal skip-self **not committed/pushed** at snapshot HEAD `e5524eb` (docs describe working-tree truth; restart MCP to load).
- No VPS deploy — not on kamakshi containers (ssh-verified 2026-07-30).
- No in-repo CI.
- Public `mode` is **agentic-only**; HTTP chat paths remain for internal/tests.
- GPT-5.6 Sol promotion still eval-gated (prod adversarial complexity default `gpt-5.5`).

### 5. How do I run it?

```bash
pipx install mcp-brain-router   # or brew / pip install -e .
mcp-brain-router-install        # keys → 0600 config → register MCP clients
# Host CLIs then expose tool: delegate
```

### 6. How do I deploy / update?

There is no remote service. Update = reinstall package (pipx upgrade / brew / editable pull) **and kill** any long-lived `mcp_brain_router.server` process so the next session loads new code. Publish channels: GitHub public, PyPI `0.1.1`, Homebrew tap `mohan-n-swamy/tap/mcp-brain-router` (ship session 2026-07-15).

### 7. What about security / compliance?

- Config secrets: `~/.config/mcp-brain-router/config.toml` mode **0600**.
- `BRAIN_ROUTER_CALLER` whitelist (`claude`/`codex`/`grok`) — spoof fix `9b45191`.
- Audit JSONL logs metadata only (**no prompt body**).
- Worker output tagged `source: external-untrusted`.
- Subscription-backed automated Claude workers: treat as personal/testing pending policy — see `COMPLIANCE.md` (prior non-Anthropic-only analysis invalidated after role/CLI migration).

### 8. Why skip-self?

Self-delegation shells a nested CLI of the **same** provider the orchestrator is already on — slow, hook-polluted, same quota pool. Working-tree code applies skip-self to **every** role (was adversary-only on committed HEAD).

### 9. Role vs complexity?

| Axis | Cascades? | When |
|------|-----------|------|
| `role=` worker/simple/thinker/adversary | Yes (quota/transient) | Default standard path |
| `complexity=` cheap/code/adversarial | No (single backend) | Explicit one-provider request only |

### 10. Where is the audit trail?

`~/.local/state/brain-router-delegations.jsonl` — fail-open append per completed `delegate`.

### 11. Can I run this on the VPS (kamakshi)?

**Not currently.** ssh 2026-07-30: no package config, no state dir, no container. Design is Mac-local stdio MCP beside Claude/Codex/Grok CLIs. Open Brain on the VPS is a **different** product.

### 12. How do I verify a change actually loaded?

1. `pytest -q` in the repo.
2. Restart MCP host / `pkill -f mcp_brain_router.server`.
3. Call `delegate(...)` and confirm backend in the last JSONL line matches the expected first eligible provider for that orchestrator.
