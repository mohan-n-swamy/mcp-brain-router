> **Snapshot:** local e5524eb (working tree DIRTY: budget shards + universal skip-self) · vps kamakshi checked / brain-router NOT deployed · package 0.1.1 · generated 2026-07-30T04:47:26Z · first-hand verified this run.

# HANDOVER — mcp-brain-router

Recreate-from-scratch checklist. Each of the 9 questions has evidence or `[GAP]`.

## The 9 questions

### 1. Runtime + versions

| Item | Value | Evidence |
|------|-------|----------|
| Language | Python ≥3.10 | `pyproject.toml` requires-python |
| Package | `mcp-brain-router` **0.1.1** | `pyproject.toml:6` |
| Git HEAD | `e5524eb` main | `git rev-parse` this run |
| Working tree | **DIRTY** (budget DEFAULT_ROLES + universal skip-self) | `git status -s` |
| Suite | **153 passed, 1 skipped** | `pytest -q` this run, 7.07s |
| Safety tag | `rebase-pre-20260730-1016` | `git tag` this run |

### 2. Env var NAMES (never values)

| Name | Where | Purpose |
|------|-------|---------|
| `BRAIN_ROUTER_CALLER` | `server.py:100` · install registration | Trusted caller: `claude`/`codex`/`grok` else `unknown` |
| `HEADROOM_BASE_URL` | `install.py:141` | Optional proxy detect at install |
| `RUN_AGENTIC_LIVE` | `tests/test_agentic.py` | Opt-in live agentic tests (`=1`) |

Keys are **not** env vars — they live in config.toml (`deepseek_key`, `glm_key`). CLI workers inherit a filtered env allowlist (`HOME`, `PATH`, `CODEX_HOME`, `CLAUDE_CONFIG_DIR`, `XDG_*`, `MOHAN_CC_*`) — `backends.py` `_base_cli_env`.

### 3. External services + auth

| Service | How reached | Auth |
|---------|-------------|------|
| DeepSeek | `https://api.deepseek.com/anthropic/v1/messages` | `x-api-key` from config |
| GLM (z.ai) | `https://api.z.ai/api/anthropic/v1/messages` | `x-api-key` from config |
| Headroom (optional) | `{headroom_base_url}/anthropic/v1/messages` | same keys via proxy |
| Codex CLI | `codex exec` | Codex CLI login / env |
| Grok CLI | `grok` | `grok login` OAuth (no key field) |
| Kimi CLI | `kimi` | OAuth (no key field) |
| GLM agentic | `cc-glm` | wraps Claude-shape + GLM |
| Anthropic agentic | `cc-brain claude` | native Claude CLI credentials |

### 4. Data stores + schema

| Store | Path | Notes |
|-------|------|-------|
| Config TOML | `~/.config/mcp-brain-router/config.toml` | **0600** required; atomic save |
| Audit JSONL | `~/.local/state/brain-router-delegations.jsonl` | fail-open; no prompt body |
| DB | none | — |

### 5. Deploy target + exact command

| Target | Command |
|--------|---------|
| Dev | `pip install -e .` then `mcp-brain-router-install` |
| User pipx | `pipx install mcp-brain-router` → `mcp-brain-router-install` |
| Homebrew | `brew install mohan-n-swamy/tap/mcp-brain-router` → `mcp-brain-router-install` |
| Run server | `mcp-brain-router` or `python -m mcp_brain_router.server` (stdio) |
| VPS kamakshi | **Not deployed** — ssh 2026-07-30 found no config/state/container |

Publish history: GitHub public + PyPI 0.1.1 + brew formula (session 2026-07-15). See rewritten `PUBLISH-RUNBOOK.md`.

### 6. Cron / scheduled jobs

**None in-repo.** No launchd/systemd/GitHub Actions for this package.

### 7. Secrets location

| Secret | Location |
|--------|----------|
| DeepSeek / GLM keys | `~/.config/mcp-brain-router/config.toml` (0600) |
| Codex/Grok/Kimi auth | native CLI stores (not this package's file) |
| PyPI token (publish) | operator machine; historically `~/secret/pypi` (session note) — **[GAP: confirm current path if republishing]** |
| Never commit | `config.toml` gitignored; example only in repo |

### 8. Domains / DNS / ports

| Item | Value |
|------|-------|
| Public HTTP service | **none** (stdio MCP) |
| Optional headroom | default example `http://localhost:8282` |
| VPS hostname | kamakshi → `vultr` (ssh); brain-router **not** bound there |
| DNS | **[GAP: N/A for product; no public route]** — score as satisfied with N/A, not missing |

### 9. How to verify it works

```bash
# Offline suite
cd "<repo>" && pytest -q
# Expect: 153 passed, 1 skipped (this run)

# CLI non-hang
mcp-brain-router --version
mcp-brain-router --help

# Live MCP (after install + restart host)
# Call: delegate(role="worker", orchestrator="<caller>", mode="agentic", cwd="/abs/path", prompt="…")
# Then: tail -1 ~/.local/state/brain-router-delegations.jsonl
# Confirm backend matches skip-self resolution for that orchestrator
```

**served==built for this package** = process restart after code change (`pkill -f mcp_brain_router.server`); there is no container SHA gate.

---

## Components

| Path | Responsibility |
|------|----------------|
| `src/mcp_brain_router/server.py` | MCP tool `delegate`, caller trust, cascade loop, audit log |
| `src/mcp_brain_router/router.py` | Roles, resolve_role, route/route_assignment, skip-self |
| `src/mcp_brain_router/backends.py` | HTTP + CLI workers, AGENTIC_SYSTEM, quota/transient classify |
| `src/mcp_brain_router/config.py` | DEFAULT_ROLES / modes, load/save 0600 TOML |
| `src/mcp_brain_router/install.py` | Interactive install + multi-client MCP register + smoke |
| `tests/` | smoke, roles, kimi, agentic |

## Business rules (load-bearing)

| Rule | Code |
|------|------|
| One public MCP tool: `delegate` | `server.py` tool registration |
| Public path agentic-only + absolute `cwd` | `server.py` validation |
| Roles cascade on quota **and** transient | `router.py` route_assignment |
| Complexity tiers do **not** cascade | `router.py` route |
| Skip-self all roles (WT) | `router.py:163-168` |
| GLM out of DEFAULT_ROLES; still on cheap/code | `config.py:39-48`, tier map |
| Audit fail-open, no prompt body | `server.py:_log_delegation` |

## Working-tree DEFAULT_ROLES (docs truth this run)

```
thinker:    claude-fable-5 → kimi → gpt-5.6-sol
adversary:  claude-opus-4-8 → kimi → gpt-5.6-sol
worker:     kimi → grok-4.5 → claude-sonnet-5 → gpt-5.6-terra
simple:     kimi → claude-haiku-4-5-20251001 → gpt-5.6-luna
```

Committed HEAD `e5524eb` differs (kimi-first thinker/adversary; worker ends terra before sonnet; simple luna-first). **Commit pending.**

## Gotchas

- Stale MCP process serves old code until killed.
- Agentic `cwd` required — server launch dir ≠ your repo.
- Grok prompts via `--prompt-file` (leading-dash `-p` clap bug).
- Never run `Config.save()` against live config from adversarial tests.
- `AGENTIC_SYSTEM` must not be replaced by chat caveman (silent no-write).

## Open decisions

- Commit + push dirty budget/skip-self patch.
- MCP restart so live sessions load WT code.
- Optional PyPI token rotation (session note 2026-07-15).

---

## Handover score: **8/9**

| # | Status |
|---|--------|
| 1 runtime | ok |
| 2 env names | ok |
| 3 external services | ok |
| 4 data stores | ok |
| 5 deploy | ok (local; VPS N/A verified) |
| 6 cron | ok (none) |
| 7 secrets | ok with **[GAP]** on current PyPI token path if republish |
| 8 domains/DNS/ports | ok as N/A for stdio product |
| 9 verify | ok |

**Gaps:** confirm PyPI token location before next publish.
