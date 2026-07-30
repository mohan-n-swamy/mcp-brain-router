> **Snapshot:** local e5524eb (working tree DIRTY: budget shards + universal skip-self) · vps kamakshi checked / brain-router NOT deployed · package 0.1.1 · generated 2026-07-30T04:47:26Z · first-hand verified this run.

# CAPABILITIES — mcp-brain-router

Canonical use-case inventory. A capability not listed here is not a supported capability.

| # | Slug | Use case (what the user can do) | Trigger | Entry point | Status | Evidence |
|---|------|----------------------------------|---------|-------------|--------|----------|
| 1 | `delegate-by-role` | Run agentic work via role shard with quota/transient cascade | MCP `delegate(role, orchestrator, prompt, cwd)` | `server.py:334` → `router.py:135` | live | tool reg `server.py:488`; `pytest` 153 pass this run |
| 2 | `delegate-by-complexity` | Explicit single-provider call (`cheap`/`code`/`adversarial`) | MCP `delegate(complexity=…)` | `server.py:132` → `router.py:419` | live | tier map in `router.py`; suite green |
| 3 | `skip-self-routing` | Never assign a role to the orchestrator's own provider | automatic inside resolve_role | `router.py:163-168` (WT) | live (WT; HEAD was adversary-only) | code read this run |
| 4 | `agentic-cli-workers` | Worker writes/runs in real `cwd` via provider CLI | agentic mode + abs cwd | `router.py:525` + `backends.call_*_agentic` | live | public path rejects non-agentic |
| 5 | `codex-adversarial-reroute` | Codex host adversarial → Anthropic adversary path | `BRAIN_ROUTER_CALLER=codex` + complexity adversarial | `server.py:176-185` | live | code path |
| 6 | `transient-cascade-advance` | 5xx/timeout advances shard instead of abort | BackendTransientError | `router.py` route_assignment; fix `536f7f9` | live | commit on main |
| 7 | `caller-identity-guard` | Whitelist caller env; reject spoof | `_read_caller` | `server.py:90-103`; fix `9b45191` | live | code path |
| 8 | `delegation-audit-jsonl` | Append metadata audit line per complete | every complete() | `server.py:80,106-129` | live | fail-open design |
| 9 | `install-and-register` | Interactive keys + register Claude/Codex/Grok MCP | `mcp-brain-router-install` | `install.py:542` | live | pyproject scripts |
| 10 | `install-smoke-backends` | Install-time smoke GLM/DeepSeek/Codex | install step | `install.py:361+` | live | install-time only |
| 11 | `model-override` | Per-call or config model override | `model=` / `[model_overrides]` | `router.py` resolve | live | config schema |
| 12 | `cli-version-help` | Non-blocking `--version` / `--help` | argv | `server.py:759+` | live | 0.1.1 release |
| 13 | `headroom-http-proxy` | Route DeepSeek/GLM HTTP via headroom | `headroom_base_url` set | `backends.py` + router | behind-flag / dormant | public path agentic-only |
| 14 | `http-chat-deepseek-glm` | Text-only HTTP chat backends | mode=chat | `backends.py:512+` | dead-code (public) | server rejects non-agentic |
| 15 | `cli-chat-codex-grok-kimi` | Text-only CLI chat | mode=chat | `backends.call_*` chat | dead-code (public) | same |
| 16 | `execute-natively-handback` | Return Anthropic for native orchestrator exec | chat Anthropic candidate | `router.py:184-193` | dead-code (public) | chat rejected on public API |
| 17 | `deepseek-as-default-tier` | DeepSeek as cheap tier | complexity cheap | was old map | dead-code | `_TIER_BACKENDS` maps cheap→glm |
| 18 | `route-sync-library` | Sync Python helper for tests/import | import | `router.py:823` | live (library) | not MCP |

## Domain notes

- **Roles (primary):** thinker / adversary / worker / simple — all agentic in `DEFAULT_ROLE_MODES`.
- **Legacy complexity:** use only for intentional single-provider calls; do not chain to fake a cascade.
- **GLM:** explicit `cheap`/`code` only on role path defaults after 2026-07-21 yank.
