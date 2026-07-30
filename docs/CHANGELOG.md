> **Snapshot:** local e5524eb (working tree DIRTY: budget shards + universal skip-self) · vps kamakshi checked / brain-router NOT deployed · package 0.1.1 · generated 2026-07-30T04:47:26Z · first-hand verified this run.

# CHANGELOG — mcp-brain-router

## Version 0.1.1 (package)

Shipped 2026-07-15: `--version`/`--help`, sdist allowlist, PyPI + brew + public GitHub.

## Git trail (newest first, this run)

| SHA | Summary |
|-----|---------|
| `e5524eb` | feat: yank GLM from default role cascades |
| `517f41e` | status: kimi shard promotion parked |
| `2fb4e2c` | feat: promote kimi (K3) into default role shards |
| `601ce28` | feat: add kimi as provider |
| `b6de59a` | docs: pipx + brew install (PyPI 0.1.1) |
| `2b9b906` | build: explicit sdist allowlist |
| `3691a26` | chore: bump to 0.1.1 |
| `8376057` | feat: --version / --help CLI flags |
| `86930e6` | chore: gitignore local publish runbook |
| `24469cd` | chore: pre-publish scrub |
| `33848a5` | status: both §9.6 bugs fixed+merged |
| `9b45191` | fix(security): validate BRAIN_ROUTER_CALLER (#15) |
| `536f7f9` | fix: transient error advances role cascade (#14) |
| `9f81c41` | status: park — crash-on-transient noted |
| `16fc2b8` | feat: role-owned provider shard + grok worker (#13) |
| `fd7d1a8` | Merge PR #12 agentic-worker reliability |
| `c947b80` | fix(agentic): AGENTIC_SYSTEM directive |
| `803796b` | fix(agentic): stdin-prompt, sandbox, nested-Codex |
| `18fd1a6` | fix(agentic): PATH-fix env for cc-glm + cc-brain (#11) |
| `f6e0d8c` | fix(agentic): fail loud on missing/invalid cwd (#10) |

## Uncommitted at snapshot (working tree)

- **Budget-optimized DEFAULT_ROLES** (2026-07-26): thinker fable-first, adversary opus-first, worker kimi→grok→sonnet→terra, simple kimi→haiku→luna.
- **Universal skip-self** in `resolve_role` (was adversary-only on HEAD).
- README / STATUS / tests aligned to the above.
- **Not pushed.** MCP restart required for live process.

## Knowledge rebase (2026-07-30)

- First `docs/` surface: PR-FAQ, EXPLAINER, HANDOVER, CHANGELOG, CAPABILITIES, `.rebase-snapshot.json`.
- Quarantined root dep tarballs → `.rebase-quarantine/2026-07-30/`.
- Wiki + index restamped; VPS kamakshi checked (not deployed).

## Component index

| Path | Job |
|------|-----|
| `server.py` | MCP entry, `delegate`, audit, cascade loop |
| `router.py` | Roles, routing, skip-self |
| `backends.py` | Provider clients |
| `config.py` | Config load/save, defaults |
| `install.py` | Installer + client registration |

## External service index

| Service | Endpoint / binary | Auth |
|---------|-------------------|------|
| DeepSeek | api.deepseek.com anthropic messages | config key |
| GLM | api.z.ai anthropic messages | config key |
| Headroom | configurable base URL | passthrough keys |
| Codex | `codex` CLI | CLI auth |
| Grok | `grok` CLI | OAuth login |
| Kimi | `kimi` CLI | OAuth login |
| Anthropic CLI | `cc-brain claude` | Claude CLI creds |
| GLM agentic | `cc-glm` | GLM via CC wrapper |
