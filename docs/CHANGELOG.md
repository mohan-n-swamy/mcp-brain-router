> **Snapshot:** main `3cab472` on kedarnath, annapurna and arunachala, read with `git log` on each host on 2026-11-25. The sections below the 2026-11-25 entry, and the other files in `docs/`, are a snapshot of 2026-07-30 and are stale.

# CHANGELOG — mcp-brain-router

## 2026-11-25: roles, failover, tools, quota (PRs 23 to 27)

The state now. The one definition of a role is `config.ROLES`; the README block and the tool description are generated from it.

| PR | What is true after it |
|---|---|
| 23 | Three ids moved to their successors: `grok-4.7`, `claude-fable-5-2`, `claude-sonnet-5-5` |
| 24 | Eight roles in one table: thinker, adversary, coder, processor, writer, explorer, qa, simple. Each slot is a model and a thinking level. `worker` is an alias of `processor`. A call with no role is refused with the table; a `roles` tool returns it. Any failure marks the model, not its provider; after a quota or timeout failure a sibling is pinged first, result kept 5 minutes (`ping.py`). Local entries are capped at 15 s and each entry leaves a reserve for the ones after it. Thinker, adversary, writer and qa have a 30-minute deadline and a 15-minute call |
| 25 | The walk reads the status line's quota files (`quota.py`): a provider at 90% or more goes last in every role; in coder, processor, explorer and simple a provider ahead of its weekly pace gives up first place; a missing or stale reading changes nothing. `scores.tsv` holds dated, sourced scores; `python -m mcp_brain_router.propose` ranks each role from them and prints where that differs from the table. It only prints |
| 26 | `tools` is `none`, `read` or `write`; coder and qa default to write, explorer to read, the rest to none. Grok and Antigravity honour it on the same terms as Claude. Write mode has no containment on any provider. Antigravity read mode reads named files and cannot search or list. The response reports the tools mode that ran |
| 27 | `mcp-brain-router delegate --role <role>` and `mcp-brain-router roles` from the shell, for scripts and scheduled jobs. Exit 0 on an answer, 1 otherwise |

Suite on the branch of PR 27: 571 passed, 2 failed, 2 skipped on kedarnath (macOS). The failures predate these PRs: `test_deck_and_router_provider_names_agree_on_every_live_card` and `test_sandbox_refusals` (Linux only). `test_real_agy_one_shot_is_lean_with_the_rig_plugin_installed` is a live token-count test that fails on some runs.

Known and open:

- A tools-off Grok call on a 16 KB review prompt timed out twice (360 s, then 570 s). The earlier failure, ending with no final message, has a unit test; no live call has shown Grok answer that prompt.
- Every Claude worker call runs with edit and shell tools, in every role. A `none` call answered by Claude reports `write`.
- The weekly model check does not yet call `propose`. No scheduled job calls the router yet.
- 38 of the 46 score rows are search quotes, marked `quote` in the table.
- Decision record: dotagents `specs/lean-rig/DECISIONS.md`, D19.

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
