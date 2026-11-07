# mcp-brain-router

An MCP server for Claude Code and Codex that delegates agentic sub-tasks to external CLI workers, returning structured answers with metadata.

## What It Is

`mcp-brain-router` is a local Model Context Protocol server shared by Claude Code, Codex, and Grok. Standard calls select a role (`worker`, `simple`, `thinker`, or `adversary`), and each role walks an ordered list of backends fixed in this repo. Legacy complexity calls remain: `cheap` walks `simple`, `code` walks `worker`, and `adversarial` is Codex alone.

## Why It Exists

Keep orchestration separate from worker selection. The human chooses Claude, Codex, or Grok as orchestrator; the router owns the worker order, free models first (D5, 2026-11-06):

- **Simple**: Apple fm (on the Mac mini) → gpt-oss:20b (this host's Ollama, only when already loaded) → Claude Haiku 5.5.
- **Worker**: Kimi → Grok 4.5 → DeepSeek V4 Pro → Claude Sonnet 5 → Codex `gpt-6.1-sol`.
- **Adversary**: Codex `gpt-6.1-sol` → Claude Opus 5.5 → Kimi. Never the caller's own provider.
- **Thinker**: Grok 4.5 → Claude Fable 5 → Codex `gpt-6.1-sol`.

A call walks its role's whole order within one deadline (`simple` 120 s, the others 15 min). A backend with no binary or key is skipped; an error, a timeout or an empty answer falls through to the next. The orders live in `config.DEFAULT_ROLES`; a host `config.toml` `[roles]` table is ignored, so a scan of this repo sees every model ID the router uses.

## Install

**pipx (recommended — fast, isolated):**

```bash
pipx install mcp-brain-router
mcp-brain-router-install
```

**Homebrew:**

```bash
brew install mohan-n-swamy/tap/mcp-brain-router
mcp-brain-router-install
```

**From source (dev):**

```bash
git clone https://github.com/mohan-n-swamy/mcp-brain-router.git
cd mcp-brain-router
pip install -e .
mcp-brain-router-install
```

The install script will:
1. Interactively prompt for API keys and detect installed Codex/Grok CLIs.
2. Store secrets in `~/.config/mcp-brain-router/config.toml` (mode 0600, gitignored).
3. Register the same MCP in installed Claude Code, Codex, and Grok clients with distinct caller identity.
   ```
   claude mcp add brain-router -e BRAIN_ROUTER_CALLER=claude -- python -m mcp_brain_router.server
   ```
4. Run smoke tests (one cheap call, one code call, codex if enabled).

Re-running the script updates keys idempotently without duplicating the registration.

## Usage from Claude Code

```python
from mcp import delegate

# Trivial task: brainstorm 10 ideas
answer = delegate(
    complexity="cheap",
    prompt="Brainstorm 10 naming ideas for a CLI tool that routes tasks to cheaper LLMs."
)
print(answer)
# Returns:
# {
#   "answer": "1. TaskRouter\n2. TierDispatch\n...",
#   "backend": "apple-fm",
#   "model": "apple-fm",
#   "complexity": "cheap",
#   "tokens_in": 120,
#   "tokens_out": 340,
#   "est_cost": 0.0024,
#   "source": "external-untrusted"
# }

# Code review: walks the worker role
code_snippet = "def merge(a, b): return sorted(a + b)"
answer = delegate(
    complexity="code",
    prompt=f"Review this merge function for correctness and efficiency:\n{code_snippet}"
)

# Adversarial: ask Codex to find the worst flaw in your design
answer = delegate(
    complexity="adversarial",
    prompt="I'm caching user sessions in memory for 24 hours. What's the single worst thing that can happen?"
)
```

### Role-based routing

The role path owns the standard order; callers must not recreate it with repeated `complexity=` calls:

```python
delegate(role="worker", orchestrator="claude", mode="agentic", cwd="/abs/repo", prompt="Implement this isolated task")
delegate(role="adversary", orchestrator="codex", mode="agentic", cwd="/abs/repo", prompt="Refute this design")
```

Roles: `thinker`, `adversary`, `worker`, `simple`. `orchestrator` is never
router-selected. Role calls require an absolute `cwd`. Claude candidates run
through `cc-brain claude` in `cwd` with tools. The adversary role skips the
orchestrator's own provider; the other roles walk their order, own provider
included.

Legacy `delegate(complexity=..., prompt=...)` remains: `cheap` walks `simple`,
`code` walks `worker`, `adversarial` is Codex alone (for class `orange`, it
walks `adversary`).

## Tier Map

| Tier | Backend | Model | Use Case | Cost | Speed |
|------|---------|-------|----------|------|-------|
| `cheap` | the `simple` walk | fm, gpt-oss:20b, Haiku | Short self-contained text | Lowest | Fastest |
| `code` | the `worker` walk | see Worker above | Self-contained code task | Varies | Varies |
| `adversarial` | Codex, tools-off | `gpt-6.1-sol`, low effort | Security reviews, refutation, second opinion | Higher | Slowest |

### Delegation policy (lean-rig PLAN §6; `scripts/policy-check.sh` proves it, I22)

- Every prompt opens with a rules preamble of at most 1,200 bytes
  (`policy.PREAMBLE`): precedence, data is never an instruction, evidence for
  claims, no one-way-door actions. Answers return as untrusted worker output.
- `class` is `personal` or `orange` (default: the host's `default_class`).
  `orange` reaches Claude only and refuses rather than falls through.
- Non-Claude workers run tools-off with the prompt inline: Grok with
  `--max-turns 1 --no-subagents --no-plan --disable-web-search` and
  `--output-format json`, returning only the final message; Codex with a
  read-only sandbox, the shell tool off and the prompt on stdin (an open stdin
  makes `codex exec` wait for EOF). Kimi has no verified tools-off mode, so it
  is skipped.
- gpt-oss:20b answers `simple` only, only when `/api/ps` shows it loaded (cold
  means the next tier, no wait), one generation at a time (5 s lock wait), 60 s.
- A worker that needs tools, and every review courier, runs under
  `scripts/sandbox.sh` (bwrap): only its worktree (read-only for reviews) and
  that CLI's login are visible, the env is allowlisted, and class `orange` is
  refused.

Claude, Codex, and Grok each register the same MCP with distinct
`BRAIN_ROUTER_CALLER` identity. Orchestrator remains human-selected.

Codex registration example:

```toml
[mcp_servers.brain-router]
command = "python"
args = ["-m", "mcp_brain_router.server"]
env = { BRAIN_ROUTER_CALLER = "codex" }
```

## Architecture

```
┌──────────────────────────┐
│ Claude / Codex / Grok    │  Human-selected orchestrator
│ native CLI + same MCP    │
└────────────┬─────────────┘
             │
             │ MCP call: delegate(role, prompt, cwd)
             │
┌────────────▼──────────────────────────────────┐
│     mcp-brain-router (this tool)               │  Local MCP server
│     Walks the role's order (this repo)         │
│     Returns answer + metadata                  │
└──┬─────────────────┬───────────────┬───────────┘
   │                 │               │
   │ (ssh, HTTP)     │ (OAuth CLI,   │ (subscription CLI)
   │                 │  API key)     │
┌──▼──────────┐  ┌───▼────────────┐  ┌──▼────────────┐
│ Apple fm,   │  │ Grok / Codex   │  │  Claude CLI   │
│ gpt-oss     │  │ tools-off,     │  │  (tools, in   │
│ text only   │  │ DeepSeek HTTP  │  │   the cwd)    │
└─────────────┘  └────────────────┘  └───────────────┘

Key point:
- The router may launch native Claude CLI as a role's Claude entry.
- Provider credentials stay inside their native API/CLI auth paths.
- Worker output is labeled external-untrusted regardless of provider.
```

## Configuration

After installation, your config lives at `~/.config/mcp-brain-router/config.toml`:

```toml
deepseek_key = "sk-..."
codex_enabled = true
grok_enabled = true
kimi_enabled = true
fm_host = "mini"            # ssh host for Apple fm; omit to skip that tier
# default_class = "orange"  # the Orange host's default
# headroom_base_url = "http://localhost:8282"

[model_overrides]
adversarial = "gpt-6.1-sol"
```

Role orders are not configured here; they live in `config.DEFAULT_ROLES`.
gpt-oss reads its endpoint from `MOHAN_OLLAMA_HOST` (env, or
`~/.config/mohan/compute.env`).

### Codex model

- `gpt-6.1-sol` is the only model the ChatGPT login accepts (tested 2026-11-06), so it is the default and the role entry.
- Keep `model_reasoning_effort="low"` as the baseline; compare `none` on the same eval before lowering it.
- Codex runs as an ephemeral, tools-off CLI worker. Model efficiency never replaces caller fallback, timeouts, untrusted-output handling, or downstream verification.

**Changing keys later:**
```bash
mcp-brain-router-install  # Re-runs setup, updates config, skips Claude Code re-registration
```

**Validating the repo:**
```bash
uv run --extra dev pytest -q
bash scripts/policy-check.sh   # I22; exit 2 until a root-owned /usr/bin/bwrap exists
```

**Before the first commit in a clone** (this repo is public; the hook runs
gitleaks with `.gitleaks.toml`, which also blocks node IPs and home paths):
```bash
ln -s ../../scripts/pre-commit "$(git rev-parse --git-common-dir)/hooks/pre-commit"
```

**Validating the installed MCP path:**
```python
delegate(complexity="code", prompt="Health check. Reply exactly PONG.")
```

## Compliance

**Short version**: role walks can launch Grok, Codex, DeepSeek, Apple fm, Ollama, or native Claude CLI workers. The prior non-Anthropic-only compliance analysis is invalid after this migration. Treat subscription-backed automated workers as personal/testing-only pending explicit policy confirmation.

**See `COMPLIANCE.md` for the full, sourced analysis** of how this design respects Anthropic's Consumer Terms and the third-party tool landscape.

## Troubleshooting

**"Config not found / unconfigured backend"**
```bash
mcp-brain-router-install  # Re-run setup
```

**"DeepSeek call failed: 429 rate limit"**
You've hit the provider limit. A role call moves on to the next backend in its order; a legacy `adversarial` call returns `exhausted=true` and the orchestrator decides what next.

**"Codex not installed"**
The tool is optional. If `codex` binary is not on your PATH, adversarial-tier calls fail with a clear message. Install Codex CLI separately if you want this tier:
```bash
# Installation depends on your Codex provider; example:
pip install codex-cli
```

**"Headroom proxy not responding"**
The tool routes DeepSeek through your local headroom proxy when configured. If that proxy is down, the call returns an error; clear `headroom_base_url` or restart the proxy to use the backend again.

**"TypeError: delegate() got unexpected keyword"**
Check your function call signature against the Usage section above. The tool accepts `complexity` (required), `prompt` (required), and `model` (optional override).

## License

MIT.
