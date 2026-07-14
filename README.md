# mcp-brain-router

An MCP server for Claude Code and Codex that delegates agentic sub-tasks to external CLI workers, returning structured answers with metadata.

## What It Is

`mcp-brain-router` is a local Model Context Protocol server shared by Claude Code, Codex, and Grok. Standard calls select a role (`worker`, `simple`, `thinker`, or `adversary`) and run an agentic CLI worker in the caller's absolute working directory. Each role owns an ordered provider shard and advances only on confirmed quota exhaustion. Legacy complexity calls (`cheap`, `code`, `adversarial`) remain single-provider.

## Why It Exists

Keep orchestration separate from worker selection. The human chooses Claude, Codex, or Grok as orchestrator; the router owns the worker shard and its quota-only fallback policy:

- **Worker**: GLM 5.2 → Grok → Codex Terra → Claude Sonnet 5.
- **Simple**: GLM 4.7 → Codex Luna → Claude Haiku.
- **Thinker**: Claude Fable → Codex Sol.
- **Adversary**: Claude Opus 4.8 → Codex Sol, excluding the orchestrator's provider.

Every role call is agentic-only and requires `cwd`. Timeouts, authentication errors, process errors, and empty output stop loud; they never trigger provider fallback.

## Install

**Zero-friction setup:**

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
#   "backend": "deepseek",
#   "model": "deepseek-v4-flash",
#   "complexity": "cheap",
#   "tokens_in": 120,
#   "tokens_out": 340,
#   "est_cost": 0.0024,
#   "source": "external-untrusted"
# }

# Code review: ask GLM to review Python logic
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

Configure ordered candidates in `~/.config/mcp-brain-router/config.toml`, then call. This role-based path owns the standard agentic cascade; callers must not recreate it with repeated `complexity=` calls:

```python
delegate(role="worker", orchestrator="claude", mode="agentic", cwd="/abs/repo", prompt="Implement this isolated task")
delegate(role="adversary", orchestrator="codex", mode="agentic", cwd="/abs/repo", prompt="Refute this design")
```

Roles: `thinker`, `adversary`, `worker`, `simple`. `orchestrator` is never
router-selected. Candidate order comes only from `[roles]`. Standard worker
order is GLM → Grok → Codex → Claude. Only genuine quota exhaustion advances;
timeouts, process errors, authentication failures, and empty answers stop loud.
All public role calls are agentic-only and require absolute `cwd`. Anthropic
candidates run through `cc-brain claude`. Only the adversary role excludes the
orchestrator's provider; every other role may use it.

Legacy `delegate(complexity=..., prompt=...)` remains supported and single-tier.
Use it only for an explicit one-provider request, not the standard cascade.

## Tier Map

| Tier | Backend | Model | Use Case | Cost | Speed |
|------|---------|-------|----------|------|-------|
| `cheap` | GLM | glm-4.7 | Explicit fast single-provider request | — | Fastest |
| `code` | GLM | glm-5.2 | Code review, algorithm design, debugging | ~$0.50/1M tokens | Fast |
| `adversarial` | Codex | gpt-5.5, low effort (via Codex CLI) | Security reviews, refutation, second opinion | Higher | Slowest |

### Enforced role policy

- `worker`: GLM 5.2 → Grok → Codex Terra → Claude Sonnet 5.
- `simple`: GLM 4.7 → Codex Luna → Claude Haiku.
- `thinker`: Claude Fable → Codex Sol.
- `adversary`: Claude Opus 4.8 → Codex Sol; candidate matching the orchestrator provider is skipped.
- Provider advancement happens only on confirmed quota exhaustion. Timeout,
  process, authentication, permission, and empty-output failures stop loud.

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
│     Routes by configured role candidates       │
│     Returns answer + metadata                  │
└──┬─────────────────┬───────────────┬───────────┘
   │                 │               │
   │ (own API key)   │ (OAuth CLI)   │ (subscription CLI)
   │                 │               │
┌──▼──────────┐  ┌───▼────────┐  ┌──▼────────────┐
│ GLM         │  │ Grok/Codex │  │  Claude CLI   │  Agentic workers
│ API + CLI   │  │ local CLI  │  │  fallback     │
│             │  │            │  │               │
└─────────────┘  └────────────┘  └───────────────┘

Key point:
- The router may launch native Claude CLI as a configured role fallback.
- Provider credentials stay inside their native API/CLI auth paths.
- Worker output is labeled external-untrusted regardless of provider.
```

## Configuration

After installation, your config lives at `~/.config/mcp-brain-router/config.toml`:

```toml
deepseek_key = "sk-..."
glm_key = "..."
codex_enabled = true
grok_enabled = true
# headroom_base_url = "http://localhost:8282"

[model_overrides]
adversarial = "gpt-5.5"

[roles]
thinker = ["claude-fable-5", "gpt-5.6-sol"]
adversary = ["claude-opus-4-8", "gpt-5.6-sol"]
worker = ["glm-5.2", "grok-4.5", "gpt-5.6-terra", "claude-sonnet-5"]
simple = ["glm-4.7", "gpt-5.6-luna", "claude-haiku-4-5-20251001"]
```

### GPT-5.6 routing policy

- `gpt-5.6-sol` is the capability-first adversarial candidate. Production default remains `gpt-5.5` until representative eval promotion.
- `gpt-5.6-terra` is an explicit cost-balanced candidate; adopt only after the same representative adversarial eval holds.
- `gpt-5.6-luna` is for efficient high-volume work, not the default refutation/security lane.
- Keep `model_reasoning_effort="low"` as the migration baseline; compare `none` on the same eval before lowering it.
- Reserve pro mode or `max` effort for quality-first tasks with measured gain; the CLI worker does not enable either by default.
- Codex runs as an ephemeral, direct CLI worker. Responses-API explicit caching and Programmatic Tool Calling do not apply to this backend unless it is deliberately migrated to that API surface.
- Model efficiency never replaces caller fallback, timeout, nested-Codex prevention, untrusted-output handling, or downstream verification.

**Changing keys later:**
```bash
mcp-brain-router-install  # Re-runs setup, updates config, skips Claude Code re-registration
```

**Validating the repo:**
```bash
pytest -q
```

**Validating the installed MCP path:**
```python
delegate(complexity="code", prompt="Health check. Reply exactly PONG.")
```

## Compliance

**Short version**: role shards can launch GLM, Grok, Codex, or native Claude CLI workers. The prior non-Anthropic-only compliance analysis is invalid after this migration. Treat subscription-backed automated workers as personal/testing-only pending explicit policy confirmation.

**See `COMPLIANCE.md` for the full, sourced analysis** of how this design respects Anthropic's Consumer Terms and the third-party tool landscape.

## Troubleshooting

**"Config not found / unconfigured backend"**
```bash
mcp-brain-router-install  # Re-run setup
```

**"DeepSeek call failed: 429 rate limit"**
You've hit the provider limit. The tool returns `exhausted=true`; the orchestrator decides whether to call another tier or handle the task natively. The router never crosses tiers itself.

**"Codex not installed"**
The tool is optional. If `codex` binary is not on your PATH, adversarial-tier calls fail with a clear message. Install Codex CLI separately if you want this tier:
```bash
# Installation depends on your Codex provider; example:
pip install codex-cli
```

**"Headroom proxy not responding"**
The tool routes DeepSeek/GLM through your local headroom proxy when configured. If that proxy is down, the call returns an error; clear `headroom_base_url` or restart the proxy to use the backend again.

**"TypeError: delegate() got unexpected keyword"**
Check your function call signature against the Usage section above. The tool accepts `complexity` (required), `prompt` (required), and `model` (optional override).

## License

MIT.
