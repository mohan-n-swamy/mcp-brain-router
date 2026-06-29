# mcp-brain-router

An MCP server for Claude Code that delegates sub-tasks to cheaper and adversarial LLMs by complexity tier, returning structured answers with metadata.

## What It Is

`mcp-brain-router` is a Model Context Protocol server that runs locally and sits between Claude Code (the orchestrator) and three external LLM providers. When Claude Code encounters a task suitable for a cheaper or independent model, it delegates via the `delegate()` tool, specifying complexity (`cheap`, `code`, or `adversarial`). The router selects the appropriate backend, sends the task using that provider's own API key, and returns the answer along with token counts and cost estimates. Anthropic's services are never touched by this tool — only by native Claude Code.

## Why It Exists

Cost. The orchestrator (Claude) excels at reasoning, planning, and judgment. Sub-tasks like brainstorming, code review, or adversarial second opinions don't require Claude's full power:

- **Cheap tier (DeepSeek V4)**: trivial tasks, brainstorming, high-volume fan-out. Haiku-equivalent speed, ~1/10th the cost.
- **Code tier (GLM-5.2)**: algorithm design, code review, debugging. Sonnet-equivalent reasoning without subscription costs.
- **Adversarial tier (Codex)**: security reviews, high-stakes refutation. Opus-level independent analysis to challenge the orchestrator's solution.

By splitting work this way, you keep your subscription token budget for planning and integration, offload grunt work to cheaper providers on *their* keys, and get a second opinion for risky decisions.

## Install

**Zero-friction setup:**

```bash
git clone https://github.com/mohan-n-swamy/mcp-brain-router.git
cd mcp-brain-router
pip install -e .
mcp-brain-router-install
```

The install script will:
1. Interactively prompt for API keys (GLM and DeepSeek are required; Codex is optional if `codex` binary is present; headroom proxy is optional).
2. Store secrets in `~/.config/mcp-brain-router/config.toml` (mode 0600, gitignored).
3. Self-register with Claude Code by printing and offering to run:
   ```
   claude mcp add brain-router -- python -m mcp_brain_router.server
   ```
4. Run smoke tests (one cheap call, one code call, codex if enabled).

Re-running the script updates keys idempotently without duplicating the registration.

## Usage from Claude Code

```python
from mcp import delegate

# Trivial task: brainstorm 10 ideas
answer = delegate(
    task="Brainstorm 10 naming ideas for a CLI tool that routes tasks to cheaper LLMs.",
    complexity="cheap"
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
    task=f"Review this merge function for correctness and efficiency:\n{code_snippet}",
    complexity="code"
)

# Adversarial: ask Codex to find the worst flaw in your design
answer = delegate(
    task="I'm caching user sessions in memory for 24 hours. What's the single worst thing that can happen?",
    complexity="adversarial"
)
```

## Tier Map

| Tier | Backend | Model | Use Case | Cost | Speed |
|------|---------|-------|----------|------|-------|
| `cheap` | DeepSeek | deepseek-v4-flash | Brainstorm, high-volume fan-out, quick checks | ~$0.07/1M tokens | Fastest |
| `code` | GLM | glm-5.2 | Code review, algorithm design, debugging | ~$0.50/1M tokens | Fast |
| `adversarial` | Codex | gpt-5.5 (via codex CLI) | Security reviews, refutation, second opinion | Higher | Slowest |

## Architecture

```
┌──────────────────────────┐
│   Claude Code (native)   │  Orchestrator
│   (native Claude + MCP)  │  (your subscription)
└────────────┬─────────────┘
             │
             │ MCP call: delegate(task, complexity)
             │
┌────────────▼──────────────────────────────────┐
│     mcp-brain-router (this tool)               │  Local MCP server
│     Routes by complexity tier                  │
│     Returns answer + metadata                  │
└──┬─────────────────┬───────────────┬───────────┘
   │                 │               │
   │ (own API key)   │ (own key)     │ (subprocess)
   │                 │               │
┌──▼──────────┐  ┌───▼────────┐  ┌──▼────────────┐
│ DeepSeek    │  │ GLM (z.ai) │  │  Codex CLI    │  External providers
│ Anthropic-  │  │ Anthropic- │  │  (local exec) │  (NOT Anthropic)
│ compatible  │  │ compatible │  │               │
└─────────────┘  └────────────┘  └───────────────┘

Key point:
- Anthropic's services are ONLY touched by native Claude Code (left side).
- This MCP only orchestrates traffic to external providers using THEIR keys.
- No proxy, no credential sharing, no subscription traffic interception.
```

## Configuration

After installation, your config lives at `~/.config/mcp-brain-router/config.toml`:

```toml
[providers]
deepseek_key = "sk-..."
glm_key = "..."
codex_enabled = true
headroom_base_url = ""  # Optional: leave empty if you don't run a local headroom proxy

[defaults]
deepseek_model = "deepseek-v4-flash"
glm_model = "glm-5.2"
codex_model = "gpt-5.5"
```

**Changing keys later:**
```bash
mcp-brain-router-install  # Re-runs setup, updates config, skips Claude Code re-registration
```

**Validating config:**
```bash
mcp-brain-router-test  # One-shot smoke test to each configured backend
```

## Compliance

**Short version**: This tool uses Claude Code (a first-party Anthropic client) to orchestrate, and delegates to non-Anthropic providers (DeepSeek, GLM, Codex) on *their own keys*. Traffic to those providers never goes through Anthropic's servers or subscription. The orchestrator remains an unmodified native Claude Code instance.

**See `COMPLIANCE.md` for the full, sourced analysis** of how this design respects Anthropic's Consumer Terms and the third-party tool landscape.

## Troubleshooting

**"Config not found / unconfigured backend"**
```bash
mcp-brain-router-install  # Re-run setup
```

**"DeepSeek call failed: 429 rate limit"**
You've hit the free-tier limit. Codex CLI calls still work; GLM may also be available depending on your z.ai plan. The tool returns the error; Claude Code will handle the fallback.

**"Codex not installed"**
The tool is optional. If `codex` binary is not on your PATH, adversarial-tier calls fail with a clear message. Install Codex CLI separately if you want this tier:
```bash
# Installation depends on your Codex provider; example:
pip install codex-cli
```

**"Headroom proxy not responding"**
The tool tries to route DeepSeek/GLM calls through your local headroom proxy (if configured). If headroom is down, it falls back to direct calls to the providers. No interruption; you'll see `"headroom_used": false` in the response metadata.

**"TypeError: delegate() got unexpected keyword"**
Check your function call signature against the Usage section above. The tool accepts `task` (required), `complexity` (required), `context` (optional), and `model` (optional override).

## License

MIT.
