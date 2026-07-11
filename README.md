# mcp-brain-router

An MCP server for Claude Code that delegates sub-tasks to cheaper and adversarial LLMs by complexity tier, returning structured answers with metadata.

## What It Is

`mcp-brain-router` is a Model Context Protocol server that runs locally and sits between an orchestrator and three external LLM providers. The orchestrator delegates via the `delegate()` tool, specifying complexity (`cheap`, `code`, or `adversarial`). Each tier maps to exactly one backend; the router never crosses tiers. It returns the answer plus terminal metadata. Anthropic's services are never touched by this tool — only by native Claude Code.

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

Configure ordered candidates in `~/.config/mcp-brain-router/config.toml`, then call:

```python
delegate(role="worker", orchestrator="opus", prompt="Implement this isolated task")
delegate(role="adversary", orchestrator="codex", prompt="Refute this design")
```

Roles: `thinker`, `adversary`, `worker`, `simple`. `orchestrator` is never
router-selected. Candidate order comes only from `[roles]`. Genuine quota
exhaustion advances to the next provider. Anthropic candidates return an
`execute_natively=true` assignment stub; this MCP never calls Anthropic.

Legacy `delegate(complexity=..., prompt=...)` remains supported and single-tier.

## Tier Map

| Tier | Backend | Model | Use Case | Cost | Speed |
|------|---------|-------|----------|------|-------|
| `cheap` | DeepSeek | deepseek-v4-flash | Brainstorm, high-volume fan-out, quick checks | ~$0.07/1M tokens | Fastest |
| `code` | GLM | glm-5.2 | Code review, algorithm design, debugging | ~$0.50/1M tokens | Fast |
| `adversarial` | Codex | gpt-5.5 (via codex CLI) | Security reviews, refutation, second opinion | Higher | Slowest |

### Default orchestration policy

- General self-contained work starts with `code` → GLM.
- A Claude orchestrator may call `adversarial` → Codex only after GLM is unusable: `exhausted=true`, an error, or a missing/empty answer.
- If Codex is exhausted, errors, or returns no answer, the orchestrator handles the task natively.
- `cheap` → DeepSeek is an explicit side lane for mechanical/high-volume work.

The MCP performs none of these cross-tier steps automatically. Set `BRAIN_ROUTER_CALLER=claude` in Claude's MCP entry and `BRAIN_ROUTER_CALLER=codex` in Codex's. The server deterministically rejects `adversarial` calls from a Codex caller with `failure_kind="nested_codex_blocked"`; after GLM, Codex must handle the task in its current native session.

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
│   Claude Code (native)   │  Orchestrator
│   (native Claude + MCP)  │  (your subscription)
└────────────┬─────────────┘
             │
             │ MCP call: delegate(prompt, complexity)
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
deepseek_key = "sk-..."
glm_key = "..."
codex_enabled = true
# headroom_base_url = "http://localhost:8282"

[model_overrides]
adversarial = "gpt-5.5"
```

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

**Short version**: This tool uses Claude Code (a first-party Anthropic client) to orchestrate, and delegates to non-Anthropic providers (DeepSeek, GLM, Codex) on *their own keys*. Traffic to those providers never goes through Anthropic's servers or subscription. The orchestrator remains an unmodified native Claude Code instance.

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
