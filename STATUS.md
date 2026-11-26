# mcp-brain-router — STATUS

**Updated:** 2026-11-26
**Branch/HEAD:** main @ ab97a2d (PRs 30, 31, 32 merged)
**Hosts:** arunachala at ab97a2d by rig-pull; annapurna `~/work/mcp-brain-router` at ab97a2d (pulled by hand: rig-pull covers the router on consumer Macs only); kedarnath checkout on main. A running MCP server keeps the old code until `/mcp` reconnect.

## Current goal

Spend Google AI Ultra where it meets the job's floor (doctrine A30): one model by several routes, the route with the most live quota room first, Antigravity then Grok then Claude when all are on pace.

## Latest verified evidence

- `config.FAMILIES`: Sonnet 5.5 and Opus 5.5 by `agy-claude-*` (Antigravity, up to high) and direct. Floors: coder, processor/worker, qa run Claude at high (agy route first); writer, thinker, adversary keep xhigh/max (direct only). Spec and review: `specs/004-model-families/`.
- Workers read files by default (thinker, adversary, processor, writer); coder and qa write; simple stays tools-off with a read upgrade for file-naming prompts.
- Suite: 613 passed, 3 failing as on main (live agy lean test, deck provider names, sandbox refusals). Cross-provider reviews by gemini-4-carbon-high on PRs 31 and 32, findings fixed.
- Live: coder resolved to `agy-claude-sonnet-5-5-high` and finished a two-file job in 41 s; `--effort xhigh` gave `antigravity:below_floor` and direct Sonnet answered; processor read a file through Gemini with `tools: read`.

## Blocker

None in the router. Codex refuses every model for the ChatGPT account (plan or login), which keeps the I7 fitness row red.

## Next action

After `/mcp` reconnect, watch a day of `quota_note` in `~/.local/state/brain-router-delegations.jsonl` for family reorders. Follow-ups: A28 by model maker (a Claude session inside agy registers as `antigravity`); a longer per-entry cap for agy on long agentic jobs; a route-id check for `agy-claude-*` against `agy models` in the test suite.
