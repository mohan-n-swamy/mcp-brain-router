# mcp-brain-router — STATUS

**Updated:** 2026-09-05
**Branch:** main
**HEAD:** b8e3ae2 — week 0 flip: gate warns (not fails) on short bands, probe skips them, gen skips native rows
**Tree:** DIRTY
**Knowledge:** wrap 2026-08-25 (CLI contract) · prior rebase 2026-07-30
**Safety tag:** rebase-pre-20260730-1016
**Deploy:** not deployed — `ssh kamakshi` this wrap: hostname vultr; no brain-router git dir; no brain-router container. Open Brain compose is unrelated.

## Current goal

Deck routing live for week 0 (spec 001). Watch it; build nothing until use says so.

## What is now true

- `routing_mode = "deck"`, `deck_min_ranked = 1` in `~/.config/mcp-brain-router/config.toml`. Ranked: B2 glm-5.3-flash + glm-4.7-flash · B3 glm-5.3-flash · B5 glm-5.3-flash · B1/B4 none (legacy walk, router warns per call).
- 29 generated agents in `~/.claude/agents` (`generated-by: agent-gen`), `agent-lint` 0/39.
- Route gate in **enforce** for claude, codex, grok, kimi, gemini via `~/.claude/state/route-gate.mode`; observations carry `caller`.
- Bench: 1 trial/cell; contenders need a probed model id (`bench/reachability.json`); refresh monthly under `--max-calls 250`.
- Luna cards unmeasured (codex CLI reports no usage) → never ranked. Design failure recorded: cost measured on flat subscriptions; API already scores capability.

## Latest verified evidence

2026-09-05: `gate-routing-mode.py` rc 0 (WARN simple→B1, adversary→B4 legacy) · `probe-band-winners.py` SC13 PASS B2/B3/B5 ALIVE · route-gate probes: curl blocked / git status allowed on all five adapters · pytest 267 passed, 1 skipped · hook selftest 78/79 (timing budget under load 11).

## Blocker

None.

## Next action

Nothing owed. If a glm answer on a B3 job is bad, that is the signal a P0 fixture is too easy. Deletion pass (API order + probe + n=1 P0, drop cost measurement) only if the deck earns it. Codex owns the claim-discipline mechanism: `/Users/mohannarayanswamy/code workshop/notes/2026-09-05-brief-codex-claim-discipline.md`.
