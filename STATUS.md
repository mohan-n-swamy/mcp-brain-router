# mcp-brain-router — STATUS

**Updated:** 2026-10-23
**Branch/HEAD:** main @ 9fdb496
**Tree:** Dirty. README.md, tests/test_effort_plumbing.py and `.session/brain-router-effort-fix/` carry earlier sessions' uncommitted edits, left untouched.

## Current goal

Apple's on-device fm model (Mac mini, over ssh) is the first tier of the `simple` role. Watch real traffic before changing anything.

## Latest verified evidence

- Live config: `fm_host = "mini"`, `simple = ["apple-fm", "claude-haiku-4-5-20251001", "grok-4.5"]`; backup `config.toml.bak-20261023-fm` (2026-10-23).
- Live smoke (`specs/003-apple-fm-simple-tier/smoke.py`): a short extract is answered by fm in ~2.6 s; oversized, unreachable-mini and tool-shaped prompts fall through to haiku with the reason in the audit line's `failures`.
- Suite: 400 passed, 1 skipped. Rigor: `specs/003-apple-fm-simple-tier/.rigor.md` 36 of 36 green. Cross-provider review by Grok 4.5: one fix (overflow on stdout), the rest recorded in `specs/003-apple-fm-simple-tier/evidence/rigor-standing-review.md`.
- Commits: `0cfdd49` (tier), `4b2dbc9` (no retry on unreachable, overflow on either stream, `docs/RUNBOOK.md`), `9fdb496` (review evidence). Pushed.
- Jev `decide` unchanged since 2026-09-18 (`docs/jev-provider.md`); office-email key answers first.

## Blocker

None for the fm tier. Known residual: with no tools, fm can invent an answer to a tool-needing prompt that names no path, file or URL (1 of 6 in the probe), and that is logged as an fm success.

## Next action

After a day of real `simple` traffic, read the status-line row `fm(mini): 24h a/b answered` and the audit `failures`, and spot-check fm answers. Jev balance in the status line waits on `brew install --formula steipete/tap/codexbar` (two download failures on 2026-10-23) and a CodexBar sign-in to the TypeSafe console.
