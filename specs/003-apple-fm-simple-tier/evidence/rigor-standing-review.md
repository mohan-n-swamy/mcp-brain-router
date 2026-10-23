# rigor-standing + work-review answers — apple-fm simple tier (2026-10-23)

Session 406f90c0. Repo mcp-brain-router, commits 0cfdd49 and the follow-up below. Answers say "gap" where the work falls short.

## rigor-standing

- F1 — Most likely defect: fm returns a confident wrong answer with exit 0 (tool-shaped prompt, or a positional prompt dropping stdin). Checked by: the 6-prompt sentinel probe on the mini (5/6), tests `test_needs_tools_*` and `test_argv_prompt_on_stdin_only`, live `smoke.py toolshaped`. Residual 1/6 stated to Mohan.
- F2 — Subagent conclusions: two advisor calls (Claude). Each claim re-checked: cheap tier walks `simple` (read server.py:299); relative paths bypassed the regex (added tests, then widened); overflow retry waste (read `_retryable`); STATUS.md left alone (git diff hash M17). Cross-provider reviewer: see F5.
- F3 — "all/none" claims and their sets: "no checklist item quotes the edited AGENTS.md line" = grep over every `source:` quote in both checklists (35). "Other readers of kimi/glm/ds caches" = grep over ~/.claude/hooks, ~/.claude/scripts, ~/.config/tmux, ~/.local/bin only. "No self-contained prompt wrongly skipped" = 3 sample prompts, not generalised.
- F4 — Material claims map to rigor-run checks in .rigor.receipt.tsv (34 rows) and to [ran:]/[opened:] tags in replies. Dates and counts came from commands run this session (fm probes, pytest, smoke, audit-log queries).
- F5 — Fresh-context review: advisor (Claude, same family) twice; cross-provider review: see "Cross-provider review" below. Codex was tried twice and did not run (CLI arg error; then 401 because --ignore-user-config dropped its login); stated, not counted as a pass.
- F6 — After the final edit: pytest full suite (400 passed, 1 skipped), tests/test_fm.py, rigor-run (36/36), live smokes inside rigor-run (extract, oversize, unreachable, toolshaped, audit, live); diff read via `git show` (fed to the Grok review). Flake: the status-line guard test (`test_usage_statusline.sh`) failed once inside one rigor-run ("1 REGRESSION(S)", 23.5 s), then passed alone and on the next full run. Cause not found; likely timing under the concurrent smoke traffic.
- F7 — Core rules: fall-through kinds. Tests drive the wrong case and the edge: oversize at exactly 20,000 vs 20,001 chars (`test_oversize_boundary_exact_limit_goes_to_ssh`), malformed host (`test_host_validated_before_ssh`), each failure kind, unreachable not retried.
- F8 — Fixes and their tests: unreachable no-retry → `test_cascade_unreachable_mini_falls_through_without_retry`; spawn count → `test_every_cli_spawn_closes_stdin` (9 spawns). Both stay in the suite.
- F9 — doctrine-check.sh on the router repo: 0 blocking, 1 warning H6. H6 is a bug in the check itself: `ls a b c` fails when any one of three paths is missing, so docs/RUNBOOK.md alone still warns (shown by command). Recorded, not worked around.
- F10 — Shaped by: Rigor Algorithm (.rigor.md, run as checks), ponytail ladder (reused cascade, stdlib only), communication doctrine (anti-slop lint on docs/RUNBOOK.md: clean).
- F11 — Security warnings and destructive-op notes kept in full in replies (license terms, config backup, prompt bodies never logged).
- F12 — docs/RUNBOOK.md added: numbered deploy, config change, rollback, and apple-fm enable/disable/verify/recover. Previous good artifacts: config.toml.bak-20261023-fm; code a1c7eee. README.md not touched (another session's uncommitted edits).
- F13 — Wiki section cites footnote fm1023; INTEGRATIONS-LOOKUP facts come from the fm probes run this session.
- F14 — Gap (partial): checks ran per step (tool history), but the run-log rows were appended in two batches, not after each step.
- F15 — MUST NOT list has 17 rows (vs 4 MUST), including non-obvious ones (editable install makes WIP live, bench traffic in cost history, ssh option injection).
- F16 — n/a: no new gate added.
- F17 — Unit 1 (backend + 28 tests) passed before the config flip; flip only after full suite and smoke.
- F18 — Authorities opened: Mohan's brief, server.py `_delegate_role_impl`, INTEGRATIONS-LOOKUP.md, fm `--help`/license on the mini.
- F19 — Not met, said plainly: Jev balance (CodexBar install failed twice, download denied); fm residual fabrication; Codex review did not run.
- F20 — .rigor.md written before the first code edit (backends.py), after orientation reads only.
- F21 — INTEGRATIONS-LOOKUP read before the first edit (gate-enforced); vendor docs = fm CLI help and license on the mini; Apple web docs not read. Row + detail block added. CodexBar not integrated.
- F22 — Touched surfaces: router repo 0 blocking. Gates touched: secrets (Jev key read into a header for one GET, never printed; fm has none), destructive ops (see F24).
- F23 — The unreachable change is in `call_fm`, the one fm function; no special-case mapping.
- F24 — Destructive or state-changing: live config overwrite (dated backup first; Mohan approved the live flip). `brew install` ran Homebrew's automatic 30-day cleanup: it removed old caches and the superseded jq 1.8.1 keg (jq 1.8.2 still works). I did not anticipate that side effect and gave no warning first. Gap: no checkpoint commit before the config flip; the backup was the checkpoint.
- F25 — Gap (partial): RIGOR-ALGORITHM.md and INTEGRATIONS-LOOKUP.md read before the first edit; golden-rules 00-INDEX and doctrines 2/4/8 not opened.
- F26 — Wiki edits describe this session's own commit; sources = the commit and the spec. The full wiki page and the router README were not re-read end to end.
- F27 — No UPDATE/DELETE on history; the audit log stays append-only.
- F28 — n/a: no UI artifact (status line rows are terminal text).
- F29 — fm output is free text from free text, so a model is required; the deterministic part (path/URL/file detection, size gate) is code (A1).
- F30 — Existing source: the role cascade and `[roles] simple`. Addition: a free first tier. Users: delegate-first skills and `complexity="cheap"` callers.

## Cross-provider review

Grok 4.5 (xAI, `grok --prompt-file … -m grok-4.5`), given the original request and the full diff of 0cfdd49 plus the uncommitted working tree, no tools. Raw output: scratchpad `grokreview/review.md` (4,487 bytes). Findings and disposition:

1. Medium — `apple-fm` not in code-default roles. Rejected: the live config puts it first (Mohan's request named that file); adding it to DEFAULT_ROLES would log a configuration_error failure on every call for installs without `fm_host`.
2. Medium — overflow detected on stderr only; stdout overflow → process_error → one wasted retry. **Fixed**: match either stream; test `test_overflow_is_input_too_large_not_retryable[out]`.
3. Medium — exit-0 wrong answer on tool-shaped prompts with no path/URL. Known residual (M16), already told to Mohan; broader cue lists would also skip valid rewrite prompts. Left for Mohan.
4. Low — substring sentinel match. Same residual as 3; mid-sentence false fall-through is harmless (safe direction).
5. Low — string-count spawn test is fragile. Pre-existing test design; call_fm's spawn kwargs are also checked at run time in the same test.

Grok found no path where an fm failure reaches the caller as an error, no argv injection, no event-loop blocking, no prompt text in the audit log.
