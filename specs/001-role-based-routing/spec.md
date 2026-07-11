# 001 — Role-based routing for brain-router

**Status:** design-approved, pre-build
**Author:** this session (2026-07-11), on Mohan's spec
**Builds on:** the uncommitted "pure tier ownership" refactor already in the working tree (47 tests green) — role-routing lands as ONE combined change with it (Mohan's call).

## WHAT / WHY (no stack)

Today `delegate(complexity)` routes work by a 3-tier axis (cheap/code/adversarial → deepseek/glm/codex). Mohan wants a **5-ROLE** model instead: each role is a distinct job in an orchestrated build loop, filled by a preference-ordered list of models, with a hard rule that the **adversary must come from a different provider than the orchestrator**.

The orchestrator is NOT chosen by the router — it is **whoever the human launched** (start Claude → Opus orchestrates; start Codex → Codex orchestrates). The router's job: given who is orchestrating, resolve the OTHER roles to concrete models, enforce the adversary-differs rule, and — when a role lands on an Anthropic model — return an *assignment* for the main session to execute (the router never calls Anthropic; COMPLIANCE.md invariant).

### The 5 roles

Resolution rule: each role's candidate list is ordered **GLM → Codex → Claude-last** (cheapest-provider first, Anthropic last). The cascade walks the list **only on exhaustion** (existing `exhausted=true` quota signal). The **adversary is the one exception** — it is resolved to differ in provider from the orchestrator, so it cannot blindly follow cost order.

| # | Role | Job | Candidate list (ordered; walked on exhaustion) |
|---|---|---|---|
| 1 | **orchestrator** | starts + ends, calls everyone, checks outputs, owns the loop | **whoever the human launched** — Opus-4.8-1M OR Codex-Sol-Terra. NOT router-picked (env input). |
| 2 | **thinker** | ultra-setting planner; lays out plan/specs (manuf-product-design) | Codex-Sol → Fable-5 |
| 3 | **adversary** | refute; **MUST differ in provider from the orchestrator** | orch=Anthropic → Codex-Terra; orch=Codex → Opus |
| 4 | **worker** | build tasks (equivalent tier) | GLM-5.2 → Codex-5.5 / Codex-Terra → Sonnet-5 |
| 5 | **simple** | trivial/mechanical execution | GLM-4.7 → Codex-Sol-Luna → Haiku |

Tier equivalences (Mohan): GLM-5.2 ≈ Sonnet-5 (worker); GLM-4.7 ≈ Haiku (simple).
Provider of each candidate: Anthropic = Opus/Sonnet/Haiku/Fable; OpenAI-Codex = every `gpt-*`/Sol/Terra/Luna id; Zhipu = GLM (4.7 & 5.2); DeepSeek = deepseek.

### Cascade (unchanged, still applies to a delegated call)
Exhaust GLM → then Codex → then native Claude. Driven by the existing `exhausted=true` quota signal (reused, not rebuilt).

## END-STATE (gold)

`delegate(role="worker", orchestrator="opus", prompt="…")` returns EITHER:
- a normal `RouteResult` with the resolved non-Anthropic backend's answer (GLM/DeepSeek/Codex), OR
- an **assignment stub** `{execute_natively: true, role, model, provider, reason}` when the resolved candidate is an Anthropic model — the main session reads it and runs that role itself. The router makes ZERO Anthropic API calls.

Given `orchestrator="opus"`, `delegate(role="adversary", …)` resolves to a **Codex** model (different provider). Given `orchestrator="codex"`, it resolves to **Opus** and returns an assignment stub (Anthropic → hand back).

The old `delegate(complexity="code")` still works unchanged (back-compat alias → maps to the worker/role table).

Model-id strings live in `config.toml [roles]` — Mohan fills them (esp. codex Sol/Terra/Luna ids, which the CLI accepts permissively so a typo can't be verified in code). Router logic is id-agnostic.

## SUCCESS CRITERIA (measurable, tech-agnostic)

1. **Role resolution correctness — 100% of a truth table.** A parametrized test enumerates every (role × orchestrator-provider) pair and asserts the resolved provider + the execute_natively flag. ≥ 10 cases, 100% pass.
2. **Adversary-differs invariant — 0 violations.** For every orchestrator provider P, the resolved adversary provider ≠ P. Asserted across all supported orchestrator providers (Anthropic, Codex), 0 same-provider results.
3. **Zero Anthropic calls — proven structurally.** A test asserts that for every role resolving to an Anthropic model, the router returns an assignment stub (execute_natively=true) and NO backend call fn (deepseek/glm/codex route) is invoked. 0 Anthropic route calls.
4. **Back-compat — 100% of existing suite still green.** All 47 currently-passing tests pass unchanged (complexity path preserved).
5. **Config-driven ids — 0 hardcoded model strings in router logic.** grep asserts role→model-id mapping is read from config, not literal in router.py (placeholders + comments only in config).
6. **Cascade preserved.** The GLM→Codex→native exhausted-signal behavior is unchanged for a delegated worker/simple call (existing exhausted tests still green).

## OUT OF SCOPE
- Adding an Anthropic backend / any Claude API call (explicitly rejected by Mohan).
- Auto-detecting the orchestrator (it is an input — the human's launch choice).
- Changing the cascade order or the quota-detection mechanism.
- Filling the exact codex Sol/Terra/Luna id strings (Mohan owns those in config).
