# Iteration 1 — role-routing completion

Primary objective: resolve five orchestration roles from config, keep adversary provider different, and return Anthropic work natively without Anthropic API calls.

Q1: Missing or malformed `[roles]`; same-provider adversary; quota loop; Anthropic accidentally reaches backend; legacy API regression.
Q2: Empty candidates, unknown IDs, orchestrator role input, provider aliases, exhausted primary, Codex-hosted adversary.
Q3: Half-resource version would hardcode mappings. Rejected: config ownership and provider separation are load-bearing, not embellishment.
Q4: Cold config, legacy config, interruption, installer rewrite, and both orchestrator providers covered. G-guards live in `tests/test_roles.py`.
Q5: Best failure: implementation ships while live config lacks `[roles]`; every real role call fails.

Refinement: install role candidates into the real 0600 config; preserve them across installer rewrites; add config round-trip and missing-table tests.
