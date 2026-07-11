# Stress-test verdict — role routing

Primary objective: config-driven role resolution with provider-separated adversary and zero Anthropic API calls.

Iterations: 2.

Verdict: implementation holds. 70 tests pass; live 0600 config resolves all four delegated roles; Codex adversary returns an Anthropic native assignment. Legacy complexity API remains green.

Residual Q5: automatic clients still use the legacy complexity contract until rig workflows/skills adopt role calls. Accepted as explicit separate integration scope, not hidden completion.

External adversarial review: prohibited by privacy policy even after user consent. Safer local substitute passed: committed-diff inspection, Anthropic-call structural grep, 22 role invariants, 70 full-suite tests, and Ruff.
