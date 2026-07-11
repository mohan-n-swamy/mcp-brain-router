# Iteration 2 — after live-config refinement

Q1: Wrong native model ID; clients continue legacy calls; external review unavailable; audit log path blocked only inside sandbox.
Q2: Claude vs Codex caller identity, real config permissions, role quota advance, native assignment before backend invocation.
Q3: Smaller version is resolver-only. Current server integration is necessary to make resolution callable and preserve legacy clients.
Q4: Three scenarios passed: legacy complexity suite, Claude role truth table, Codex adversary native handback. Failures return structured validation/backend results.
Q5: Best residual failure: existing Claude/Codex workflows still call `complexity`; feature exists but will not become default automatically.

Tradeoff: repo feature ships independently; client/doctrine adoption is a separate rig change requiring its own enforcement review. Direct role calls work now after MCP restart.

Q6: Deterministic resolution; repeated stochastic model-output eval is not the objective. N/A.
Q7: No unattended metric-descent loop. N/A.
