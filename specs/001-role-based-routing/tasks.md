# 001 — Tasks (dependency-ordered)

Each task = ONE surface. Fit-check before + after (§9.3). Commit per task.

- [ ] **T1 — Provider + Role enums + id→provider table** (`router.py`)
  - Add `Role`, `Provider` enums; `provider_for_model(id) -> Provider` (prefix table).
  - Verify: unit test — every known id prefix maps to right provider; unknown → raises.

- [ ] **T2 — `[roles]` config parsing** (`config.py`, `config.toml`)
  - Parse `[roles]` ordered lists; add `roles` field (Optional). Placeholder ids + comments in config.toml (Mohan fills).
  - Verify: config round-trip test; missing `[roles]` → empty, no crash (back-compat).

- [ ] **T3 — `resolve_role()` + `Assignment` dataclass** (`router.py`) — CORE
  - Pure fn: (role, orchestrator_provider, config, exhausted_providers) → Assignment.
  - Adversary: first candidate provider ≠ orchestrator. Others: first live in cost-order.
  - Anthropic-resolved → `execute_natively=True` stub (returns BEFORE any backend call). `floor:` comment.
  - Verify: SC-1 truth-table (≥10 cases), SC-2 adversary-differs (0 violations), SC-3 zero-Anthropic-calls.

- [ ] **T4 — delegate() gains `role` + `orchestrator`** (`server.py`)
  - role given → resolve_role → assignment stub OR route() the backend. complexity path unchanged.
  - Verify: SC-4 all 47 existing tests green; new role path returns correct stub/result.

- [ ] **T5 — Full suite + adversarial verify + G-guard**
  - Verify: SC-5 (grep: no hardcoded ids in router logic), SC-6 (cascade/exhausted tests green).
  - §9.6 adversarial pass (differ-from-orchestrator is the load-bearing invariant).
  - G-guard: a test that fails if any adversary resolution shares the orchestrator's provider.

- [ ] **T6 — Branch, PR** (never direct-to-main per global rule)
  - Combined change (refactor + role-routing) → branch → PR. Mohan fills config ids before merge.
