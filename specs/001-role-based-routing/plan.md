# 001 — Plan (HOW + stack)

## Stack
Python 3.12, FastMCP stdio, dataclasses, pytest. No new deps (rung ≤4). Builds on the working-tree "pure tier ownership" refactor.

## Architecture

Add a `Role` axis ALONGSIDE the existing `Complexity` axis. `complexity` stays a back-compat alias. The router gains a role-resolution layer that sits in front of the existing backend-call layer.

### New: `Role` enum + provider model
```
class Role(str, Enum): ORCHESTRATOR, THINKER, ADVERSARY, WORKER, SIMPLE
class Provider(str, Enum): ANTHROPIC, CODEX, ZHIPU, DEEPSEEK
```

### New: role → ordered candidate list (from config)
`config.toml [roles]` holds, per role, an ordered list of model-id strings. Each id maps to a Provider via a prefix table (`gpt-*`/sol/terra/luna→CODEX, glm*→ZHIPU, deepseek*→DEEPSEEK, opus/sonnet/haiku/fable→ANTHROPIC). Mohan fills the exact id strings; the router reads them.

### Resolution fn (pure, unit-testable)
`resolve_role(role, orchestrator_provider, config, exhausted_providers=set()) -> Assignment`
- ADVERSARY: pick first candidate whose provider ≠ orchestrator_provider AND not exhausted.
- others: walk the ordered candidate list, skip exhausted providers, take first live one.
- If resolved model's provider == ANTHROPIC → return `Assignment(execute_natively=True, model, provider, role, reason)`. Router will NOT call it.
- Else → return `Assignment(execute_natively=False, backend, model, ...)` → existing route() calls the backend.

### delegate() gains `role` + `orchestrator` params
`delegate(prompt, role=None, orchestrator=None, complexity=None, model=None)`:
- if `role` given → resolve_role → either return the assignment stub (native) or call the backend.
- elif `complexity` given → existing path unchanged (back-compat).
- The assignment stub is a NEW return shape the orchestrator (main Claude session) reads.

### COMPLIANCE invariant (structural)
Anthropic-resolved roles never reach `_route_deepseek/glm/codex`. The resolve step returns BEFORE any backend call fn. Test asserts no route-fn invoked for Anthropic roles (mock/spy).

## Files
- `router.py` — add Role/Provider enums, `resolve_role`, `Assignment` dataclass, provider-from-id table. Keep existing route() intact.
- `config.py` — parse `[roles]` table; add `roles` field.
- `server.py` — delegate() gets `role` + `orchestrator` params; returns assignment stub or routed result.
- `config.toml` — `[roles]` table with PLACEHOLDER ids + comments (Mohan fills).
- `tests/test_roles.py` — new: truth-table, adversary-differs, zero-Anthropic-calls.
- existing tests — unchanged (back-compat proof).

## Non-goals / floors
- No Anthropic backend. No API call to Claude. `floor:` comment where the assignment stub is returned.
- Exact codex ids are config placeholders — router is id-agnostic.
