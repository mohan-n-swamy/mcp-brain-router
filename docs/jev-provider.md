# Jev decision provider

Jev answers typed questions through the Brain Router `decide` MCP tool. It cannot write code, use tools, or produce free-form text. Existing agentic `delegate` roles remain unchanged.

## Configuration

Set `TYPESAFE_API_KEY_FILE` to an owner-readable key file, or add the top-level `jev_key_file = "/absolute/path/to/key"` to the router config. `TYPESAFE_API_KEY` takes precedence when set. The key is read for each call; rotate it in its existing file. Never copy it into the repository. Restart the MCP connection after installing this change so clients discover `decide`. This integration uses FastMCP, verified in the installed runtime; the obsolete low-level server fallback remains delegate-only.

Default model: `jev-1.13.0` (pinned). Endpoint: `https://api.typesafe.ai/v1/systemone`. Account: a key file kept outside the repository.

## Call

```json
{"state":"Add Jev to Brain Router and show it in Warp.","questions":{"work":{"type":"choice","instructions":"Classify the requested work.","criteria":{"integration":"Connect a service to existing software","research":"Read and summarize sources","writing":"Draft prose"}},"requires_code":{"type":"noul","instructions":"Does this request require code changes?"}}}
```

The result includes `backend: jev`, typed `answers`, actual input/output tokens, and elapsed milliseconds. Batch related questions so state is sent once. No automatic retry, routing threshold, or speculative cache is added. Callers must respect `retry_after` and decide how to handle uncertain answers. HTTP failures are sanitized; raw bodies and credentials never enter the audit log. Calls use the existing `brain-router-delegations.jsonl` audit stream. Warp displays the latest Jev call without making API calls during rendering.

## Verification and rollback

Run `pytest -q tests/test_jev.py tests/test_jev_mcp.py`. A live smoke test must exercise a fresh stdio MCP server and the supplied credential. Render Warp with `multi-model-status.sh --all` and confirm the Jev row.

Rollback: remove `jev_key_file` (or unset the TypeSafe environment variables) to disable calls; restore the backed-up Warp renderer and launcher to remove its row. No role ordering needs restoration.

## Attribution

Original adapter against TypeSafe AI's published HTTP contract: https://docs.typesafe.ai/api and https://docs.typesafe.ai/models (verified 2026-09-18). Hosted proprietary service; no vendor SDK or site code copied. Discovery source: https://galigutta.github.io/jev-use-cases/ by galigutta (community use-case map; content license unverified).

## Verification receipt — 2026-09-18

- Adapter/MCP/config checks: 18 passed. Full router suite: 297 passed, 1 skipped. Ruff and diff whitespace checks passed.
- Fresh stdio MCP + real Jev: HTTP 200; Choice, Noul, and Score returned valid answers for the requested integration; 407 input / 68 output tokens, 993 ms.
- Warp: empty, success, and error rows passed; actual isolated tmux pane showed the Jev row and final time row at height 11. Existing full Warp test remains blocked by its missing startup-config fixture.
- Independent review via Grok found no live-path blocker. Fixed empty-key classification, invalid nested-state audit handling, and fractional-score regression coverage.
- Prior config values and 0600 permissions preserved. Current MCP connections require reconnect; no commit or push performed.
