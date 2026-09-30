# Decide backends: Jev and self-hosted laya behind one switch

`decide` answers typed questions through an ordered list of slots. Each slot is a
backend URL, a provider and an owner-only key file. The first usable slot answers;
a slot that is rate-limited, rejected, unavailable, timed out or unreachable hands
the same call to the next one. Validation and response errors do not fail over.

Added 2026-09-28 (DECIDE-SWITCH, `rig-consolidation/slices/DECIDE-SWITCH/`), on top
of the two-key Jev pool (JEV-LOAD, `docs/jev-provider.md`).

## Configuration

Add a top-level `decide_backends` list to the router config. The order is the switch.

```toml
decide_backends = [
  {name = "jev-office",   url = "https://api.typesafe.ai/v1/systemone",   provider = "jev",  key_file = "~/secret/jev-key-1"},
  {name = "jev-personal", url = "https://api.typesafe.ai/v1/systemone",   provider = "jev",  key_file = "~/secret/jev-key-2"},
  {name = "laya-mini",    url = "http://100.100.10.20:8090/v1/systemone", provider = "laya", key_file = "~/secret/laya-key", model = "typed-decisions"},
]
```

This is the live order chosen by Mohan on 2026-09-28: zero-shot laya scored 4/10 against
Jev's 10/10 on labelled routing cases (`rig-consolidation/slices/DECIDE-SWITCH/evidence/side-by-side-2026-09-28.tsv`),
so laya is the free fallback when both Jev keys fail. Move it up only after fine-tuning and a new side-by-side.

- `model` (optional) is the checkpoint the slot asks for. A laya slot may name `english`,
  `multilingual` or `typed-decisions` (measured best: 7/10 on labelled routing cases against 4/10 for
  the base checkpoint); a Jev slot only accepts `jev-*`. Omit it to pass the caller's model through.
- `provider` is `jev` or `laya`. `laya` accepts laya-serve's own model name in the
  response; everything else about the answer is validated exactly as for Jev.
- `url` must be https, or plain http to loopback or a Tailscale `100.64.0.0/10`
  address (tailnet traffic is WireGuard-encrypted). Anything else is refused as a
  configuration error before any call.
- Precedence: `TYPESAFE_API_KEY` / `TYPESAFE_API_KEY_FILE` env, then
  `decide_backends`, then `jev_key_files`, then `jev_key_file`.
- Restart the MCP connection (`/mcp` reconnect) after changing the list.

Each audit line in `brain-router-delegations.jsonl` carries `decide_slot` (the slot
name), `backend` (`jev` or `laya`) and `key_slot` (its index), never a key.

## laya confidence is a different scale

laya's `confidence` on choice and score answers is 1 minus normalised entropy, not
Jev's formula; its `answer_confidence` is the calibrated `max(p)`. Never reuse a
threshold tuned on one backend for the other.

## Rollback

1. Remove `decide_backends` from the router config (the key-file pool takes over),
   or move the Jev slots to the top of the list.
2. `/mcp` reconnect.
3. On the laya host: `launchctl bootout gui/$(id -u)/<your-label>.laya-serve`
   (Mac mini) or `docker rm -f laya-serve` (VPS).

## Verification

`uv run pytest -q tests/test_decide_backends.py tests/test_jev_pool.py tests/test_jev.py tests/test_jev_mcp.py`,
then a live `decide` from a fresh MCP process and check the audit line's `decide_slot`.

## Decide spool (optional, off by default)

Set `BRAIN_ROUTER_DECIDE_SPOOL=<path>` in a client's MCP registration to keep one JSONL row per
answered Jev call: state, questions, site (sorted question ids) and Jev's answers under
`jev` with `trainable: false`. The file is created mode 0600; a write failure is logged and
never affects `decide`. The audit line carries the row's `spool_id`, never the text. Rows are
for local shadow comparison: TypeSafe's MCA §2.3(b) bars using Jev output to train a model
that imitates it, so do not build training labels from the `jev` field. Test:
`pytest -q tests/test_decide_spool.py`.
