# RUNBOOK — mcp-brain-router

Operations only. Install paths, env names and the full verify list are in
[`HANDOVER.md`](HANDOVER.md); Jev keys in [`jev-provider.md`](jev-provider.md);
decide slots in [`decide-backends.md`](decide-backends.md).

Live config: `~/.config/mcp-brain-router/config.toml`, mode 0600 (the router refuses any other mode).
Audit log: `~/.local/state/brain-router-delegations.jsonl`, one line per call, no prompt text.

## Deploy a code change

The live install is editable (`pip install -e .`), so the working tree is what runs.

1. `pytest -q` in the repo. Stop on any failure.
2. Commit and push.
3. Restart: `pkill -f mcp_brain_router.server`. Clients respawn the server on their next call; a session that held the old server needs `/mcp` reconnect or a new session.
4. Verify: make one `delegate` call, then `tail -1 ~/.local/state/brain-router-delegations.jsonl` shows the expected `backend` and no `error`.

## Change the live config

1. Back up with a dated name: `cp -p ~/.config/mcp-brain-router/config.toml ~/.config/mcp-brain-router/config.toml.bak-$(date +%Y%m%d)-<reason>`
2. Edit. Keep `fm_host`, keys and flags above the first `[table]`.
3. Check the mode: `stat -f %Lp ~/.config/mcp-brain-router/config.toml` prints `600`; if not, `chmod 600` it.
4. Restart as in Deploy step 3. Config is read once per server process.

## Roll back

1. Config: `cp -p ~/.config/mcp-brain-router/config.toml.bak-<stamp> ~/.config/mcp-brain-router/config.toml`, then restart.
2. Code: `git revert <sha>`, `pytest -q`, push, restart.

## apple-fm tier (first in `simple`, since 2026-10-23)

Apple's on-device model on the Mac mini, called as `ssh mini "fm respond --no-stream --instructions …"` with the prompt on stdin. Every failure falls through to the next `simple` tier; `failures` in the audit line says why (`apple-fm:<kind>`).

1. **Enable:** on the mini, `fm license` once (fm exits 69 until agreed). Check `ssh mini "fm available"` prints "System model available". Set `fm_host = "mini"` and put `"apple-fm"` first in `[roles] simple`. Restart.
2. **Disable:** remove `"apple-fm"` from `simple`, or delete `fm_host`. Restart. Backup from the flip: `config.toml.bak-20261023-fm`.
3. **Verify:** `uv run python specs/003-apple-fm-simple-tier/smoke.py live` prints `PASS live` (backend `apple-fm`, ~2.6 s).
4. **Recover by failure kind:**
   - `unreachable`: the mini is asleep or off Tailscale. Check `ssh -o BatchMode=yes mini true`. Each call costs one 5 s connect wait until it is back.
   - `configuration_error`: `fm_host` missing, or fm exit 69. Run `fm license` on the mini.
   - `input_too_large`: expected for prompts over 20,000 characters (8,192-token window).
   - `needs_tools`: expected for prompts that name a path, file or URL, or that fm says need tools.
   - `timeout` or `process_error` repeating: run `ssh mini "fm available"` and a manual `echo test | ssh mini "fm respond --no-stream --instructions 'Reply ok'"`.
5. **Watch:** the Claude status line row `fm(mini): 24h a/b answered · p50` reads the same audit log.
