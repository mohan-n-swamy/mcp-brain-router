"""apple-fm: Apple's on-device model on the Mac mini as the first `simple` tier.

Every fm failure must raise BackendError so the role cascade moves on; the
prompt goes on stdin only (a positional prompt makes fm drop stdin, exit 0).
All subprocess calls are mocked — no ssh, no network.
"""
from __future__ import annotations

import json
import subprocess
from unittest.mock import MagicMock, patch

import pytest

from mcp_brain_router import backends, server
from mcp_brain_router.backends import BackendError
from mcp_brain_router.config import DEFAULT_ROLES, Config
from mcp_brain_router.router import Provider, provider_for_model

PROMPT = "Extract the invoice number: Invoice 4471 from Acme Labs."


def _proc(rc=0, out="", err=""):
    return MagicMock(returncode=rc, stdout=out, stderr=err)


def _call(prompt=PROMPT, host="mini", proc=None, side_effect=None):
    with patch.object(backends.subprocess, "run", return_value=proc, side_effect=side_effect) as run:
        try:
            return backends.call_fm(prompt, host), run
        except BackendError as e:
            return e, run


# ---------------------------------------------------------------- argv / stdin

def test_argv_prompt_on_stdin_only():
    result, run = _call(proc=_proc(out="4471\n"))
    assert result == {"content": "4471", "usage": None}
    argv = run.call_args.args[0]
    kw = run.call_args.kwargs
    assert argv[:6] == ["/usr/bin/ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5", "mini"]
    assert argv[6].startswith("fm respond --no-stream --instructions ")
    assert len(argv) == 7
    assert kw["input"] == PROMPT
    assert kw["timeout"] == backends.FM_TIMEOUT_SECONDS == 20
    assert all(PROMPT not in a for a in argv)


def test_argv_carries_fm_system_not_agentic_or_caveman():
    result, run = _call(proc=_proc(out="ok"))
    remote = run.call_args.args[0][-1]
    assert "NEEDS_TOOLS" in remote
    for frame in (backends.AGENTIC_SYSTEM, backends.CAVEMAN_SYSTEM):
        assert frame[:40] not in remote
        assert frame[:40] not in run.call_args.kwargs["input"]


@pytest.mark.parametrize("host", ["-oProxyCommand=touch /tmp/x", "mini; rm -rf ~", "", None, "mini host"])
def test_host_validated_before_ssh(host):
    result, run = _call(host=host, proc=_proc(out="ok"))
    assert isinstance(result, BackendError)
    assert result.failure_kind == "validation_error"
    run.assert_not_called()


# ------------------------------------------------------------- failure kinds

def test_oversize_skips_ssh():
    result, run = _call(prompt="x " * (backends.FM_MAX_PROMPT_CHARS // 2 + 1))
    assert result.failure_kind == "input_too_large"
    run.assert_not_called()


def test_oversize_boundary_exact_limit_goes_to_ssh():
    at_limit = "a" * backends.FM_MAX_PROMPT_CHARS
    result, run = _call(prompt=at_limit, proc=_proc(out="ok"))
    assert result == {"content": "ok", "usage": None}
    result, run = _call(prompt=at_limit + "a")
    assert result.failure_kind == "input_too_large"
    run.assert_not_called()


@pytest.mark.parametrize("stream", ["err", "out"])
def test_overflow_is_input_too_large_not_retryable(stream):
    msg = "Error: The session's transcript exceeded the model's context size."
    result, _ = _call(proc=_proc(rc=1, **{stream: msg}))
    assert result.failure_kind == "input_too_large"
    assert result.failure_kind not in server.RETRYABLE_KINDS


@pytest.mark.parametrize("out", ["NEEDS_TOOLS\n", "**NEEDS_TOOLS**", "NEEDS_TOOLS — I can't read files."])
def test_needs_tools_reply_is_a_failure(out):
    result, _ = _call(proc=_proc(out=out))
    assert result.failure_kind == "needs_tools"


@pytest.mark.parametrize("prompt", [
    "Read /Users/me/notes.md and summarize it.",
    "Summarize ~/code/STATUS.md",
    "Fetch https://example.com/pricing and list the plans.",
    "Edit src/mcp_brain_router/router.py to add a docstring.",
    "Summarize notes.md in two lines.",
    "Run ./build and report errors.",
])
def test_needs_tools_path_or_url_skips_ssh(prompt):
    result, run = _call(prompt=prompt)
    assert result.failure_kind == "needs_tools"
    run.assert_not_called()


@pytest.mark.parametrize("proc,kind", [
    (_proc(out="   \n"), "empty_output"),
    (_proc(rc=255, err="ssh: connect to host mini: Operation timed out"), "unreachable"),
    (_proc(rc=69, err="license not agreed"), "configuration_error"),
    (_proc(rc=3, err="boom"), "process_error"),
])
def test_failure_kinds(proc, kind):
    result, _ = _call(proc=proc)
    assert isinstance(result, BackendError)
    assert result.failure_kind == kind
    assert result.backend == "apple-fm"


def test_timeout():
    result, _ = _call(side_effect=subprocess.TimeoutExpired("ssh", 20))
    assert result.failure_kind == "timeout"


# ------------------------------------------------------------------ provider

def test_provider_apple_fm_and_no_existing_id_moves():
    assert provider_for_model("apple-fm") is Provider.APPLE
    ids = {m for models in DEFAULT_ROLES.values() for m in models} | {
        "claude-haiku-4-5-20251001", "claude-sonnet-5", "claude-fable-5-1", "claude-opus-5-5",
        "grok-4.5", "gpt-6-astra", "gpt-5.6-terra", "glm-5.3", "kimi", "deepseek-v4-flash",
    }
    assert all(provider_for_model(m) is not Provider.APPLE for m in ids)


# ------------------------------------------------------------------- cascade

def _config(**kw):
    base = dict(fm_host="mini", roles={"simple": ["apple-fm", "claude-haiku-4-5-20251001"]})
    base.update(kw)
    return Config(**base)


async def _delegate(config, fm_effect, prompt=PROMPT):
    haiku = MagicMock(return_value={"content": "from haiku", "usage": None})
    with (
        patch.object(server, "_load_config", return_value=config),
        patch.object(backends, "call_fm", side_effect=fm_effect) as fm,
        patch.object(backends, "call_anthropic_agentic", haiku),
    ):
        response = await server._delegate_role_impl("simple", prompt, "opus", cwd="/tmp")
    return response, fm, haiku


@pytest.mark.asyncio
async def test_cascade_fm_answers():
    response, fm, haiku = await _delegate(_config(), [{"content": "4471", "usage": None}])
    assert response["answer"] == "4471"
    assert response["backend"] == "apple-fm"
    assert response["tried"] == ["apple-fm"]
    haiku.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["input_too_large", "needs_tools", "empty_output", "timeout"])
async def test_cascade_falls_through_without_retry(kind):
    err = BackendError("fm", backend="apple-fm", failure_kind=kind, elapsed_ms=20_000 if kind == "timeout" else 5)
    response, fm, haiku = await _delegate(_config(), [err, err])
    assert response["answer"] == "from haiku"
    assert response["tried"] == ["apple-fm", "anthropic-cli"]
    assert response["failures"] == [f"apple-fm:{kind}"]
    assert fm.call_count == 1


@pytest.mark.asyncio
async def test_cascade_unreachable_mini_falls_through_without_retry():
    err = BackendError("ssh", backend="apple-fm", failure_kind="unreachable", elapsed_ms=5_000)
    response, fm, _ = await _delegate(_config(), [err, err])
    assert response["answer"] == "from haiku"
    assert response["tried"] == ["apple-fm", "anthropic-cli"]
    assert response["retried"] == []
    assert fm.call_count == 1
    assert "unreachable" not in server.RETRYABLE_KINDS
    assert "error" not in response


@pytest.mark.asyncio
async def test_cascade_no_fm_host_falls_through():
    response, fm, _ = await _delegate(_config(fm_host=None), [])
    assert response["answer"] == "from haiku"
    assert response["failures"] == ["apple-fm:configuration_error"]
    fm.assert_not_called()


@pytest.mark.asyncio
async def test_cascade_audit_row_has_fm_usage_and_no_prompt():
    secret_prompt = "Classify: the courier was late " + "z" * 50
    err = BackendError("fm", backend="apple-fm", failure_kind="needs_tools", elapsed_ms=1)
    await _delegate(_config(), [err], prompt=secret_prompt)
    rows = [json.loads(l) for l in server._DELEGATION_LOG.read_text().splitlines()]
    assert rows[-1]["tried"] == ["apple-fm", "anthropic-cli"]
    assert rows[-1]["failures"] == ["apple-fm:needs_tools"]
    assert rows[-1]["prompt_len"] == len(secret_prompt)
    assert "courier" not in server._DELEGATION_LOG.read_text()


def test_config_round_trips_fm_host(tmp_path, monkeypatch):
    from mcp_brain_router import config as cfg
    monkeypatch.setattr(cfg, "CONFIG_DIR", tmp_path)
    monkeypatch.setattr(cfg, "CONFIG_FILE", tmp_path / "config.toml")
    Config(fm_host="mini", roles={"simple": ["apple-fm"]}).save()
    assert Config.load().fm_host == "mini"
