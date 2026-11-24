"""D15 (Mohan, 2026-11-24): Antigravity (`agy`) is a rig provider. Gemini ids only."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
from unittest.mock import MagicMock, patch

import pytest

from mcp_brain_router import backends, router
from mcp_brain_router.backends import (BackendError, BackendQuotaError,
                                       BackendTransientError, call_antigravity)
from mcp_brain_router.config import Config, DEFAULT_ROLES
from mcp_brain_router.router import Provider, Role, provider_for_model, resolve_role

OK = {"conversation_id": "c", "status": "SUCCESS", "response": "OK\n", "num_turns": 1,
      "usage": {"input_tokens": 100, "output_tokens": 2, "thinking_tokens": 0,
                "cache_read_tokens": 0, "total_tokens": 102}}


def _proc(stdout="", returncode=0, stderr=""):
    return MagicMock(returncode=returncode, stdout=stdout, stderr=stderr)


def _run(proc):
    with patch("mcp_brain_router.backends.subprocess.run", return_value=proc) as m:
        out = call_antigravity("hi", "gemini-3.8-flash-low")
    return out, m.call_args


# ---- model ids (N1, A28) ---------------------------------------------------

@pytest.mark.parametrize("model", ["claude-opus-5-5-high", "claude-sonnet-5-5-low",
                                   "gpt-oss-120b-medium", "gpt-6.1-sol", "", "gemini", "x gemini-4",
                                   "gemini-4 --effort low", "gemini-4;rm"])
def test_only_gemini_ids_run(model):
    with patch("mcp_brain_router.backends.subprocess.run") as m, pytest.raises(BackendError) as e:
        call_antigravity("hi", model)
    assert e.value.failure_kind == "validation_error"
    m.assert_not_called()


# ---- argv: no permission bypass (N4), no effort flag, stdin closed ----------

def test_argv_stdin_timeout_and_no_bypass():
    out, call = _run(_proc(json.dumps(OK)))
    argv = call.args[0]
    assert "--dangerously-skip-permissions" not in argv
    assert "--effort" not in argv  # the id suffix decides; a clashing flag is an agy error
    assert argv[argv.index("--model") + 1] == "gemini-3.8-flash-low"
    assert argv[argv.index("--output-format") + 1] == "json"
    assert argv[argv.index("--print-timeout") + 1].endswith("s")
    assert call.kwargs["stdin"] is subprocess.DEVNULL
    assert out == {"content": "OK", "usage": {"input_tokens": 100, "output_tokens": 2,
                                              "source": "cli-json"}}


def test_print_timeout_follows_the_call_deadline():
    token = backends.DEADLINE.set(backends.time.monotonic() + 30)
    try:
        _, call = _run(_proc(json.dumps(OK)))
    finally:
        backends.DEADLINE.reset(token)
    argv = call.args[0]
    assert 29 <= int(argv[argv.index("--print-timeout") + 1][:-1]) <= 30


def test_workdir_holds_a_deny_all_tool_hook():
    seen = {}

    def fake(argv, **kw):
        path = os.path.join(kw["cwd"], ".agents", "hooks.json")
        seen["hooks"] = json.loads(open(path).read())
        seen["cwd"] = kw["cwd"]
        return _proc(json.dumps(OK))

    with patch("mcp_brain_router.backends.subprocess.run", side_effect=fake):
        call_antigravity("hi", "gemini-3.8-flash-low")
    group = seen["hooks"]["brain-router-tools-off"]["PreToolUse"][0]
    assert group["matcher"] == "*"
    assert '"decision":"deny"' in group["hooks"][0]["command"]
    assert not os.path.exists(seen["cwd"])  # temp dir is removed


# ---- failure mapping --------------------------------------------------------

def test_missing_binary_is_a_configuration_error():
    with patch("mcp_brain_router.backends.subprocess.run", side_effect=FileNotFoundError):
        with pytest.raises(BackendError) as e:
            call_antigravity("hi", "gemini-3.8-flash-low")
    assert e.value.failure_kind == "configuration_error"


def test_print_timeout_is_transient():
    err = 'AGY_ERROR: {"status":"DEADLINE_EXCEEDED","reason":"print_timeout"}'
    bad = {"status": "ERROR", "response": "", "error": "print timeout after 1s"}
    with patch("mcp_brain_router.backends.subprocess.run",
               return_value=_proc(json.dumps(bad), 3, err)):
        with pytest.raises(BackendTransientError) as e:
            call_antigravity("hi", "gemini-3.8-flash-low")
    assert e.value.failure_kind == "timeout"


def test_subprocess_timeout_is_transient():
    with patch("mcp_brain_router.backends.subprocess.run",
               side_effect=subprocess.TimeoutExpired("agy", 1)):
        with pytest.raises(BackendTransientError):
            call_antigravity("hi", "gemini-3.8-flash-low")


@pytest.mark.parametrize("text", ["Out of credits", "RESOURCE_EXHAUSTED: quota"])
def test_quota_text_is_quota(text):
    bad = {"status": "ERROR", "response": "", "error": text}
    with patch("mcp_brain_router.backends.subprocess.run",
               return_value=_proc(json.dumps(bad), 1, text)):
        with pytest.raises(BackendQuotaError):
            call_antigravity("hi", "gemini-3.8-flash-low")


def test_error_status_with_exit_zero_raises():
    bad = {"status": "ERROR", "response": "", "error": "boom"}
    with pytest.raises(BackendError) as e:
        _run(_proc(json.dumps(bad)))
    assert e.value.failure_kind == "process_error"


def test_empty_and_invalid_output_raise():
    with pytest.raises(BackendError) as e:
        _run(_proc(json.dumps({**OK, "response": " \n"})))
    assert e.value.failure_kind == "empty_output"
    with pytest.raises(BackendError) as e:
        _run(_proc("not json"))
    assert e.value.failure_kind == "invalid_output"


# ---- binary resolution from a bare PATH --------------------------------------

def test_agy_resolves_from_the_fallback_dirs_on_a_bare_path(tmp_path, monkeypatch):
    fake = tmp_path / "agy"
    fake.write_text("#!/bin/sh\n")
    fake.chmod(0o755)
    monkeypatch.setenv("PATH", "/usr/bin:/bin")
    assert backends._resolve_bin("agy", (str(tmp_path),)) == str(fake)
    assert "/opt/homebrew/bin" in backends._AGY_FALLBACK_DIRS
    assert "~/.local/bin" in backends._AGY_FALLBACK_DIRS


def test_missing_agy_is_a_free_skip(monkeypatch):
    monkeypatch.setattr(backends, "_AGY_BIN", "agy")  # bare name: nothing found
    with pytest.raises(router.BackendUnavailableError):
        router._validate_credentials("antigravity", Config())
    monkeypatch.setattr(backends, "_AGY_BIN", "/bin/sh")
    router._validate_credentials("antigravity", Config())  # no raise


def test_agy_env_prepends_its_bin_and_keeps_real_home(monkeypatch):
    monkeypatch.setattr(backends, "_AGY_BIN", "/opt/homebrew/bin/agy")
    env = backends._agy_env()
    assert env["PATH"].split(os.pathsep)[0] == "/opt/homebrew/bin"
    assert env.get("HOME") == os.environ.get("HOME")  # a lean HOME signs agy out


# ---- wiring ---------------------------------------------------------------

def test_gemini_ids_map_to_antigravity():
    for m in ("gemini-4-neon-high", "gemini-3.8-flash-low", "gemini-4-neon-xhigh"):
        assert provider_for_model(m) is Provider.ANTIGRAVITY
    assert Provider.ANTIGRAVITY.value == "antigravity"
    assert provider_for_model("claude-opus-5-5") is Provider.ANTHROPIC
    assert provider_for_model("grok-4.5") is Provider.XAI


def test_effort_is_reported_unapplied():
    assert "antigravity" in backends.EFFORT_UNSUPPORTED


def _cfg():
    return Config(grok_enabled=True, fm_host="h",
                  roles={r: list(o) for r, o in DEFAULT_ROLES.items()})


def test_adversary_skips_the_callers_own_provider(monkeypatch):
    monkeypatch.setattr(backends, "_AGY_BIN", "/bin/sh")
    claude = resolve_role(Role.ADVERSARY, "claude", _cfg(), mode="agentic")
    assert (claude.model, claude.provider) == ("gemini-4-neon-high", Provider.ANTIGRAVITY)
    for caller in ("antigravity", "gemini-4-neon-high"):
        a = resolve_role(Role.ADVERSARY, caller, _cfg(), mode="agentic")
        assert a.provider is not Provider.ANTIGRAVITY
        assert a.model == "grok-4.5"
    g = resolve_role(Role.ADVERSARY, "grok", _cfg(), mode="agentic")
    assert g.provider is Provider.ANTIGRAVITY


def test_worker_second_entry_is_gemini():
    a = resolve_role(Role.WORKER, "claude", _cfg(), exhausted_providers=[Provider.XAI],
                     mode="agentic")
    assert (a.model, a.backend) == ("gemini-4-neon-medium", "antigravity")


def test_antigravity_is_a_known_caller():
    from mcp_brain_router import server
    assert "antigravity" in server._KNOWN_CALLERS


def test_every_gemini_role_id_is_one_agy_lists():
    """N3, live: the table's gemini ids against `agy models`."""
    agy = shutil.which("agy") or backends._AGY_BIN
    if not os.path.isabs(agy) or not os.access(agy, os.X_OK):
        pytest.skip("agy not installed")
    out = subprocess.run([agy, "models"], stdin=subprocess.DEVNULL, capture_output=True,
                         text=True, timeout=60).stdout
    have = {line.split("\t")[0] for line in out.splitlines()}
    miss = [m for v in DEFAULT_ROLES.values() for m in v if m.startswith("gemini") and m not in have]
    assert not miss, miss


# ---- the real agy (skips when the binary is missing) --------------------------

def _agy_or_skip():
    if not os.path.isabs(backends._AGY_BIN) or not os.access(backends._AGY_BIN, os.X_OK):
        pytest.skip("agy not installed")


def test_real_agy_answers():
    _agy_or_skip()
    r = call_antigravity("Reply with the single word OK.", "gemini-3.8-flash-low")
    assert "OK" in r["content"]
    assert r["usage"]["input_tokens"] > 0


def test_real_agy_cannot_run_tools(tmp_path):
    """A prompt that needs a file write: the deny hook must stop it."""
    _agy_or_skip()
    target = tmp_path / "must-not-exist.txt"
    call_antigravity(f"Use your tools to create the file {target} containing hi. "
                     "Then say what happened.", "gemini-3.8-flash-low")
    assert not target.exists()
