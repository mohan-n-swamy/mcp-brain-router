"""D16 (Mohan, 2026-11-24): Antigravity (`agy`) is a rig provider. Gemini ids only."""
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
        out = call_antigravity("hi", "gemini-4-hydrogen-low")
    return out, m.call_args


# ---- model ids (N1, A28) ---------------------------------------------------

@pytest.mark.parametrize("model", ["claude-opus-5-5-high", "claude-sonnet-5-5-low",
                                   "gpt-oss-20b", "gpt-6.1-sol", "", "gemini", "x gemini-4",
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
    assert argv[argv.index("--model") + 1] == "gemini-4-hydrogen-low"
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
        seen["plugins_text"] = open(os.path.join(kw["cwd"], ".agents", "plugins.json")).read()
        return _proc(json.dumps(OK))

    with patch("mcp_brain_router.backends.subprocess.run", side_effect=fake):
        call_antigravity("hi", "gemini-4-hydrogen-low")
    group = seen["hooks"]["brain-router-tools-off"]["PreToolUse"][0]
    assert group["matcher"] == "*"
    assert '"decision":"deny"' in group["hooks"][0]["command"]
    assert json.loads(seen["plugins_text"]) == {"exclude": ["rig", "ponytail"]}
    assert not os.path.exists(seen["cwd"])  # temp dir is removed


# ---- failure mapping --------------------------------------------------------

def test_missing_binary_is_a_configuration_error():
    with patch("mcp_brain_router.backends.subprocess.run", side_effect=FileNotFoundError):
        with pytest.raises(BackendError) as e:
            call_antigravity("hi", "gemini-4-hydrogen-low")
    assert e.value.failure_kind == "configuration_error"


def test_print_timeout_is_transient():
    err = 'AGY_ERROR: {"status":"DEADLINE_EXCEEDED","reason":"print_timeout"}'
    bad = {"status": "ERROR", "response": "", "error": "print timeout after 1s"}
    with patch("mcp_brain_router.backends.subprocess.run",
               return_value=_proc(json.dumps(bad), 3, err)):
        with pytest.raises(BackendTransientError) as e:
            call_antigravity("hi", "gemini-4-hydrogen-low")
    assert e.value.failure_kind == "timeout"


def test_print_timeout_named_only_in_stdout_is_transient():
    out = json.dumps({"status": "ERROR", "response": "", "finished_reason": "print_timeout"})
    with patch("mcp_brain_router.backends.subprocess.run", return_value=_proc(out, 3, "")):
        with pytest.raises(BackendTransientError):
            call_antigravity("hi", "gemini-4-hydrogen-low")


def test_print_timeout_with_exit_zero_is_transient():
    out = json.dumps({"status": "ERROR", "response": "", "finished_reason": "print_timeout"})
    with patch("mcp_brain_router.backends.subprocess.run", return_value=_proc(out, 0, "")):
        with pytest.raises(BackendTransientError):
            call_antigravity("hi", "gemini-4-hydrogen-low")


def test_subprocess_timeout_is_transient():
    with patch("mcp_brain_router.backends.subprocess.run",
               side_effect=subprocess.TimeoutExpired("agy", 1)):
        with pytest.raises(BackendTransientError):
            call_antigravity("hi", "gemini-4-hydrogen-low")


@pytest.mark.parametrize("text", ["Out of credits", "RESOURCE_EXHAUSTED: quota"])
def test_quota_text_is_quota(text):
    bad = {"status": "ERROR", "response": "", "error": text}
    with patch("mcp_brain_router.backends.subprocess.run",
               return_value=_proc(json.dumps(bad), 1, text)):
        with pytest.raises(BackendQuotaError):
            call_antigravity("hi", "gemini-4-hydrogen-low")


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
    fake_dir = tmp_path / ".local" / "bin"
    fake_dir.mkdir(parents=True)
    fake = fake_dir / "agy"
    fake.write_text("#!/bin/sh\n")
    fake.chmod(0o755)
    monkeypatch.setenv("PATH", "/usr/bin:/bin")
    monkeypatch.setenv("HOME", str(tmp_path))  # "~" must expand to the fake home
    assert "~/.local/bin" in backends._AGY_FALLBACK_DIRS
    assert backends._resolve_bin("agy", ("/nonexistent-dir", "~/.local/bin")) == str(fake)
    assert backends._resolve_bin("agy", (str(fake_dir),)) == str(fake)
    assert backends._resolve_bin("agy", ("/nonexistent-dir",)) == "agy"


def test_missing_agy_is_a_free_skip(monkeypatch):
    monkeypatch.setattr(backends, "_AGY_BIN", "agy")  # bare name: nothing found
    with pytest.raises(router.BackendUnavailableError):
        router._validate_credentials("antigravity", Config())
    monkeypatch.setattr(backends, "_AGY_BIN", "/bin/sh")
    router._validate_credentials("antigravity", Config())  # no raise


def test_agy_env_prepends_its_bin(monkeypatch):
    monkeypatch.setattr(backends, "_AGY_BIN", "/opt/homebrew/bin/agy")
    assert backends._agy_env()["PATH"].split(os.pathsep)[0] == "/opt/homebrew/bin"




# ---- wiring ---------------------------------------------------------------

def test_gemini_ids_map_to_antigravity():
    for m in ("gemini-4-neon-high", "gemini-4-hydrogen-low", "gemini-4-neon-xhigh"):
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
    assert (claude.model, claude.provider) == ("gemini-4-neon-xhigh", Provider.ANTIGRAVITY)
    for caller in ("antigravity", "gemini-4-neon-xhigh"):
        a = resolve_role(Role.ADVERSARY, caller, _cfg(), mode="agentic")
        assert a.provider is not Provider.ANTIGRAVITY
        assert a.model == "claude-fable-5-2"
    g = resolve_role(Role.ADVERSARY, "grok", _cfg(), mode="agentic")
    assert (g.model, g.provider) == ("claude-fable-5-2", Provider.ANTHROPIC)


def test_adversary_start_per_caller(monkeypatch):
    monkeypatch.setattr(backends, "_AGY_BIN", "/bin/sh")
    for caller, model in (("claude", "gemini-4-neon-xhigh"), ("grok", "claude-fable-5-2"),
                          ("antigravity", "claude-fable-5-2")):
        a = resolve_role(Role.ADVERSARY, caller, _cfg(), mode="agentic")
        assert a.model == model, (caller, a.model)
    # A Claude caller never gets either Claude entry, even with Neon gone.
    c = resolve_role(Role.ADVERSARY, "claude", _cfg(), mode="agentic",
                     exhausted_providers=[Provider.ANTIGRAVITY])
    assert c.model == "grok-4.7"
    # A Grok caller whose Fable failed gets Neon, then Opus.
    g = resolve_role(Role.ADVERSARY, "grok", _cfg(), mode="agentic",
                     skip_models=["claude-fable-5-2", "gemini-4-neon-xhigh"])
    assert g.model == "claude-opus-5-5"


def test_worker_first_entry_is_gemini_and_grok_is_the_backup():
    a = resolve_role(Role.WORKER, "claude", _cfg(), mode="agentic")
    assert (a.model, a.backend) == ("gemini-4-neon-medium", "antigravity")
    b = resolve_role(Role.WORKER, "claude", _cfg(),
                     exhausted_providers=[Provider.ANTIGRAVITY, Provider.ANTHROPIC], mode="agentic")
    assert b.model == "grok-4.7"


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
    r = call_antigravity("Reply with the single word OK.", "gemini-4-hydrogen-low")
    assert "OK" in r["content"]
    assert r["usage"]["input_tokens"] > 0


def _agy_tool_steps(workdir, target, hooked):
    """Run agy in stream-json, forced to need a shell tool; return {tool step: (state, error)}."""
    if hooked:
        backends._write_agy_hook(workdir)
    prompt = (f"Run the shell command `echo hi > {target}` with your shell tool, then say "
              "what happened. You cannot answer without running it.")
    out = subprocess.run(
        [backends._AGY_BIN, "-p", prompt, "--model", "gemini-4-hydrogen-low",
         "--output-format", "stream-json", "--print-timeout", "90s"],
        stdin=subprocess.DEVNULL, capture_output=True, text=True, cwd=workdir,
        env=backends._agy_env(), timeout=120).stdout
    steps = {}
    for line in out.splitlines():
        ev = json.loads(line)
        su = ev.get("step_update") or {}
        if su.get("step_type") == "tool":
            steps[su["step_index"]] = (su["state"], (su.get("tool_info") or {}).get("error"))
    return steps


def test_real_agy_deny_hook_fires_and_refuses_the_tool(tmp_path):
    """Proof, not absence: agy's own stream shows the tool call was attempted and
    refused by the pre-tool hook. The control run (no hook) shows the same prompt
    does run a tool, so an inert hook cannot pass this."""
    _agy_or_skip()
    work = tmp_path / "hooked"
    work.mkdir()
    target = tmp_path / "must-not-exist.txt"
    steps = _agy_tool_steps(str(work), target, hooked=True)
    assert steps, "the prompt never reached a tool call, so the hook was not tested"
    for state, err in steps.values():
        assert state == "ERROR" and "denied by pre-tool hook" in (err or {}).get("message", ""), steps
    assert not target.exists()


def test_real_agy_control_runs_the_tool_without_the_hook(tmp_path):
    _agy_or_skip()
    work = tmp_path / "bare"
    work.mkdir()
    target = tmp_path / "control.txt"
    steps = _agy_tool_steps(str(work), target, hooked=False)
    assert any(state == "DONE" for state, _ in steps.values()), steps
    assert target.exists()


def test_real_agy_one_shot_is_lean_with_the_rig_plugin_installed():
    _agy_or_skip()
    out = subprocess.run([backends._AGY_BIN, "plugin", "list"], stdin=subprocess.DEVNULL,
                         capture_output=True, text=True, timeout=60).stdout
    if "- rig" not in out:
        pytest.skip("rig plugin not installed")
    r = call_antigravity("Reply with the single word OK.", "gemini-4-hydrogen-low")
    assert 0 < r["usage"]["input_tokens"] < 15000, r["usage"]
