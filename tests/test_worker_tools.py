"""`tools` (none, read, write) on delegate: Antigravity, Grok and Claude workers.

Offline: every subprocess is faked. The live proof against the real CLIs is in the
slice's EVIDENCE.txt, not here."""
from __future__ import annotations

import json
import os
import subprocess
from unittest.mock import MagicMock, patch

import pytest

from mcp_brain_router import backends, server
from mcp_brain_router.backends import BackendError, call_antigravity, call_grok
from mcp_brain_router.config import DEFAULT_ROLES, ROLES, Config
from mcp_brain_router.router import RouteResult

AGY_OK = {"status": "SUCCESS", "response": "done", "usage": {"input_tokens": 5, "output_tokens": 1}}
GROK_OK = {"text": "done", "stopReason": "end_turn", "num_turns": 3}


def _proc(stdout="", returncode=0, stderr=""):
    return MagicMock(returncode=returncode, stdout=stdout, stderr=stderr)


class FakeRun:
    """subprocess.run stand-in that records argv, cwd and what the workdir held."""

    def __init__(self, stdout):
        self.stdout, self.argv, self.cwd, self.hooks, self.files = stdout, None, None, None, None

    def __call__(self, argv, **kw):
        self.argv, self.cwd = list(argv), kw.get("cwd")
        hook = os.path.join(self.cwd, ".agents", "hooks.json")
        self.hooks = open(hook, encoding="utf-8").read() if os.path.exists(hook) else None
        self.files = sorted(os.listdir(self.cwd))
        return _proc(self.stdout)


def _agy(**kw):
    fake = FakeRun(json.dumps(AGY_OK))
    with patch("mcp_brain_router.backends.subprocess.run", fake):
        call_antigravity("task", "gemini-4-neon-medium", **kw)
    return fake


def _grok(**kw):
    one_turn = {**GROK_OK, "num_turns": 1}
    fake = FakeRun(json.dumps(GROK_OK if kw.get("tools", "none") != "none" else one_turn))
    with patch("mcp_brain_router.backends.subprocess.run", fake):
        call_grok("task", "grok-4.7", **kw)
    return fake


def _hook_verdict(hooks_json: str, tool: str) -> str:
    """Run the hook command agy would run, with a PreToolUse payload for `tool`."""
    entry = next(iter(json.loads(hooks_json).values()))["PreToolUse"][0]["hooks"][0]
    out = subprocess.run(entry["command"], shell=True, capture_output=True, text=True,
                         input=json.dumps({"toolCall": {"name": tool, "args": {}}}))
    return out.stdout


# ---- tools_argument -------------------------------------------------------

@pytest.mark.parametrize("bad", ["", "all", "WRITE", "read,write", "rw"])
async def test_tools_argument_rejects_anything_but_the_three_values(bad):
    r = await server._delegate_role_impl("coder", "x", "claude", mode="agentic", cwd="/tmp", tools=bad)
    assert r["failure_kind"] == "validation_error" and "tools must be one of" in r["error"]


@pytest.mark.parametrize("bad", ["all", "WRITE", ""])
def test_tools_argument_rejected_at_each_backend_too(bad, tmp_path):
    for fn in (call_grok, call_antigravity):
        with patch("mcp_brain_router.backends.subprocess.run") as run, pytest.raises(BackendError) as e:
            fn("x", "grok-4.7" if fn is call_grok else "gemini-4-neon-medium", tools=bad, cwd=str(tmp_path))
        assert e.value.failure_kind == "validation_error"
        run.assert_not_called()


def test_tools_argument_defaults_come_from_the_role_table():
    assert {n: r.tools for n, r in ROLES.items()} == {
        "thinker": "read", "adversary": "read", "coder": "write", "processor": "read",
        "writer": "read", "explorer": "read", "qa": "write", "simple": "none"}
    assert "worker" not in ROLES and DEFAULT_ROLES["worker"] == DEFAULT_ROLES["processor"]


@pytest.mark.parametrize(("role", "passed", "expected"), [
    ("coder", None, "write"), ("qa", None, "write"), ("explorer", None, "read"),
    ("worker", None, "read"), ("thinker", None, "read"), ("adversary", None, "read"),
    ("writer", None, "read"), ("simple", None, "none"), ("worker", "none", "none"),
    ("worker", "write", "write"), ("coder", "none", "none"), ("explorer", "write", "write"),
])
async def test_tools_argument_resolved_once_and_passed_down(role, passed, expected):
    cfg = Config(grok_enabled=True, roles={role if role != "worker" else "processor": ["grok-4.7"]})
    seen = []

    async def fake_route(assignment, prompt, config, mode="chat", cwd=None, effort=None, tools="none"):
        seen.append(tools)
        return RouteResult(content="ok", model="grok-4.7", backend="grok", complexity=None,
                                  headroom_used=False)

    with patch.object(server, "_load_config", return_value=cfg), \
            patch.object(server, "_log_delegation"), patch.object(server, "route_assignment", fake_route):
        r = await server._delegate_role_impl(role, "x", "claude", mode="agentic", cwd="/tmp",
                                             data_class="personal", tools=passed)
    assert seen == [expected] and r["tools"] == expected


# ---- tools_off_unchanged --------------------------------------------------

def test_tools_off_unchanged_antigravity_empty_folder_deny_all_hook():
    fake = _agy()
    assert fake.files == [".agents"]  # an empty temp folder holding only the hook files
    assert os.path.basename(fake.cwd).startswith("brain-router-agy-")
    assert fake.hooks == backends._AGY_DENY_HOOK
    assert "--add-dir" not in fake.argv and "--mode" not in fake.argv
    assert "--dangerously-skip-permissions" not in fake.argv


def test_tools_off_unchanged_antigravity_keeps_lean_plugins_file():
    seen = {}

    def run(argv, **kw):
        seen["p"] = open(os.path.join(kw["cwd"], ".agents", "plugins.json"), encoding="utf-8").read()
        return _proc(json.dumps(AGY_OK))

    with patch("mcp_brain_router.backends.subprocess.run", run):
        call_antigravity("task", "gemini-4-neon-medium")
    assert seen["p"] == backends._AGY_LEAN_PLUGINS


def test_tools_off_unchanged_grok_one_turn_no_tools_in_a_temp_folder(tmp_path):
    fake = _grok(cwd=str(tmp_path))  # a cwd is ignored when tools are none
    assert os.path.basename(fake.cwd).startswith("brain-router-grok-")
    a = fake.argv
    assert a[a.index("--max-turns") + 1] == "1"
    assert "--no-subagents" in a and "--disable-web-search" in a
    assert "--permission-mode" not in a and "-p" not in a and "--prompt-file" in a


def test_tools_off_unchanged_a_call_that_does_not_pass_tools_is_tools_off():
    import inspect
    for fn in (call_grok, call_antigravity, backends.call_anthropic_agentic):
        assert inspect.signature(fn).parameters["tools"].default == "none"
    from mcp_brain_router.router import route_assignment
    assert inspect.signature(route_assignment).parameters["tools"].default == "none"


async def test_tools_off_unchanged_walk_with_explicit_tools_none_calls_backends_bare():
    cfg = Config(grok_enabled=True, roles={"processor": ["grok-4.7"]})
    with patch.object(server, "_load_config", return_value=cfg), patch.object(server, "_log_delegation"), \
            patch("mcp_brain_router.router.backends.call_grok", return_value={"content": "ok"}) as g:
        r = await server._delegate_role_impl("worker", "x", "claude", mode="agentic", cwd="/tmp",
                                             data_class="personal", tools="none")
    assert g.call_args.kwargs.get("tools", "none") == "none" and "cwd" not in g.call_args.kwargs
    assert r["tools"] == "none"


async def test_tools_off_unchanged_explicit_tools_none_runs_antigravity_tools_off():
    cfg = Config(roles={"processor": ["gemini-4-hydrogen-medium"]})
    with patch.object(server, "_load_config", return_value=cfg), patch.object(server, "_log_delegation"), \
            patch("mcp_brain_router.router.backends.call_antigravity",
                  return_value={"content": "ok", "usage": None}) as agy:
        r = await server._delegate_role_impl("worker", "x", "claude", mode="agentic", cwd="/tmp",
                                             data_class="personal", tools="none")
    assert agy.call_args.kwargs.get("tools", "none") == "none" and "cwd" not in agy.call_args.kwargs
    assert r["backend"] == "antigravity" and r["tools"] == "none"


async def test_worker_with_no_tools_argument_passes_read_and_cwd_to_backends():
    cfg = Config(grok_enabled=True, roles={"processor": ["grok-4.7"]})
    with patch.object(server, "_load_config", return_value=cfg), patch.object(server, "_log_delegation"), \
            patch("mcp_brain_router.router.backends.call_grok", return_value={"content": "ok"}) as g:
        r = await server._delegate_role_impl("worker", "x", "claude", mode="agentic", cwd="/tmp",
                                             data_class="personal")
    assert g.call_args.kwargs["tools"] == "read" and g.call_args.kwargs["cwd"] == "/tmp"
    assert r["tools"] == "read"


async def test_simple_prompt_naming_a_path_runs_antigravity_with_read_not_haiku():
    cfg = Config(roles={"simple": ["gemini-4-hydrogen-medium", "claude-haiku-5-5"]})
    agentic = MagicMock(return_value={"content": "from claude", "usage": None})
    with patch.object(server, "_load_config", return_value=cfg), patch.object(server, "_log_delegation"), \
            patch("mcp_brain_router.router.backends.call_antigravity",
                  return_value={"content": "from agy", "usage": None}) as agy, \
            patch("mcp_brain_router.router.backends.call_anthropic_agentic", agentic):
        r = await server._delegate_role_impl("simple", "summarise ./notes.md", "grok", mode="agentic",
                                             cwd="/tmp", data_class="personal")
    agentic.assert_not_called()
    assert agy.call_args.kwargs == {"tools": "read", "cwd": "/tmp"}
    assert r["backend"] == "antigravity" and r["tools"] == "read"
    assert "antigravity:needs_tools" not in r["failures"]


async def test_simple_prompt_naming_a_url_still_skips_antigravity_with_needs_tools():
    cfg = Config(roles={"simple": ["gemini-4-hydrogen-medium", "claude-haiku-5-5"]})
    with patch.object(server, "_load_config", return_value=cfg), patch.object(server, "_log_delegation"), \
            patch("mcp_brain_router.router.backends.call_antigravity") as agy, \
            patch("mcp_brain_router.router.backends.call_anthropic_agentic",
                  return_value={"content": "from claude", "usage": None}):
        r = await server._delegate_role_impl("simple", "summarise https://example.com/a", "grok",
                                             mode="agentic", cwd="/tmp", data_class="personal")
    agy.assert_not_called()
    assert r["backend"] == "anthropic-cli" and "antigravity:needs_tools" in r["failures"]


async def test_tools_off_unchanged_the_path_skip_does_not_apply_when_tools_are_asked():
    cfg = Config(roles={"simple": ["gemini-4-hydrogen-medium", "claude-haiku-5-5"]})
    with patch.object(server, "_load_config", return_value=cfg), patch.object(server, "_log_delegation"), \
            patch("mcp_brain_router.router.backends.call_antigravity",
                  return_value={"content": "from agy", "usage": None}) as agy:
        r = await server._delegate_role_impl("simple", "summarise /tmp/notes.md", "grok", mode="agentic",
                                             cwd="/tmp", data_class="personal", tools="read")
    assert r["backend"] == "antigravity" and agy.call_args.kwargs == {"tools": "read", "cwd": "/tmp"}


async def test_text_only_backends_are_skipped_on_a_tools_job():
    cfg = Config(fm_host="h", roles={"simple": ["apple-fm", "claude-haiku-5-5"]})
    with patch.object(server, "_load_config", return_value=cfg), patch.object(server, "_log_delegation"), \
            patch("mcp_brain_router.router.backends.call_fm") as fm, \
            patch("mcp_brain_router.router.backends.call_anthropic_agentic",
                  return_value={"content": "c", "usage": None}):
        r = await server._delegate_role_impl("simple", "x", "grok", mode="agentic", cwd="/tmp",
                                             data_class="personal", tools="write")
    fm.assert_not_called()
    assert r["backend"] == "anthropic-cli" and "apple-fm:needs_tools" in r["failures"]


# ---- the Grok max-turns failure --------------------------------------------

def test_grok_tools_off_gets_zero_tools_so_the_one_turn_is_an_answer():
    """Seen 2026-11-24T22:27Z: with tools left on, `--max-turns 1` was spent on a tool
    call and the run ended stopReason=max_turn_requests with no text. The fake CLI
    does what grok 1.0.92 does: it spends its only turn on a tool unless every tool
    is removed."""
    def cli(argv, **kw):
        tools = argv[argv.index("--tools") + 1].split(",") if "--tools" in argv else ["ALL"]
        denied = argv[argv.index("--disallowed-tools") + 1].split(",") if "--disallowed-tools" in argv else []
        if [t for t in tools if t not in denied]:
            return _proc(json.dumps({"text": "I'll read the file.", "stopReason": "max_turn_requests",
                                     "num_turns": 1}), returncode=1)
        return _proc(json.dumps({"text": "the review", "stopReason": "end_turn", "num_turns": 1}))

    with patch("mcp_brain_router.backends.subprocess.run", cli):
        assert call_grok("a 34 KB review prompt", "grok-4.7")["content"] == "the review"


def test_grok_tools_off_still_reports_a_cut_turn_as_no_final_message():
    cut = _proc(json.dumps({"text": "", "stopReason": "max_turn_requests"}), returncode=1)
    with patch("mcp_brain_router.backends.subprocess.run", return_value=cut), pytest.raises(BackendError) as e:
        call_grok("x", "grok-4.7")
    assert e.value.failure_kind == "no_final_message"


# ---- write_runs_in_cwd ------------------------------------------------------

def test_write_runs_in_cwd_antigravity_with_no_hook_and_nothing_written_there(tmp_path):
    fake = _agy(tools="write", cwd=str(tmp_path))
    assert fake.cwd == os.path.realpath(tmp_path)
    assert fake.hooks is None and fake.files == []  # no .agents/ in the caller's folder
    assert os.listdir(tmp_path) == []
    a = fake.argv
    assert a[a.index("--mode") + 1] == "accept-edits" and "--add-dir" not in a


def test_write_runs_in_cwd_grok_with_tools_on_and_the_prompt_in_a_file(tmp_path):
    fake = _grok(tools="write", cwd=str(tmp_path))
    a = fake.argv
    assert fake.cwd == os.path.realpath(tmp_path)
    tools = a[a.index("--tools") + 1].split(",")
    assert {"read_file", "search_replace", "write", "run_terminal_command"} <= set(tools)
    assert a[a.index("--permission-mode") + 1] == "bypassPermissions"
    assert "--max-turns" not in a and "--prompt-file" in a and "-p" not in a
    assert "--no-subagents" in a and "--no-plan" in a


def test_write_runs_in_cwd_grok_accepts_a_multi_turn_answer(tmp_path):
    with patch("mcp_brain_router.backends.subprocess.run", return_value=_proc(json.dumps(GROK_OK))):
        assert call_grok("x", "grok-4.7", tools="write", cwd=str(tmp_path))["content"] == "done"


async def test_write_runs_in_cwd_through_the_walk_passes_the_callers_cwd(tmp_path):
    cfg = Config(grok_enabled=True, roles={"coder": ["grok-4.7"]})
    with patch.object(server, "_load_config", return_value=cfg), patch.object(server, "_log_delegation"), \
            patch("mcp_brain_router.router.backends.call_grok", return_value={"content": "ok"}) as g:
        r = await server._delegate_role_impl("coder", "x", "claude", mode="agentic", cwd=str(tmp_path),
                                             data_class="personal")
    assert g.call_args.kwargs["cwd"] == str(tmp_path) and g.call_args.kwargs["tools"] == "write"
    assert r["tools"] == "write"


# ---- read_only --------------------------------------------------------------

def test_read_only_antigravity_hook_allows_read_file_and_denies_everything_else(tmp_path):
    fake = _agy(tools="read", cwd=str(tmp_path))
    assert fake.hooks and fake.hooks != backends._AGY_DENY_HOOK
    assert fake.cwd != os.path.realpath(tmp_path)  # a temp folder, not the caller's
    assert os.listdir(tmp_path) == []  # nothing written into the caller's folder
    a = fake.argv
    assert a[a.index("--add-dir") + 1] == os.path.realpath(tmp_path) and "--mode" not in a
    assert _hook_verdict(fake.hooks, "read_file") == ""
    for tool in ("write_file", "edit_file", "run_command", "search_web", "call_mcp_tool", "mystery", ""):
        assert json.loads(_hook_verdict(fake.hooks, tool))["decision"] == "deny", tool


def test_read_only_antigravity_hook_fails_closed_on_garbage_input(tmp_path):
    fake = _agy(tools="read", cwd=str(tmp_path))
    entry = next(iter(json.loads(fake.hooks).values()))["PreToolUse"][0]["hooks"][0]
    out = subprocess.run(entry["command"], shell=True, capture_output=True, text=True, input="not json")
    assert json.loads(out.stdout)["decision"] == "deny"


def test_read_only_grok_gets_read_tools_only(tmp_path):
    fake = _grok(tools="read", cwd=str(tmp_path))
    a = fake.argv
    assert fake.cwd == os.path.realpath(tmp_path)
    assert set(a[a.index("--tools") + 1].split(",")) == {"read_file", "list_dir", "grep"}
    denied = set(a[a.index("--disallowed-tools") + 1].split(","))
    assert {"search_replace", "write", "run_terminal_command"} <= denied
    assert a[a.index("--permission-mode") + 1] == "dontAsk"
    assert "bypassPermissions" not in a and "acceptEdits" not in a


def test_read_only_claude_gets_read_grep_glob_only(tmp_path):
    fake = FakeRun(json.dumps({"result": "ok"}))
    with patch("mcp_brain_router.backends.subprocess.run", lambda argv, **kw: fake(argv, **{**kw, "cwd": str(tmp_path)})):
        backends.call_anthropic_agentic("x", "claude-haiku-5-5", str(tmp_path), tools="read")
    a = fake.argv
    assert a[a.index("--tools") + 1] == "Read,Grep,Glob"
    assert a[a.index("--allowedTools") + 1] == "Read,Grep,Glob"
    assert a[a.index("--permission-mode") + 1] == "dontAsk"


def test_claude_write_and_none_are_todays_call(tmp_path):
    for tools in ("none", "write"):
        fake = FakeRun(json.dumps({"result": "ok"}))
        with patch("mcp_brain_router.backends.subprocess.run",
                   lambda argv, **kw: fake(argv, **{**kw, "cwd": str(tmp_path)})):
            backends.call_anthropic_agentic("x", "claude-haiku-5-5", str(tmp_path), tools=tools)
        a = fake.argv
        assert a[a.index("--allowedTools") + 1] == "Read,Edit,Write,Bash"
        assert a[a.index("--permission-mode") + 1] == "bypassPermissions"


# ---- needs_cwd --------------------------------------------------------------

@pytest.mark.parametrize("tools", ["read", "write"])
@pytest.mark.parametrize("cwd", [None, "", "relative/dir", "/no/such/folder-xyz"])
def test_needs_cwd_backends_refuse_before_any_process_starts(tools, cwd):
    for fn, model in ((call_grok, "grok-4.7"), (call_antigravity, "gemini-4-neon-medium")):
        with patch("mcp_brain_router.backends.subprocess.run") as run, pytest.raises(BackendError) as e:
            fn("x", model, tools=tools, cwd=cwd)
        assert e.value.failure_kind == "validation_error"
        run.assert_not_called()


async def test_needs_cwd_the_delegate_call_with_tools_and_no_cwd_is_refused():
    cfg = Config(grok_enabled=True, roles={"coder": ["grok-4.7"]})
    with patch.object(server, "_load_config", return_value=cfg), patch.object(server, "_log_delegation"), \
            patch("mcp_brain_router.backends.subprocess.run") as run:
        r = await server._delegate_role_impl("coder", "x", "claude", mode="agentic", cwd=None,
                                             data_class="personal", tools="write")
    assert r["failure_kind"] == "validation_error" and "cwd" in r["error"]
    run.assert_not_called()


# ---- the response field, and orange ------------------------------------------

@pytest.mark.parametrize(("claude_tools", "field"), [("none", "write"), ("write", "write"), ("read", "read")])
async def test_response_says_what_a_claude_worker_ran_with(claude_tools, field):
    cfg = Config(roles={"coder": ["claude-sonnet-5-5"]})
    with patch.object(server, "_load_config", return_value=cfg), patch.object(server, "_log_delegation"), \
            patch("mcp_brain_router.router.backends.call_anthropic_agentic",
                  return_value={"content": "c", "usage": None}):
        r = await server._delegate_role_impl("coder", "x", "grok", mode="agentic", cwd="/tmp",
                                             data_class="personal", tools=claude_tools)
    assert r["backend"] == "anthropic-cli" and r["tools"] == field


@pytest.mark.parametrize("tools", ["none", "read", "write"])
async def test_orange_stays_claude_whatever_tools_says(tools):
    cfg = Config(grok_enabled=True, roles={"coder": ["grok-4.7", "gemini-4-neon-high", "claude-sonnet-5-5"]})
    with patch.object(server, "_load_config", return_value=cfg), patch.object(server, "_log_delegation"), \
            patch("mcp_brain_router.router.backends.call_grok") as g, \
            patch("mcp_brain_router.router.backends.call_antigravity") as agy, \
            patch("mcp_brain_router.router.backends.call_anthropic_agentic",
                  return_value={"content": "c", "usage": None}):
        r = await server._delegate_role_impl("coder", "x", "grok", mode="agentic", cwd="/tmp",
                                             data_class="orange", tools=tools)
    g.assert_not_called()
    agy.assert_not_called()
    assert r["backend"] == "anthropic-cli" and r["class"] == "orange"


# ---- review findings --------------------------------------------------------

async def test_role_missing_from_table_resolves_to_tools_none():
    r = await server._delegate_role_impl("orchestrator", "x", "claude", mode="agentic", cwd="/tmp")
    assert r["failure_kind"] == "validation_error" and "orchestrator" in r["error"]
    cfg = Config(grok_enabled=True, roles={"worker": ["grok-4.7"]})
    seen = []

    async def fake_route(assignment, prompt, config, mode="chat", cwd=None, effort=None, tools="none"):
        seen.append(tools)
        return RouteResult(content="ok", model="grok-4.7", backend="grok", complexity=None,
                           headroom_used=False)

    with patch.dict(server.ROLES, {}, clear=True), \
            patch.object(server, "_load_config", return_value=cfg), \
            patch.object(server, "_log_delegation"), \
            patch.object(server, "route_assignment", fake_route):
        out = await server._delegate_role_impl("worker", "x", "claude", mode="agentic", cwd="/tmp",
                                               data_class="personal")
    assert seen == ["none"] and out["tools"] == "none"


async def test_legacy_complexity_validates_tools_and_refuses_non_none():
    srv = server.create_server()
    cfg = Config(grok_enabled=True, roles={"processor": ["grok-4.7"]})
    with patch.object(server, "_load_config", return_value=cfg), \
            patch.object(server, "_log_delegation"), \
            patch.object(server, "_read_caller", return_value="claude"), \
            patch("mcp_brain_router.router.backends.call_grok",
                  return_value={"content": "ok", "usage": None}) as g:
        for bad in ("bogus", "", "WRITE"):
            out = await srv.call_tool("delegate", {"complexity": "code", "prompt": "x", "cwd": "/tmp", "tools": bad})
            res = out[1] if isinstance(out, tuple) else out
            assert res["failure_kind"] == "validation_error" and "tools must be one of" in res["error"]
        for mode_tools in ("read", "write"):
            out = await srv.call_tool("delegate", {"complexity": "code", "prompt": "x", "cwd": "/tmp", "tools": mode_tools})
            res = out[1] if isinstance(out, tuple) else out
            assert res["failure_kind"] == "validation_error" and "role=" in res["error"]
        g.assert_not_called()
        out = await srv.call_tool("delegate", {"complexity": "code", "prompt": "x", "cwd": "/tmp", "tools": "none"})
        res = out[1] if isinstance(out, tuple) else out
        assert res["answer"] == "ok" and res["tools"] == "none"


def test_grok_none_and_read_disallow_edit_and_shell_tools_and_read_uses_dont_ask(tmp_path):
    edit_and_shell = {"search_replace", "write", "run_terminal_command"}
    off = _grok(tools="none", cwd=str(tmp_path)).argv
    off_denied = set(off[off.index("--disallowed-tools") + 1].split(","))
    assert edit_and_shell <= off_denied and "read_file" in off_denied
    assert "--permission-mode" not in off

    read = _grok(tools="read", cwd=str(tmp_path)).argv
    read_denied = set(read[read.index("--disallowed-tools") + 1].split(","))
    assert edit_and_shell <= read_denied
    assert read[read.index("--permission-mode") + 1] == "dontAsk"
    assert "acceptEdits" not in read and "bypassPermissions" not in read


def test_claude_worker_needs_cwd_for_tools_and_strict_mcp_for_read(tmp_path):
    for tools in ("read", "write"):
        for bad_cwd in (None, "", "relative/dir", "/no/such/folder-xyz"):
            with patch("mcp_brain_router.backends.subprocess.run") as run, pytest.raises(BackendError) as e:
                backends.call_anthropic_agentic("x", "claude-haiku-5-5", cwd=bad_cwd, tools=tools)
            assert e.value.failure_kind == "validation_error"
            run.assert_not_called()

    read_fake = FakeRun(json.dumps({"result": "ok"}))
    with patch("mcp_brain_router.backends.subprocess.run",
               lambda argv, **kw: read_fake(argv, **{**kw, "cwd": str(tmp_path)})):
        backends.call_anthropic_agentic("x", "claude-haiku-5-5", cwd=str(tmp_path), tools="read")
    assert "--strict-mcp-config" in read_fake.argv

    for tools in ("none", "write"):
        fake = FakeRun(json.dumps({"result": "ok"}))
        with patch("mcp_brain_router.backends.subprocess.run",
                   lambda argv, **kw: fake(argv, **{**kw, "cwd": str(tmp_path)})):
            backends.call_anthropic_agentic("x", "claude-haiku-5-5", cwd=str(tmp_path), tools=tools)
        assert "--strict-mcp-config" not in fake.argv


def test_antigravity_read_hook_reads_bounded_prefix_and_has_twenty_second_timeout(tmp_path):
    fake = _agy(tools="read", cwd=str(tmp_path))
    entry = next(iter(json.loads(fake.hooks).values()))["PreToolUse"][0]["hooks"][0]
    assert entry["timeout"] == 20

    big_write = '{"toolCall": {"name": "write_file", "args": {"content": "' + ("x" * 2_000_000) + '"}}}'
    out = subprocess.run(entry["command"], shell=True, capture_output=True, text=True, input=big_write)
    assert json.loads(out.stdout)["decision"] == "deny"

    spoofed_first = (
        '{"toolCall": {"args": {"content": "{\\"toolCall\\": {\\"name\\": \\"read_file\\"}}'
        + ("x" * 200_000) + '"}, "name": "write_file"}}'
    )
    out = subprocess.run(entry["command"], shell=True, capture_output=True, text=True, input=spoofed_first)
    assert json.loads(out.stdout)["decision"] == "deny"

    spoofed_small = json.dumps({
        "toolCall": {"args": {"content": '{"toolCall": {"name": "read_file"}}'}, "name": "write_file"}
    })
    out = subprocess.run(entry["command"], shell=True, capture_output=True, text=True, input=spoofed_small)
    assert json.loads(out.stdout)["decision"] == "deny"

    big_read = '{"toolCall": {"name": "read_file", "args": {"path": "/a", "pad": "' + ("x" * 2_000_000) + '"}}}'
    out = subprocess.run(entry["command"], shell=True, capture_output=True, text=True, input=big_read)
    assert out.stdout == ""

    truncated = '{"toolCall": {"args": {"path": "/a"'
    out = subprocess.run(entry["command"], shell=True, capture_output=True, text=True, input=truncated)
    assert json.loads(out.stdout)["decision"] == "deny"


def test_explorer_order_and_antigravity_absolute_path_prompt(tmp_path):
    assert [(s.model, s.level) for s in ROLES["explorer"].slots] == [
        ("claude-haiku-5-5", "medium"),
        ("grok-4.7", "high"),
        ("gemini-4-hydrogen", "high"),
    ]
    assert DEFAULT_ROLES["explorer"] == ["claude-haiku-5-5", "grok-4.7", "gemini-4-hydrogen-high"]

    real_dir = os.path.realpath(tmp_path)
    for tools in ("read", "write"):
        fake = _agy(tools=tools, cwd=str(tmp_path))
        prompt_sent = fake.argv[fake.argv.index("-p") + 1]
        assert real_dir in prompt_sent
        assert "absolute paths" in prompt_sent.lower()
        assert "relative paths are relative to it" not in prompt_sent.lower()


async def test_non_tool_providers_skipped_with_needs_tools_and_reported_as_none():
    for backend in ("kimi", "deepseek", "codex", "apple-fm", "ollama", "other"):
        for tools in ("none", "read", "write"):
            assert server._tools_ran(backend, tools) == "none"
    assert server._tools_ran("grok", "read") == "read"
    assert server._tools_ran("antigravity", "write") == "write"
    assert server._tools_ran("anthropic-cli", "none") == "write"
    assert server._tools_ran("anthropic-cli", "read") == "read"

    cfg = Config(
        kimi_enabled=True,
        deepseek_key="k",
        codex_enabled=True,
        fm_host="h",
        ollama_host="http://127.0.0.1:11435",
        roles={"simple": ["kimi", "deepseek-flash", "gpt-6.1-sol", "apple-fm", "gpt-oss:20b", "claude-haiku-5-5"]},
    )
    with patch.object(server, "_load_config", return_value=cfg), \
            patch.object(server, "_log_delegation"), \
            patch("mcp_brain_router.router.backends.call_kimi") as m_kimi, \
            patch("mcp_brain_router.router.backends.call_deepseek") as m_ds, \
            patch("mcp_brain_router.router.backends.call_codex") as m_codex, \
            patch("mcp_brain_router.router.backends.call_fm") as m_fm, \
            patch("mcp_brain_router.router.backends.call_ollama") as m_ollama, \
            patch("mcp_brain_router.router.backends.call_anthropic_agentic",
                  return_value={"content": "c", "usage": None}):
        r = await server._delegate_role_impl("simple", "x", "grok", mode="agentic", cwd="/tmp",
                                             data_class="personal", tools="read")
    for mock in (m_kimi, m_ds, m_codex, m_fm, m_ollama):
        mock.assert_not_called()
    assert r["backend"] == "anthropic-cli" and r["tools"] == "read"
    for expected in ("kimi:needs_tools", "deepseek:needs_tools", "codex:needs_tools",
                     "apple-fm:needs_tools", "ollama:needs_tools"):
        assert expected in r["failures"]


async def test_simple_url_prompt_skips_antigravity_in_read_mode():
    cfg = Config(roles={"simple": ["gemini-4-hydrogen-medium", "claude-haiku-5-5"]})
    with patch.object(server, "_load_config", return_value=cfg), \
            patch.object(server, "_log_delegation"), \
            patch("mcp_brain_router.router.backends.call_antigravity") as agy, \
            patch("mcp_brain_router.router.backends.call_anthropic_agentic",
                  return_value={"content": "from claude", "usage": None}):
        r = await server._delegate_role_impl(
            "simple", "summarize https://example.com/article", "grok",
            mode="agentic", cwd="/tmp", data_class="personal", tools="read",
        )
    agy.assert_not_called()
    assert r["backend"] == "anthropic-cli" and "antigravity:needs_tools" in r["failures"]


# ---- #31 cross-provider review (gemini-4-carbon-high), findings 2, 3, 4 ----

async def test_explicit_tools_none_with_a_url_runs_antigravity_tools_off():
    cfg = Config(roles={"simple": ["gemini-4-hydrogen-medium", "claude-haiku-5-5"]})
    agentic = MagicMock(return_value={"content": "from claude", "usage": None})
    with patch.object(server, "_load_config", return_value=cfg), patch.object(server, "_log_delegation"), \
            patch("mcp_brain_router.router.backends.call_antigravity",
                  return_value={"content": "error", "usage": None}) as agy, \
            patch("mcp_brain_router.router.backends.call_anthropic_agentic", agentic):
        r = await server._delegate_role_impl("simple", "Classify log: GET https://example.com 500", "grok",
                                             mode="agentic", cwd="/tmp", data_class="personal", tools="none")
    agentic.assert_not_called()
    assert agy.called and agy.call_args.kwargs == {} and r["tools"] == "none"


async def test_simple_path_prompt_that_falls_to_haiku_gets_read_not_write():
    cfg = Config(roles={"simple": ["claude-haiku-5-5"]})
    agentic = MagicMock(return_value={"content": "from claude", "usage": None})
    with patch.object(server, "_load_config", return_value=cfg), patch.object(server, "_log_delegation"), \
            patch("mcp_brain_router.router.backends.call_anthropic_agentic", agentic):
        r = await server._delegate_role_impl("simple", "summarise ./untrusted.md", "grok", mode="agentic",
                                             cwd="/tmp", data_class="personal")
    assert agentic.call_args.kwargs.get("tools") == "read" and r["tools"] == "read"


async def test_a_local_needs_tools_verdict_gives_antigravity_read_even_when_the_regex_misses():
    cfg = Config(roles={"simple": ["apple-fm", "gemini-4-hydrogen-medium"]}, fm_host="fm-host")
    with patch.object(server, "_load_config", return_value=cfg), patch.object(server, "_log_delegation"), \
            patch("mcp_brain_router.router.backends.call_fm",
                  side_effect=backends.BackendError("NEEDS_TOOLS", backend="apple-fm",
                                                    failure_kind="needs_tools")), \
            patch("mcp_brain_router.router.backends.call_antigravity",
                  return_value={"content": "12 rows", "usage": None}) as agy:
        r = await server._delegate_role_impl("simple", "Count rows in scores.tsv", "grok", mode="agentic",
                                             cwd="/tmp", data_class="personal")
    assert agy.call_args.kwargs == {"tools": "read", "cwd": "/tmp"} and r["tools"] == "read"
