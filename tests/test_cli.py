"""The shell entry: `mcp-brain-router delegate` runs the same role walk as the MCP
tool, for scripts and scheduled jobs that cannot call an MCP tool."""
from __future__ import annotations

import io
import json
import sys
from unittest.mock import AsyncMock, patch

import pytest

from mcp_brain_router import server


def _run(argv, stdin=""):
    with patch.object(sys, "argv", ["mcp-brain-router", *argv]), patch.object(sys, "stdin", io.StringIO(stdin)):
        server.main()


def test_delegate_prints_the_result_and_exits_zero_on_an_answer(capsys, tmp_path):
    f = tmp_path / "p.txt"
    f.write_text("the task")
    walk = AsyncMock(return_value={"answer": "done", "backend": "apple-fm"})
    with patch.object(server, "_delegate_role_impl", walk), pytest.raises(SystemExit) as e:
        _run(["delegate", "--role", "simple", "--prompt-file", str(f), "--class", "personal", "--cwd", str(tmp_path)])
    assert e.value.code == 0
    assert json.loads(capsys.readouterr().out)["answer"] == "done"
    args, kw = walk.await_args
    assert args[:3] == ("simple", "the task", "apple") and args[4] == str(tmp_path)
    assert kw == {"effort": None, "data_class": "personal", "tools": None}


def test_delegate_reads_stdin_and_exits_one_without_an_answer(capsys):
    walk = AsyncMock(return_value={"error": "all eligible role providers failed", "backend": "none"})
    with patch.object(server, "_delegate_role_impl", walk), pytest.raises(SystemExit) as e:
        _run(["delegate", "--role", "worker", "--tools", "none"], stdin="from stdin")
    assert e.value.code == 1
    assert walk.await_args.args[1] == "from stdin" and walk.await_args.kwargs["tools"] == "none"
    assert "error" in json.loads(capsys.readouterr().out)


def test_an_unknown_role_or_tools_value_is_refused_by_the_parser():
    for argv in (["delegate", "--role", "nonsense", "--prompt", "x"],
                 ["delegate", "--role", "simple", "--tools", "everything", "--prompt", "x"]):
        with pytest.raises(SystemExit) as e:
            _run(argv)
        assert e.value.code == 2


def test_roles_prints_the_table(capsys):
    _run(["roles"])
    names = {r["name"] for r in json.loads(capsys.readouterr().out)["roles"]}
    assert {"thinker", "coder", "processor", "simple"} <= names
