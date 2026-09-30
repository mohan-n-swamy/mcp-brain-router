"""JEV-SHADOW: the decide spool records answered Jev calls without changing decide()."""
import copy
import json
import os
import stat
from unittest.mock import AsyncMock

import pytest

from mcp_brain_router import jev, server

Q = {"urgent": {"type": "noul", "instructions": "Urgent?"},
     "area": {"type": "choice", "instructions": "Which area?", "criteria": {"ops": None, "code": None}}}
JEV_OK = {"model": "jev-1.13.0", "usage": {"input_tokens": 9, "output_tokens": 2}, "answers": {
    "urgent": {"type": "noul", "noul": 0.8},
    "area": {"type": "choice", "choice": "ops", "confidence": 0.7, "probabilities": {"ops": 0.9, "code": 0.1}}}}
STATE = "private note text that must stay in the spool only"


@pytest.fixture
def wired(monkeypatch, tmp_path):
    monkeypatch.setenv("TYPESAFE_API_KEY", "test-key")
    monkeypatch.delenv("BRAIN_ROUTER_BENCH_RUN_ID", raising=False)
    monkeypatch.delenv("BRAIN_ROUTER_DECIDE_SPOOL", raising=False)
    audit = tmp_path / "audit.jsonl"
    monkeypatch.setattr(server, "_DELEGATION_LOG", audit)
    # a fresh copy per call, so an in-place change by the spool cannot hide in a shared object
    monkeypatch.setattr(server.jev, "evaluate", AsyncMock(side_effect=lambda *a, **k: copy.deepcopy(JEV_OK)))
    return audit


def _strip(resp):
    return {k: v for k, v in resp.items() if k != "elapsed_ms"}


@pytest.mark.asyncio
async def test_response_unchanged_when_spool_is_on(wired, monkeypatch, tmp_path):
    off = await server._decide_impl(STATE, Q, "jev-1.13.0")
    monkeypatch.setenv("BRAIN_ROUTER_DECIDE_SPOOL", str(tmp_path / "spool" / "spool.jsonl"))
    on = await server._decide_impl(STATE, Q, "jev-1.13.0")
    assert _strip(on) == _strip(off)
    assert "spool_id" not in on


@pytest.mark.asyncio
async def test_row_holds_input_and_untrainable_jev_answer(wired, monkeypatch, tmp_path):
    spool = tmp_path / "spool" / "spool.jsonl"
    monkeypatch.setenv("BRAIN_ROUTER_DECIDE_SPOOL", str(spool))
    await server._decide_impl(STATE, Q, "jev-1.13.0")
    row = json.loads(spool.read_text())
    assert (row["state"], row["questions"], row["site"]) == (STATE, Q, "area,urgent")
    assert row["jev"] == {"answers": JEV_OK["answers"], "trainable": False}
    assert stat.S_IMODE(spool.stat().st_mode) == 0o600
    audit_text = wired.read_text()
    assert json.loads(audit_text)["spool_id"] == row["spool_id"]
    assert STATE not in audit_text and "test-key" not in spool.read_text()


@pytest.mark.asyncio
async def test_spool_is_off_by_default(wired, tmp_path):
    await server._decide_impl(STATE, Q, "jev-1.13.0")
    assert json.loads(wired.read_text())["spool_id"] is None
    assert not list(tmp_path.glob("**/spool*"))


@pytest.mark.asyncio
async def test_fail_open_when_spool_is_unwritable(wired, monkeypatch, tmp_path):
    blocker = tmp_path / "not-a-dir"
    blocker.write_text("x")
    monkeypatch.setenv("BRAIN_ROUTER_DECIDE_SPOOL", str(blocker / "spool.jsonl"))
    resp = await server._decide_impl(STATE, Q, "jev-1.13.0")
    assert resp["answers"] == JEV_OK["answers"] and "error" not in resp
    assert json.loads(wired.read_text())["spool_id"] is None


@pytest.mark.asyncio
async def test_no_row_for_errors_or_bench(wired, monkeypatch, tmp_path):
    spool = tmp_path / "spool.jsonl"
    monkeypatch.setenv("BRAIN_ROUTER_DECIDE_SPOOL", str(spool))
    monkeypatch.setattr(server.jev, "evaluate",
                        AsyncMock(side_effect=jev.JevError("Jev rate limit exceeded", failure_kind="rate_limited")))
    await server._decide_impl(STATE, Q, "jev-1.13.0")
    monkeypatch.setattr(server.jev, "evaluate", AsyncMock(return_value=JEV_OK))
    monkeypatch.setenv("BRAIN_ROUTER_BENCH_RUN_ID", "bench-1")
    await server._decide_impl(STATE, Q, "jev-1.13.0")
    assert not spool.exists()


def test_no_row_for_laya_backend(monkeypatch, tmp_path):
    spool = tmp_path / "spool.jsonl"
    monkeypatch.setenv("BRAIN_ROUTER_DECIDE_SPOOL", str(spool))
    monkeypatch.delenv("BRAIN_ROUTER_BENCH_RUN_ID", raising=False)
    assert server._spool_decide({"backend": "laya", "answers": {}}, STATE, Q) is None
    assert not spool.exists()


@pytest.mark.asyncio
@pytest.mark.parametrize("case", ["symlink", "audit_log"])
async def test_fail_open_refuses_unsafe_paths(wired, monkeypatch, tmp_path, case):
    if case == "symlink":
        target = tmp_path / "elsewhere.jsonl"
        target.write_text("")
        (tmp_path / "spool.jsonl").symlink_to(target)
        path = tmp_path / "spool.jsonl"
    else:
        path = wired
    monkeypatch.setenv("BRAIN_ROUTER_DECIDE_SPOOL", str(path))
    resp = await server._decide_impl(STATE, Q, "jev-1.13.0")
    assert resp["answers"] == JEV_OK["answers"] and "error" not in resp
    assert STATE not in wired.read_text()
    if case == "symlink":
        assert target.read_text() == ""


@pytest.mark.asyncio
async def test_existing_loose_spool_is_tightened(wired, monkeypatch, tmp_path):
    spool = tmp_path / "spool.jsonl"
    spool.write_text("")
    os.chmod(spool, 0o644)
    monkeypatch.setenv("BRAIN_ROUTER_DECIDE_SPOOL", str(spool))
    await server._decide_impl(STATE, Q, "jev-1.13.0")
    assert stat.S_IMODE(spool.stat().st_mode) == 0o600 and spool.read_text()
