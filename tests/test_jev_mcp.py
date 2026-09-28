"""Public MCP and config contract for the Jev provider."""
import json
from unittest.mock import AsyncMock

import pytest

from mcp_brain_router import config, server


@pytest.mark.asyncio
async def test_registered_decide_reads_key_and_audits_usage(monkeypatch, tmp_path):
    key = tmp_path / "key"
    key.write_text("test-key")
    key.chmod(0o600)
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    monkeypatch.delenv("TYPESAFE_API_KEY_FILE", raising=False)
    monkeypatch.setattr(server, "_config", config.Config(jev_key_file=str(key)))
    audit = tmp_path / "audit.jsonl"
    monkeypatch.setattr(server, "_DELEGATION_LOG", audit)
    evaluate = AsyncMock(return_value={"model":"jev-1.13.0", "answers":{"q":{"type":"noul","noul":0.9}}, "usage":{"input_tokens":12,"output_tokens":3}})
    monkeypatch.setattr(server.jev, "evaluate", evaluate)
    mcp = server.create_server()
    names = {t.name for t in await mcp.list_tools()}
    assert {"delegate", "decide"} <= names
    questions = {"q":{"type":"noul","instructions":"Needs code?"}}
    await mcp.call_tool("decide", {"state":"private state", "questions":questions})
    evaluate.assert_awaited_once_with("private state", questions, "test-key", "jev-1.13.0")
    row = json.loads(audit.read_text())
    assert (row["backend"], row["tokens_in"], row["tokens_out"], row["usage_source"]) == ("jev",12,3,"api")
    assert "test-key" not in audit.read_text() and "private state" not in audit.read_text()
    key.chmod(0o644)
    result = await server._decide_impl("s", questions, "jev-1.13.0")
    assert result["failure_kind"] == "configuration_error"
    assert evaluate.await_count == 1


def test_key_path_config_roundtrip(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "CONFIG_DIR", tmp_path)
    monkeypatch.setattr(config, "CONFIG_FILE", tmp_path / "config.toml")
    config.Config(jev_key_file="/secret/key").save()
    assert config.Config.load().jev_key_file == "/secret/key"


@pytest.mark.asyncio
async def test_empty_key_and_unserializable_state_are_bounded(monkeypatch, tmp_path):
    key = tmp_path / "key"
    key.write_text("")
    key.chmod(0o600)
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    monkeypatch.setenv("TYPESAFE_API_KEY_FILE", str(key))
    monkeypatch.setattr(server, "_config", config.Config())
    monkeypatch.setattr(server, "_DELEGATION_LOG", tmp_path / "audit.jsonl")
    questions = {"q":{"type":"noul","instructions":"Is this true?"}}
    result = await server._decide_impl("s", questions, "jev-1.13.0")
    assert result["failure_kind"] == "configuration_error"
    key.write_text("test-key")
    result = await server._decide_impl({"invalid":{1,2}}, questions, "jev-1.13.0")
    assert result["failure_kind"] == "validation_error"
    assert len((tmp_path / "audit.jsonl").read_text().splitlines()) == 2
