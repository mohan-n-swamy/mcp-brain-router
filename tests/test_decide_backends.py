"""DECIDE-SWITCH: decide_backends slots (Jev and self-hosted laya) behind one pool."""
import json
from unittest.mock import AsyncMock

import httpx
import pytest

from mcp_brain_router import config, jev, jev_pool, server

Q = {"cause": {"type": "choice", "instructions": "What caused it?",
               "criteria": {"timeout": "too slow", "syntax": "does not parse", "network": "remote failed"}},
     "urgent": {"type": "noul", "instructions": "Urgent?"}}
LAYA_BODY = {  # shape built by laya Agent._decode_answers + serve.py (laya 0.3.20)
    "model": "english",
    "answers": {"cause": {"type": "choice", "choice": "timeout",
                          "probabilities": {"timeout": 0.9123, "syntax": 0.0512, "network": 0.0364},
                          "confidence": 0.61, "answer_confidence": 0.9123, "action": {"act_probability": 0.97}},
                "urgent": {"type": "noul", "noul": 0.8123, "confidence": 0.8123,
                           "answer_confidence": 0.8123, "action": {"act_probability": 0.9}}},
    "usage": {"input_tokens": 120, "output_tokens": 0},
    "routing": {"model": "english", "reason": "latin script"},
}
JEV_OK = {"model": "jev-1.13.0", "answers": {
    "cause": {"type": "choice", "choice": "timeout", "confidence": 0.9,
              "probabilities": {"timeout": 0.95, "syntax": 0.03, "network": 0.02}},
    "urgent": {"type": "noul", "noul": 0.7}}, "usage": {"input_tokens": 50, "output_tokens": 3}}
LAYA_KEY, JEV_KEY = "LAYA-SECRET-333", "JEV-SECRET-444"
LAYA_URL = "http://100.100.10.20:8090/v1/systemone"


@pytest.fixture(autouse=True)
def _clear_cooldowns():
    jev_pool._cooldown_until.clear()
    yield
    jev_pool._cooldown_until.clear()


@pytest.fixture
def backends(tmp_path):
    out = []
    for name, url, provider, key in (("laya-mini", LAYA_URL, "laya", LAYA_KEY),
                                     ("jev-office", jev.JEV_ENDPOINT, "jev", JEV_KEY)):
        p = tmp_path / name
        p.write_text(key)
        p.chmod(0o600)
        out.append({"name": name, "url": url, "provider": provider, "key_file": str(p)})
    return out


def _wire(monkeypatch, tmp_path, cfg, side_effect=None):
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    monkeypatch.delenv("TYPESAFE_API_KEY_FILE", raising=False)
    monkeypatch.setattr(server, "_config", cfg)
    audit = tmp_path / "audit.jsonl"
    monkeypatch.setattr(server, "_DELEGATION_LOG", audit)
    evaluate = AsyncMock(side_effect=side_effect)
    if side_effect is not None:
        monkeypatch.setattr(server.jev, "evaluate", evaluate)
    return audit, evaluate


def _mock_http(monkeypatch, handler):
    real = httpx.AsyncClient
    monkeypatch.setattr(jev.httpx, "AsyncClient",
                        lambda **kw: real(transport=httpx.MockTransport(handler), **kw))


@pytest.mark.asyncio
async def test_laya_slot_end_to_end_through_real_adapter(monkeypatch, tmp_path, backends):
    seen = []

    def handler(request):
        seen.append((str(request.url), request.headers.get("authorization")))
        return httpx.Response(200, json=LAYA_BODY)
    _mock_http(monkeypatch, handler)
    audit, _ = _wire(monkeypatch, tmp_path, config.Config(decide_backends=backends))
    result = await server._decide_impl("the build timed out", Q, "jev-1.13.0")
    assert result["answers"]["cause"]["choice"] == "timeout" and result["decide_slot"] == "laya-mini"
    assert seen == [(LAYA_URL, "Bearer " + LAYA_KEY)]


@pytest.mark.asyncio
async def test_jev_provider_still_rejects_non_jev_model(monkeypatch):
    with pytest.raises(jev.JevError) as info:
        jev._validate_response(dict(LAYA_BODY), Q, "jev")
    assert info.value.failure_kind == "response_error"
    assert jev._validate_response(dict(LAYA_BODY), Q, "laya")["model"] == "english"


@pytest.mark.parametrize("url,ok", [
    ("https://api.typesafe.ai/v1/systemone", True), ("http://127.0.0.1:8090/v1/systemone", True),
    ("http://localhost:8090/v1/systemone", True), ("http://100.100.10.20:8090/v1/systemone", True),
    ("http://100.100.10.30:8090/v1/systemone", True), ("http://192.168.1.20:8090/v1/systemone", False),
    ("http://example.com/v1/systemone", False), ("ftp://100.100.10.20/x", False), ("not a url", False)])
def test_endpoint_scheme_https_or_loopback_or_tailnet(url, ok, tmp_path):
    k = tmp_path / "k"
    k.write_text("x")
    entry = [{"name": "a", "url": url, "provider": "laya", "key_file": str(k)}]
    if ok:
        assert jev_pool.normalize_backends(entry)[0].url == url
    else:
        with pytest.raises(jev_pool.SlotConfigError):
            jev_pool.normalize_backends(entry)
        with pytest.raises(jev.JevError):
            jev.validate_endpoint(url)


@pytest.mark.asyncio
async def test_endpoint_scheme_bad_config_is_configuration_error(monkeypatch, tmp_path, backends):
    backends[0]["url"] = "http://example.com/v1/systemone"
    _, evaluate = _wire(monkeypatch, tmp_path, config.Config(decide_backends=backends), [JEV_OK])
    result = await server._decide_impl("s", Q, "jev-1.13.0")
    assert result["failure_kind"] == "configuration_error" and evaluate.await_count == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["network_error", "timeout", "unavailable"])
async def test_tunnel_down_falls_over_to_jev(monkeypatch, tmp_path, backends, kind):
    _, evaluate = _wire(monkeypatch, tmp_path, config.Config(decide_backends=backends),
                        [jev.JevError("down", failure_kind=kind), JEV_OK])
    result = await server._decide_impl("s", Q, "jev-1.13.0")
    assert result["decide_slot"] == "jev-office" and result["backend"] == "jev" and result["key_slot"] == 1
    first, second = evaluate.await_args_list
    assert first.kwargs == {"endpoint": LAYA_URL, "provider": "laya"} and second.kwargs == {}


@pytest.mark.asyncio
async def test_backend_logged_in_audit(monkeypatch, tmp_path, backends):
    audit, _ = _wire(monkeypatch, tmp_path, config.Config(decide_backends=backends), [dict(LAYA_BODY)])
    await server._decide_impl("s", Q, "jev-1.13.0")
    row = json.loads(audit.read_text())
    assert (row["decide_slot"], row["backend"], row["key_slot"]) == ("laya-mini", "laya", 0)


@pytest.mark.asyncio
async def test_no_key_material_in_response_or_audit(monkeypatch, tmp_path, backends):
    audit, _ = _wire(monkeypatch, tmp_path, config.Config(decide_backends=backends), [
        jev.JevError("down", failure_kind="network_error"), JEV_OK,
        jev.JevError("down", failure_kind="network_error"), jev.JevError("auth", failure_kind="authentication_error")])
    responses = [await server._decide_impl("s", Q, "jev-1.13.0") for _ in range(2)]
    blob = json.dumps(responses) + audit.read_text()
    assert LAYA_KEY not in blob and JEV_KEY not in blob
    assert responses[1]["failure_kind"] == "all_slots_exhausted"


@pytest.mark.asyncio
async def test_decide_backends_wins_over_jev_key_files(monkeypatch, tmp_path, backends):
    cfg = config.Config(decide_backends=backends, jev_key_files=[backends[1]["key_file"]])
    _, evaluate = _wire(monkeypatch, tmp_path, cfg, [dict(LAYA_BODY)])
    result = await server._decide_impl("s", Q, "jev-1.13.0")
    assert result["decide_slot"] == "laya-mini" and evaluate.await_args.kwargs["provider"] == "laya"


def test_decide_backends_config_roundtrip(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "CONFIG_DIR", tmp_path)
    monkeypatch.setattr(config, "CONFIG_FILE", tmp_path / "config.toml")
    b = [{"name": "laya-mini", "url": LAYA_URL, "provider": "laya", "key_file": "/k/a"},
         {"name": "jev-office", "url": jev.JEV_ENDPOINT, "provider": "jev", "key_file": "/k/b"}]
    config.Config(decide_backends=b).save()
    assert config.Config.load().decide_backends == b


def test_duplicate_or_incomplete_backends_rejected():
    with pytest.raises(jev_pool.SlotConfigError):
        jev_pool.normalize_backends([{"name": "a", "url": LAYA_URL, "provider": "laya", "key_file": "/k"}] * 2)
    with pytest.raises(jev_pool.SlotConfigError):
        jev_pool.normalize_backends([{"name": "a", "url": LAYA_URL, "provider": "other", "key_file": "/k"}])
    with pytest.raises(jev_pool.SlotConfigError):
        jev_pool.normalize_backends([{"name": "a", "url": LAYA_URL}])


@pytest.mark.asyncio
async def test_slot_model_laya_sends_named_checkpoint(monkeypatch, tmp_path, backends):
    sent = []

    def handler(request):
        sent.append(json.loads(request.content)["model"])
        return httpx.Response(200, json=LAYA_BODY)
    _mock_http(monkeypatch, handler)
    backends[0]["model"] = "typed-decisions"
    _wire(monkeypatch, tmp_path, config.Config(decide_backends=backends))
    result = await server._decide_impl("s", Q, "jev-1.13.0")
    assert result["decide_slot"] == "laya-mini" and sent == ["typed-decisions"]


@pytest.mark.asyncio
async def test_slot_model_absent_keeps_caller_model(monkeypatch, tmp_path, backends):
    sent = []

    def handler(request):
        sent.append(json.loads(request.content)["model"])
        return httpx.Response(200, json=LAYA_BODY)
    _mock_http(monkeypatch, handler)
    _wire(monkeypatch, tmp_path, config.Config(decide_backends=backends))
    await server._decide_impl("s", Q, "jev-1.13.0")
    assert sent == ["jev-1.13.0"]


@pytest.mark.asyncio
async def test_slot_model_jev_still_requires_jev_name():
    with pytest.raises(jev.JevError) as info:
        await jev.evaluate("s", Q, "k", "typed-decisions")
    assert info.value.failure_kind == "validation_error"
    with pytest.raises(jev.JevError):
        await jev.evaluate("s", Q, "k", "Bad Model!", endpoint="http://127.0.0.1:8090/v1/systemone", provider="laya")


def test_slot_model_empty_rejected():
    with pytest.raises(jev_pool.SlotConfigError):
        jev_pool.normalize_backends([{"name": "a", "url": LAYA_URL, "provider": "laya", "key_file": "/k", "model": " "}])
