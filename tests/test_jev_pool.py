"""JEV-LOAD key pool: failover, cooldown, bounds, audit slot, no key material."""
import json
from unittest.mock import AsyncMock

import pytest

from mcp_brain_router import config, jev, jev_pool, server

Q = {"q": {"type": "noul", "instructions": "Is this true?"}}
OK = {"model": "jev-1.13.0", "answers": {"q": {"type": "noul", "noul": 0.9}},
      "usage": {"input_tokens": 10, "output_tokens": 2}}
KEY_A, KEY_B = "SECRET-KEY-A-111", "SECRET-KEY-B-222"


@pytest.fixture(autouse=True)
def _clear_cooldowns():
    jev_pool._cooldown_until.clear()
    yield
    jev_pool._cooldown_until.clear()


@pytest.fixture
def keys(tmp_path):
    paths = []
    for name, value in (("key_a", KEY_A), ("key_b", KEY_B)):
        p = tmp_path / name
        p.write_text(value)
        p.chmod(0o600)
        paths.append(str(p))
    return paths


def _err(kind, retry_after=None):
    return jev.JevError("Jev call failed", failure_kind=kind, retry_after=retry_after)


def _wire(monkeypatch, tmp_path, cfg, side_effect):
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    monkeypatch.delenv("TYPESAFE_API_KEY_FILE", raising=False)
    monkeypatch.setattr(server, "_config", cfg)
    audit = tmp_path / "audit.jsonl"
    monkeypatch.setattr(server, "_DELEGATION_LOG", audit)
    evaluate = AsyncMock(side_effect=side_effect)
    monkeypatch.setattr(server.jev, "evaluate", evaluate)
    return audit, evaluate


def _keys_used(evaluate):
    return [c.args[2] for c in evaluate.await_args_list]


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", sorted(jev_pool.FAILOVER_KINDS))
async def test_each_failover_kind_moves_to_next_slot(monkeypatch, tmp_path, keys, kind):
    _, evaluate = _wire(monkeypatch, tmp_path, config.Config(jev_key_files=keys), [_err(kind), OK])
    result = await server._decide_impl("s", Q, "jev-1.13.0")
    assert result["answers"] == OK["answers"] and result["key_slot"] == 1
    assert _keys_used(evaluate) == [KEY_A, KEY_B]


@pytest.mark.asyncio
async def test_key_slot_logged_in_audit(monkeypatch, tmp_path, keys):
    audit, _ = _wire(monkeypatch, tmp_path, config.Config(jev_key_files=keys), [_err("rate_limited"), OK])
    await server._decide_impl("s", Q, "jev-1.13.0")
    row = json.loads(audit.read_text().splitlines()[-1])
    assert row["key_slot"] == 1 and row["backend"] == "jev"


@pytest.mark.asyncio
async def test_key_slot_logged_zero_when_first_slot_serves(monkeypatch, tmp_path, keys):
    audit, evaluate = _wire(monkeypatch, tmp_path, config.Config(jev_key_files=keys), [OK])
    result = await server._decide_impl("s", Q, "jev-1.13.0")
    assert result["key_slot"] == 0 and _keys_used(evaluate) == [KEY_A]
    assert json.loads(audit.read_text())["key_slot"] == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["validation_error", "response_error", "backend_error"])
async def test_no_failover_on_validation_or_response_error(monkeypatch, tmp_path, keys, kind):
    _, evaluate = _wire(monkeypatch, tmp_path, config.Config(jev_key_files=keys), [_err(kind), OK])
    result = await server._decide_impl("s", Q, "jev-1.13.0")
    assert result["failure_kind"] == kind and result["key_slot"] == 0
    assert _keys_used(evaluate) == [KEY_A]


@pytest.mark.asyncio
async def test_cooldown_skips_slot_until_retry_after(keys):
    now = [100.0]
    clock = lambda: now[0]  # noqa: E731
    fn = AsyncMock(side_effect=[_err("rate_limited", retry_after=30), OK])
    _, slot, _ = await jev_pool.evaluate("s", Q, "jev-1.13.0", keys, evaluate_fn=fn, clock=clock)
    assert slot == 1
    fn = AsyncMock(side_effect=[OK])
    _, slot, attempts = await jev_pool.evaluate("s", Q, "jev-1.13.0", keys, evaluate_fn=fn, clock=clock)
    assert slot == 1 and attempts == [{"slot": 0, "outcome": "cooldown"}]
    assert _keys_used(fn) == [KEY_B]
    now[0] = 131.0
    fn = AsyncMock(side_effect=[OK])
    _, slot, _ = await jev_pool.evaluate("s", Q, "jev-1.13.0", keys, evaluate_fn=fn, clock=clock)
    assert slot == 0 and _keys_used(fn) == [KEY_A]


@pytest.mark.asyncio
async def test_bounded_attempts_one_per_slot(keys):
    fn = AsyncMock(side_effect=_err("unavailable"))
    with pytest.raises(jev.JevError) as info:
        await jev_pool.evaluate("s", Q, "jev-1.13.0", keys, evaluate_fn=fn)
    assert info.value.failure_kind == "all_slots_exhausted"
    assert fn.await_count == 2 and _keys_used(fn) == [KEY_A, KEY_B]


@pytest.mark.asyncio
async def test_all_slots_exhausted_is_explicit(monkeypatch, tmp_path, keys):
    audit, _ = _wire(monkeypatch, tmp_path, config.Config(jev_key_files=keys),
                     [_err("rate_limited", retry_after=9), _err("rate_limited", retry_after=5)])
    result = await server._decide_impl("s", Q, "jev-1.13.0")
    assert result["failure_kind"] == "all_slots_exhausted"
    assert "all 2 Jev key slots failed" in result["error"]
    assert 0 < result["retry_after"] <= 5
    assert "answers" not in result
    assert json.loads(audit.read_text())["failure_kind"] == "all_slots_exhausted"


@pytest.mark.asyncio
async def test_single_slot_keeps_original_failure_kind(monkeypatch, tmp_path, keys):
    _wire(monkeypatch, tmp_path, config.Config(jev_key_file=keys[0]), [_err("rate_limited", retry_after=7)])
    result = await server._decide_impl("s", Q, "jev-1.13.0")
    assert result["failure_kind"] == "rate_limited" and result["retry_after"] == 7


@pytest.mark.asyncio
async def test_unusable_key_file_slot_is_skipped(monkeypatch, tmp_path, keys):
    (tmp_path / "key_a").chmod(0o644)
    _, evaluate = _wire(monkeypatch, tmp_path, config.Config(jev_key_files=keys), [OK])
    result = await server._decide_impl("s", Q, "jev-1.13.0")
    assert result["key_slot"] == 1 and _keys_used(evaluate) == [KEY_B]
    assert result["key_attempts"] == [{"slot": 0, "outcome": "configuration_error"}]
    (tmp_path / "key_b").chmod(0o644)
    result = await server._decide_impl("s", Q, "jev-1.13.0")
    assert result["failure_kind"] == "configuration_error" and evaluate.await_count == 1


@pytest.mark.asyncio
async def test_bad_jev_key_files_value_is_configuration_error(monkeypatch, tmp_path):
    _, evaluate = _wire(monkeypatch, tmp_path, config.Config(jev_key_files="not-a-list"), [OK])
    result = await server._decide_impl("s", Q, "jev-1.13.0")
    assert result["failure_kind"] == "configuration_error" and evaluate.await_count == 0


@pytest.mark.asyncio
async def test_env_key_file_overrides_pool(monkeypatch, tmp_path, keys):
    _, evaluate = _wire(monkeypatch, tmp_path, config.Config(jev_key_files=keys), [OK])
    monkeypatch.setenv("TYPESAFE_API_KEY_FILE", keys[1])
    result = await server._decide_impl("s", Q, "jev-1.13.0")
    assert result["key_slot"] == 0 and _keys_used(evaluate) == [KEY_B]


@pytest.mark.asyncio
async def test_no_key_material_in_response_or_audit(monkeypatch, tmp_path, keys):
    audit, _ = _wire(monkeypatch, tmp_path, config.Config(jev_key_files=keys), [
        _err("rate_limited", retry_after=3), OK,            # failover success
        _err("authentication_error"), _err("timeout"),      # exhausted
        _err("validation_error"),                           # no failover
    ])
    jev_pool._cooldown_until.clear()
    responses = [await server._decide_impl("s", Q, "jev-1.13.0")]
    jev_pool._cooldown_until.clear()
    responses.append(await server._decide_impl("s", Q, "jev-1.13.0"))
    responses.append(await server._decide_impl("s", Q, "jev-1.13.0"))
    blob = json.dumps(responses) + audit.read_text()
    assert KEY_A not in blob and KEY_B not in blob
    assert [r.get("failure_kind") for r in responses] == [None, "all_slots_exhausted", "validation_error"]


def test_jev_key_files_config_roundtrip(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "CONFIG_DIR", tmp_path)
    monkeypatch.setattr(config, "CONFIG_FILE", tmp_path / "config.toml")
    config.Config(jev_key_file="/secret/a", jev_key_files=["/secret/a", "/secret/b"]).save()
    loaded = config.Config.load()
    assert loaded.jev_key_files == ["/secret/a", "/secret/b"] and loaded.jev_key_file == "/secret/a"
