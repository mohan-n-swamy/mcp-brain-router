"""Fail-open role cascade (rig-consolidation D-R-01, 2026-09-02).

Before: the role loop advanced only on quota / transient results, so a
process_error from the first tier ended the call. Live 2026-09-02: glm failed
after ~190 s on 35 of 35 `simple` calls in 48 h and nothing reached kimi.
"""
from __future__ import annotations

import inspect
import json
import subprocess
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mcp_brain_router import backends, server
from mcp_brain_router.backends import BackendError
from mcp_brain_router.config import Config
from mcp_brain_router.router import RouteResult


@pytest.fixture
def role_config():
    return Config(
        deepseek_key="ds",
        glm_key="glm",
        codex_enabled=True,
        roles={
            "worker": ["glm-5.2", "grok-4.5", "gpt-5.6-terra", "sonnet-worker"],
            "simple": ["glm-4.7", "luna-simple", "haiku-simple"],
        },
    )


def _ok(model="grok-4.5", backend="grok"):
    return RouteResult(
        content="built", model=model, backend=backend, complexity=None, headroom_used=False
    )


def _err(kind="process_error", elapsed_ms=190_000, backend="glm"):
    return BackendError(
        f"{backend} {kind}", backend=backend, failure_kind=kind, elapsed_ms=elapsed_ms
    )


def _quota():
    return RouteResult(
        content="quota", model="", backend="none", complexity=None, headroom_used=False,
        exhausted=True, failure_kind="quota_exhausted", failure_reason="quota",
    )


async def _run(role_config, side_effect, role="worker", orchestrator="opus"):
    with (
        patch.object(server, "_load_config", return_value=role_config),
        patch(
            "mcp_brain_router.server.route_assignment",
            new_callable=AsyncMock,
            side_effect=side_effect,
        ) as route_call,
    ):
        response = await server._delegate_role_impl(role, "build", orchestrator, cwd="/tmp")
    return response, route_call


@pytest.mark.asyncio
async def test_delegate_role_advances_on_process_error(role_config):
    response, route_call = await _run(role_config, [_err(), _ok()])
    assert response["answer"] == "built"
    assert response["backend"] == "grok"
    assert response["tried"] == ["glm", "grok"]
    assert response["retried"] == []
    assert "failure_kind" not in response
    assert route_call.await_count == 2


@pytest.mark.asyncio
async def test_all_candidates_failed_reports_last_failure_kind(role_config):
    # orchestrator opus → anthropic skipped (skip-self): glm, grok, codex
    response, route_call = await _run(
        role_config, [_err(), _err(backend="grok"), _err("invalid_output", backend="codex")]
    )
    assert response["backend"] == "none"
    assert response["exhausted"] is True
    assert response["failure_kind"] == "invalid_output"
    assert response["tried"] == ["glm", "grok", "codex"]
    assert route_call.await_count == 3


@pytest.mark.asyncio
async def test_mixed_quota_and_process_error_is_not_labelled_quota(role_config):
    response, _ = await _run(role_config, [_quota(), _err(backend="grok"), _quota()])
    assert response["exhausted"] is True
    # the LAST failure was quota, but not EVERY failure was: honest kind wins
    assert response["failure_kind"] == "process_error"


@pytest.mark.asyncio
async def test_all_quota_keeps_quota_contract(role_config):
    response, _ = await _run(role_config, [_quota(), _quota(), _quota()])
    assert response["exhausted"] is True
    assert response["failure_kind"] == "quota_exhausted"


@pytest.mark.asyncio
async def test_fast_failure_retried_once(role_config):
    response, route_call = await _run(role_config, [_err(elapsed_ms=800), _ok("glm-5.2", "glm")])
    assert response["answer"] == "built"
    assert response["tried"] == ["glm", "glm"]
    assert response["retried"] == ["glm"]
    assert route_call.await_count == 2
    assert [c.args[0].model for c in route_call.await_args_list] == ["glm-5.2", "glm-5.2"]


@pytest.mark.asyncio
async def test_fast_failure_twice_then_advances(role_config):
    response, route_call = await _run(
        role_config, [_err(elapsed_ms=800), _err(elapsed_ms=900), _ok()]
    )
    assert response["answer"] == "built"
    assert response["tried"] == ["glm", "glm", "grok"]
    assert response["retried"] == ["glm"]
    assert route_call.await_count == 3


@pytest.mark.asyncio
async def test_slow_failure_never_retried(role_config):
    response, route_call = await _run(role_config, [_err(elapsed_ms=190_000), _ok()])
    assert response["retried"] == []
    assert response["tried"] == ["glm", "grok"]


@pytest.mark.asyncio
async def test_timeout_never_retried(role_config):
    # a timeout surfaces as a transient (exhausted=True) result from route_assignment
    timeout = RouteResult(
        content="t", model="", backend="none", complexity=None, headroom_used=False,
        exhausted=True, failure_kind="timeout", failure_reason="timed out",
    )
    response, route_call = await _run(role_config, [timeout, _ok()])
    assert response["retried"] == []
    assert response["tried"] == ["glm", "grok"]
    # and a raised timeout-kind error with tiny elapsed is still not retried
    response, route_call = await _run(role_config, [_err("timeout", elapsed_ms=10), _ok()])
    assert response["retried"] == []


@pytest.mark.asyncio
async def test_configuration_error_never_retried(role_config):
    response, _ = await _run(role_config, [_err("configuration_error", elapsed_ms=5), _ok()])
    assert response["retried"] == []
    assert response["tried"] == ["glm", "grok"]


def test_retry_window_is_well_under_the_agentic_timeout():
    assert server.RETRY_FAST_MS <= 15_000
    assert server.RETRY_FAST_MS < backends.AGENTIC_TIMEOUT_SECONDS * 1000 / 10
    assert not server._retryable("process_error", None)
    assert not server._retryable("process_error", server.RETRY_FAST_MS)
    assert server._retryable("network_error", 1)


def test_retry_predicate_does_not_read_tokens_out():
    assert "tokens_out" not in inspect.getsource(server._retryable)
    loop = inspect.getsource(server._delegate_role_impl)
    retry_lines = [l for l in loop.splitlines() if "_retryable(" in l]
    assert retry_lines and all("tokens_out" not in l for l in retry_lines)


@pytest.mark.asyncio
async def test_log_record_carries_role_and_retried(role_config, tmp_path):
    log = tmp_path / "log.jsonl"
    with patch.object(server, "_DELEGATION_LOG", log):
        await _run(role_config, [_err(elapsed_ms=800), _ok("glm-5.2", "glm")])
    rows = [json.loads(l) for l in log.read_text().splitlines()]
    assert len(rows) == 1
    assert rows[0]["role"] == "worker"
    assert rows[0]["retried"] == ["glm"]
    assert rows[0]["tried"] == ["glm", "glm"]


def test_every_cli_spawn_closes_stdin():
    """Live 2026-09-02 failure text: 'no stdin data received in 3s … redirect
    stdin explicitly: < /dev/null' — the child inherited the MCP stdio pipe."""
    src = inspect.getsource(backends)
    calls = src.count("result = subprocess.run(")
    assert calls == 8, calls
    assert src.count("stdin=subprocess.DEVNULL,") == calls
    cases = (
        (backends.call_glm_agentic, ("p", "glm-5.3", "/tmp")),
        (backends.call_grok_agentic, ("p", "grok-4.5", "/tmp")),
        (backends.call_kimi_agentic, ("p", "kimi", "/tmp")),
        (backends.call_codex_agentic, ("p", "gpt-5.6-terra", "/tmp")),
        (backends.call_anthropic_agentic, ("p", "claude-sonnet-5", "/tmp")),
    )
    for fn, args in cases:
        with patch("mcp_brain_router.backends.subprocess.run") as mrun, patch(
            "mcp_brain_router.backends._resolve_agentic_cwd", return_value="/tmp"
        ):
            mrun.return_value = MagicMock(returncode=0, stdout="ok", stderr="")
            try:
                fn(*args)
            except BackendError:
                pass  # some workers validate the stdout shape; the spawn kwargs are what we check
            assert mrun.call_args.kwargs.get("stdin") is subprocess.DEVNULL, fn.__name__


def test_cli_snippet_keeps_tail():
    banner = "Claude Code -> GLM (glm-5.3[1m], fast=glm-5.3-flash) via z.ai. " * 6
    text = backends._cli_snippet(banner + "real error: connect ECONNREFUSED at the end")
    assert "real error: connect ECONNREFUSED at the end" in text
    assert len(text) <= 240
