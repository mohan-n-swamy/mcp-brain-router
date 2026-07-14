from unittest.mock import AsyncMock, patch

import pytest

from mcp_brain_router import server
from mcp_brain_router.backends import BackendError, BackendQuotaError
from mcp_brain_router.config import Config
from mcp_brain_router.router import (
    Provider,
    Role,
    RouteResult,
    orchestrator_provider,
    resolve_role,
)


@pytest.fixture
def role_config():
    return Config(
        deepseek_key="ds",
        glm_key="glm",
        codex_enabled=True,
        roles={
            "thinker": ["fable-planner", "sol-planner"],
            "adversary": ["opus-refuter", "sol-refuter"],
            "worker": [
                "glm-5.2",
                "grok-4.5",
                "gpt-5.6-terra",
                "sonnet-worker",
            ],
            "simple": ["glm-4.7", "luna-simple", "haiku-simple"],
        },
    )


@pytest.mark.parametrize(
    ("role", "orchestrator", "provider", "native"),
    [
        (Role.THINKER, "opus", Provider.ANTHROPIC, True),
        (Role.THINKER, "codex", Provider.ANTHROPIC, True),
        (Role.ADVERSARY, "opus", Provider.CODEX, False),
        (Role.ADVERSARY, "codex", Provider.ANTHROPIC, True),
        (Role.WORKER, "opus", Provider.ZHIPU, False),
        (Role.WORKER, "codex", Provider.ZHIPU, False),
        (Role.SIMPLE, "opus", Provider.ZHIPU, False),
        (Role.SIMPLE, "codex", Provider.ZHIPU, False),
        (Role.WORKER, Provider.ANTHROPIC, Provider.ZHIPU, False),
        (Role.SIMPLE, Provider.CODEX, Provider.ZHIPU, False),
    ],
)
def test_role_truth_table(role_config, role, orchestrator, provider, native):
    assignment = resolve_role(role, orchestrator, role_config)
    assert assignment.provider is provider
    assert assignment.execute_natively is native


@pytest.mark.parametrize("orchestrator", [Provider.ANTHROPIC, Provider.CODEX])
def test_adversary_always_differs(role_config, orchestrator):
    assignment = resolve_role(Role.ADVERSARY, orchestrator, role_config)
    assert assignment.provider is not orchestrator


@pytest.mark.parametrize("orchestrator", ["claude", "codex", "grok"])
def test_only_adversary_excludes_orchestrator_provider(role_config, orchestrator):
    adversary = resolve_role(Role.ADVERSARY, orchestrator, role_config)
    worker = resolve_role(Role.WORKER, orchestrator, role_config)
    assert adversary.provider is not orchestrator_provider(orchestrator)
    assert worker.provider is Provider.ZHIPU


def test_exhaustion_walks_candidate_providers(role_config):
    assignment = resolve_role(
        Role.WORKER,
        Provider.ANTHROPIC,
        role_config,
        exhausted_providers={Provider.ZHIPU},
    )
    assert assignment.provider is Provider.XAI
    assert assignment.model == "grok-4.5"


def test_anthropic_fallback_is_native(role_config):
    assignment = resolve_role(
        Role.WORKER,
        Provider.CODEX,
        role_config,
        exhausted_providers={Provider.ZHIPU, Provider.XAI, Provider.CODEX},
    )
    assert assignment.provider is Provider.ANTHROPIC
    assert assignment.backend is None
    assert assignment.execute_natively is True


def test_orchestrator_role_is_never_resolved(role_config):
    with pytest.raises(ValueError, match="selected by the human"):
        resolve_role(Role.ORCHESTRATOR, "opus", role_config)


def test_same_provider_only_adversary_rejected():
    config = Config(roles={"adversary": ["gpt-5.6-sol", "terra-refuter"]})
    with pytest.raises(ValueError, match="differs"):
        resolve_role(Role.ADVERSARY, "codex", config)


def test_unknown_model_id_is_loud():
    config = Config(roles={"worker": ["mystery-model"]})
    with pytest.raises(ValueError, match="unknown model id"):
        resolve_role(Role.WORKER, "opus", config)


def test_orchestrator_aliases():
    assert orchestrator_provider("claude") is Provider.ANTHROPIC
    assert orchestrator_provider("codex") is Provider.CODEX
    assert orchestrator_provider("grok") is Provider.XAI


@pytest.mark.asyncio
async def test_delegate_role_rejects_chat_even_if_explicit(role_config):
    with patch.object(server, "_load_config", return_value=role_config):
        response = await server._delegate_role_impl(
            "worker", "build", "opus", mode="chat", cwd="/tmp"
        )
    assert response["failure_kind"] == "validation_error"
    assert "agentic-only" in response["error"]


@pytest.mark.asyncio
async def test_native_assignment_never_calls_backend(role_config):
    with (
        patch("mcp_brain_router.router.backends.call_deepseek", new_callable=AsyncMock) as ds,
        patch("mcp_brain_router.router.backends.call_glm", new_callable=AsyncMock) as glm,
        patch("mcp_brain_router.router.backends.call_codex") as codex,
    ):
        assignment = resolve_role(
            Role.ADVERSARY,
            Provider.CODEX,
            role_config,
        )
        assert assignment.execute_natively is True
        ds.assert_not_called()
        glm.assert_not_called()
        codex.assert_not_called()


@pytest.mark.asyncio
async def test_delegate_role_runs_anthropic_cli_for_codex_adversary(role_config, tmp_path):
    success = RouteResult(
        content="refuted",
        model="claude-opus-4-8",
        backend="anthropic-cli",
        complexity=None,
        headroom_used=False,
    )
    with (
        patch.object(server, "_load_config", return_value=role_config),
        patch(
            "mcp_brain_router.server.route_assignment",
            new_callable=AsyncMock,
            return_value=success,
        ) as route_call,
    ):
        response = await server._delegate_role_impl(
            "adversary", "refute", "codex", cwd=str(tmp_path)
        )

    assert response["execute_natively"] is False
    assert response["backend"] == "anthropic-cli"
    assert response["provider"] == "anthropic"
    assert response["role"] == "adversary"
    route_call.assert_awaited_once()


@pytest.mark.asyncio
async def test_delegate_role_walks_quota_through_full_worker_cascade(role_config):
    exhausted = RouteResult(
        content="quota",
        model="",
        backend="none",
        complexity=None,
        headroom_used=False,
        exhausted=True,
    )
    success = RouteResult(
        content="built",
        model="sonnet-worker",
        backend="anthropic-cli",
        complexity=None,
        headroom_used=False,
    )
    with (
        patch.object(server, "_load_config", return_value=role_config),
        patch(
            "mcp_brain_router.server.route_assignment",
            new_callable=AsyncMock,
            side_effect=[exhausted, exhausted, exhausted, success],
        ) as route_call,
    ):
        response = await server._delegate_role_impl(
            "worker", "build", "opus", cwd="/tmp"
        )

    assert response["answer"] == "built"
    assert response["provider"] == "anthropic"
    assert response["tried"] == ["glm", "grok", "codex", "anthropic-cli"]
    assert route_call.await_count == 4
    assert [
        call.args[0].model for call in route_call.await_args_list
    ] == ["glm-5.2", "grok-4.5", "gpt-5.6-terra", "sonnet-worker"]


@pytest.mark.asyncio
async def test_delegate_role_does_not_advance_on_non_quota_error(role_config):
    with (
        patch.object(server, "_load_config", return_value=role_config),
        patch(
            "mcp_brain_router.server.route_assignment",
            new_callable=AsyncMock,
            side_effect=BackendError(
                "GLM process failed",
                backend="glm",
                failure_kind="process_error",
            ),
        ) as route_call,
    ):
        response = await server._delegate_role_impl(
            "worker", "build", "opus", cwd="/tmp"
        )

    assert response["failure_kind"] == "process_error"
    assert response["backend"] == "glm"
    assert route_call.await_count == 1


@pytest.mark.asyncio
async def test_anthropic_cli_quota_becomes_exhausted_result(role_config, tmp_path):
    assignment = resolve_role(
        Role.THINKER, "codex", role_config, mode="agentic"
    )
    with patch(
        "mcp_brain_router.router._route_agentic",
        new_callable=AsyncMock,
        side_effect=BackendQuotaError("Anthropic", 429, "usage limit reached"),
    ):
        from mcp_brain_router.router import route_assignment

        result = await route_assignment(
            assignment, "think", role_config, mode="agentic", cwd=str(tmp_path)
        )
    assert result.exhausted is True
    assert result.failure_kind == "quota_exhausted"
    assert result.tried == ["anthropic-cli"]


@pytest.mark.asyncio
async def test_all_role_candidates_exhausted_keeps_quota_contract(role_config):
    exhausted = RouteResult(
        content="quota",
        model="",
        backend="none",
        complexity=None,
        headroom_used=False,
        exhausted=True,
        reset_at="2026-07-15 09:00",
        failure_reason="quota",
    )
    with (
        patch.object(server, "_load_config", return_value=role_config),
        patch(
            "mcp_brain_router.server.route_assignment",
            new_callable=AsyncMock,
            side_effect=[exhausted, exhausted, exhausted, exhausted],
        ),
    ):
        response = await server._delegate_role_impl(
            "worker", "build", "opus", cwd="/tmp"
        )
    assert response["exhausted"] is True
    assert response["failure_kind"] == "quota_exhausted"
    assert response["tried"] == ["glm", "grok", "codex", "anthropic-cli"]
    assert response["reset_at"] == "2026-07-15 09:00"


@pytest.mark.asyncio
async def test_registered_caller_identity_cannot_be_spoofed(
    role_config, monkeypatch
):
    monkeypatch.setenv("BRAIN_ROUTER_CALLER", "codex")
    with patch.object(server, "_load_config", return_value=role_config):
        response = await server._delegate_role_impl(
            "adversary", "refute", "claude", cwd="/tmp"
        )
    assert response["failure_kind"] == "validation_error"
    assert "does not match registered caller identity" in response["error"]


@pytest.mark.asyncio
async def test_delegate_role_requires_orchestrator(role_config):
    with patch.object(server, "_load_config", return_value=role_config):
        response = await server._delegate_role_impl("worker", "build", "")
    assert response["failure_kind"] == "validation_error"
    assert "orchestrator is required" in response["error"]
