from unittest.mock import AsyncMock, patch

import pytest

from mcp_brain_router import server
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
            "thinker": ["sol-planner", "fable-planner"],
            "adversary": ["terra-refuter", "opus-refuter"],
            "worker": ["glm-5.2", "gpt-5.5", "sonnet-worker"],
            "simple": ["glm-4.7", "luna-simple", "haiku-simple"],
        },
    )


@pytest.mark.parametrize(
    ("role", "orchestrator", "provider", "native"),
    [
        (Role.THINKER, "opus", Provider.CODEX, False),
        (Role.THINKER, "codex", Provider.CODEX, False),
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


def test_exhaustion_walks_candidate_providers(role_config):
    assignment = resolve_role(
        Role.WORKER,
        Provider.ANTHROPIC,
        role_config,
        exhausted_providers={Provider.ZHIPU},
    )
    assert assignment.provider is Provider.CODEX
    assert assignment.model == "gpt-5.5"


def test_anthropic_fallback_is_native(role_config):
    assignment = resolve_role(
        Role.WORKER,
        Provider.CODEX,
        role_config,
        exhausted_providers={Provider.ZHIPU, Provider.CODEX},
    )
    assert assignment.provider is Provider.ANTHROPIC
    assert assignment.backend is None
    assert assignment.execute_natively is True


def test_orchestrator_role_is_never_resolved(role_config):
    with pytest.raises(ValueError, match="selected by the human"):
        resolve_role(Role.ORCHESTRATOR, "opus", role_config)


def test_same_provider_only_adversary_rejected():
    config = Config(roles={"adversary": ["gpt-5.5", "terra-refuter"]})
    with pytest.raises(ValueError, match="differs"):
        resolve_role(Role.ADVERSARY, "codex", config)


def test_unknown_model_id_is_loud():
    config = Config(roles={"worker": ["mystery-model"]})
    with pytest.raises(ValueError, match="unknown model id"):
        resolve_role(Role.WORKER, "opus", config)


def test_orchestrator_aliases():
    assert orchestrator_provider("claude") is Provider.ANTHROPIC
    assert orchestrator_provider("codex") is Provider.CODEX


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
async def test_delegate_role_returns_native_stub_without_route(role_config):
    with (
        patch.object(server, "_load_config", return_value=role_config),
        patch("mcp_brain_router.server.route_assignment", new_callable=AsyncMock) as route_call,
    ):
        response = await server._delegate_role_impl("adversary", "refute", "codex")

    assert response["execute_natively"] is True
    assert response["provider"] == "anthropic"
    assert response["role"] == "adversary"
    route_call.assert_not_called()


@pytest.mark.asyncio
async def test_delegate_role_walks_quota_then_codex(role_config):
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
        model="gpt-5.5",
        backend="codex",
        complexity=None,
        headroom_used=False,
    )
    with (
        patch.object(server, "_load_config", return_value=role_config),
        patch(
            "mcp_brain_router.server.route_assignment",
            new_callable=AsyncMock,
            side_effect=[exhausted, success],
        ) as route_call,
    ):
        response = await server._delegate_role_impl("worker", "build", "opus")

    assert response["answer"] == "built"
    assert response["provider"] == "codex"
    assert response["tried"] == ["glm", "codex"]
    assert route_call.await_count == 2


@pytest.mark.asyncio
async def test_delegate_role_requires_orchestrator(role_config):
    with patch.object(server, "_load_config", return_value=role_config):
        response = await server._delegate_role_impl("worker", "build", "")
    assert response["failure_kind"] == "validation_error"
    assert "orchestrator is required" in response["error"]
