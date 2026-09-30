"""ROUTER: GLM-only tiers walk a role list when no GLM key is configured."""
from unittest.mock import AsyncMock, patch

import pytest

from mcp_brain_router import server
from mcp_brain_router.config import Config
from mcp_brain_router.router import Complexity, RouteResult


@pytest.mark.asyncio
@pytest.mark.parametrize("tier,role", [("cheap", "simple"), ("code", "worker")])
async def test_tier_without_glm_key_walks_role_list(tier, role):
    role_call = AsyncMock(return_value={"answer": "ok", "backend": "anthropic-cli", "role": role})
    with (
        patch.object(server, "_load_config", return_value=Config(codex_enabled=True)),
        patch.object(server, "_read_caller", return_value="claude"),
        patch.object(server, "route", new_callable=AsyncMock) as route_call,
        patch.object(server, "_delegate_role_impl", role_call),
        patch.object(server, "_log_delegation"),
    ):
        response = await server._delegate_impl(tier, "p", cwd="/tmp")
    route_call.assert_not_awaited()
    role_call.assert_awaited_once_with(role, "p", "claude", mode="agentic", cwd="/tmp")
    assert response["answer"] == "ok"


@pytest.mark.asyncio
async def test_tier_with_glm_key_keeps_glm_route():
    result = RouteResult(content="done", model="glm-5.3", backend="glm",
                                complexity=Complexity.CODE, headroom_used=False, tried=["glm"])
    with (
        patch.object(server, "_load_config", return_value=Config(glm_key="glm-k")),
        patch.object(server, "route", new_callable=AsyncMock, return_value=result) as route_call,
        patch.object(server, "_delegate_role_impl", new_callable=AsyncMock) as role_call,
        patch.object(server, "_log_delegation"),
    ):
        response = await server._delegate_impl("code", "p", cwd="/tmp")
    route_call.assert_awaited_once()
    role_call.assert_not_awaited()
    assert response["backend"] == "glm"


@pytest.mark.asyncio
async def test_adversarial_tier_is_untouched():
    with (
        patch.object(server, "_load_config", return_value=Config(codex_enabled=True)),
        patch.object(server, "_read_caller", return_value="claude"),
        patch.object(server, "route", new_callable=AsyncMock) as route_call,
        patch.object(server, "_delegate_role_impl", new_callable=AsyncMock) as role_call,
        patch.object(server, "_log_delegation"),
    ):
        await server._delegate_impl("adversarial", "p", cwd="/tmp")
    route_call.assert_awaited_once()
    role_call.assert_not_awaited()
