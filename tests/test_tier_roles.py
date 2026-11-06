"""Legacy tiers: cheap and code walk a role list (GLM left the rig, D5)."""
from unittest.mock import AsyncMock, patch

import pytest

from mcp_brain_router import server
from mcp_brain_router.config import Config


@pytest.mark.asyncio
@pytest.mark.parametrize("tier,role", [("cheap", "simple"), ("code", "worker")])
async def test_tier_walks_role_list(tier, role):
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
    role_call.assert_awaited_once_with(role, "p", "claude", mode="agentic", cwd="/tmp",
                                       data_class="personal")
    assert response["answer"] == "ok"


@pytest.mark.asyncio
async def test_adversarial_tier_keeps_codex_route():
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


@pytest.mark.asyncio
async def test_orange_adversarial_tier_walks_the_adversary_role():
    role_call = AsyncMock(return_value={"error": "refused", "backend": "none"})
    with (
        patch.object(server, "_load_config", return_value=Config(codex_enabled=True)),
        patch.object(server, "_read_caller", return_value="claude"),
        patch.object(server, "route", new_callable=AsyncMock) as route_call,
        patch.object(server, "_delegate_role_impl", role_call),
        patch.object(server, "_log_delegation"),
    ):
        await server._delegate_impl("adversarial", "p", cwd="/tmp", data_class="orange")
    route_call.assert_not_awaited()
    role_call.assert_awaited_once_with("adversary", "p", "claude", mode="agentic", cwd="/tmp",
                                       data_class="orange")


@pytest.mark.asyncio
async def test_unknown_caller_cannot_walk_a_role():
    with (
        patch.object(server, "_load_config", return_value=Config(codex_enabled=True)),
        patch.object(server, "_read_caller", return_value="unknown"),
        patch.object(server, "route", new_callable=AsyncMock) as route_call,
        patch.object(server, "_delegate_role_impl", new_callable=AsyncMock) as role_call,
        patch.object(server, "_log_delegation"),
    ):
        response = await server._delegate_impl("cheap", "p", cwd="/tmp")
    role_call.assert_not_awaited()
    route_call.assert_not_awaited()
    assert response["failure_kind"] == "validation_error"
