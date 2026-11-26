"""Legacy tiers walk a role list: cheap=simple, code=worker, adversarial=adversary
(D16 took Codex out of every role; only an explicit model name reaches it)."""
from unittest.mock import AsyncMock, patch

import pytest

from mcp_brain_router import server
from mcp_brain_router.config import Config


@pytest.mark.asyncio
@pytest.mark.parametrize("tier,role", [("cheap", "simple"), ("code", "worker"),
                                       ("adversarial", "adversary")])
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
                                       data_class="personal", tools="none")
    assert response["answer"] == "ok"


@pytest.mark.asyncio
async def test_adversarial_tier_with_explicit_model_reaches_codex():
    with (
        patch.object(server, "_load_config", return_value=Config(codex_enabled=True)),
        patch.object(server, "_read_caller", return_value="claude"),
        patch.object(server, "route", new_callable=AsyncMock) as route_call,
        patch.object(server, "_delegate_role_impl", new_callable=AsyncMock) as role_call,
        patch.object(server, "_log_delegation"),
    ):
        await server._delegate_impl("adversarial", "p", model="gpt-6.1-sol", cwd="/tmp")
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
                                       data_class="orange", tools="none")


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
