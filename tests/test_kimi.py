"""Tests for the Kimi provider (Kimi Code CLI, OAuth login, enable flag).

Kimi stays in D5's worker and adversary orders, but non-Claude workers run
tools-off (lean-rig §6) and Kimi's `-p` mode auto-approves its own tools with
no verified way to turn them off. So the router refuses it before any spawn:
a free skip, and the role walk moves on. No live CLI, no network.
"""

from unittest.mock import patch

import pytest

from mcp_brain_router import backends
from mcp_brain_router.backends import BackendError
from mcp_brain_router.config import Config
from mcp_brain_router.router import (
    BackendUnavailableError,
    Complexity,
    Provider,
    Role,
    _get_backend_default_model,
    provider_for_model,
    resolve_role,
    route_assignment,
)


def _kimi_config(**overrides) -> Config:
    base = dict(
        codex_enabled=True,
        grok_enabled=True,
        kimi_enabled=True,
        roles={"worker": ["kimi", "grok-4.5", "claude-sonnet-5"]},
    )
    base.update(overrides)
    return Config(**base)


class TestKimiProviderMapping:
    def test_kimi_prefix_maps_to_kimi_provider(self):
        assert provider_for_model("kimi") is Provider.KIMI
        assert provider_for_model("Kimi") is Provider.KIMI  # case-insensitive

    def test_kimi_default_model_is_just_kimi(self):
        assert _get_backend_default_model("kimi", Config()) == "kimi"

    def test_kimi_is_coding_provider_not_complexity_tier(self):
        with pytest.raises(ValueError):
            Complexity("kimi")
        assignment = resolve_role(Role.WORKER, "opus", _kimi_config())
        assert assignment.provider is Provider.KIMI
        assert assignment.backend == "kimi"
        assert assignment.execute_natively is False


class TestKimiRefusedToolsOff:
    @pytest.mark.asyncio
    async def test_kimi_disabled_raises_unavailable(self):
        assignment = resolve_role(Role.WORKER, "opus", _kimi_config())
        with pytest.raises(BackendUnavailableError, match="Kimi not enabled"):
            await route_assignment(
                assignment, "p", _kimi_config(kimi_enabled=False), cwd="/tmp"
            )

    def test_call_kimi_refuses_before_any_spawn(self):
        with patch("mcp_brain_router.backends.subprocess.run") as mrun:
            with pytest.raises(BackendError) as exc:
                backends.call_kimi("p", "kimi")
        mrun.assert_not_called()
        assert exc.value.failure_kind == "tools_unsupported"
        assert exc.value.backend == "kimi"

    @pytest.mark.asyncio
    async def test_enabled_kimi_is_still_refused_so_the_walk_advances(self):
        assignment = resolve_role(Role.WORKER, "opus", _kimi_config())
        with patch("mcp_brain_router.backends.subprocess.run") as mrun:
            with pytest.raises(BackendError) as exc:
                await route_assignment(
                    assignment, "p", _kimi_config(), mode="agentic", cwd="/tmp"
                )
        mrun.assert_not_called()
        assert exc.value.failure_kind == "tools_unsupported"


class TestKimiConfigFlag:
    def test_kimi_enabled_round_trips(self, tmp_path, monkeypatch):
        cfg_file = tmp_path / "config.toml"
        monkeypatch.setattr("mcp_brain_router.config.CONFIG_FILE", cfg_file)
        Config(deepseek_key="k", kimi_enabled=True).save()
        assert Config.load().kimi_enabled is True

    def test_kimi_enabled_defaults_false(self, tmp_path, monkeypatch):
        cfg_file = tmp_path / "config.toml"
        cfg_file.write_text('deepseek_key = "k"\n')
        cfg_file.chmod(0o600)
        monkeypatch.setattr("mcp_brain_router.config.CONFIG_FILE", cfg_file)
        assert Config.load().kimi_enabled is False
