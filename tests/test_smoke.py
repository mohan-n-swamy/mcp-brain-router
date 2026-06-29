"""
Smoke tests for mcp-brain-router.

Tests:
- route() function with complexity enum
- Config persistence with correct file permissions (0600)
- Missing key error handling
- Codex subprocess args passed as list (no shell injection)
- Router exceptions (MissingCredentialError, BackendUnavailableError)
"""

import os
import stat
import tempfile
import json
from pathlib import Path
from unittest.mock import patch, AsyncMock, MagicMock
import pytest

from mcp_brain_router.router import (
    route,
    Complexity,
    RouteResult,
    BackendError,
    MissingCredentialError,
    BackendUnavailableError,
)
from mcp_brain_router.config import Config, ConfigError, ensure_config_dir


class TestRouteFunction:
    """Test the route() async function."""

    @pytest.mark.asyncio
    async def test_route_cheap_uses_deepseek(self):
        """Test that complexity=CHEAP routes to DeepSeek."""
        config = Config(
            deepseek_key="test_key",
            glm_key="test_glm",
            codex_enabled=False,
        )

        with patch("mcp_brain_router.router.backends.call_deepseek", new_callable=AsyncMock) as mock_ds:
            mock_ds.return_value = {
                "content": "response",
                "usage": {"input_tokens": 10, "output_tokens": 20},
            }

            result = await route(Complexity.CHEAP, "test prompt", config=config)

            assert result.backend == "deepseek"
            assert result.content == "response"
            assert mock_ds.called

    @pytest.mark.asyncio
    async def test_route_code_uses_glm(self):
        """Test that complexity=CODE routes to GLM."""
        config = Config(
            deepseek_key="test_ds",
            glm_key="test_glm",
            codex_enabled=False,
        )

        with patch("mcp_brain_router.router.backends.call_glm", new_callable=AsyncMock) as mock_glm:
            mock_glm.return_value = {
                "content": "glm_response",
                "usage": {"input_tokens": 5, "output_tokens": 15},
            }

            result = await route(Complexity.CODE, "test prompt", config=config)

            assert result.backend == "glm"
            assert result.content == "glm_response"
            assert mock_glm.called

    @pytest.mark.asyncio
    async def test_route_adversarial_uses_codex(self):
        """Test that complexity=ADVERSARIAL routes to Codex."""
        config = Config(
            deepseek_key="test_ds",
            glm_key="test_glm",
            codex_enabled=True,
        )

        with patch("mcp_brain_router.router.backends.call_codex") as mock_codex:
            mock_codex.return_value = {
                "content": "codex_response",
                "usage": None,
            }

            result = await route(Complexity.ADVERSARIAL, "test prompt", config=config)

            assert result.backend == "codex"
            assert result.content == "codex_response"
            assert mock_codex.called


class TestConfigPersistence:
    """Test config save/load with correct file permissions."""

    @pytest.fixture
    def temp_config_dir(self):
        """Create a temporary config directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    def test_config_save_creates_file_with_0600_perms(self, temp_config_dir, monkeypatch):
        """Test that config file is saved with 0600 (rw-------) permissions."""
        config_file = temp_config_dir / "config.toml"

        # Patch CONFIG_FILE to use temp location
        monkeypatch.setattr("mcp_brain_router.config.CONFIG_FILE", config_file)

        config = Config(
            deepseek_key="test_ds",
            glm_key="test_glm",
            codex_enabled=False,
        )

        config.save()

        assert config_file.exists(), f"Config file not created at {config_file}"

        # Check file permissions
        file_stat = config_file.stat()
        file_perms = stat.S_IMODE(file_stat.st_mode)

        assert file_perms == 0o600, f"Config file permissions are {oct(file_perms)}, expected 0o600"

    def test_config_load_roundtrip(self, temp_config_dir, monkeypatch):
        """Test that config can be saved and loaded with same values."""
        config_file = temp_config_dir / "config.toml"
        monkeypatch.setattr("mcp_brain_router.config.CONFIG_FILE", config_file)

        # Save
        original = Config(
            deepseek_key="test_ds_key",
            glm_key="test_glm_key",
            codex_enabled=True,
            headroom_base_url="http://localhost:8000",
        )
        original.save()

        # Load
        loaded = Config.load()

        assert loaded.deepseek_key == original.deepseek_key
        assert loaded.glm_key == original.glm_key
        assert loaded.codex_enabled == original.codex_enabled
        assert loaded.headroom_base_url == original.headroom_base_url


class TestMissingKeyErrors:
    """Test that missing keys raise appropriate errors."""

    @pytest.mark.asyncio
    async def test_missing_deepseek_key_raises_error(self):
        """Test that missing DeepSeek key raises MissingCredentialError."""
        config = Config(
            deepseek_key=None,
            glm_key="test_glm",
            codex_enabled=False,
        )

        with pytest.raises(MissingCredentialError) as exc_info:
            await route(Complexity.CHEAP, "test prompt", config=config)

        assert "deepseek" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_missing_glm_key_raises_error(self):
        """Test that missing GLM key raises MissingCredentialError."""
        config = Config(
            deepseek_key="test_ds",
            glm_key=None,
            codex_enabled=False,
        )

        with pytest.raises(MissingCredentialError) as exc_info:
            await route(Complexity.CODE, "test prompt", config=config)

        assert "glm" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_codex_disabled_raises_error(self):
        """Test that adversarial tier with codex disabled raises error."""
        config = Config(
            deepseek_key="test_ds",
            glm_key="test_glm",
            codex_enabled=False,
        )

        with pytest.raises(BackendUnavailableError) as exc_info:
            await route(Complexity.ADVERSARIAL, "test prompt", config=config)

        assert "codex" in str(exc_info.value).lower()


class TestConfigError:
    """Test ConfigError exception."""

    def test_config_not_found_raises_config_error(self, monkeypatch):
        """Test that missing config file raises ConfigError."""
        monkeypatch.setattr(
            "mcp_brain_router.config.CONFIG_FILE",
            Path("/nonexistent/config.toml")
        )

        with pytest.raises(ConfigError):
            Config.load()

    def test_config_insecure_perms_raises_config_error(self, temp_config_dir, monkeypatch):
        """Test that insecure file permissions raise ConfigError."""
        config_file = temp_config_dir / "config.toml"
        config_file.write_text("deepseek_key = 'test'")
        config_file.chmod(0o644)  # world-readable, not 0600

        monkeypatch.setattr("mcp_brain_router.config.CONFIG_FILE", config_file)

        with pytest.raises(ConfigError) as exc_info:
            Config.load()

        assert "permission" in str(exc_info.value).lower()

    @pytest.fixture
    def temp_config_dir(self):
        """Create a temporary config directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)


class TestCodexSubprocessArgs:
    """Test that Codex subprocess args are passed as list (no shell injection)."""

    @pytest.mark.asyncio
    async def test_codex_args_passed_as_list(self):
        """Test that codex args are a list, not a shell string."""
        config = Config(
            deepseek_key="test_ds",
            glm_key="test_glm",
            codex_enabled=True,
        )

        with patch("mcp_brain_router.backends.subprocess.run") as mock_run:
            mock_result = MagicMock()
            mock_result.returncode = 0
            mock_result.stdout = "output"
            mock_result.stderr = ""
            mock_run.return_value = mock_result

            result = await route(Complexity.ADVERSARIAL, "test prompt", config=config)

            # Verify subprocess.run was called with args as a list
            assert mock_run.called
            call_args = mock_run.call_args
            args_list = call_args[0][0]  # First positional arg to subprocess.run

            # Args should be a list, not a string
            assert isinstance(args_list, (list, tuple))
            assert args_list[0] == "codex"

    @pytest.mark.asyncio
    async def test_codex_shell_injection_protection(self):
        """Test that malicious prompts can't break shell."""
        config = Config(
            deepseek_key="test_ds",
            glm_key="test_glm",
            codex_enabled=True,
        )

        malicious_prompt = "'; rm -rf /; echo '"

        with patch("mcp_brain_router.backends.subprocess.run") as mock_run:
            mock_result = MagicMock()
            mock_result.returncode = 0
            mock_result.stdout = "safe"
            mock_result.stderr = ""
            mock_run.return_value = mock_result

            result = await route(Complexity.ADVERSARIAL, malicious_prompt, config=config)

            # Verify args are passed as list (preventing shell interpretation)
            call_args = mock_run.call_args
            args_list = call_args[0][0]

            # The malicious string should be a single arg, not split
            assert isinstance(args_list, (list, tuple))
            # Args list should be passed with shell=False (default), not shell=True
            call_kwargs = call_args[1]
            assert call_kwargs.get("shell", False) is False


# ============================================================================
# Integration smoke tests
# ============================================================================

class TestIntegrationSmoke:
    """Light integration tests without real API calls."""

    @pytest.fixture
    def temp_config_dir(self):
        """Create a temporary config directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    def test_ensure_config_dir_creates_dir(self, temp_config_dir, monkeypatch):
        """Test that ensure_config_dir creates the config directory."""
        monkeypatch.setattr("mcp_brain_router.config.CONFIG_DIR", temp_config_dir)

        result = ensure_config_dir()

        assert result == temp_config_dir
        assert temp_config_dir.exists()

    @pytest.mark.asyncio
    async def test_route_result_has_all_fields(self):
        """Test that RouteResult contains all expected fields."""
        config = Config(
            deepseek_key="test_key",
            glm_key="test_glm",
            codex_enabled=False,
        )

        with patch("mcp_brain_router.router.backends.call_deepseek", new_callable=AsyncMock) as mock_ds:
            mock_ds.return_value = {
                "content": "test response",
                "usage": {"input_tokens": 10, "output_tokens": 20},
            }

            result = await route(Complexity.CHEAP, "test prompt", config=config)

            assert hasattr(result, "content")
            assert hasattr(result, "model")
            assert hasattr(result, "backend")
            assert hasattr(result, "complexity")
            assert hasattr(result, "headroom_used")
            assert hasattr(result, "usage")
