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
from mcp_brain_router import backends


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


# ============================================================================
# Model Validation & Security Tests
# ============================================================================

class TestModelValidation:
    """Test model name validation against argument injection."""

    def test_valid_model_names(self):
        """Test that valid model names pass validation."""
        valid_models = [
            "gpt-5.5",
            "deepseek-v4-flash",
            "glm-5.2",
            "gpt_5_5",
            "model.v1",
        ]
        for model in valid_models:
            backends._validate_model_name(model)  # Should not raise

    def test_invalid_model_names_raise_error(self):
        """Test that models with shell metacharacters are rejected."""
        invalid_models = [
            "gpt-5.5; rm -rf",
            "gpt-5.5 || whoami",
            "gpt-5.5`whoami`",
            "gpt-5.5$(whoami)",
            "gpt-5.5 --flag",
            "gpt-5.5@special",
            "gpt-5.5&whoami",
        ]
        for model in invalid_models:
            with pytest.raises(backends.BackendError) as exc_info:
                backends._validate_model_name(model)
            assert "Invalid model name" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_codex_model_injection_rejected(self):
        """Test that Codex rejects injected model args."""
        config = Config(
            deepseek_key="test_ds",
            glm_key="test_glm",
            codex_enabled=True,
        )

        # Try to inject a flag via model name
        with pytest.raises(backends.BackendError) as exc_info:
            backends.call_codex("test prompt", "gpt-5.5; rm -rf /")

        assert "Invalid model name" in str(exc_info.value)


class TestHttpxUsage:
    """Test that backends use httpx, not aiohttp."""

    def test_httpx_imported_in_backends(self):
        """Verify httpx is imported at module level in backends."""
        import sys
        # backends module must import httpx successfully
        import mcp_brain_router.backends
        assert "httpx" in dir(mcp_brain_router.backends)

    def test_no_aiohttp_in_backends(self):
        """Verify backends.py does NOT use aiohttp."""
        with open("/Users/mohannarayanswamy/code workshop/claude projects/Personal/mcp-brain-router/src/mcp_brain_router/backends.py") as f:
            content = f.read()
            # Should not import aiohttp anywhere
            assert "import aiohttp" not in content
            assert "from aiohttp" not in content
            # Should import httpx
            assert "import httpx" in content or "httpx.AsyncClient" in content


class TestHeadroomUrlSubstitution:
    """Test that headroom URL substitution works correctly."""

    def test_deepseek_via_headroom_url_construction(self):
        """Verify DeepSeek via headroom constructs the correct URL."""
        # This is a code inspection test: verify the function builds the URL correctly
        import inspect
        source = inspect.getsource(backends.call_deepseek_via_headroom)
        # Should construct URL as {headroom_url}/anthropic/v1/messages
        assert "f\"{headroom_url}/anthropic/v1/messages\"" in source or \
               "f'{headroom_url}/anthropic/v1/messages'" in source

    def test_glm_via_headroom_url_construction(self):
        """Verify GLM via headroom constructs the correct URL."""
        import inspect
        source = inspect.getsource(backends.call_glm_via_headroom)
        # Should construct URL as {headroom_url}/anthropic/v1/messages
        assert "f\"{headroom_url}/anthropic/v1/messages\"" in source or \
               "f'{headroom_url}/anthropic/v1/messages'" in source


class TestFallbackChain:
    """Quota-fallback chain: on 429/5xx a tier falls through to the next
    non-Anthropic backend; all-exhausted returns a structured signal."""

    def _cfg(self):
        return Config(deepseek_key="sk-ds", glm_key="glm-k", codex_enabled=True)

    @pytest.mark.asyncio
    async def test_code_glm_429_falls_to_deepseek(self):
        """CODE primary GLM hits 429 -> chain falls to DeepSeek, returns it."""
        with patch("mcp_brain_router.router.backends.call_glm", new_callable=AsyncMock) as mglm, \
             patch("mcp_brain_router.router.backends.call_deepseek", new_callable=AsyncMock) as mds:
            mglm.side_effect = backends.BackendQuotaError("GLM", 429, "limit reached")
            mds.return_value = {"content": "PONG", "usage": {"input_tokens": 1, "output_tokens": 1}}

            result = await route(Complexity.CODE, "p", config=self._cfg())

            assert result.backend == "deepseek"
            assert result.content == "PONG"
            assert result.exhausted is False
            assert result.tried == ["glm", "deepseek"]
            assert mglm.called and mds.called

    @pytest.mark.asyncio
    async def test_all_backends_exhausted_returns_signal(self):
        """Both backends 429 -> exhausted=True signal (NOT an exception),
        carrying reset_at so the orchestrator can decide to handle natively."""
        with patch("mcp_brain_router.router.backends.call_glm", new_callable=AsyncMock) as mglm, \
             patch("mcp_brain_router.router.backends.call_deepseek", new_callable=AsyncMock) as mds:
            mglm.side_effect = backends.BackendQuotaError("GLM", 429, "reset at 2026-06-29 15:56:33", reset_at="2026-06-29 15:56:33")
            mds.side_effect = backends.BackendQuotaError("DeepSeek", 429, "rate limited")

            result = await route(Complexity.CODE, "p", config=self._cfg())

            assert result.exhausted is True
            assert result.backend == "none"
            assert result.tried == ["glm", "deepseek"]
            assert result.reset_at == "2026-06-29 15:56:33"
            # The content is a human/agent-readable "handle natively" hint.
            assert "natively" in result.content.lower()

    @pytest.mark.asyncio
    async def test_hard_error_does_not_silently_fallback(self):
        """A non-retryable BackendError (e.g. auth/4xx) must propagate, NOT
        get masked by a fallback — config errors should be loud."""
        with patch("mcp_brain_router.router.backends.call_glm", new_callable=AsyncMock) as mglm, \
             patch("mcp_brain_router.router.backends.call_deepseek", new_callable=AsyncMock) as mds:
            mglm.side_effect = backends.BackendError("GLM API error 401: bad key")
            mds.return_value = {"content": "should-not-be-used", "usage": {}}

            with pytest.raises(backends.BackendError):
                await route(Complexity.CODE, "p", config=self._cfg())
            assert not mds.called  # never fell through on a hard error

    @pytest.mark.asyncio
    async def test_codex_timeout_returns_exhausted_not_raw_error(self):
        """ADVERSARIAL primary (codex) subprocess timeout is TRANSIENT — the
        single-backend chain ends with exhausted=True ("handle natively"),
        never a raw error dict / exception leaking to the orchestrator.
        Regression guard for the 2026-07-02 live failure (codex MCP-boot
        stall -> 60s timeout -> {"error": ...} with backend "unknown")."""
        with patch("mcp_brain_router.router.backends.call_codex") as mcx:
            mcx.side_effect = backends.BackendTransientError(
                "Codex subprocess timed out (90s)"
            )

            result = await route(Complexity.ADVERSARIAL, "p", config=self._cfg())

            assert result.exhausted is True
            assert result.backend == "none"
            assert result.tried == ["codex"]
            assert "natively" in result.content.lower()

    @pytest.mark.asyncio
    async def test_cheap_chain_is_deepseek_then_glm(self):
        """CHEAP primary DeepSeek 429 -> falls to GLM."""
        with patch("mcp_brain_router.router.backends.call_deepseek", new_callable=AsyncMock) as mds, \
             patch("mcp_brain_router.router.backends.call_glm", new_callable=AsyncMock) as mglm:
            mds.side_effect = backends.BackendQuotaError("DeepSeek", 429, "limit")
            mglm.return_value = {"content": "PONG", "usage": {}}

            result = await route(Complexity.CHEAP, "p", config=self._cfg())

            assert result.backend == "glm"
            assert result.tried == ["deepseek", "glm"]

    @pytest.mark.asyncio
    async def test_adversarial_single_entry_chain(self):
        """ADVERSARIAL chain is codex-only; success returns codex."""
        with patch("mcp_brain_router.router.backends.call_codex") as mcx:
            mcx.return_value = {"content": "PONG", "usage": {}}
            result = await route(Complexity.ADVERSARIAL, "p", config=self._cfg())
            assert result.backend == "codex"
            assert result.tried == ["codex"]
            assert result.exhausted is False

    @pytest.mark.asyncio
    async def test_missing_fallback_credential_is_skipped_not_fatal(self):
        """If a FALLBACK backend lacks credentials, it is skipped (unavailable),
        and if the primary was exhausted with no usable fallback -> signal."""
        cfg = Config(deepseek_key=None, glm_key="glm-k", codex_enabled=True)
        with patch("mcp_brain_router.router.backends.call_glm", new_callable=AsyncMock) as mglm:
            mglm.side_effect = backends.BackendQuotaError("GLM", 429, "limit")
            # CODE chain = [glm, deepseek]; deepseek has no key -> skipped.
            result = await route(Complexity.CODE, "p", config=cfg)
            assert result.exhausted is True
            assert result.tried == ["glm"]  # deepseek skipped (no key)


class TestErrorClassification:
    """_classify_http_error maps statuses to the right exception type."""

    def test_429_is_quota_error(self):
        err = backends._classify_http_error(
            "GLM", 429,
            '{"error":{"message":"Usage limit reached for 5 hour. reset at 2026-06-29 15:56:33"}}',
        )
        assert isinstance(err, backends.BackendQuotaError)
        assert err.status_code == 429
        assert err.reset_at == "2026-06-29 15:56:33"

    def test_5xx_is_quota_error(self):
        err = backends._classify_http_error("DeepSeek", 503, "{}")
        assert isinstance(err, backends.BackendQuotaError)

    def test_4xx_auth_is_hard_error(self):
        err = backends._classify_http_error(
            "GLM", 401, '{"error":{"message":"invalid api key"}}',
        )
        assert isinstance(err, backends.BackendError)
        assert not isinstance(err, backends.BackendQuotaError)

    def test_hard_error_does_not_leak_raw_body(self):
        err = backends._classify_http_error("GLM", 400, "raw-secret-echo-body")
        # Unparseable body -> withheld, not echoed.
        assert "raw-secret-echo-body" not in str(err)


class TestCodexArgvSafety:
    """G-guards from the 2026-07-02 adversarial pass (PR #1 review)."""

    def test_codex_argv_has_end_of_options_separator(self):
        """A prompt starting with "-" must reach codex as the PROMPT, not be
        parsed as a CLI flag (prompt="-h" printed codex help pre-fix). The
        "--" separator must sit between options and the prompt."""
        with patch("mcp_brain_router.backends.subprocess.run") as mrun:
            mrun.return_value = MagicMock(returncode=0, stdout="ok")
            backends.call_codex("-h --evil-flag", "gpt-5.5")
            argv = mrun.call_args[0][0]
        sep = argv.index("--")
        assert argv[sep + 1] == "-h --evil-flag"  # prompt is positional, after --
        assert argv[-1] == "-h --evil-flag"

    def test_codex_argv_skips_mcp_boot(self):
        """mcp_servers={} must be present — codex otherwise boots every MCP
        server in ~/.codex/config.toml and blows the subprocess timeout."""
        with patch("mcp_brain_router.backends.subprocess.run") as mrun:
            mrun.return_value = MagicMock(returncode=0, stdout="ok")
            backends.call_codex("p", "gpt-5.5")
            argv = mrun.call_args[0][0]
        assert "-c" in argv and "mcp_servers={}" in argv
        assert "--skip-git-repo-check" in argv

    def test_install_smoke_test_uses_same_codex_base(self):
        """install.py's codex smoke test must share CODEX_EXEC_BASE so the
        two invocations can never drift (found drifted in adversarial pass)."""
        import inspect
        from mcp_brain_router import install
        src = inspect.getsource(install._test_backend_codex)
        assert "CODEX_EXEC_BASE" in src

    def test_error_family_is_unified(self):
        """router's credential/availability errors must subclass
        backends.BackendError, so server.py's `except BackendError` (imported
        from backends) catches the WHOLE family — pre-fix these were two
        unrelated classes and codex errors fell into the generic handler."""
        import mcp_brain_router.router as router_mod
        assert issubclass(router_mod.BackendError, backends.BackendError)
        assert issubclass(router_mod.MissingCredentialError, backends.BackendError)
        assert issubclass(backends.BackendTransientError, backends.BackendError)
