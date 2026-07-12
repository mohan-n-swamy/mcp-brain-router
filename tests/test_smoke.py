"""
Smoke tests for mcp-brain-router.

Tests:
- route() function with complexity enum
- Config persistence with correct file permissions (0600)
- Missing key error handling
- Codex subprocess args passed as list (no shell injection)
- Router exceptions (MissingCredentialError, BackendUnavailableError)
"""

import json
import stat
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mcp_brain_router import backends, server
from mcp_brain_router.config import Config, ConfigError, ensure_config_dir
from mcp_brain_router.router import (
    BackendUnavailableError,
    Complexity,
    MissingCredentialError,
    RouteResult,
    route,
)


class TestRouteFunction:
    """Test the route() async function."""

    @pytest.mark.asyncio
    async def test_route_cheap_uses_glm(self):
        """002: CHEAP tier now routes to GLM (DeepSeek removed from routing)."""
        config = Config(
            deepseek_key="test_key",
            glm_key="test_glm",
            codex_enabled=False,
        )

        with patch(
            "mcp_brain_router.router.backends.call_glm", new_callable=AsyncMock
        ) as mock_glm:
            mock_glm.return_value = {
                "content": "response",
                "usage": {"input_tokens": 10, "output_tokens": 20},
            }

            result = await route(Complexity.CHEAP, "test prompt", config=config)

            assert result.backend == "glm"
            assert result.content == "response"
            assert mock_glm.called

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
            roles={
                "worker": ["glm-5.2", "gpt-5.6-sol", "sonnet-worker"],
                "simple": ["glm-4.7", "luna-simple", "haiku-simple"],
            },
        )
        original.save()

        # Load
        loaded = Config.load()

        assert loaded.deepseek_key == original.deepseek_key
        assert loaded.glm_key == original.glm_key
        assert loaded.codex_enabled == original.codex_enabled
        assert loaded.headroom_base_url == original.headroom_base_url
        assert loaded.roles == original.roles

    def test_config_without_roles_loads_empty(self, temp_config_dir, monkeypatch):
        config_file = temp_config_dir / "config.toml"
        config_file.write_text('glm_key = "test"\n')
        config_file.chmod(0o600)
        monkeypatch.setattr("mcp_brain_router.config.CONFIG_FILE", config_file)

        assert Config.load().roles == {}


class TestMissingKeyErrors:
    """Test that missing keys raise appropriate errors."""

    @pytest.mark.asyncio
    async def test_missing_glm_key_raises_error_for_cheap(self):
        """002: CHEAP tier now routes to GLM, so a missing GLM key is the error."""
        config = Config(
            deepseek_key=None,
            glm_key=None,
            codex_enabled=False,
        )

        with pytest.raises(MissingCredentialError) as exc_info:
            await route(Complexity.CHEAP, "test prompt", config=config)

        assert "glm" in str(exc_info.value).lower()

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
        monkeypatch.setattr("mcp_brain_router.config.CONFIG_FILE", Path("/nonexistent/config.toml"))

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

            await route(Complexity.ADVERSARIAL, "test prompt", config=config)

            # Verify subprocess.run was called with args as a list
            assert mock_run.called
            call_args = mock_run.call_args
            args_list = call_args[0][0]  # First positional arg to subprocess.run

            # Args should be a list, not a string
            assert isinstance(args_list, (list, tuple))
            # First arg is the codex binary — now resolved to an ABSOLUTE path
            # (2026-07-05 PATH fix: MCP process has empty env, bare "codex"
            # wasn't found). Accept the resolved path or the bare fallback.
            assert args_list[0] == "codex" or args_list[0].endswith("/codex")
            # subprocess.run must receive an env carrying an augmented PATH so
            # codex + its `env node` shebang both resolve under the empty MCP env
            passed_env = mock_run.call_args.kwargs.get("env")
            assert passed_env is not None and "PATH" in passed_env

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

            await route(Complexity.ADVERSARIAL, malicious_prompt, config=config)

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

        with patch(
            "mcp_brain_router.router.backends.call_glm", new_callable=AsyncMock
        ) as mock_glm:
            mock_glm.return_value = {
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
            "gpt-5.6-sol",
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
            "gpt-5.6-sol; rm -rf",
            "gpt-5.6-sol || whoami",
            "gpt-5.6-sol`whoami`",
            "gpt-5.6-sol$(whoami)",
            "gpt-5.6-sol --flag",
            "gpt-5.6-sol@special",
            "gpt-5.6-sol&whoami",
        ]
        for model in invalid_models:
            with pytest.raises(backends.BackendError) as exc_info:
                backends._validate_model_name(model)
            assert "Invalid model name" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_codex_model_injection_rejected(self):
        """Test that Codex rejects injected model args."""
        # Try to inject a flag via model name
        with pytest.raises(backends.BackendError) as exc_info:
            backends.call_codex("test prompt", "gpt-5.6-sol; rm -rf /")

        assert "Invalid model name" in str(exc_info.value)


class TestHttpxUsage:
    """Test that backends use httpx, not aiohttp."""

    def test_httpx_imported_in_backends(self):
        """Verify httpx is imported at module level in backends."""
        # backends module must import httpx successfully
        import mcp_brain_router.backends

        assert "httpx" in dir(mcp_brain_router.backends)

    def test_no_aiohttp_in_backends(self):
        """Verify backends.py does NOT use aiohttp."""
        with open(
            "/Users/mohannarayanswamy/code workshop/claude projects/Personal/mcp-brain-router/src/mcp_brain_router/backends.py"
        ) as f:
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
        assert (
            'f"{headroom_url}/anthropic/v1/messages"' in source
            or "f'{headroom_url}/anthropic/v1/messages'" in source
        )

    def test_glm_via_headroom_url_construction(self):
        """Verify GLM via headroom constructs the correct URL."""
        import inspect

        source = inspect.getsource(backends.call_glm_via_headroom)
        # Should construct URL as {headroom_url}/anthropic/v1/messages
        assert (
            'f"{headroom_url}/anthropic/v1/messages"' in source
            or "f'{headroom_url}/anthropic/v1/messages'" in source
        )

    def test_glm_paths_use_reasoning_safe_text_extraction(self):
        """Both GLM paths skip thinking blocks instead of assuming content[0]."""
        import inspect

        assert "_extract_text" in inspect.getsource(backends.call_glm)
        assert "_extract_text" in inspect.getsource(backends.call_glm_via_headroom)

    def test_extract_text_skips_thinking_blocks(self):
        data = {
            "content": [
                {"type": "thinking", "thinking": "hidden"},
                {"type": "text", "text": "usable answer"},
            ]
        }
        assert backends._extract_text(data, "GLM") == "usable answer"


class TestPureTierRouting:
    """Each tier owns one backend; orchestrators own cross-tier cascades."""

    def _cfg(self):
        return Config(deepseek_key="sk-ds", glm_key="glm-k", codex_enabled=True)

    @pytest.mark.asyncio
    async def test_code_glm_429_does_not_call_deepseek(self):
        """CODE quota exhaustion is returned without crossing into CHEAP."""
        with (
            patch("mcp_brain_router.router.backends.call_glm", new_callable=AsyncMock) as mglm,
            patch("mcp_brain_router.router.backends.call_deepseek", new_callable=AsyncMock) as mds,
        ):
            mglm.side_effect = backends.BackendQuotaError("GLM", 429, "limit reached")

            result = await route(Complexity.CODE, "p", config=self._cfg())

            assert result.backend == "none"
            assert result.exhausted is True
            assert result.tried == ["glm"]
            assert result.failure_kind == "quota_exhausted"
            assert "limit reached" in result.failure_reason
            assert mglm.called
            assert not mds.called

    @pytest.mark.asyncio
    async def test_quota_exhaustion_carries_reset_metadata(self):
        """A real 429 carries reset metadata for orchestrator policy."""
        with patch("mcp_brain_router.router.backends.call_glm", new_callable=AsyncMock) as mglm:
            mglm.side_effect = backends.BackendQuotaError(
                "GLM", 429, "reset at 2026-06-29 15:56:33", reset_at="2026-06-29 15:56:33"
            )

            result = await route(Complexity.CODE, "p", config=self._cfg())

            assert result.exhausted is True
            assert result.backend == "none"
            assert result.tried == ["glm"]
            assert result.reset_at == "2026-06-29 15:56:33"
            assert "another tier" in result.content.lower()

    @pytest.mark.asyncio
    async def test_hard_error_does_not_silently_fallback(self):
        """A non-retryable BackendError (e.g. auth/4xx) must propagate, NOT
        get masked by a fallback — config errors should be loud."""
        with (
            patch("mcp_brain_router.router.backends.call_glm", new_callable=AsyncMock) as mglm,
            patch("mcp_brain_router.router.backends.call_deepseek", new_callable=AsyncMock) as mds,
        ):
            mglm.side_effect = backends.BackendError("GLM API error 401: bad key")
            mds.return_value = {"content": "should-not-be-used", "usage": {}}

            with pytest.raises(backends.BackendError):
                await route(Complexity.CODE, "p", config=self._cfg())
            assert not mds.called  # never fell through on a hard error

    @pytest.mark.asyncio
    async def test_codex_timeout_is_not_quota_exhaustion(self):
        """A Codex process timeout propagates as a typed transient error."""
        with patch("mcp_brain_router.router.backends.call_codex") as mcx:
            mcx.side_effect = backends.BackendTransientError(
                "codex",
                "Codex subprocess timed out after 180s",
                failure_kind="timeout",
                elapsed_ms=180000,
            )

            with pytest.raises(backends.BackendTransientError) as exc_info:
                await route(Complexity.ADVERSARIAL, "p", config=self._cfg())
            assert exc_info.value.failure_kind == "timeout"
            assert exc_info.value.backend == "codex"

    @pytest.mark.asyncio
    async def test_cheap_glm_429_does_not_cross_tier(self):
        """002: CHEAP now routes to GLM; a 429 returns exhausted without crossing."""
        with patch("mcp_brain_router.router.backends.call_glm", new_callable=AsyncMock) as mglm:
            mglm.side_effect = backends.BackendQuotaError("GLM", 429, "limit")

            result = await route(Complexity.CHEAP, "p", config=self._cfg())

            assert result.exhausted is True
            assert result.tried == ["glm"]
            assert result.failure_kind == "quota_exhausted"
            assert mglm.called

    @pytest.mark.asyncio
    async def test_adversarial_tier_is_codex_only(self):
        """ADVERSARIAL success returns only Codex."""
        with patch("mcp_brain_router.router.backends.call_codex") as mcx:
            mcx.return_value = {"content": "PONG", "usage": {}}
            result = await route(Complexity.ADVERSARIAL, "p", config=self._cfg())
            assert result.backend == "codex"
            assert result.tried == ["codex"]
            assert result.exhausted is False


class TestTerminalMetadata:
    """Server contract distinguishes quota, timeout, and hard errors."""

    def _cfg(self):
        return Config(deepseek_key="sk-ds", glm_key="glm-k", codex_enabled=True)

    @pytest.mark.asyncio
    async def test_quota_result_is_logged_with_reason_and_elapsed(self):
        quota_result = RouteResult(
            content="glm quota exhausted; call another tier",
            model="",
            backend="none",
            complexity=Complexity.CODE,
            headroom_used=False,
            exhausted=True,
            tried=["glm"],
            reset_at="2026-07-11 12:00:00",
            failure_kind="quota_exhausted",
            failure_reason="GLM quota error 429",
        )
        with (
            patch.object(server, "_load_config", return_value=self._cfg()),
            patch.object(server, "route", new_callable=AsyncMock, return_value=quota_result),
            patch.object(server, "_log_delegation") as log_call,
        ):
            response = await server._delegate_impl("code", "p")

        assert response["exhausted"] is True
        assert response["failure_kind"] == "quota_exhausted"
        assert response["failure_reason"] == "GLM quota error 429"
        assert response["elapsed_ms"] >= 0
        log_call.assert_called_once()

    @pytest.mark.asyncio
    async def test_timeout_is_error_not_exhausted_and_is_logged(self):
        timeout = backends.BackendTransientError(
            "codex",
            "Codex subprocess timed out after 180s",
            failure_kind="timeout",
            elapsed_ms=180000,
        )
        with (
            patch.object(server, "_load_config", return_value=self._cfg()),
            patch.object(server, "route", new_callable=AsyncMock, side_effect=timeout),
            patch.object(server, "_log_delegation") as log_call,
        ):
            response = await server._delegate_impl("adversarial", "p")

        assert response["exhausted"] is False
        assert response["backend"] == "codex"
        assert response["failure_kind"] == "timeout"
        assert response["failure_reason"] == "Codex subprocess timed out after 180s"
        assert response["elapsed_ms"] == 180000
        log_call.assert_called_once()

    @pytest.mark.asyncio
    async def test_codex_caller_cannot_spawn_nested_codex(self):
        with (
            patch.dict("os.environ", {"BRAIN_ROUTER_CALLER": "codex"}),
            patch.object(server, "_load_config") as load_config,
            patch.object(server, "route", new_callable=AsyncMock) as route_call,
            patch.object(server, "_log_delegation") as log_call,
        ):
            response = await server._delegate_impl("adversarial", "p")

        assert response["exhausted"] is False
        assert response["backend"] == "router"
        assert response["caller"] == "codex"
        assert response["failure_kind"] == "nested_codex_blocked"
        load_config.assert_not_called()
        route_call.assert_not_awaited()
        log_call.assert_called_once()
        assert log_call.call_args.args[0]["caller"] == "codex"

    @pytest.mark.asyncio
    async def test_claude_caller_may_use_adversarial_tier(self):
        result = RouteResult(
            content="PONG",
            model="gpt-5.6-sol",
            backend="codex",
            complexity=Complexity.ADVERSARIAL,
            headroom_used=False,
        )
        with (
            patch.dict("os.environ", {"BRAIN_ROUTER_CALLER": "claude"}),
            patch.object(server, "_load_config", return_value=self._cfg()),
            patch.object(server, "route", new_callable=AsyncMock, return_value=result) as route_call,
            patch.object(server, "_log_delegation"),
        ):
            response = await server._delegate_impl("adversarial", "p")

        assert response["answer"] == "PONG"
        assert response["caller"] == "claude"
        route_call.assert_awaited_once()

    def test_audit_log_persists_terminal_metadata(self, tmp_path, monkeypatch):
        log_path = tmp_path / "delegations.jsonl"
        monkeypatch.setattr(server, "_DELEGATION_LOG", log_path)
        server._log_delegation(
            {
                "complexity": "adversarial",
                "backend": "codex",
                "exhausted": False,
                "failure_kind": "timeout",
                "failure_reason": "timed out",
                "elapsed_ms": 180001,
            },
            prompt_len=3,
        )

        record = json.loads(log_path.read_text().strip())
        assert record["failure_kind"] == "timeout"
        assert record["failure_reason"] == "timed out"
        assert record["elapsed_ms"] == 180001


class TestErrorClassification:
    """_classify_http_error maps statuses to the right exception type."""

    def test_429_is_quota_error(self):
        err = backends._classify_http_error(
            "GLM",
            429,
            '{"error":{"message":"Usage limit reached for 5 hour. reset at 2026-06-29 15:56:33"}}',
        )
        assert isinstance(err, backends.BackendQuotaError)
        assert err.status_code == 429
        assert err.reset_at == "2026-06-29 15:56:33"

    def test_5xx_is_transient_provider_error_not_quota(self):
        err = backends._classify_http_error("DeepSeek", 503, "{}")
        assert isinstance(err, backends.BackendTransientError)
        assert not isinstance(err, backends.BackendQuotaError)
        assert err.failure_kind == "provider_error"
        assert err.status_code == 503

    def test_4xx_auth_is_hard_error(self):
        err = backends._classify_http_error(
            "GLM",
            401,
            '{"error":{"message":"invalid api key"}}',
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
            backends.call_codex("-h --evil-flag", "gpt-5.6-sol")
            argv = mrun.call_args[0][0]
        sep = argv.index("--")
        # Prompt is ONE positional arg after "--" (caveman system directive is
        # prepended inside that same arg — still not parsed as a CLI flag).
        assert sep == len(argv) - 2  # exactly one positional after the separator
        assert argv[-1].endswith("-h --evil-flag")  # untrusted input reaches codex verbatim
        assert "--evil-flag" not in argv[:-1]  # never leaks out as its own argv token

    def test_codex_argv_skips_mcp_boot(self):
        """mcp_servers={} must be present — codex otherwise boots every MCP
        server in ~/.codex/config.toml and blows the subprocess timeout."""
        with patch("mcp_brain_router.backends.subprocess.run") as mrun:
            mrun.return_value = MagicMock(returncode=0, stdout="ok")
            backends.call_codex("p", "gpt-5.6-sol")
            argv = mrun.call_args[0][0]
        assert "-c" in argv and "mcp_servers={}" in argv
        assert "--skip-git-repo-check" in argv

    def test_codex_argv_uses_lean_worker_profile(self):
        """Delegated Codex must not load the full interactive user rig."""
        with patch("mcp_brain_router.backends.subprocess.run") as mrun:
            mrun.return_value = MagicMock(returncode=0, stdout="ok")
            backends.call_codex("p", "gpt-5.6-sol")
            argv = mrun.call_args[0][0]

        for flag in ("--ignore-user-config", "--ignore-rules", "--ephemeral"):
            assert flag in argv
        disabled = [argv[i + 1] for i, arg in enumerate(argv[:-1]) if arg == "--disable"]
        assert disabled == ["plugins", "hooks", "memories", "apps", "multi_agent"]
        assert argv[argv.index("-C") + 1] == "/private/tmp"
        assert 'model_reasoning_effort="low"' in argv

    def test_codex_timeout_carries_kind_reason_and_elapsed(self):
        """Timeout metadata must survive to the server; it is never quota."""
        timeout = backends.subprocess.TimeoutExpired(cmd=["codex"], timeout=180)
        with (
            patch("mcp_brain_router.backends.subprocess.run", side_effect=timeout),
            patch("mcp_brain_router.backends.time.perf_counter", side_effect=[10.0, 370.0]),
        ):
            with pytest.raises(backends.BackendTransientError) as exc_info:
                backends.call_codex("p", "gpt-5.6-sol")

        err = exc_info.value
        assert err.backend == "codex"
        assert err.failure_kind == "timeout"
        assert err.elapsed_ms == 360000
        assert "360s" in str(err)  # 004 C410: cap raised 180->360

    def test_install_smoke_test_uses_same_codex_base(self):
        """install.py's codex smoke test must share CODEX_EXEC_BASE so the
        two invocations can never drift (found drifted in adversarial pass)."""
        import inspect

        from mcp_brain_router import install

        src = inspect.getsource(install._test_backend_codex)
        assert "CODEX_EXEC_BASE" in src

    def test_installer_smokes_the_selected_codex_model(self):
        """A Terra/Luna/custom selection must not be falsely validated by Sol."""
        from mcp_brain_router import install

        with (
            patch.object(install, "_test_backend_glm", return_value=True),
            patch.object(install, "_test_backend_deepseek", return_value=True),
            patch.object(install, "_test_backend_codex", return_value=True) as codex,
        ):
            install.run_smoke_test("glm-key", "deepseek-key", True, "gpt-5.6-terra", None)

        codex.assert_called_once_with("gpt-5.6-terra")

    def test_claude_registration_declares_caller_identity(self):
        """Fresh Claude registrations must activate the nested-Codex guard."""
        import inspect

        from mcp_brain_router import install

        src = inspect.getsource(install.register_in_claude_code)
        assert "BRAIN_ROUTER_CALLER=claude" in src

    def test_error_family_is_unified(self):
        """router's credential/availability errors must subclass
        backends.BackendError, so server.py's `except BackendError` (imported
        from backends) catches the WHOLE family — pre-fix these were two
        unrelated classes and codex errors fell into the generic handler."""
        import mcp_brain_router.router as router_mod

        assert issubclass(router_mod.BackendError, backends.BackendError)
        assert issubclass(router_mod.MissingCredentialError, backends.BackendError)
        assert issubclass(backends.BackendTransientError, backends.BackendError)


class TestCavemanSystemDirective:
    """Every delegated backend must carry the caveman-ultra reply directive so
    backend REPLIES stay terse (cuts output tokens on every delegate). One
    source of truth: backends.CAVEMAN_SYSTEM. G-guard for the 2026-07-05 change."""

    def test_constant_exists_and_terse_intent(self):
        assert backends.CAVEMAN_SYSTEM
        assert "caveman" in backends.CAVEMAN_SYSTEM.lower()

    def test_codex_prompt_prepends_caveman(self):
        with patch("mcp_brain_router.backends.subprocess.run") as mrun:
            mrun.return_value = MagicMock(returncode=0, stdout="ok")
            backends.call_codex("do X", "gpt-5.6-sol")
            argv = mrun.call_args[0][0]
        assert argv[-1].startswith(backends.CAVEMAN_SYSTEM)  # directive leads
        assert argv[-1].endswith("do X")  # task preserved verbatim
