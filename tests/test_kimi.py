"""Tests for the Kimi provider (Kimi Code CLI, OAuth login, enable flag).

Mirrors the Grok coverage: kimi is a CLI-subprocess coding provider selected
by role routing (not a public complexity tier), with a chat path
(`kimi -p <prompt> --output-format text`) and an agentic path (same + `--auto`
in the REAL cwd). All subprocess calls are mocked — no live CLI, no network.
"""

import os
from unittest.mock import MagicMock, patch

import pytest

from mcp_brain_router import backends
from mcp_brain_router.backends import (
    BackendError,
    BackendQuotaError,
    BackendTransientError,
)
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
        glm_key="glm-k",
        codex_enabled=True,
        grok_enabled=True,
        kimi_enabled=True,
        roles={"worker": ["kimi", "glm-5.2", "claude-sonnet-5"]},
    )
    base.update(overrides)
    return Config(**base)


# ============================================================================
# Provider mapping / role resolution
# ============================================================================


class TestKimiProviderMapping:
    def test_kimi_prefix_maps_to_kimi_provider(self):
        assert provider_for_model("kimi") is Provider.KIMI
        assert provider_for_model("Kimi") is Provider.KIMI  # case-insensitive

    def test_kimi_default_model_is_just_kimi(self):
        """No -m flag is passed to the CLI (its configured default runs), so
        the router model id for kimi is just 'kimi'."""
        assert _get_backend_default_model("kimi", Config()) == "kimi"

    def test_kimi_is_coding_provider_not_complexity_tier(self):
        """Kimi resolves as KIMI for role routing; no public Kimi tier exists."""
        with pytest.raises(ValueError):
            Complexity("kimi")
        assignment = resolve_role(Role.WORKER, "opus", _kimi_config())
        assert assignment.provider is Provider.KIMI
        assert assignment.backend == "kimi"
        assert assignment.execute_natively is False
        assert assignment.model == "kimi"


# ============================================================================
# Credential validation
# ============================================================================


class TestKimiCredentialValidation:
    @pytest.mark.asyncio
    async def test_kimi_disabled_raises_unavailable(self):
        """Availability = enable flag + binary on PATH (like grok). Disabled
        flag → BackendUnavailableError before any subprocess."""
        assignment = resolve_role(Role.WORKER, "opus", _kimi_config())
        with pytest.raises(BackendUnavailableError, match="Kimi not enabled"):
            await route_assignment(
                assignment, "p", _kimi_config(kimi_enabled=False), cwd="/tmp"
            )

    @pytest.mark.asyncio
    async def test_kimi_enabled_passes_validation(self):
        assignment = resolve_role(Role.WORKER, "opus", _kimi_config())
        with patch(
            "mcp_brain_router.router.backends.call_kimi_agentic"
        ) as magentic:
            magentic.return_value = {"content": "done", "usage": None}
            result = await route_assignment(
                assignment, "p", _kimi_config(), mode="agentic", cwd="/tmp"
            )
        assert result.backend == "kimi"
        magentic.assert_called_once()


# ============================================================================
# Backend subprocess argv shape (chat + agentic)
# ============================================================================


class TestKimiArgv:
    def test_kimi_chat_uses_p_and_text_output(self):
        """Chat: `kimi -p <CAVEMAN+prompt> --output-format text`. The prompt is
        ONE argv token (CAVEMAN_SYSTEM-prefixed so it can never start with a
        dash). NO -m flag — the CLI's configured default model runs."""
        with patch("mcp_brain_router.backends.subprocess.run") as mrun:
            mrun.return_value = MagicMock(returncode=0, stdout="pong")
            out = backends.call_kimi("ping", "kimi")
            argv = mrun.call_args[0][0]
        assert out["content"] == "pong"
        assert argv[0] == "kimi" or argv[0].endswith("/kimi")
        prompt_token = argv[argv.index("-p") + 1]
        assert prompt_token.endswith("ping")
        assert prompt_token.startswith(backends.CAVEMAN_SYSTEM[:20])
        assert argv[argv.index("--output-format") + 1] == "text"
        assert "-m" not in argv  # default model (k3) — never overridden
        assert "--auto" not in argv  # chat mode never edits
        assert mrun.call_args.kwargs.get("shell", False) is False

    def test_kimi_agentic_uses_real_cwd_no_interactive_flags(self):
        """Agentic: same chat argv, run with cwd=<orchestrator cwd> so the
        worker writes into the caller's repo. Kimi's prompt mode already
        auto-approves tools non-interactively (live-verified 2026-07-19);
        `--auto`/`--yolo` are REJECTED when combined with `-p`, so neither is
        passed. AGENTIC_SYSTEM (not caveman) is prepended — the file write is
        the deliverable."""
        with patch("mcp_brain_router.backends.subprocess.run") as mrun, patch(
            "mcp_brain_router.backends._resolve_agentic_cwd", return_value="/tmp"
        ):
            mrun.return_value = MagicMock(returncode=0, stdout="done")
            out = backends.call_kimi_agentic("do K", "kimi", "/tmp")
            argv = mrun.call_args[0][0]
        assert out["content"] == "done"
        assert argv[0] == "kimi" or argv[0].endswith("/kimi")
        prompt_token = argv[argv.index("-p") + 1]
        assert prompt_token.endswith("do K")
        assert prompt_token.startswith(backends.AGENTIC_SYSTEM[:20])
        assert "--auto" not in argv  # rejected by the CLI when combined with -p
        assert "--yolo" not in argv
        assert "-m" not in argv
        assert mrun.call_args.kwargs.get("cwd") == "/tmp"
        assert mrun.call_args.kwargs.get("shell", False) is False

    def test_kimi_passes_path_fixed_env(self):
        """G-guard (env-drop): kimi is a bundled binary (no node shim) so
        _kimi_env only prepends kimi's own bin dir — but both calls MUST pass
        env= so the empty-env MCP process resolves kimi."""
        kimi_dir = os.path.dirname(backends._KIMI_BIN) if "/" in backends._KIMI_BIN else ""
        for fn, args in (
            (backends.call_kimi, ("p", "kimi")),
            (backends.call_kimi_agentic, ("p", "kimi", "/tmp")),
        ):
            with patch("mcp_brain_router.backends.subprocess.run") as mrun, patch(
                "mcp_brain_router.backends._resolve_agentic_cwd", return_value="/tmp"
            ):
                mrun.return_value = MagicMock(returncode=0, stdout="ok")
                fn(*args)
                env = mrun.call_args.kwargs.get("env")
            assert env is not None, f"{fn.__name__} passed no env= (env-drop regression)"
            if kimi_dir:
                assert kimi_dir in env.get("PATH", ""), (
                    f"{fn.__name__} env PATH missing kimi bin dir {kimi_dir}"
                )

    def test_kimi_agentic_nonexistent_cwd_raises_clean_backenderror(self):
        """Same §9.6 injection-safety guard as the other agentic workers: a bad
        cwd fails LOUD with BackendError before subprocess.run."""
        bad = "/no/such/dir/kimi_probe_does_not_exist_98765"
        with patch("mcp_brain_router.backends.subprocess.run") as mrun:
            with pytest.raises(BackendError, match="not a directory"):
                backends.call_kimi_agentic("p", "kimi", bad)
            mrun.assert_not_called()


# ============================================================================
# Error classification
# ============================================================================


class TestKimiErrorClassification:
    @pytest.mark.parametrize(
        "message",
        ["Usage limit reached", "quota exhausted", "status 429", "rate_limit"],
    )
    def test_kimi_quota_messages_are_classified(self, message):
        """Non-zero exit + a quota marker in stdout/stderr → BackendQuotaError
        (kimi exits non-zero on failure; markers matched case-insensitively,
        same as grok)."""
        with patch("mcp_brain_router.backends.subprocess.run") as mrun:
            mrun.return_value = MagicMock(returncode=1, stdout="", stderr=message)
            with pytest.raises(BackendQuotaError):
                backends.call_kimi("p", "kimi")

    def test_kimi_nonzero_exit_without_quota_marker_is_backend_error(self):
        with patch("mcp_brain_router.backends.subprocess.run") as mrun:
            mrun.return_value = MagicMock(returncode=1, stdout="", stderr="boom")
            with pytest.raises(BackendError) as exc:
                backends.call_kimi("p", "kimi")
        assert exc.value.failure_kind == "process_error"
        assert not isinstance(exc.value, BackendQuotaError)

    def test_kimi_timeout_is_transient_not_quota(self):
        import subprocess as sp

        with patch("mcp_brain_router.backends.subprocess.run") as mrun:
            mrun.side_effect = sp.TimeoutExpired(cmd="kimi", timeout=1)
            with pytest.raises(BackendTransientError) as exc:
                backends.call_kimi("p", "kimi")
        assert exc.value.failure_kind == "timeout"


# ============================================================================
# route_assignment dispatch (chat + agentic) and cascade contract
# ============================================================================


class TestKimiRouteAssignment:
    @pytest.mark.asyncio
    async def test_kimi_agentic_dispatches_to_call_kimi_agentic(self):
        assignment = resolve_role(Role.WORKER, "opus", _kimi_config())
        with patch(
            "mcp_brain_router.router.backends.call_kimi_agentic"
        ) as magentic:
            magentic.return_value = {"content": "built", "usage": None}
            result = await route_assignment(
                assignment, "build", _kimi_config(), mode="agentic", cwd="/tmp"
            )
        assert result.backend == "kimi"
        assert result.content == "built"
        assert result.complexity is Complexity.CODE
        assert result.tried == ["kimi"]
        magentic.assert_called_once()
        # cwd threads to the backend's 3rd positional (agentic file-placement)
        assert magentic.call_args[0][2] == "/tmp"

    @pytest.mark.asyncio
    async def test_kimi_chat_dispatches_to_call_kimi(self):
        assignment = resolve_role(Role.WORKER, "opus", _kimi_config())
        with patch("mcp_brain_router.router.backends.call_kimi") as mchat:
            mchat.return_value = {"content": "PONG", "usage": None}
            result = await route_assignment(
                assignment, "ping", _kimi_config(), mode="chat", cwd="/tmp"
            )
        assert result.backend == "kimi"
        assert result.content == "PONG"
        assert result.tried == ["kimi"]
        mchat.assert_called_once()

    @pytest.mark.asyncio
    async def test_kimi_quota_becomes_exhausted_result(self):
        """BackendQuotaError → exhausted quota_exhausted result (tried=["kimi"]),
        so the role cascade advances to the next candidate."""
        assignment = resolve_role(Role.WORKER, "opus", _kimi_config())
        with patch(
            "mcp_brain_router.router._route_agentic",
            side_effect=BackendQuotaError("Kimi", 429, "usage limit reached"),
        ):
            result = await route_assignment(
                assignment, "build", _kimi_config(), mode="agentic", cwd="/tmp"
            )
        assert result.exhausted is True
        assert result.failure_kind == "quota_exhausted"
        assert result.tried == ["kimi"]

    @pytest.mark.asyncio
    async def test_kimi_transient_advances_cascade_not_quota(self):
        """A transient timeout on the kimi path must ADVANCE the cascade
        (exhausted=True) WITHOUT being labeled quota."""
        assignment = resolve_role(Role.WORKER, "opus", _kimi_config())
        with patch(
            "mcp_brain_router.router._route_agentic",
            side_effect=BackendTransientError(
                "Kimi", "kimi agentic subprocess timed out after 360s",
                failure_kind="timeout",
            ),
        ):
            result = await route_assignment(
                assignment, "build", _kimi_config(), mode="agentic", cwd="/tmp"
            )
        assert result.exhausted is True  # advances, does not abort
        assert result.failure_kind == "transient_error"
        assert result.failure_kind != "quota_exhausted"
        assert result.tried == ["kimi"]


# ============================================================================
# Config load/save round-trip
# ============================================================================


class TestKimiConfigFlag:
    def test_kimi_enabled_round_trips(self, tmp_path, monkeypatch):
        cfg_file = tmp_path / "config.toml"
        monkeypatch.setattr("mcp_brain_router.config.CONFIG_FILE", cfg_file)
        Config(glm_key="k", kimi_enabled=True).save()
        assert Config.load().kimi_enabled is True

    def test_kimi_enabled_defaults_false(self, tmp_path, monkeypatch):
        cfg_file = tmp_path / "config.toml"
        cfg_file.write_text('glm_key = "k"\n')
        cfg_file.chmod(0o600)
        monkeypatch.setattr("mcp_brain_router.config.CONFIG_FILE", cfg_file)
        assert Config.load().kimi_enabled is False
