"""Tests for the agentic worker mode (spec 002-agentic-worker-mode).

Covers SC-1..SC-8. The default `pytest` run stays fast + offline: the
slow/live-subprocess tests (SC-1, SC-5) are marked `@pytest.mark.slow` and
guarded behind the RUN_AGENTIC_LIVE env flag, so they are SKIPPED unless the
operator opts in. All routing/resolution logic tests (SC-2/3/4/6/8) run with
mocks — no live network or subprocess.
"""

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mcp_brain_router import backends, server
from mcp_brain_router.config import Config, DEFAULT_ROLE_MODES
from mcp_brain_router.router import (
    Complexity,
    Provider,
    Role,
    default_mode_for_role,
    resolve_role,
    route,
    route_assignment,
)

_LIVE = pytest.mark.slow
_RUN_LIVE = os.environ.get("RUN_AGENTIC_LIVE") == "1"


def _agentic_config() -> Config:
    """Config carrying the spec 002 candidate rosters."""
    return Config(
        glm_key="glm-k",
        codex_enabled=True,
        roles={
            "worker": ["glm-5.2", "gpt-5.6-terra", "claude-sonnet-5"],
            "simple": ["glm-4.7", "gpt-5.6-luna", "claude-haiku-4-5"],
            "adversary": ["gpt-5.6-sol", "claude-opus-4-8"],
            "thinker": ["gpt-5.6-sol", "claude-opus-4-8"],
        },
    )


# ============================================================================
# SC-2: cheap routes to GLM (DeepSeek removed); chat mode unchanged
# ============================================================================


class TestSC2CheapRoutesToGlm:
    @pytest.mark.asyncio
    async def test_cheap_resolves_to_glm_backend(self):
        """`cheap` resolves to the glm backend (deepseek removed, SC-2)."""
        config = Config(glm_key="glm-k", codex_enabled=False)
        with patch(
            "mcp_brain_router.router.backends.call_glm", new_callable=AsyncMock
        ) as mglm:
            mglm.return_value = {"content": "ok", "usage": {}}
            result = await route(Complexity.CHEAP, "p", config=config)
        assert result.backend == "glm"
        assert result.model == "glm-4.7"  # cheap default = FAST glm

    @pytest.mark.asyncio
    async def test_chat_mode_unchanged_returns_text_no_subprocess(self):
        """chat mode never touches an agentic harness — text-only path."""
        config = Config(glm_key="glm-k", codex_enabled=False)
        with (
            patch(
                "mcp_brain_router.router.backends.call_glm", new_callable=AsyncMock
            ) as mglm,
            patch("mcp_brain_router.router.backends.call_glm_agentic") as magentic,
        ):
            mglm.return_value = {"content": "text", "usage": {}}
            result = await route(Complexity.CODE, "p", config=config, mode="chat")
        assert result.backend == "glm"
        assert result.content == "text"
        mglm.assert_called()
        magentic.assert_not_called()


# ============================================================================
# SC-3: default mode policy (worker=agentic, others=chat)
# ============================================================================


class TestSC3DefaultPolicy:
    @pytest.mark.parametrize(
        ("role", "expected"),
        [
            (Role.WORKER, "agentic"),
            (Role.ADVERSARY, "chat"),
            (Role.THINKER, "chat"),
            (Role.SIMPLE, "chat"),
        ],
    )
    def test_default_mode_per_role(self, role, expected):
        assert default_mode_for_role(role) == expected

    def test_config_default_role_modes_worker_agentic(self):
        assert DEFAULT_ROLE_MODES["worker"] == "agentic"
        assert DEFAULT_ROLE_MODES["adversary"] == "chat"

    def test_role_modes_merged_on_load_even_when_section_absent(self, tmp_path, monkeypatch):
        """A config.toml that omits [role_modes] still resolves worker=agentic."""
        cfg_file = tmp_path / "config.toml"
        cfg_file.write_text('glm_key = "k"\n')
        cfg_file.chmod(0o600)
        monkeypatch.setattr("mcp_brain_router.config.CONFIG_FILE", cfg_file)
        loaded = Config.load()
        assert loaded.role_modes["worker"] == "agentic"
        assert loaded.role_modes["adversary"] == "chat"

    def test_role_modes_user_override_wins(self, tmp_path, monkeypatch):
        """A config.toml [role_modes] override sits on top of the default."""
        cfg_file = tmp_path / "config.toml"
        cfg_file.write_text(
            'glm_key = "k"\n[role_modes]\nworker = "chat"\n'
        )
        cfg_file.chmod(0o600)
        monkeypatch.setattr("mcp_brain_router.config.CONFIG_FILE", cfg_file)
        loaded = Config.load()
        assert loaded.role_modes["worker"] == "chat"  # user override
        assert loaded.role_modes["adversary"] == "chat"  # default preserved


# ============================================================================
# SC-4: codex-orchestrator adversary → anthropic-cli (NOT native, NOT codex)
# ============================================================================


class TestSC4CodexOrchestratorAdversary:
    def test_agentic_adversary_targets_anthropic_cli(self):
        """With orchestrator=codex, adversary in agentic mode → anthropic-cli
        backend, NOT execute_natively, NOT codex (SC-4)."""
        assignment = resolve_role(
            Role.ADVERSARY,
            Provider.CODEX,
            _agentic_config(),
            mode="agentic",
        )
        assert assignment.provider is Provider.ANTHROPIC
        assert assignment.backend == "anthropic-cli"
        assert assignment.execute_natively is False
        assert assignment.model == "claude-opus-4-8"

    def test_chat_mode_adversary_still_native(self):
        """chat mode preserves the pre-change native hand-back (SC-2 regression)."""
        assignment = resolve_role(
            Role.ADVERSARY, Provider.CODEX, _agentic_config(), mode="chat"
        )
        assert assignment.execute_natively is True
        assert assignment.backend is None


# ============================================================================
# SC-6: agentic argv shape (no secret/PII leak, lean codex flags minus sandbox)
# ============================================================================


class TestSC6AgenticArgv:
    def test_glm_agentic_uses_cc_glm_argv(self):
        with patch("mcp_brain_router.backends.subprocess.run") as mrun:
            mrun.return_value = MagicMock(returncode=0, stdout="ok")
            backends.call_glm_agentic("do X", "glm-5.2")
            argv = mrun.call_args[0][0]
        # argv is a list, not a shell string — no injection surface
        assert isinstance(argv, list)
        assert argv[0] == "cc-glm" or argv[0].endswith("/cc-glm")
        assert "-p" in argv
        assert "glm-5.2" in argv  # tier glm model passed through
        # caveman directive + prompt travel together as one argv token
        prompt_token = argv[argv.index("-p") + 1]
        assert prompt_token.endswith("do X")
        # no shell interpretation
        assert mrun.call_args.kwargs.get("shell", False) is False

    def test_anthropic_agentic_uses_cc_brain_claude_argv(self):
        with patch("mcp_brain_router.backends.subprocess.run") as mrun:
            mrun.return_value = MagicMock(returncode=0, stdout="ok")
            backends.call_anthropic_agentic("do Y", "claude-opus-4-8")
            argv = mrun.call_args[0][0]
        assert isinstance(argv, list)
        assert argv[0] == "cc-brain" or argv[0].endswith("/cc-brain")
        assert argv[1] == "claude"
        assert "claude-opus-4-8" in argv  # resolved candidate model passed through
        assert "-p" in argv
        assert mrun.call_args.kwargs.get("shell", False) is False

    def test_codex_agentic_drops_sandbox_and_ignore_rules(self):
        """SC-6: codex exec in REAL cwd — no -C /private/tmp, no --ignore-rules,
        but keeps mcp_servers={} + low reasoning + skip-git-repo-check."""
        with patch("mcp_brain_router.backends.subprocess.run") as mrun:
            mrun.return_value = MagicMock(returncode=0, stdout="ok")
            backends.call_codex_agentic("do Z", "gpt-5.6-terra")
            argv = mrun.call_args[0][0]
        assert isinstance(argv, list)
        # lean worker profile preserved
        assert "-c" in argv and "mcp_servers={}" in argv
        assert "--skip-git-repo-check" in argv
        assert 'model_reasoning_effort="low"' in argv
        for flag in ("--ignore-user-config", "--ephemeral"):
            assert flag in argv
        # SC-6: the two flags that would prevent editing the repo are DROPPED
        assert "-C" not in argv
        assert "/private/tmp" not in argv
        assert "--ignore-rules" not in argv
        # runs in the real cwd (no cwd isolation to /private/tmp)
        assert mrun.call_args.kwargs.get("cwd") is not None
        # prompt travels after a "--" separator (same protection as call_codex)
        sep = argv.index("--")
        assert argv[-1].endswith("do Z")

    def test_no_api_key_in_agentic_argv(self):
        """No secret/API key ever appears in the constructed argv (SC-6)."""
        with patch("mcp_brain_router.backends.subprocess.run") as mrun:
            mrun.return_value = MagicMock(returncode=0, stdout="ok")
            backends.call_glm_agentic("p", "glm-5.2")
            glm_argv = mrun.call_args[0][0]
            backends.call_codex_agentic("p", "gpt-5.6-terra")
            codex_argv = mrun.call_args[0][0]
            backends.call_anthropic_agentic("p", "claude-sonnet-5")
            anth_argv = mrun.call_args[0][0]
        joined = " ".join(glm_argv + codex_argv + anth_argv)
        assert "sk-" not in joined
        assert "api_key" not in joined.lower()


# ============================================================================
# SC-1 (offline logic): route(complexity='code', mode='agentic') dispatches
# to call_glm_agentic with cc-glm — no live GLM call needed.
# ============================================================================


class TestSC1AgenticDispatch:
    @pytest.mark.asyncio
    async def test_code_agentic_dispatches_to_glm_agentic(self):
        """SC-1 offline: code+agentic routes to call_glm_agentic (cc-glm)."""
        config = Config(glm_key="glm-k", codex_enabled=True)
        with patch(
            "mcp_brain_router.router.backends.call_glm_agentic"
        ) as magentic:
            magentic.return_value = {"content": "done", "usage": None}
            result = await route(
                Complexity.CODE, "build it", config=config, mode="agentic"
            )
        assert result.backend == "glm"
        assert result.content == "done"
        magentic.assert_called_once()
        # the harness binary the dispatch targets is cc-glm
        sent_argv = magentic.call_args[0]  # (prompt, model)
        assert "build it" in sent_argv[0]

    @pytest.mark.asyncio
    async def test_adversarial_agentic_dispatches_to_codex_agentic(self):
        config = Config(glm_key="glm-k", codex_enabled=True)
        with patch(
            "mcp_brain_router.router.backends.call_codex_agentic"
        ) as magentic:
            magentic.return_value = {"content": "refuted", "usage": None}
            result = await route(
                Complexity.ADVERSARIAL, "refute this", config=config, mode="agentic"
            )
        assert result.backend == "codex"
        magentic.assert_called_once()


# ============================================================================
# SC-8: exhaustion cascade → Opus CLI as the agentic worker
# (codex orchestrator, both GLM + codex exhausted)
# ============================================================================


class TestSC8ExhaustionCascade:
    def test_worker_agentic_anthropic_cli_when_glm_codex_exhausted(self):
        """SC-8: codex orchestrator + {zhipu,codex} exhausted → worker targets
        anthropic-cli in agentic mode (NOT a dead-end, NOT native)."""
        assignment = resolve_role(
            Role.WORKER,
            Provider.CODEX,
            _agentic_config(),
            exhausted_providers={Provider.ZHIPU, Provider.CODEX},
            mode="agentic",
        )
        assert assignment.provider is Provider.ANTHROPIC
        assert assignment.backend == "anthropic-cli"
        assert assignment.execute_natively is False
        assert assignment.model == "claude-sonnet-5"

    def test_adversary_agentic_anthropic_cli_when_codex_exhausted(self):
        """SC-8: codex orchestrator + codex exhausted → adversary anthropic-cli."""
        assignment = resolve_role(
            Role.ADVERSARY,
            Provider.CODEX,
            _agentic_config(),
            exhausted_providers={Provider.CODEX},
            mode="agentic",
        )
        assert assignment.backend == "anthropic-cli"
        assert assignment.model == "claude-opus-4-8"

    @pytest.mark.asyncio
    async def test_route_assignment_anthropic_cli_dispatches_agentic(self):
        """An anthropic-cli assignment executes via call_anthropic_agentic."""
        assignment = resolve_role(
            Role.ADVERSARY, Provider.CODEX, _agentic_config(), mode="agentic"
        )
        with patch(
            "mcp_brain_router.router.backends.call_anthropic_agentic"
        ) as magentic:
            magentic.return_value = {"content": "opus did it", "usage": None}
            result = await route_assignment(assignment, "p", _agentic_config())
        assert result.backend == "anthropic-cli"
        assert result.content == "opus did it"
        magentic.assert_called_once()


# ============================================================================
# SC-3 (end-to-end): role path resolves mode via config default
# ============================================================================


class TestRoleModeResolution:
    @pytest.mark.asyncio
    async def test_worker_role_defaults_to_agentic(self):
        """role=worker, no explicit mode → agentic dispatch (cc-glm)."""
        config = _agentic_config()
        with (
            patch.object(server, "_load_config", return_value=config),
            patch(
                "mcp_brain_router.router.backends.call_glm_agentic"
            ) as magentic,
            patch.object(server, "_log_delegation"),
        ):
            magentic.return_value = {"content": "wrote file", "usage": None}
            response = await server._delegate_role_impl("worker", "build", "opus")
        assert response["backend"] == "glm"
        assert response["mode"] == "agentic"
        magentic.assert_called_once()

    @pytest.mark.asyncio
    async def test_adversary_role_defaults_to_chat(self):
        """role=adversary, no explicit mode → chat (codex, NOT agentic)."""
        config = _agentic_config()
        with (
            patch.object(server, "_load_config", return_value=config),
            patch(
                "mcp_brain_router.router.backends.call_codex"
            ) as mcodex,
            patch("mcp_brain_router.router.backends.call_codex_agentic") as magentic,
            patch.object(server, "_log_delegation"),
        ):
            mcodex.return_value = {"content": "refuted", "usage": None}
            response = await server._delegate_role_impl("adversary", "refute", "opus")
        assert response["backend"] == "codex"
        assert response["mode"] == "chat"
        mcodex.assert_called_once()
        magentic.assert_not_called()


# ============================================================================
# SC-1 / SC-5 (LIVE): real subprocess writes a scratch file. SKIPPED by default.
# Opt in with RUN_AGENTIC_LIVE=1.
# ============================================================================


class TestSC1SC5LiveSubprocess:
    """Slow/live: confirm each harness launches headless and writes a file.

    Skipped unless RUN_AGENTIC_LIVE=1 — the default pytest run stays fast/offline.
    Each test is self-contained: it writes a PID-tagged scratch file and the
    assertion reads the FILESYSTEM, not the returned text (SC-1)."""

    @pytest.fixture(autouse=True)
    def _require_live(self):
        if not _RUN_LIVE:
            pytest.skip("live subprocess test; set RUN_AGENTIC_LIVE=1 to run")

    @pytest.mark.asyncio
    async def test_glm_agentic_writes_file(self, tmp_path):
        probe = tmp_path / f"rt_probe_{os.getpid()}.txt"
        await route(
            Complexity.CODE,
            f"create {probe} containing OK",
            config=Config(glm_key="live", codex_enabled=True),
            mode="agentic",
        )
        assert probe.exists(), "cc-glm agentic worker did not write the file"
        assert probe.read_text().strip() == "OK"
