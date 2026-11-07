"""Delegation policy (lean-rig PLAN §6, invariant I22); scripts/policy-check.sh runs it.

Proves: the D5 role orders, the data class (orange reaches Claude only and
refuses), the preamble, one deadline per call, the walk past three failed
backends, gpt-oss's cold skip and lock, Grok's final message from a narrated
answer, Codex's closed stdin and the review sandbox's denials.

Backends are mocked at the CLI or HTTP edge, so the real route_assignment and
role walk run. The Grok payloads are trimmed copies of live grok 1.0.77 runs
(2026-11-06, tools-off flags, --output-format json).
"""
from __future__ import annotations

import asyncio
import fcntl
import json
import os
import subprocess
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from mcp_brain_router import backends, policy, router, server
from mcp_brain_router.backends import BackendError
from mcp_brain_router.config import DEFAULT_ROLES, Config

ROOT = Path(__file__).resolve().parents[1]
SANDBOX = ROOT / "scripts" / "sandbox.sh"

# Live grok 1.0.77: the narration lands in `thought`, the answer in `text`.
GROK_NARRATED = {"stopReason": "end_turn", "num_turns": 1, "text": "7",
                 "thought": "Counting the lines between the markers.",
                 "usage": {"input_tokens": 13819, "output_tokens": 100},
                 "total_cost_usd": 0.0305804}
# Live: a prompt naming a file. Narration, then a tool call cut by --max-turns;
# the CLI exits 1 with "Error: max turns reached".
GROK_CUT = {"stopReason": "max_turn_requests", "num_turns": 1, "text": "",
            "thought": "I will count the lines in DECISIONS.md and answer with only the number.",
            "usage": {"input_tokens": 21094, "output_tokens": 84},
            "total_cost_usd": 0.0428456}
# The 2026-11-06 finding: agentic text joined across turns reached the caller.
GROK_JOINED = {"stopReason": "end_turn", "num_turns": 2,
               "text": "I'll count the lines in DECISIONS.md now.15"}


@pytest.fixture(autouse=True)
def _gpt_oss_lock_in_tmp(tmp_path, monkeypatch):
    """A warm gpt-oss call takes the generation lock; never the live one."""
    monkeypatch.setattr(backends, "OLLAMA_LOCK", str(tmp_path / "gpt-oss.lock"))


def _cfg(**kw) -> Config:
    base = dict(deepseek_key="ds", codex_enabled=True, grok_enabled=True, kimi_enabled=True,
                fm_host="fm-host", ollama_host="http://127.0.0.1:11435",
                roles={role: list(order) for role, order in DEFAULT_ROLES.items()})
    base.update(kw)
    return Config(**base)


def _proc(stdout="", returncode=0, stderr=""):
    return MagicMock(returncode=returncode, stdout=stdout, stderr=stderr)


class Calls:
    """Records the prompt each backend received and the deadline it ran under."""

    def __init__(self):
        self.seen: dict[str, tuple[str, float | None]] = {}

    def note(self, name: str, prompt: str) -> None:
        self.seen[name] = (prompt, backends.DEADLINE.get())


async def _walk(cfg, role, *, orchestrator="claude", data_class=None, cwd="/tmp"):
    with patch.object(server, "_load_config", return_value=cfg):
        return await server._delegate_role_impl(role, "the task", orchestrator, cwd=cwd,
                                                data_class=data_class)


# ── Role orders (D5) ────────────────────────────────────────────────────────

@pytest.mark.parametrize("role,order", [
    ("simple", ["apple-fm", "gpt-oss:20b", "claude-haiku-5-5"]),
    ("worker", ["kimi", "grok-4.5", "deepseek-v4-pro", "claude-sonnet-5", "gpt-6.1-sol"]),
    ("adversary", ["gpt-6.1-sol", "claude-opus-5-5", "kimi"]),
])
def test_d5_role_order(role, order):
    assert DEFAULT_ROLES[role] == order


def test_host_roles_table_cannot_reorder(tmp_path, monkeypatch):
    from mcp_brain_router import config as config_mod
    monkeypatch.setattr(config_mod, "CONFIG_DIR", tmp_path)
    monkeypatch.setattr(config_mod, "CONFIG_FILE", tmp_path / "config.toml")
    (tmp_path / "config.toml").write_text('[roles]\nworker = ["claude-sonnet-5"]\n')
    (tmp_path / "config.toml").chmod(0o600)
    assert config_mod.Config.load().roles["worker"] == DEFAULT_ROLES["worker"]


def test_codex_default_model_is_sol():
    assert router._get_backend_default_model("codex", _cfg()) == "gpt-6.1-sol"


# ── Data class ──────────────────────────────────────────────────────────────

def test_resolve_class():
    assert policy.resolve_class(None, None) == "personal"
    assert policy.resolve_class(None, "orange") == "orange"
    assert policy.resolve_class(" Orange ", "personal") == "orange"
    with pytest.raises(ValueError):
        policy.resolve_class("work", None)


def _forbid(*names):
    """Patch these backend entry points to fail the test if the walk reaches them."""
    def boom(*a, **k):
        raise AssertionError("a non-Claude backend was called for class orange")
    return [patch.object(backends, n, AsyncMock(side_effect=boom) if n == "call_deepseek"
                         else MagicMock(side_effect=boom)) for n in names]


async def test_orange_reaches_claude_only():
    patches = _forbid("call_kimi", "call_grok", "call_deepseek", "call_codex", "call_fm")
    for p in patches:
        p.start()
    try:
        with patch.object(backends, "call_anthropic_agentic",
                          return_value={"content": "ok", "usage": None}):
            out = await _walk(_cfg(), "worker", data_class="orange")
    finally:
        for p in patches:
            p.stop()
    assert (out["backend"], out["model"], out["class"]) == ("anthropic-cli", "claude-sonnet-5", "orange")
    assert out["tried"] == ["anthropic-cli"]


async def test_orange_refuses_rather_than_falls_through():
    patches = _forbid("call_kimi", "call_grok", "call_deepseek", "call_codex")
    for p in patches:
        p.start()
    try:
        failing = BackendError("claude down", backend="anthropic-cli",
                               failure_kind="process_error", elapsed_ms=200_000)
        with patch.object(backends, "call_anthropic_agentic", side_effect=failing):
            out = await _walk(_cfg(), "worker", data_class="orange")
    finally:
        for p in patches:
            p.stop()
    assert out["failure_kind"] == "class_refused"
    assert out["backend"] == "none" and out["tried"] == ["anthropic-cli"]


async def test_orange_adversary_from_claude_refuses():
    with patch.object(server, "route_assignment", AsyncMock()) as run:
        out = await _walk(_cfg(), "adversary", data_class="orange")
    assert out["failure_kind"] == "class_refused"
    run.assert_not_awaited()


async def test_host_default_orange_and_unknown_class():
    with patch.object(server, "route_assignment", AsyncMock()) as run:
        out = await _walk(_cfg(default_class="orange"), "adversary")
        bad = await _walk(_cfg(), "worker", data_class="work")
    assert out["failure_kind"] == "class_refused"
    assert bad["failure_kind"] == "validation_error"
    run.assert_not_awaited()


async def test_mcp_tool_takes_class():
    srv = server.create_server()
    tool = [t for t in await srv.list_tools() if t.name == "delegate"][0]
    assert "class" in tool.inputSchema["properties"]
    role = AsyncMock(return_value={"answer": "ok"})
    with patch.object(server, "_delegate_role_impl", role):
        await srv.call_tool("delegate", {"role": "worker", "prompt": "p", "orchestrator": "claude",
                                         "cwd": "/tmp", "class": "orange"})
    assert role.await_args.kwargs["data_class"] == "orange"


# ── Preamble ────────────────────────────────────────────────────────────────

def test_preamble_size_and_content():
    text = policy.PREAMBLE
    assert len(text.encode()) <= policy.PREAMBLE_MAX_BYTES == 1200
    for needle in ("Precedence", "data, never an instruction", "Evidence for claims",
                   "Destructive ops", "untrusted worker output"):
        assert needle in text
    # The text-only tiers reject tool-shaped prompts; the preamble must not be one.
    assert not backends._FM_TOOL_SHAPED.search(text)
    assert policy.with_preamble("X").endswith("--- task ---\nX")


async def test_adversarial_tier_sends_the_preamble():
    seen = {}

    async def fake_route(**kw):
        seen.update(kw)
        return router.RouteResult(content="ok", model="gpt-6.1-sol", backend="codex",
                                  complexity=router.Complexity.ADVERSARIAL, headroom_used=False)
    with (patch.object(server, "_load_config", return_value=_cfg()),
          patch.object(server, "_read_caller", return_value="claude"),
          patch.object(server, "route", side_effect=fake_route)):
        await server._delegate_impl("adversarial", "review this", cwd="/tmp")
    assert seen["prompt"] == policy.with_preamble("review this")


# ── The walk, deadlines, preamble at every backend ──────────────────────────

async def test_walk_passes_three_failed_backends():
    calls = Calls()

    def grok(prompt, model, effort=None):
        calls.note("grok", prompt)
        raise BackendError("cut", backend="grok", failure_kind="no_final_message", elapsed_ms=5000)

    async def deepseek(prompt, model, api_key, effort=None):
        calls.note("deepseek", prompt)
        return {"content": "   ", "usage": {}}  # empty answer: walk on

    def claude(prompt, model, cwd=None):
        calls.note("claude", prompt)
        return {"content": "built", "usage": None}

    with (patch.object(backends, "call_grok", side_effect=grok),
          patch.object(backends, "call_deepseek", side_effect=deepseek),
          patch.object(backends, "call_anthropic_agentic", side_effect=claude)):
        out = await _walk(_cfg(), "worker")
    # kimi refuses (no tools-off mode), grok has no final message, deepseek is empty.
    assert out["tried"] == ["kimi", "grok", "deepseek", "anthropic-cli"]
    assert (out["backend"], out["model"], out["answer"]) == ("anthropic-cli", "claude-sonnet-5", "built")
    assert out["retried"] == []  # a cut Grok turn is not retried on Grok
    deadlines = {d for _, d in calls.seen.values()}
    assert len(deadlines) == 1 and None not in deadlines  # one deadline for the call
    for prompt, _ in calls.seen.values():
        assert prompt.startswith(policy.PREAMBLE)


async def test_missing_key_or_binary_is_a_free_skip():
    with patch.object(backends, "call_anthropic_agentic",
                      return_value={"content": "ok", "usage": None}):
        out = await _walk(_cfg(deepseek_key=None, grok_enabled=False, kimi_enabled=False), "worker")
    assert out["backend"] == "anthropic-cli"
    assert all(f.endswith("configuration_error") for f in out["failures"])


def test_deadlines():
    assert policy.deadline_seconds("simple") == 120
    assert policy.deadline_seconds("worker") == 15 * 60
    token = backends.DEADLINE.set(time.monotonic() + 10)
    try:
        assert 9 <= backends._timeout(360) <= 10
    finally:
        backends.DEADLINE.reset(token)
    assert backends._timeout(360) == 360


async def test_walk_stops_at_the_deadline():
    def slow_grok(prompt, model, effort=None):
        time.sleep(0.7)
        raise BackendError("slow", backend="grok", failure_kind="process_error",
                           elapsed_ms=200_000)  # not a fast flake: no same-provider retry

    deepseek = AsyncMock(return_value={"content": "late", "usage": {}})
    with (patch.object(server, "deadline_seconds", lambda role: 1.5),
          patch.object(backends, "call_grok", side_effect=slow_grok),
          patch.object(backends, "call_deepseek", deepseek)):
        out = await _walk(_cfg(), "worker")
    assert out["failure_kind"] == "deadline_exceeded"
    deepseek.assert_not_awaited()


# ── gpt-oss: cold skip, one generation at a time ────────────────────────────

def _ollama(handler):
    real = httpx.AsyncClient

    def factory(*a, **kw):
        kw["transport"] = httpx.MockTransport(handler)
        return real(*a, **kw)
    return patch.object(backends.httpx, "AsyncClient", factory)


def _ps(loaded):
    def handler(request):
        if request.url.path == "/api/ps":
            return httpx.Response(200, json={"models": [{"name": m} for m in loaded]})
        assert request.url.path == "/api/generate", request.url.path
        assert loaded, "a cold model must never be asked to generate"
        return httpx.Response(200, json={"done": True, "done_reason": "stop",
                                         "response": "label_b", "eval_count": 9})
    return handler


async def test_gpt_oss_cold_skips_without_waiting():
    started = time.monotonic()
    with _ollama(_ps([])), pytest.raises(BackendError) as err:
        await backends.call_ollama("Classify: late report", "http://127.0.0.1:11435", "gpt-oss:20b")
    assert err.value.failure_kind == "cold"
    assert time.monotonic() - started < 1


async def test_cold_gpt_oss_means_haiku():
    fm_down = BackendError("mini down", backend="apple-fm", failure_kind="unreachable", elapsed_ms=5)
    with (_ollama(_ps([])), patch.object(backends, "call_fm", side_effect=fm_down),
          patch.object(backends, "call_anthropic_agentic",
                       return_value={"content": "label_b", "usage": None})):
        out = await _walk(_cfg(), "simple")
    assert out["tried"] == ["apple-fm", "ollama", "anthropic-cli"]
    assert out["model"] == "claude-haiku-5-5"


async def test_gpt_oss_warm_answers():
    with _ollama(_ps(["gpt-oss:20b"])):
        out = await backends.call_ollama("Classify: late report", "http://127.0.0.1:11435",
                                         "gpt-oss:20b")
    assert out["content"] == "label_b"


async def test_gpt_oss_lock_second_caller_skips_after_5s():
    assert backends.OLLAMA_LOCK_WAIT_SECONDS == 5 and backends.OLLAMA_TIMEOUT_SECONDS == 60
    lock = backends.OLLAMA_LOCK  # the autouse fixture's tmp path
    paths = []

    def handler(request):
        paths.append(request.url.path)
        return httpx.Response(200, json={"models": [{"name": "gpt-oss:20b"}]})
    fd = os.open(lock, os.O_RDWR | os.O_CREAT, 0o600)
    fcntl.flock(fd, fcntl.LOCK_EX)  # the first caller's generation
    try:
        started = time.monotonic()
        with _ollama(handler), pytest.raises(BackendError) as err:
            await backends.call_ollama("Classify: x", "http://127.0.0.1:11435", "gpt-oss:20b")
        waited = time.monotonic() - started
    finally:
        os.close(fd)
    assert err.value.failure_kind == "busy"
    assert 4.9 <= waited < 7
    assert paths == ["/api/ps"]  # no generation while another holds the lock


def test_gpt_oss_serves_simple_only():
    cfg = _cfg(roles={"worker": ["gpt-oss:20b", "claude-sonnet-5"]})
    assignment = router.resolve_role(router.Role.WORKER, "claude", cfg, mode="agentic")
    assert assignment.model == "claude-sonnet-5"


# ── Grok: tools-off, final message only ─────────────────────────────────────

def _grok(stdout, returncode=0, stderr=""):
    run = patch.object(backends.subprocess, "run", return_value=_proc(stdout, returncode, stderr))
    home = patch.object(backends, "_ensure_grok_lean_home", return_value="/nonexistent-grok-home")
    return run, home


def test_grok_returns_the_final_message_not_the_narration():
    run, home = _grok(json.dumps(GROK_NARRATED))
    with run as mrun, home:
        out = backends.call_grok("Count the lines.", "grok-4.5")
    assert out["content"] == "7"
    argv, kw = mrun.call_args.args[0], mrun.call_args.kwargs
    for flag in ("--max-turns", "1", "--no-subagents", "--no-plan", "--disable-web-search",
                 "--prompt-file", "--output-format", "json"):
        assert flag in argv
    assert kw["stdin"] is subprocess.DEVNULL
    assert kw["env"]["GROK_HOME"] == "/nonexistent-grok-home"


def test_grok_cut_turn_is_no_final_message():
    run, home = _grok(json.dumps(GROK_CUT), returncode=1, stderr="Error: max turns reached")
    with run, home, pytest.raises(BackendError) as err:
        backends.call_grok("Count the lines in DECISIONS.md.", "grok-4.5")
    assert err.value.failure_kind == "no_final_message"
    assert "I will count" not in str(err.value)


def test_grok_joined_narration_never_returns():
    with pytest.raises(BackendError) as err:
        backends.grok_final_message(json.dumps(GROK_JOINED))
    assert err.value.failure_kind == "no_final_message"


# ── Codex: tools-off, stdin closed ──────────────────────────────────────────

def test_codex_prompt_on_stdin_tools_off():
    with patch.object(backends.subprocess, "run", return_value=_proc("ok")) as mrun:
        backends.call_codex("review this", "gpt-6.1-sol")
    argv, kw = mrun.call_args.args[0], mrun.call_args.kwargs
    assert argv[-1] == "-" and "review this" in kw["input"] and "stdin" not in kw
    joined = " ".join(argv)
    for part in ("--sandbox read-only", "--disable shell_tool",
                 "--disable external_agent_memory_import", "-m gpt-6.1-sol"):
        assert part in joined
    assert os.path.basename(argv[argv.index("-C") + 1]).startswith("brain-router-codex-")


# ── Review sandbox denials ──────────────────────────────────────────────────

BWRAP = os.environ.get("BRAIN_ROUTER_BWRAP", "/usr/bin/bwrap")
needs_bwrap = pytest.mark.skipif(not os.access(BWRAP, os.X_OK),
                                 reason=f"no bwrap at {BWRAP} (gate 2)")


def _sandbox(*args, env=None):
    full = dict(os.environ, **(env or {}))
    return subprocess.run(["bash", str(SANDBOX), *args], capture_output=True, text=True,
                          env=full, timeout=60, stdin=subprocess.DEVNULL)


@pytest.fixture
def tree(tmp_path):
    (tmp_path / "wt").mkdir()
    (tmp_path / "wt" / "f").write_text("inside\n")
    (tmp_path / "outside").mkdir()
    (tmp_path / "outside" / "canary").write_text("secret\n")
    return tmp_path


@needs_bwrap
def test_sandbox_review_denials(tree):
    probe = (f"cat f; touch g 2>/dev/null && echo WROTE; test -e {tree}/outside/canary && echo OUT; "
             f"test -e {Path.home()}/.ssh && echo SSH; echo key=${{OPENBRAIN_KEY:-unset}}")
    out = _sandbox("--review", "--class", "personal", str(tree / "wt"), "--", "/bin/sh", "-c", probe,
                   env={"OPENBRAIN_KEY": "canary-key"})
    assert out.returncode == 0, out.stderr
    assert out.stdout.split() == ["inside", "key=unset"]
    assert not (tree / "wt" / "g").exists()


@needs_bwrap
def test_sandbox_worker_writes_only_its_worktree(tree):
    out = _sandbox("--class", "personal", str(tree / "wt"), "--", "/bin/sh", "-c",
                   "echo x > g && echo ok; touch /usr/zz 2>/dev/null && echo USR; exit 0")
    assert out.returncode == 0, out.stderr
    assert out.stdout.split() == ["ok"] and (tree / "wt" / "g").exists()


@needs_bwrap
def test_sandbox_refusals(tree):
    home = tree / "home"
    (home / ".config" / "mcp-brain-router").mkdir(parents=True)
    (home / ".config" / "mcp-brain-router" / "config.toml").write_text('default_class = "orange"\n')
    wt = str(tree / "wt")
    assert _sandbox("--class", "orange", wt, "--", "/bin/true").returncode == 3
    assert _sandbox(wt, "--", "/bin/true", env={"HOME": str(home)}).returncode == 3
    assert _sandbox("--class", "personal", str(Path.home()), "--", "/bin/true").returncode == 3
    assert _sandbox("--class", "personal", wt, "--", "/bin/true",
                    env={"BRAIN_ROUTER_BWRAP": "/nonexistent"}).returncode == 2
