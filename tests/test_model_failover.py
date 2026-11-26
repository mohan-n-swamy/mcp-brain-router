"""A failed model never rejects its provider; a dead entry never eats the deadline.

Backends are faked at the CLI edge (no network, no real CLI), so the real role walk,
route_assignment and ping cache run. Each fake records the model and prompt it got.
"""
from __future__ import annotations

import time
from unittest.mock import patch

import pytest

from mcp_brain_router import backends, ping, server
from mcp_brain_router.backends import BackendError, BackendQuotaError, BackendTransientError
from mcp_brain_router.config import DEFAULT_ROLES, Config

pytestmark = pytest.mark.asyncio


@pytest.fixture(autouse=True)
def _agy_present(monkeypatch):
    monkeypatch.setattr(backends, "_AGY_BIN", "/bin/sh")


def _cfg() -> Config:
    return Config(grok_enabled=True, fm_host="fm-host", ollama_host="http://127.0.0.1:11435",
                  roles={r: list(o) for r, o in DEFAULT_ROLES.items()})


async def _walk(role, orchestrator="grok"):
    with patch.object(server, "_load_config", return_value=_cfg()):
        return await server._delegate_role_impl(role, "the task", orchestrator, cwd="/tmp")


def _is_ping(prompt: str) -> bool:
    return ping.PROMPT in prompt


class Fakes:
    """Per-model behaviour: a value answers, an exception is raised. Calls are recorded
    as (model, 'ping' | 'real')."""

    def __init__(self, behaviour: dict):
        self.behaviour, self.calls = behaviour, []

    def _go(self, prompt, model):
        self.calls.append((model, "ping" if _is_ping(prompt) else "real"))
        b = self.behaviour.get(model, "answer")
        if callable(b):
            b = b(prompt)
        if isinstance(b, Exception):
            raise b
        return {"content": b if isinstance(b, str) and b != "answer" else "answer", "usage": None}

    def install(self):
        return (patch.object(backends, "call_antigravity", lambda p, m, effort=None: self._go(p, m)),
                patch.object(backends, "call_anthropic_agentic", lambda p, m, cwd=None, effort=None: self._go(p, m)),
                patch.object(backends, "call_grok", lambda p, m, effort=None: self._go(p, m)))


async def _run(fakes: Fakes, role, **kw):
    a, b, c = fakes.install()
    with a, b, c:
        return await _walk(role, **kw)


# agy-Claude sits between Fable and Opus in writer; the sibling tests take it out.
AGYC = "agy-claude-opus-5-5-high"


def _agyc_down():
    return BackendError("down", backend="antigravity", failure_kind="process_error")


def _agy_timeout():
    return BackendTransientError("antigravity", "timed out", failure_kind="timeout", elapsed_ms=90_000)


async def test_transient_keeps_sibling():
    """A timeout on one Antigravity model still reaches the next Antigravity model."""
    f = Fakes({"gemini-4-hydrogen-high": _agy_timeout()})
    out = await _run(f, "processor")
    assert out["model"] == "gemini-4-neon-medium" and out["provider"] == "antigravity"
    assert [m for m, _ in f.calls] == ["gemini-4-hydrogen-high", "gemini-4-neon-medium", "gemini-4-neon-medium"]
    assert [k for _, k in f.calls] == ["real", "ping", "real"]


async def test_quota_sibling_ping_ok_runs_the_sibling():
    f = Fakes({"claude-fable-5-2": BackendQuotaError("anthropic", 429, "out", reset_at="2026-11-26T00:00:00Z"),
               AGYC: _agyc_down()})
    out = await _run(f, "writer")
    assert out["model"] == "claude-opus-5-5"
    assert f.calls == [("claude-fable-5-2", "real"), (AGYC, "real"),
                       ("claude-opus-5-5", "ping"), ("claude-opus-5-5", "real")]


async def test_quota_sibling_ping_fails_skips_the_sibling():
    def opus(prompt):
        if _is_ping(prompt):
            return BackendError("no answer", backend="anthropic-cli", failure_kind="process_error")
        return "answer"

    f = Fakes({"claude-fable-5-2": BackendQuotaError("anthropic", 429, "out"), "claude-opus-5-5": opus,
               AGYC: _agyc_down()})
    out = await _run(f, "writer")
    assert out["model"] == "gemini-4-carbon-high"
    assert ("claude-opus-5-5", "real") not in f.calls
    assert "anthropic-cli:ping_failed" in out["failures"]


async def test_quota_reset_at_is_still_reported():
    f = Fakes({m: BackendQuotaError("anthropic", 429, "out", reset_at="2026-11-26T00:00:00Z")
               for m in ("claude-fable-5-2", "claude-opus-5-5")})
    f.behaviour["agy-claude-opus-5-5-high"] = BackendQuotaError("antigravity", 429, "out")
    f.behaviour["gemini-4-carbon-high"] = BackendQuotaError("antigravity", 429, "out")
    out = await _run(f, "writer")
    assert out["failure_kind"] == "quota_exhausted" and out["reset_at"]


async def test_ping_cached_fresh_ok_skips_the_ping_and_fresh_down_skips_the_model():
    f = Fakes({"claude-fable-5-2": BackendQuotaError("anthropic", 429, "out"), AGYC: _agyc_down()})
    await _run(f, "writer")  # pings Opus, which answers: cached ok
    assert ("claude-opus-5-5", "ping") in f.calls
    f.calls.clear()
    await _run(f, "writer")  # Opus is fresh ok: no ping
    assert f.calls == [("claude-fable-5-2", "real"), (AGYC, "real"), ("claude-opus-5-5", "real")]

    ping.record("claude-opus-5-5", False)  # a fresh "down": skipped without a call
    f.calls.clear()
    out = await _run(f, "writer")
    assert all(m != "claude-opus-5-5" for m, _ in f.calls) and out["model"] == "gemini-4-carbon-high"


async def test_ping_expires(monkeypatch):
    t0 = 1_000_000.0
    monkeypatch.setattr(ping, "now", lambda: t0)
    ping.record("claude-opus-5-5", False)
    assert ping.fresh("claude-opus-5-5") == "down"
    monkeypatch.setattr(ping, "now", lambda: t0 + ping.TTL_S + 1)
    assert ping.fresh("claude-opus-5-5") is None
    f = Fakes({"claude-fable-5-2": BackendQuotaError("anthropic", 429, "out"), AGYC: _agyc_down()})
    out = await _run(f, "writer")  # the stale "down" is ignored: Opus is pinged and runs
    assert out["model"] == "claude-opus-5-5" and ("claude-opus-5-5", "ping") in f.calls


def test_carbon_ping_gets_the_longer_cap():
    assert ping.cap_for("gemini-4-carbon-high") == 45 and ping.cap_for("gemini-4-neon-high") == 30


async def test_last_entry_reserved(monkeypatch):
    """2026-11-24 21:33Z: apple-fm timeout, ollama unreachable, agy 95 s, deadline gone,
    Haiku never tried. Now Haiku still gets at least its reserve inside the 120 s."""
    real, offset, budgets = time.monotonic, [0.0], {}
    monkeypatch.setattr(time, "monotonic", lambda: real() + offset[0])

    def spend(name, burn=True):
        budgets[name] = backends._timeout(10_000)
        if burn:
            offset[0] += budgets[name]  # the entry hangs until its allowance runs out

    def fm(prompt, host):
        spend("fm")
        raise BackendTransientError("apple-fm", "timed out", failure_kind="timeout")

    async def ollama(prompt, host, model):
        spend("ollama", burn=False)
        raise BackendError("unreachable", backend="ollama", failure_kind="unreachable")

    def agy(prompt, model, effort=None):
        spend("agy")
        raise BackendTransientError("antigravity", "timed out", failure_kind="timeout")

    def claude(prompt, model, cwd=None, effort=None):
        spend("haiku", burn=False)
        return {"content": "answer", "usage": None}

    with (patch.object(backends, "call_fm", fm), patch.object(backends, "call_ollama", ollama),
          patch.object(backends, "call_antigravity", agy),
          patch.object(backends, "call_anthropic_agentic", claude)):
        out = await _walk("simple", orchestrator="claude")
    assert out["model"] == "claude-haiku-5-5", out
    assert budgets["fm"] <= ping.LOCAL_ENTRY_CAP_S + 0.5
    assert budgets["haiku"] >= ping.RESERVE_S - 1


def test_slow_role_call_cap_outlasts_the_cli_cap():
    """A 30-minute role raises the per-call cap; a deadline still bounds it."""
    assert backends._timeout(backends.CODEX_TIMEOUT_SECONDS) == backends.CODEX_TIMEOUT_SECONDS
    tok = backends.CALL_CAP.set(900)
    try:
        assert backends._timeout(backends.CODEX_TIMEOUT_SECONDS) == 900
        d = backends.DEADLINE.set(time.monotonic() + 60)
        try:
            assert backends._timeout(backends.CODEX_TIMEOUT_SECONDS) <= 60
        finally:
            backends.DEADLINE.reset(d)
    finally:
        backends.CALL_CAP.reset(tok)


async def test_fast_failure_retries_the_model_without_pinging_itself():
    """A fast process error retries the same model once, with no ping in between."""
    n = {"c": 0}

    def flaky(prompt):
        n["c"] += 1
        if n["c"] == 1:
            return BackendError("blip", backend="antigravity", failure_kind="process_error", elapsed_ms=200)
        return "answer"

    f = Fakes({"gemini-4-hydrogen-high": flaky})
    out = await _run(f, "processor")
    assert out["model"] == "gemini-4-hydrogen-high"
    assert f.calls == [("gemini-4-hydrogen-high", "real"), ("gemini-4-hydrogen-high", "real")]


async def test_a_non_outage_failure_does_not_mark_the_model_down():
    """A bad argument or a prompt-specific error says nothing about the model being up."""
    f = Fakes({"gemini-4-hydrogen-high": BackendError("bad", backend="antigravity",
                                                      failure_kind="validation_error", elapsed_ms=90_000)})
    out = await _run(f, "processor")
    assert out["model"] == "gemini-4-neon-medium"
    assert ping.fresh("gemini-4-hydrogen-high") is None
    assert ("gemini-4-neon-medium", "ping") not in f.calls  # the provider was never suspect


async def test_a_call_through_mcp_gets_a_budget_under_the_client_limit():
    """The client stops waiting at 600 s; a 30-minute role must not outlive it."""
    seen = {}

    def slow(prompt):
        seen["left"] = backends.DEADLINE.get() - time.monotonic()
        return "answer"

    f = Fakes({"claude-fable-5-2": slow})
    out = await _run(f, "writer")
    assert out["model"] == "claude-fable-5-2" and 0 < seen["left"] <= 540


async def test_the_shell_entry_lifts_the_budget():
    seen = {}

    def slow(prompt):
        seen["left"] = backends.DEADLINE.get() - time.monotonic()
        return "answer"

    tok = server.CALL_BUDGET_S.set(None)
    try:
        await _run(Fakes({"claude-fable-5-2": slow}), "writer")
    finally:
        server.CALL_BUDGET_S.reset(tok)
    assert seen["left"] > 600
