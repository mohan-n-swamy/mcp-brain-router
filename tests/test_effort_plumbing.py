"""C06: a requested reasoning effort reaches each provider's OWN parameter, and an
absent one changes nothing.

One assertion per adapter that the value lands where that provider reads it.
And for every adapter, the harder property: with effort=None the constructed
request equals the pre-change request, compared as serialised bodies rather than
by inspection (C06 STOP: a default that was ADDED rather than a parameter
THREADED would change every call in the rig).

kimi is asserted the other way round. It is refused before any spawn (no
verified tools-off mode, lean-rig §6), and it is in EFFORT_UNSUPPORTED so the
router reports effort_applied=False rather than an unverifiable claim (C05 STOP).
"""
from __future__ import annotations

import json
from types import SimpleNamespace

import httpx
import pytest

from mcp_brain_router import backends
from mcp_brain_router.backends import BackendError


# ---------------------------------------------------------------- capture helpers

class _Run:
    """Captures argv passed to subprocess.run and returns a benign result."""
    def __init__(self, stdout="ok"):
        self.calls: list[list[str]] = []
        self.stdout = stdout
    def __call__(self, argv, **kw):
        self.calls.append(list(argv))
        return SimpleNamespace(returncode=0, stdout=self.stdout, stderr="")


class _Post:
    """Captures the json= body of httpx.AsyncClient.post."""
    def __init__(self):
        self.bodies: list[dict] = []
    async def __call__(self, url, json=None, headers=None, **kw):
        self.bodies.append(json)
        req = httpx.Request("POST", url)
        return httpx.Response(200, request=req, json={
            "model": "x", "content": [{"type": "text", "text": "ok"}],
            "usage": {"input_tokens": 1, "output_tokens": 1}})


# A clean tools-off grok answer: one turn, end_turn, the final message in `text`.
GROK_JSON = json.dumps({"text": "ok", "stopReason": "end_turn", "num_turns": 1,
                        "usage": {"input_tokens": 1, "output_tokens": 1}})


# ---------------------------------------------------------------- one each

@pytest.mark.asyncio
async def test_deepseek_payload_carries_effort(monkeypatch):
    post = _Post(); monkeypatch.setattr(httpx.AsyncClient, "post", post)
    await backends.call_deepseek("hi", "deepseek-flash", "k", effort="high")
    assert post.bodies[-1]["thinking"] == {"type": "enabled"}


@pytest.mark.asyncio
async def test_deepseek_low_effort_disables_thinking(monkeypatch):
    post = _Post(); monkeypatch.setattr(httpx.AsyncClient, "post", post)
    await backends.call_deepseek("hi", "deepseek-flash", "k", effort="low")
    assert post.bodies[-1]["thinking"] == {"type": "disabled"}


def test_codex_argv_carries_effort(monkeypatch):
    run = _Run(); monkeypatch.setattr(backends.subprocess, "run", run)
    backends.call_codex("hi", "gpt-6.1-sol", effort="xhigh")
    assert 'model_reasoning_effort="xhigh"' in run.calls[-1]
    assert 'model_reasoning_effort="low"' not in run.calls[-1]


def test_grok_argv_carries_effort(monkeypatch):
    run = _Run(stdout=GROK_JSON); monkeypatch.setattr(backends.subprocess, "run", run)
    backends.call_grok("hi", "grok-4.5", effort="high")
    argv = run.calls[-1]
    assert argv[argv.index("--reasoning-effort") + 1] == "high"


def test_kimi_accepts_effort_and_applies_nothing(monkeypatch):
    run = _Run(); monkeypatch.setattr(backends.subprocess, "run", run)
    for effort in (None, "high"):
        with pytest.raises(BackendError):
            backends.call_kimi("hi", "kimi", effort=effort)
    assert run.calls == []
    assert "kimi" in backends.EFFORT_UNSUPPORTED


# ---------------------------------------------------------------- effort=None changes NOTHING

def _norm(argv: list[str]) -> list[str]:
    """Mask what legitimately differs between two identical calls: grok's
    --prompt-file and codex's -C are fresh temp paths each time."""
    out = list(argv)
    for flag in ("--prompt-file", "-C"):
        if flag in out:
            out[out.index(flag) + 1] = "<tmp>"
    return out


@pytest.mark.parametrize("fn,model,stdout", [
    ("call_codex", "gpt-6.1-sol", "ok"), ("call_grok", "grok-4.5", GROK_JSON),
])
def test_chat_cli_argv_identical_without_effort(monkeypatch, fn, model, stdout):
    run = _Run(stdout=stdout); monkeypatch.setattr(backends.subprocess, "run", run)
    f = getattr(backends, fn)
    f("hi", model)
    f("hi", model, effort=None)
    assert _norm(run.calls[0]) == _norm(run.calls[1])


@pytest.mark.asyncio
async def test_http_payload_identical_without_effort(monkeypatch):
    post = _Post(); monkeypatch.setattr(httpx.AsyncClient, "post", post)
    await backends.call_deepseek("hi", "deepseek-flash", "k")
    await backends.call_deepseek("hi", "deepseek-flash", "k", effort=None)
    assert json.dumps(post.bodies[0], sort_keys=True) == json.dumps(post.bodies[1], sort_keys=True)
    assert "thinking" not in post.bodies[0]


def test_codex_constants_are_the_effort_none_argv():
    """install.py's smoke test shares this constant; it must equal the builder at
    effort=None or the smoke test and the live call could drift apart."""
    assert backends.CODEX_EXEC_BASE == backends._codex_exec_base()
    assert 'model_reasoning_effort="low"' in backends.CODEX_EXEC_BASE


# ---------------------------------------------------------------- refuter finding, 2026-09-05

@pytest.mark.asyncio
async def test_effort_reaches_grok_through_the_role_path(monkeypatch):
    """route_assignment once dropped effort for providers on the generic branch.
    The worker walk's first live backend is grok, so pin it end to end."""
    from mcp_brain_router import router as r
    from mcp_brain_router.config import Config
    seen = {}
    def fake_grok(prompt, model, effort=None):
        seen["effort"] = effort
        return {"content": "ok", "usage": None}
    monkeypatch.setattr(backends, "call_grok", fake_grok)
    cfg = Config(roles={"worker": ["grok-4.5"]}, grok_enabled=True)
    a = r.resolve_role(r.Role.WORKER, "claude", cfg, mode="agentic")
    await r.route_assignment(a, "hi", cfg, mode="agentic", cwd="/tmp", effort="high")
    assert seen.get("effort") == "high"
