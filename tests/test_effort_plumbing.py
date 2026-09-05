"""C06: a requested reasoning effort reaches each provider's OWN parameter, and an
absent one changes nothing.

One assertion per adapter that the value lands where that provider reads it --
five providers, five assertions, not one. And for every adapter, the harder
property: with effort=None the constructed request equals the pre-change request,
compared as serialised bodies rather than by inspection (C06 STOP: a default that
was ADDED rather than a parameter THREADED would change every call in the rig).

kimi is asserted the other way round. Its CLI has no effort control at all, so
the honest behaviour is: accept the parameter, apply nothing, and let the router
report effort_applied=False. Silently dropping it would make an unverifiable
claim look verified (C05 STOP).
"""
from __future__ import annotations

import json
import subprocess
from types import SimpleNamespace

import httpx
import pytest

from mcp_brain_router import backends


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
        # Anthropic Messages shape -- both HTTP adapters post to a /v1/messages
        # endpoint and _extract_text reads content blocks, not OpenAI choices.
        return httpx.Response(200, request=req, json={
            "model": "x", "content": [{"type": "text", "text": "ok"}],
            "usage": {"input_tokens": 1, "output_tokens": 1}})


# One stdout that satisfies every agentic family's parser at once: the Claude-Code
# family reads `result`; grok reads `text` + snake_case `stopReason` and refuses
# num_turns < 2 as "no tool effect".
CC_JSON = json.dumps({"result": "ok", "text": "ok", "stopReason": "end_turn", "num_turns": 2,
                      "usage": {"input_tokens": 1, "output_tokens": 1}})


# ---------------------------------------------------------------- the five, one each

@pytest.mark.asyncio
async def test_deepseek_payload_carries_effort(monkeypatch):
    post = _Post(); monkeypatch.setattr(httpx.AsyncClient, "post", post)
    await backends.call_deepseek("hi", "deepseek-chat", api_key="k", effort="high") \
        if "api_key" in backends.call_deepseek.__code__.co_varnames else \
        await backends.call_deepseek("hi", "deepseek-chat", "k", effort="high")
    assert post.bodies[-1]["thinking"] == {"type": "enabled"}


@pytest.mark.asyncio
async def test_glm_payload_carries_effort(monkeypatch):
    post = _Post(); monkeypatch.setattr(httpx.AsyncClient, "post", post)
    if "api_key" in backends.call_glm.__code__.co_varnames:
        await backends.call_glm("hi", "glm-5.3", api_key="k", effort="medium")
    else:
        await backends.call_glm("hi", "glm-5.3", "k", effort="medium")
    assert post.bodies[-1]["thinking"] == {"type": "enabled"}


@pytest.mark.asyncio
async def test_glm_low_effort_disables_thinking(monkeypatch):
    post = _Post(); monkeypatch.setattr(httpx.AsyncClient, "post", post)
    await backends.call_glm("hi", "glm-5.3", "k", effort="low")
    assert post.bodies[-1]["thinking"] == {"type": "disabled"}


def test_codex_argv_carries_effort(monkeypatch):
    run = _Run(); monkeypatch.setattr(backends.subprocess, "run", run)
    backends.call_codex("hi", "gpt-5.6-sol", effort="xhigh")
    assert 'model_reasoning_effort="xhigh"' in run.calls[-1]
    assert 'model_reasoning_effort="low"' not in run.calls[-1]


def test_grok_argv_carries_effort(monkeypatch):
    run = _Run(); monkeypatch.setattr(backends.subprocess, "run", run)
    backends.call_grok("hi", "grok-4.5", effort="high")
    argv = run.calls[-1]
    assert argv[argv.index("--reasoning-effort") + 1] == "high"


def test_kimi_accepts_effort_and_applies_nothing(monkeypatch):
    """kimi --help lists no effort control. The parameter is accepted so the router
    can thread uniformly; the argv must be identical with and without it; and the
    provider is in EFFORT_UNSUPPORTED so the router reports effort_applied=False."""
    run = _Run(); monkeypatch.setattr(backends.subprocess, "run", run)
    backends.call_kimi("hi", "kimi")
    backends.call_kimi("hi", "kimi", effort="high")
    assert run.calls[0] == run.calls[1]
    assert "kimi" in backends.EFFORT_UNSUPPORTED


# ---------------------------------------------------------------- agentic path (what roles use)

def test_agentic_adapters_carry_effort(monkeypatch):
    run = _Run(stdout=CC_JSON); monkeypatch.setattr(backends.subprocess, "run", run)
    monkeypatch.setattr(backends, "_resolve_agentic_cwd", lambda cwd, _p: "/tmp")
    backends.call_glm_agentic("hi", "glm-5.3", cwd="/tmp", effort="high")
    glm = run.calls[-1]; assert glm[glm.index("--effort") + 1] == "high"
    backends.call_grok_agentic("hi", "grok-4.5", cwd="/tmp", effort="low")
    grok = run.calls[-1]; assert grok[grok.index("--reasoning-effort") + 1] == "low"
    backends.call_codex_agentic("hi", "gpt-5.6-terra", cwd="/tmp", effort="medium")
    assert 'model_reasoning_effort="medium"' in run.calls[-1]


# ---------------------------------------------------------------- effort=None changes NOTHING

def _norm(argv: list[str]) -> list[str]:
    """Mask the one element that legitimately differs between two identical calls:
    grok's --prompt-file is a fresh temp path each time."""
    out = list(argv)
    if "--prompt-file" in out:
        out[out.index("--prompt-file") + 1] = "<tmp>"
    return out


@pytest.mark.parametrize("fn,model", [
    ("call_codex", "gpt-5.6-sol"), ("call_grok", "grok-4.5"), ("call_kimi", "kimi"),
])
def test_chat_cli_argv_identical_without_effort(monkeypatch, fn, model):
    run = _Run(); monkeypatch.setattr(backends.subprocess, "run", run)
    f = getattr(backends, fn)
    f("hi", model)
    f("hi", model, effort=None)
    assert _norm(run.calls[0]) == _norm(run.calls[1])


@pytest.mark.parametrize("fn,model", [
    ("call_glm_agentic", "glm-5.3"), ("call_grok_agentic", "grok-4.5"),
    ("call_kimi_agentic", "kimi"), ("call_codex_agentic", "gpt-5.6-terra"),
])
def test_agentic_argv_identical_without_effort(monkeypatch, fn, model):
    run = _Run(stdout=CC_JSON); monkeypatch.setattr(backends.subprocess, "run", run)
    monkeypatch.setattr(backends, "_resolve_agentic_cwd", lambda cwd, _p: "/tmp")
    f = getattr(backends, fn)
    f("hi", model, cwd="/tmp")
    f("hi", model, cwd="/tmp", effort=None)
    assert _norm(run.calls[0]) == _norm(run.calls[1])


@pytest.mark.asyncio
async def test_http_payload_identical_without_effort(monkeypatch):
    post = _Post(); monkeypatch.setattr(httpx.AsyncClient, "post", post)
    kw = {"api_key": "k"} if "api_key" in backends.call_glm.__code__.co_varnames else {}
    args = () if kw else ("k",)
    await backends.call_glm("hi", "glm-5.3", *args, **kw)
    await backends.call_glm("hi", "glm-5.3", *args, effort=None, **kw)
    assert json.dumps(post.bodies[0], sort_keys=True) == json.dumps(post.bodies[1], sort_keys=True)
    assert "thinking" not in post.bodies[0]


def test_codex_constants_are_the_effort_none_argv():
    """install.py's smoke test shares these constants; they must equal the builder
    at effort=None or the smoke test and the live call could drift apart."""
    assert backends.CODEX_EXEC_BASE == backends._codex_exec_base(agentic=False)
    assert backends.CODEX_EXEC_BASE_AGENTIC == backends._codex_exec_base(agentic=True)
    assert 'model_reasoning_effort="low"' in backends.CODEX_EXEC_BASE


# ---------------------------------------------------------------- refuter finding, 2026-09-05

@pytest.mark.asyncio
async def test_effort_reaches_glm_through_the_role_path(monkeypatch):
    """The fatal one: route_assignment -> route() -> _route_agentic dropped effort for
    every provider that goes through the generic branch -- which is where the live
    worker cascade lands (glm). Only the adapter unit tests had covered glm."""
    from mcp_brain_router import router as r
    from mcp_brain_router.config import Config
    seen = {}
    def fake_glm(prompt, model, cwd=None, effort=None):
        seen["effort"] = effort
        return {"content": "ok", "usage": {"input_tokens": 1, "output_tokens": 1}}
    monkeypatch.setattr(backends, "call_glm_agentic", fake_glm)
    monkeypatch.setattr(r, "_validate_credentials", lambda *a, **k: None, raising=False)
    cfg = Config(roles={"worker": ["glm-5.3"]}, glm_key="k")
    a = r.resolve_role(r.Role.WORKER, "claude", cfg, mode="agentic")
    await r.route_assignment(a, "hi", cfg, mode="agentic", cwd="/tmp", effort="high")
    assert seen.get("effort") == "high"
