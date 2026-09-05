"""run_card_trial must dispatch a card as (model, effort), never as its slug."""
import asyncio
import importlib.util
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))


def _load():
    spec = importlib.util.spec_from_file_location("rb_card_test", REPO / "bin" / "run-bench.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class _R:
    content = "PONG"; exhausted = False; usage = {"input_tokens": 1, "output_tokens": 1}
    backend = "codex"; model = "gpt-5.6-luna"; failure_kind = None


def _run(monkeypatch, card):
    rb = _load()
    from mcp_brain_router import router
    seen = {}

    async def fake_route(complexity, prompt, model_override=None, config=None,
                         mode="chat", cwd=None, effort=None):
        seen.update(model=model_override, effort=effort); return _R()

    async def fake_agentic(backend, prompt, model, config, cwd=None, effort=None):
        seen.update(model=model, effort=effort, backend=backend); return _R()

    monkeypatch.setattr(router, "route", fake_route)
    monkeypatch.setattr(router, "_route_agentic", fake_agentic)
    monkeypatch.setattr(rb, "build_prompt", lambda f, cwd: "p")
    t = asyncio.run(rb.run_card_trial(card, {"id": "x"}, "/tmp"))
    return seen, t


def test_effort_variant_card_dispatches_base_model_plus_effort(monkeypatch):
    seen, t = _run(monkeypatch, {"slug": "gpt-5-6-luna-xhigh", "model": "gpt-5-6-luna",
                                 "effort": "xhigh", "provider": "openai"})
    assert seen == {"model": "gpt-5.6-luna", "effort": "xhigh"}
    assert t["ok"]


def test_base_effort_is_cli_default(monkeypatch):
    seen, _ = _run(monkeypatch, {"slug": "gpt-5-6-luna", "model": "gpt-5-6-luna",
                                 "effort": "base", "provider": "openai"})
    assert seen == {"model": "gpt-5.6-luna", "effort": None}


def test_non_reasoning_maps_to_low_on_direct_backend(monkeypatch):
    seen, _ = _run(monkeypatch, {"slug": "grok-4-5-non-reasoning", "model": "grok-4-5",
                                 "effort": "non-reasoning", "provider": "xai"})
    assert seen == {"model": "grok-4.5", "effort": "low", "backend": "grok"}


def test_failed_trial_keeps_error_text(monkeypatch):
    rb = _load()
    from mcp_brain_router import router

    async def boom(*a, **k):
        raise RuntimeError("model is not supported when using Codex")

    monkeypatch.setattr(router, "route", boom)
    monkeypatch.setattr(rb, "build_prompt", lambda f, cwd: "p")
    t = asyncio.run(rb.run_card_trial({"slug": "gpt-oss-120b", "model": "gpt-oss-120b",
                                       "effort": "base", "provider": "openai"}, {"id": "x"}, "/tmp"))
    assert t["ok"] is False
    assert "not supported" in t["error"]
