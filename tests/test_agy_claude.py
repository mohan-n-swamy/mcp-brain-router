"""Claude served through Antigravity (`agy-claude-*`, Mohan 2026-11-26): it bills the
Google AI Ultra quota, never reviews a Claude caller (A28) and never takes class orange."""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from mcp_brain_router.backends import BackendError, call_antigravity
from mcp_brain_router.config import DEFAULT_ROLES, Config, Slot
from mcp_brain_router.router import Provider, Role, provider_for_model, resolve_role

OK = {"status": "SUCCESS", "response": "ok\n", "usage": {"input_tokens": 1, "output_tokens": 1}}


@pytest.mark.parametrize("model", ["agy-claude-opus-5-5-high", "agy-claude-sonnet-5-5-medium"])
def test_agy_claude_bills_antigravity_in_both_maps(model):
    assert provider_for_model(model) is Provider.ANTIGRAVITY
    assert Slot(model.rsplit("-", 1)[0], "high").provider == "antigravity"


def test_slot_id_carries_level_suffix():
    assert Slot("agy-claude-sonnet-5-5", "high").id == "agy-claude-sonnet-5-5-high"


def test_prefix_is_stripped_before_agy_sees_it():
    proc = MagicMock(returncode=0, stdout=json.dumps(OK), stderr="")
    with patch("mcp_brain_router.backends.subprocess.run", return_value=proc) as m:
        out = call_antigravity("hi", "agy-claude-sonnet-5-5-high")
    argv = m.call_args.args[0]
    assert argv[argv.index("--model") + 1] == "claude-sonnet-5-5-high"
    assert out["content"] == "ok"


@pytest.mark.parametrize("model", ["agy-gpt-oss-120b", "agy-", "agy-claude", "agy-claude-x;rm"])
def test_other_agy_ids_are_refused(model):
    with patch("mcp_brain_router.backends.subprocess.run") as m, pytest.raises(BackendError) as e:
        call_antigravity("hi", model)
    assert e.value.failure_kind == "validation_error"
    m.assert_not_called()


def test_no_agy_claude_in_adversary():
    assert not [m for m in DEFAULT_ROLES["adversary"] if m.startswith("agy-")]


def test_adversary_skips_agy_claude_for_claude_caller_even_if_listed():
    cfg = Config(roles={"adversary": ["agy-claude-opus-5-5-high", "gemini-4-neon-xhigh"]})
    a = resolve_role(Role.ADVERSARY, "claude", cfg, mode="agentic")
    assert a.model == "gemini-4-neon-xhigh"


def test_adversary_may_use_agy_claude_for_grok_caller():
    cfg = Config(roles={"adversary": ["agy-claude-opus-5-5-high", "gemini-4-neon-xhigh"]})
    a = resolve_role(Role.ADVERSARY, "grok", cfg, mode="agentic")
    assert a.model == "agy-claude-opus-5-5-high"


def test_class_orange_never_reaches_agy_claude():
    cfg = Config(roles={"coder": ["agy-claude-sonnet-5-5-high", "claude-sonnet-5-5"]})
    a = resolve_role(Role("coder"), "claude", cfg, mode="agentic", claude_only=True)
    assert a.provider is Provider.ANTHROPIC and a.model == "claude-sonnet-5-5"


def test_coder_leads_with_agy_claude():
    a = resolve_role(Role("coder"), "claude", Config(roles=dict(DEFAULT_ROLES)), mode="agentic")
    assert a.model == "agy-claude-sonnet-5-5-high" and a.backend == "antigravity"



def test_only_agy_claude_takes_the_agy_rule():
    """Finding 4 (cross-provider review): a stray agy-* id is not claimed by the agy-claude rule."""
    assert provider_for_model("agy-gpt-oss-120b") is not Provider.ANTIGRAVITY
