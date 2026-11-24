"""One ROLES table (config.ROLES) is the only definition of the roles. What a caller
reads (tool description, refusal, README block) is generated from it, `worker` still
resolves as the coder, a call with no role is refused with the table, and every role
has a candidate for every known caller."""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from mcp_brain_router import server
from mcp_brain_router.config import DEFAULT_ROLES, ROLES, Config, render_roles
from mcp_brain_router.policy import deadline_seconds
from mcp_brain_router.router import Role, resolve_role

README = Path(__file__).resolve().parents[1] / "README.md"


def _cfg() -> Config:
    return Config(grok_enabled=True, fm_host="h", ollama_host="http://x",
                  roles={r: list(o) for r, o in DEFAULT_ROLES.items()})


async def _delegate_tool():
    return next(t for t in await server.create_server().list_tools() if t.name == "delegate")


async def test_description_names_every_role_and_job():
    text = (await _delegate_tool()).description
    for r in ROLES.values():
        assert r.name in text and r.job in text, r.name


def test_worker_alias_is_the_coder():
    cfg = _cfg()
    assert DEFAULT_ROLES["worker"] == DEFAULT_ROLES["coder"]
    assert deadline_seconds("worker") == deadline_seconds("coder")
    for caller in ("claude", "grok", "antigravity"):
        w = resolve_role(Role.WORKER, caller, cfg, mode="agentic")
        c = resolve_role(Role.CODER, caller, cfg, mode="agentic")
        assert w.model == c.model


async def test_missing_role_is_refused_with_the_table():
    out = await server.create_server().call_tool("delegate", {"prompt": "p"})
    out = out[1] if isinstance(out, tuple) else out
    assert out["failure_kind"] == "validation_error"
    for name in ROLES:
        assert name in out["error"]


async def test_unknown_role_is_refused_with_the_table():
    out = await server._delegate_role_impl("planner", "p", "claude", cwd="/tmp")
    assert out["failure_kind"] == "validation_error" and "thinker" in out["error"]


@pytest.mark.parametrize("role", sorted(ROLES))
@pytest.mark.parametrize("caller", ["claude", "grok", "antigravity"])
def test_every_caller_has_a_candidate(role, caller):
    assert resolve_role(Role(role), caller, _cfg(), mode="agentic").model


async def test_roles_tool_returns_the_table():
    out = await server.create_server().call_tool("roles", {})
    out = out[1] if isinstance(out, tuple) else out
    assert {r["name"] for r in out["roles"]} == set(ROLES)


def test_readme_block_is_the_rendered_table():
    block = re.search(r"<!-- roles:start -->\n(.*?)\n<!-- roles:end -->", README.read_text(), re.S)
    assert block and block.group(1) == render_roles()
