"""propose prints; it never edits. No test asserts the ranking equals ROLES: the rule may disagree."""
import json
import subprocess
import sys

# rigor: small
from mcp_brain_router import propose
from mcp_brain_router.config import ROLES, RoleDef, Slot
from mcp_brain_router.scores import Score


def _row(model, level, value, measure="aa_index_v5"):
    return Score(model, level, measure, value, "https://example.test", "2026-11-25", "quote")


def test_command_runs_and_json_is_readable():
    for args, check in (([], lambda s: "thinker" in s), (["--json"], lambda s: json.loads(s)["roles"])):
        r = subprocess.run([sys.executable, "-m", "mcp_brain_router.propose", *args], capture_output=True, text=True)
        assert r.returncode == 0 and check(r.stdout), r.stderr
    got = json.loads(r.stdout)
    assert [x["role"] for x in got["roles"]] == list(ROLES)
    assert {"table", "ranked", "differences", "not_in_this_role"} <= set(got["roles"][0])


def test_unscored_candidate_keeps_its_place_and_models_without_rows_are_flagged():
    # explorer is haiku@medium > sonnet@high; give only sonnet a score: nothing to reorder against, haiku stays first
    p = propose.propose([_row("claude-sonnet-5-5", "high", 41)])
    ex = next(r for r in p["roles"] if r["role"] == "explorer")
    assert ex["ranked"][0] == "claude-haiku-5-5@medium" and not ex["differs"]
    assert any("no score" in n and "claude-haiku-5-5" in n for n in ex["notes"])
    assert "claude-haiku-5-5" in p["models_without_any_row"] and "claude-sonnet-5-5" not in p["models_without_any_row"]


def test_rule_can_disagree_and_volume_roles_skip_the_top_thinker_model():
    rows = [_row("gemini-4-hydrogen", "xhigh", 10, "aa_index_v5"), _row("claude-haiku-5-5", "medium", 90)]
    ex = next(r for r in propose.propose(rows)["roles"] if r["role"] == "explorer")
    # grok@high and hydrogen@high are unscored: each keeps its place
    assert ex["ranked"] == ["claude-haiku-5-5@medium", "gemini-4-hydrogen@high", "grok-4.7@high"]
    rows = [_row("gemini-4-hydrogen", "high", 99), _row("gemini-4-neon", "medium", 1)]
    pr = next(r for r in propose.propose(rows)["roles"] if r["role"] == "processor")
    assert pr["ranked"][0] == "gemini-4-hydrogen@high" and pr["differs"]


def test_propose_has_no_write_call():
    src = open(propose.__file__).read()
    assert "write_text" not in src and "open(" not in src.replace("open(propose", "")


# rigor: small
def test_table_uses_real_slots_and_threshold_uses_lowest_scored_slot(monkeypatch):
    fake_roles = {
        "thinker": ROLES["thinker"],
        "processor": RoleDef("processor", "", "", (Slot("claude-sonnet-5-5", "high"),
                                                   Slot("claude-fable-5-2", "max")), False, 60),
        "writer": RoleDef("writer", "", "", (Slot("claude-opus-5-5", "high"),
                                             Slot("claude-sonnet-5-5", "high"),
                                             Slot("apple-fm")), False, 60),
    }
    monkeypatch.setattr(propose, "ROLES", fake_roles)
    rows = [_row("claude-opus-5-5", "high", 40), _row("claude-sonnet-5-5", "high", 60),
            _row("grok-4.7", "high", 50)]
    p = propose.propose(rows)
    pr = next(r for r in p["roles"] if r["role"] == "processor")
    assert pr["table"] == ["claude-sonnet-5-5@high", "claude-fable-5-2@max"]
    assert pr["ranked"] == ["claude-sonnet-5-5@high"]
    assert pr["differs"] is True
    wr = next(r for r in p["roles"] if r["role"] == "writer")
    assert wr["not_in_this_role"] == ["grok-4.7@high (50)"]


def test_unmapped_role_does_not_raise(monkeypatch):
    fake_roles = dict(ROLES) | {
        "custom": RoleDef("custom", "", "", (Slot("claude-sonnet-5-5", "high"),), False, 60),
    }
    monkeypatch.setattr(propose, "ROLES", fake_roles)
    p = propose.propose([_row("claude-sonnet-5-5", "high", 50)])
    cr = next(r for r in p["roles"] if r["role"] == "custom")
    assert cr["measure"] is None
    assert cr["table"] == ["claude-sonnet-5-5@high"] and cr["ranked"] == ["claude-sonnet-5-5@high"]
    assert not cr["differs"] and cr["not_in_this_role"] == []
    assert "custom" in propose.render(p)
