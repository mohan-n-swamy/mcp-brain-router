"""C07: the --seed-round approval gate (must-not #11).

--seed-round is the only mode that spends on cards nobody has measured yet, so
the spend may start only on a plan a human actually SAW: --seed-round --dry-run
prints a plan hash, and only that exact hash opens the gate. These tests drive
main() itself with execute_seed stubbed, so a gate that opens without approval
fails here instead of three seconds into a live run -- which is exactly how the
gate's absence was first noticed.
"""

import importlib.util
import os
import re
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent


def _load_bench():
    # bin/ is not a package; load the script by path so the gate is tested as
    # shipped, not through a re-implementation that could drift from it.
    spec = importlib.util.spec_from_file_location("run_bench_under_test",
                                                  REPO / "bin" / "run-bench.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


bench = _load_bench()


def _fixtures() -> list[dict]:
    # Two synthetic frozen fixtures, one per scoring path, so plan_seed covers
    # both a mechanical row and a judge row without touching the spec vault.
    return [
        {"id": "x1-ref", "band": "B1", "size_class": "small", "scoring": "reference",
         "rotating": False,
         "input": {"prompt": "list every wikilink on the page"},
         "reference": {"answer": ["a", "b"]},
         "acceptance": [{"id": "a1", "p0": True, "check": "all items present"}]},
        {"id": "x2-rubric", "band": "B3", "size_class": "large", "scoring": "rubric",
         "rotating": False,
         "input": {"prompt": "review the module for bugs"},
         "acceptance": [{"id": "a1", "p0": True, "check": "names the off-by-one"}]},
    ]


def _sets() -> list[dict]:
    # A fixed contender set: the real contender_sets() reads live cards.json,
    # and the gate's job does not depend on which cards happen to be in the deck.
    def card(slug, provider, price, capability):
        return {"slug": slug, "provider": provider, "price_blended": price,
                "capability": capability}

    return [
        {"band": "B1", "floor": 10.0, "excluded": {},
         "contenders": [card("glm-5-3-flash", "zhipu", 0.10, 70.0),
                        card("kimi-k3", "moonshot", 0.18, 75.0)]},
        {"band": "B3", "floor": 55.0, "excluded": {},
         "contenders": [card("glm-5-3-flash", "zhipu", 0.10, 70.0)]},
    ]


@pytest.fixture(autouse=True)
def gate(monkeypatch):
    """No live state, no spend: fixtures and contenders are synthetic, headroom
    printing reads nothing, and execute_seed -- the only function that spends --
    is replaced by a stub that records it was reached."""
    reached: list[str] = []

    async def fake_execute_seed(fixtures, sets, p, run_id):
        reached.append(run_id)
        return 0

    monkeypatch.setattr(bench, "load_fixtures", lambda frozen_only: _fixtures())
    monkeypatch.setattr(bench, "contender_sets", _sets)
    monkeypatch.setattr(bench, "read_headroom", lambda: {"providers": {}})
    monkeypatch.setattr(bench, "execute_seed", fake_execute_seed)
    yield reached
    # main() writes the run id straight into os.environ, which monkeypatch
    # cannot restore; the tests must not leak it into the next process state.
    os.environ.pop("BRAIN_ROUTER_BENCH_RUN_ID", None)


def _main(monkeypatch, *argv: str) -> int:
    monkeypatch.setattr(sys, "argv", ["run-bench.py", *argv])
    return bench.main()


def test_seed_round_without_approve_refuses_rc_2(monkeypatch, capsys, gate):
    rc = _main(monkeypatch, "--seed-round")
    assert rc == 2
    assert "REFUSED" in capsys.readouterr().err
    assert gate == []  # the refusal must happen before any spend path is entered


def test_seed_round_wrong_hash_refuses_rc_2(monkeypatch, capsys, gate):
    rc = _main(monkeypatch, "--seed-round", "--approve", "000000000000")
    assert rc == 2
    err = capsys.readouterr().err
    assert "REFUSED" in err
    # the refusal names the hash it wanted and the one it got, or the operator
    # cannot tell a stale plan from a typo
    assert "000000000000" in err
    assert gate == []


def test_dry_run_prints_plan_hash_and_never_reaches_execute_seed(monkeypatch, capsys, gate):
    rc = _main(monkeypatch, "--seed-round", "--dry-run")
    assert rc == 0
    out = capsys.readouterr().out
    assert re.search(r"^plan hash: [0-9a-f]{12}$", out, re.MULTILINE)
    assert "nothing has been spent" in out.lower()
    assert gate == []


def test_exact_hash_from_dry_run_opens_the_gate(monkeypatch, capsys, gate):
    rc = _main(monkeypatch, "--seed-round", "--dry-run")
    assert rc == 0
    h = re.search(r"plan hash: ([0-9a-f]{12})", capsys.readouterr().out).group(1)

    rc = _main(monkeypatch, "--seed-round", "--approve", h)
    assert rc == 0
    assert len(gate) == 1  # reached the (stubbed) spend exactly once, no retries


def test_hash_changes_when_a_fixture_prompt_changes():
    h0 = bench.seed_plan_hash(_fixtures(), _sets())
    fx = _fixtures()
    fx[1]["input"]["prompt"] = "review the module for bugs, harder"
    # A changed prompt invalidates the series; a stale approval must not survive it
    assert bench.seed_plan_hash(fx, _sets()) != h0


def test_hash_changes_when_the_contender_set_changes():
    h0 = bench.seed_plan_hash(_fixtures(), _sets())
    sets = _sets()
    sets[1]["contenders"].append({"slug": "grok-5", "provider": "xai",
                                  "price_blended": 0.20, "capability": 72.0})
    # One more card is more planned calls; the approved plan no longer matches
    assert bench.seed_plan_hash(_fixtures(), sets) != h0


def test_execute_seed_flush_writes_results(monkeypatch, tmp_path):
    """The seed flush must run the whole way: on 2026-09-05 the live run died on
    NameError inside flush() after three glm calls, because the unscored-rows
    count read a name that exists only in the baseline path. Drive execute_seed
    with a stubbed trial and assert the file lands with meta.complete and
    unscored_rows derived from the merged rows."""
    import asyncio
    import json

    bench = _load_bench()  # fresh module: the autouse gate stubs execute_seed
    results = tmp_path / "results.json"
    monkeypatch.setattr(bench, "RESULTS", results)
    monkeypatch.setattr(bench, "read_headroom", lambda: {"providers": {}})
    monkeypatch.setattr(bench, "git_head", lambda: "test")
    monkeypatch.setattr(bench, "scratch_cwd", lambda run_id: str(tmp_path))
    monkeypatch.setattr(bench, "card_dispatch", lambda c: ("route:test", "test"))

    async def fake_trial(card, fixture, cwd):
        return {"ok": True, "answer": "x", "cost_usd": 0.01, "wall_s": 1.0}

    monkeypatch.setattr(bench, "run_card_trial", fake_trial)
    monkeypatch.setattr(bench, "mechanically_scorable", lambda f: True)
    monkeypatch.setattr(bench, "score_reference",
                        lambda f, a: {"p0_pass": True, "score": 1.0})
    monkeypatch.setattr(bench, "_score_trials",
                        lambda f, trials: {"acceptance": {"p0_pass": True},
                                           "trials_ok": len(trials)})
    sets = [{"band": "B1", "floor": 10.0, "excluded": {},
             "contenders": [{"slug": "c1", "provider": "zhipu",
                             "price_blended": 0.1, "capability": 70.0}]}]
    fx = [f for f in _fixtures() if f["band"] == "B1"]
    rc = asyncio.run(bench.execute_seed(fx, sets, {"total_calls": 3}, "seed-test"))
    assert rc == 0
    doc = json.loads(results.read_text())
    assert doc["meta"]["complete"] is True
    assert doc["meta"]["unscored_rows"] == 0
    assert [r["card"] for r in doc["results"]] == ["c1"]
