"""Seed contender set: unreachable model ids are excluded, unprobed ids refuse,
and cells already measured for this fixture set are skipped."""
import importlib.util
import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent


def _load():
    spec = importlib.util.spec_from_file_location("rb_reach_test", REPO / "bin" / "run-bench.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _card(slug, provider, price, cap, model=None, effort="base"):
    return {"slug": slug, "model": model or slug, "effort": effort, "provider": provider,
            "capability": cap, "price_blended": price}


@pytest.fixture
def rb(monkeypatch, tmp_path):
    rb = _load()
    cards = tmp_path / "cards.json"
    cards.write_text(json.dumps({"cards": [
        _card("a-cheap", "zhipu", 0.10, 70.0),
        _card("b-dead", "xai", 0.12, 70.0),
        _card("c-mid", "openai", 0.19, 70.0),
        _card("d-far", "openai", 0.30, 70.0),   # outside 2x of 0.10, inside 2x of 0.19
    ]}))
    monkeypatch.setattr(rb, "CARDS", cards)
    monkeypatch.setattr(rb, "REACH", tmp_path / "reachability.json")
    monkeypatch.setattr(rb, "RESULTS", tmp_path / "results.json")
    return rb


def test_unprobed_contender_refuses(rb):
    with pytest.raises(rb.Refused, match="never probed"):
        rb.contender_sets()


def test_dead_id_excluded_and_window_recomputed_from_survivors(rb):
    rb.save_reachability({"zhipu/a-cheap": {"ok": True}, "xai/b-dead": {"ok": False},
                          "openai/c-mid": {"ok": True}, "openai/d-far": {"ok": True}})
    sets = rb.contender_sets()
    b1 = sets[0]
    assert [c["slug"] for c in b1["contenders"]] == ["a-cheap", "c-mid"]
    assert b1["unreachable"] == ["b-dead"]
    assert b1["unprobed"] == []


def test_require_probed_false_lists_unprobed_instead_of_refusing(rb):
    rb.save_reachability({"zhipu/a-cheap": {"ok": False}})
    sets = rb.contender_sets(require_probed=False)
    b1 = sets[0]
    # a-cheap dead -> cheapest survivor is b-dead at 0.12 -> window admits d-far too
    assert [c["slug"] for c in b1["contenders"]] == ["b-dead", "c-mid"]
    assert b1["unprobed"] == ["openai/c-mid", "xai/b-dead"]


def test_measured_cells_skip_only_same_fixture_set(rb):
    fx = [{"id": "j1", "band": "B1", "size_class": "small", "scoring": "reference",
           "rotating": False, "input": {"prompt": "p"}, "reference": {"answer": "x"},
           "acceptance": [{"id": "a1", "p0": True, "check": "c"}]}]
    row = {"card": "a-cheap", "band": "B1", "size_class": "small", "id": "j1", "trials_ok": 3}
    rb.RESULTS.write_text(json.dumps({"meta": {"fixture_set_hash": rb.fixture_set_hash(fx)},
                                      "results": [row, dict(row, card="c-mid", trials_ok=0)]}))
    assert rb.measured_cells(fx) == {("a-cheap", "B1", "small", "j1")}
    rb.RESULTS.write_text(json.dumps({"meta": {"fixture_set_hash": "other"}, "results": [row]}))
    assert rb.measured_cells(fx) == set()


def test_plan_seed_zero_calls_for_measured_cell(rb, monkeypatch):
    fx = [{"id": "j1", "band": "B1", "size_class": "small", "scoring": "reference",
           "rotating": False, "input": {"prompt": "p"}, "reference": {"answer": "x"},
           "acceptance": [{"id": "a1", "p0": True, "check": "c"}]}]
    monkeypatch.setattr(rb, "mechanically_scorable", lambda f: True)
    monkeypatch.setattr(rb, "measured_cells", lambda f: {("a-cheap", "B1", "small", "j1")})
    sets = [{"band": "B1", "floor": 10.0, "excluded": {},
             "contenders": [_card("a-cheap", "zhipu", 0.1, 70.0), _card("c-mid", "openai", 0.19, 70.0)]}]
    p = rb.plan_seed(fx, sets)
    assert p["skipped_cells"] == 1
    assert p["total_calls"] == 3
    assert [r["skip"] for r in p["rows"]] == [True, False]
