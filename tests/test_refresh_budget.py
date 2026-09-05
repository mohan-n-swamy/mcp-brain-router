"""C08 unattended spend: probe + benchmark run only inside --max-calls; over
budget STOPs with the plan and spends nothing; zero-call plans are skipped."""
import importlib.util
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO = Path(__file__).resolve().parent.parent


def _load():
    spec = importlib.util.spec_from_file_location("refresh_under_test", REPO / "bin" / "refresh-deck.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _stub(monkeypatch, rd, dry_stdout):
    calls = []

    def fake_run(cmd, **kw):
        calls.append([str(c) for c in cmd])
        if "--dry-run" in cmd:
            return SimpleNamespace(returncode=0, stdout=dry_stdout, stderr="")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(rd.subprocess, "run", fake_run)
    return calls


SEED_PLAN = "...\nTOTAL PROVIDER CALLS                                                       183\nplan hash: 986266f296ab\n"
PROBE_PLAN = "unprobed contender model ids: 2\nplan hash: probe-abc\nTo spend 2 tiny calls: ...\n"


def test_within_budget_spends_with_the_dry_run_hash(monkeypatch):
    rd = _load()
    calls = _stub(monkeypatch, rd, SEED_PLAN)
    rd.spend_within_budget("bench", "--seed-round", 250, remeasure=True)
    live = [c for c in calls if "--dry-run" not in c]
    assert len(live) == 1
    assert live[0][-4:] == ["--seed-round", "--approve", "986266f296ab", "--remeasure"]


def test_over_budget_stops_and_spends_nothing(monkeypatch, capsys):
    rd = _load()
    calls = _stub(monkeypatch, rd, SEED_PLAN)
    with pytest.raises(SystemExit) as e:
        rd.spend_within_budget("bench", "--seed-round", 100, remeasure=False)
    assert e.value.code == 4
    assert all("--dry-run" in c for c in calls)
    assert "--approve 986266f296ab" in capsys.readouterr().err


def test_zero_calls_skips(monkeypatch):
    rd = _load()
    calls = _stub(monkeypatch, rd, "unprobed contender model ids: 0\nnothing to probe\n")
    rd.spend_within_budget("probe", "--probe-cards", 0, remeasure=False)
    assert all("--dry-run" in c for c in calls)


def test_probe_plan_parsed(monkeypatch):
    rd = _load()
    _stub(monkeypatch, rd, PROBE_PLAN)
    assert rd.bench_plan("--probe-cards", False) == (2, "probe-abc")
