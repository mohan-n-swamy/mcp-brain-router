"""peg-costs on seed-round rows: price by the row's own card, never a card
that produced no answer."""
import importlib.util
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def _load():
    spec = importlib.util.spec_from_file_location("peg_under_test", REPO / "bin" / "peg-costs.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _run(monkeypatch, tmp_path, capsys, target_rows):
    peg = _load()
    monkeypatch.setattr(peg, "price_of", lambda slug: ({"kimi-k3": 1.0, "kimi-k2-turbo": 4.0}[slug], 60.0))
    t = tmp_path / "results.json"
    d = tmp_path / "baseline.json"
    t.write_text(json.dumps({"meta": {}, "results": target_rows}))
    d.write_text(json.dumps({"meta": {"run_id": "donor"}, "results": [
        {"id": "j1", "backend": "glm", "trials": [{"tokens_in": 900_000, "tokens_out": 100_000}]}]}))
    monkeypatch.setattr("sys.argv", ["peg", "--target", str(t), "--donor", str(d), "--write"])
    assert peg.main() == 0
    return json.loads(t.read_text())["results"], capsys.readouterr().out


def test_seed_row_priced_by_its_own_card(monkeypatch, tmp_path, capsys):
    rows, _ = _run(monkeypatch, tmp_path, capsys, [
        {"id": "j1", "card": "kimi-k2-turbo", "backend": "kimi", "cost_basis": "unmeasured",
         "total_cost_usd": None, "trials_ok": 3}])
    assert rows[0]["cost_basis"] == "pegged"
    assert rows[0]["peg"]["card"] == "kimi-k2-turbo"
    assert rows[0]["total_cost_usd"] == 4.0  # 1M tokens x $4/1M, not the $1 default card


def test_zero_ok_trials_never_pegged(monkeypatch, tmp_path, capsys):
    rows, out = _run(monkeypatch, tmp_path, capsys, [
        {"id": "j1", "card": "kimi-k3", "backend": "kimi", "cost_basis": "unmeasured",
         "total_cost_usd": None, "trials_ok": 0}])
    assert rows[0]["cost_basis"] == "unmeasured"
    assert rows[0]["total_cost_usd"] is None
    assert "0 ok trials" in out
