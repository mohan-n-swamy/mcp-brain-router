"""scores.tsv: every row carries a source, a date and a verified flag; a row without one is rejected."""
import pytest

from mcp_brain_router import scores

HEADER = "\t".join(scores.COLUMNS) + "\n"
GOOD = "claude-opus-5-5\thigh\taa_index_v5\t52\thttps://example.test/a\t2026-11-25\tquote\n"


def _file(tmp_path, body):
    p = tmp_path / "s.tsv"
    p.write_text(HEADER + body)
    return p


def test_rows_are_sourced():
    rows = scores.load()
    assert len(rows) > 40
    assert all(r.source_url.startswith("https://") and r.read_date and r.verified in scores.VERIFIED for r in rows)
    # the four top-level Artificial Analysis rows were read on their page; nothing else is marked page
    pages = {(r.model, r.measure) for r in rows if r.verified == "page"}
    assert pages == {(m, me) for m in ("gemini-4-neon", "gemini-4-carbon", "claude-sonnet-5-5", "gemini-4-hydrogen")
                     for me in ("aa_index_v5", "aa_minutes_per_task")}


@pytest.mark.parametrize("col", ["source_url", "read_date", "verified"])
def test_a_row_missing_source_date_or_flag_is_rejected(tmp_path, col):
    cells = GOOD.rstrip("\n").split("\t")
    cells[scores.COLUMNS.index(col)] = ""
    with pytest.raises(ValueError, match=col):
        scores.load(_file(tmp_path, "\t".join(cells) + "\n"))


def test_bad_flag_and_value_are_rejected(tmp_path):
    with pytest.raises(ValueError, match="verified"):
        scores.load(_file(tmp_path, GOOD.replace("quote", "trust-me")))
    with pytest.raises(ValueError, match="number"):
        scores.load(_file(tmp_path, GOOD.replace("\t52\t", "\tfifty\t")))
    assert len(scores.load(_file(tmp_path, GOOD))) == 1


# rigor: small
def test_empty_level_is_allowed_and_non_finite_value_is_rejected(tmp_path):
    empty_level = "claude-opus-5-5\t\tvals_terminal_bench_4_0\t65.15\thttps://example.test/a\t2026-11-25\tquote\n"
    rows = scores.load(_file(tmp_path, empty_level))
    assert len(rows) == 1 and rows[0].level == "" and rows[0].value == 65.15
    for bad in ("nan", "NaN", "inf", "-inf", "Infinity"):
        with pytest.raises(ValueError, match="finite"):
            scores.load(_file(tmp_path, GOOD.replace("\t52\t", f"\t{bad}\t")))
