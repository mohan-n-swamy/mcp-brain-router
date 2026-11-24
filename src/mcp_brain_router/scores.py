"""Loader for scores.tsv: one row per (model, level, measure), each with a source, a date and a flag."""
from __future__ import annotations

# rigor: small
import csv
import math
from dataclasses import dataclass
from pathlib import Path

SCORES_FILE = Path(__file__).with_name("scores.tsv")
COLUMNS = ("model", "level", "measure", "value", "source_url", "read_date", "verified")
VERIFIED = ("page", "quote")  # page: read on the source page itself; quote: via a search-engine quote or a research agent


@dataclass(frozen=True)
class Score:
    model: str
    level: str
    measure: str
    value: float
    source_url: str
    read_date: str
    verified: str


def load(path=None) -> list[Score]:
    """Every row, or ValueError naming the first bad one: a score nobody can recheck must not rank anything."""
    rows = []
    with open(path or SCORES_FILE, newline="") as f:
        rd = csv.DictReader(f, delimiter="\t")
        if tuple(rd.fieldnames or ()) != COLUMNS:
            raise ValueError(f"scores header must be {COLUMNS}, got {rd.fieldnames}")
        # rigor: small
        for n, r in enumerate(rd, 2):
            for c in COLUMNS:
                if c != "level" and not (r.get(c) or "").strip():
                    raise ValueError(f"scores row {n}: missing {c}")
            if not r["source_url"].startswith("https://"):
                raise ValueError(f"scores row {n}: source_url is not a URL")
            if r["verified"] not in VERIFIED:
                raise ValueError(f"scores row {n}: verified must be one of {VERIFIED}")
            try:
                val = float(r["value"])
            except ValueError:
                raise ValueError(f"scores row {n}: value {r['value']!r} is not a number") from None
            if not math.isfinite(val):
                raise ValueError(f"scores row {n}: value {r['value']!r} is not a finite number")
            rows.append(Score(r["model"], (r.get("level") or "").strip(), r["measure"], val, r["source_url"],
                              r["read_date"], r["verified"]))
    return rows
