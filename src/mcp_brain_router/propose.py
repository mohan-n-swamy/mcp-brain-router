"""Print where a rule-ranked order differs from config.ROLES. Prints only: never edits config.py or any file.

    python -m mcp_brain_router.propose [--json]

The rule, per role, over the role's own slots, using scores.tsv (every row carries a source, a date, a flag):
  thinker, adversary, writer   rank by aa_index_v5 at the slot's level, highest first.
  coder, qa                    rank by vals_terminal_bench_4_0 (the model's board row), highest first;
                               ties broken by aa_index_v5 at the slot's level.
  processor, explorer, simple  rank by aa_index_v5 at the slot's level, highest first, but a model that is
                               first in thinker is skipped (volume roles do not use the top model).
A slot with no score for the measure is listed as "no score" and keeps its current position.
Separately, per role: any (model, level) in the table that outscores the role's LAST slot and is not in the
role ("not in this role"), and every model id in ROLES with no row at all. The ranking can disagree with
ROLES; that is the output, not an error. Exit code is 0 whenever the command runs.
"""
from __future__ import annotations

# rigor: small
from itertools import zip_longest
import json
import sys

from .config import ROLES
from .scores import load

AA, TB = "aa_index_v5", "vals_terminal_bench_4_0"
MEASURE = {"thinker": AA, "adversary": AA, "writer": AA, "coder": TB, "qa": TB,
           "processor": AA, "explorer": AA, "simple": AA}
VOLUME = ("processor", "explorer", "simple")


def label(model, level):
    return f"{model}@{level}" if level else model


# rigor: small
def propose(rows=None) -> dict:
    rows = load() if rows is None else rows
    aa = {(r.model, r.level): r.value for r in rows if r.measure == AA}
    tb = {r.model: r.value for r in rows if r.measure == TB}
    top = ROLES["thinker"].slots[0].model if "thinker" in ROLES and ROLES["thinker"].slots else None

    def fam(s):
        """004: a route scores as its model family (agy-claude-sonnet-5-5 is claude-sonnet-5-5)."""
        return getattr(s, "family", None) or s.model

    def score(measure, s):
        """The sort key of a slot, or None when it has no score for the role's measure."""
        if measure is None:
            return None
        a = aa.get((fam(s), s.level or ""))
        if measure == AA:
            return None if a is None else (a,)
        if measure == TB:
            return None if fam(s) not in tb else (tb[fam(s)], a if a is not None else float("-inf"))
        return None

    out_roles = []
    for role, rd in ROLES.items():
        measure = MEASURE.get(role)
        table = [label(s.model, s.level) for s in rd.slots]
        slots, notes = list(rd.slots), []
        if measure is None:
            notes.append(f"no measure mapping for role {role}")
        if role in VOLUME and top is not None:
            skipped = [s for s in slots if s.model == top]
            slots = [s for s in slots if s.model != top]
            notes += [f"skipped {label(s.model, s.level)}: top model of thinker" for s in skipped]
        keys = [score(measure, s) for s in slots]
        scored = sorted((i for i, k in enumerate(keys) if k is not None), key=lambda i: keys[i], reverse=True)
        ranked, it = list(slots), iter(scored)
        for i, k in enumerate(keys):  # unscored slots keep their place; scored ones fill the rest by rank
            if k is None:
                if measure is not None:
                    notes.append(f"no score: {label(slots[i].model, slots[i].level)} kept at position {i + 1}")
            else:
                ranked[i] = slots[next(it)]
        new = [label(s.model, s.level) for s in ranked]
        diffs = [f"position {i + 1}: {a} -> {b}"
                 for i, (a, b) in enumerate(zip_longest(table, new, fillvalue="(removed)")) if a != b]
        scored_keys = [k for k in keys if k is not None]
        last = min(scored_keys) if scored_keys else None
        not_in = []
        if last is not None:
            in_role = {fam(s) for s in rd.slots} if measure == TB else {(fam(s), s.level or "") for s in rd.slots}
            if measure == TB:
                cand = {(m, ""): (v, float("-inf")) for m, v in tb.items() if m not in in_role}
            else:
                cand = {k: (v,) for k, v in aa.items() if k not in in_role}
            not_in = [f"{label(m, l)} ({v[0]})" for (m, l), v in sorted(cand.items(), key=lambda kv: -kv[1][0])
                      if v[0] > last[0] and not (role in VOLUME and m == top)]
        out_roles.append({"role": role, "measure": measure, "table": table, "ranked": new,
                          "differs": bool(diffs), "differences": diffs, "notes": notes, "not_in_this_role": not_in})
    have = {r.model for r in rows}
    missing = sorted({getattr(s, "family", None) or s.model for rd in ROLES.values() for s in rd.slots} - have)
    return {"roles": out_roles, "models_without_any_row": missing,
            "scores_read_dates": sorted({r.read_date for r in rows})}


def render(p: dict) -> str:
    out = []
    for r in p["roles"]:
        out += [f"{r['role']} (by {r['measure'] or 'none'})", "  table : " + " > ".join(r["table"]),
                "  ranked: " + " > ".join(r["ranked"]),
                "  differs: " + ("; ".join(r["differences"]) if r["differs"] else "no")]
        out += [f"  note: {n}" for n in r["notes"]]
        out += ["  not in this role: " + ", ".join(r["not_in_this_role"])] if r["not_in_this_role"] else []
    out.append("models in ROLES with no row in scores.tsv: " + (", ".join(p["models_without_any_row"]) or "none"))
    out.append("score rows read on: " + ", ".join(p["scores_read_dates"]))
    return "\n".join(out)


def main(argv=None) -> int:
    p = propose()
    print(json.dumps(p, indent=2) if "--json" in (sys.argv[1:] if argv is None else argv) else render(p))
    return 0


if __name__ == "__main__":
    sys.exit(main())
