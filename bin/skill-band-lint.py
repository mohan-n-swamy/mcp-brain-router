#!/usr/bin/env python3
"""Coverage and count-agreement lint for SKILL-BANDS.md (C09).

Two jobs, because the pack has already failed at both:

  coverage   every skill in either tree has a disposition, and every classified
             name still exists. Deduplicated by realpath: the two trees
             cross-link heavily, so naive counting roughly doubles the total.

  agreement  the disposition counts stated in SKILL-BANDS.md match the rows
             actually present, AND match every other file in the pack that
             quotes them. An adversarial review found spec.md saying 21/41
             while SKILL-BANDS.md said 19/15/42 — SC8 passed anyway, because
             it only checked mention coverage. A count repeated in four files
             is a count that will drift.

Exit 1 on any failure.
"""
import os
import re
import sys
from pathlib import Path

PACK = Path.home() / "code workshop/specs/001-agent-capability-routing"
BANDS = PACK / "SKILL-BANDS.md"
TREES = [Path.home() / ".claude/skills", Path.home() / ".agents/skills"]
# Files that quote the disposition counts and must agree with SKILL-BANDS.md.
QUOTING = ["spec.md", "components/C10-agent-generator.md", "verification.md", "prd.md"]


def inventory() -> set[str]:
    seen: set[str] = set()
    names: set[str] = set()
    for root in TREES:
        if not root.is_dir():
            continue
        for d in sorted(os.listdir(root)):
            p = root / d
            if not p.is_dir():
                continue
            real = os.path.realpath(p)
            if real in seen:
                continue
            seen.add(real)
            names.add(d)
    return names


def main() -> int:
    text = BANDS.read_text(encoding="utf-8")
    errors = 0

    # ---- coverage ----------------------------------------------------------
    inv = inventory()
    named = set(re.findall(r"`([a-z0-9][a-z0-9-]*)`", text))
    unclassified = sorted(inv - named)
    stale = sorted(n for n in (named - inv) if n in _plausible_skill_names(text))

    for u in unclassified:
        print(f"unclassified skill (exists in a tree, absent from SKILL-BANDS.md): {u}")
        errors += 1
    for s in stale:
        print(f"stale entry (classified but no longer in either tree): {s}")
        errors += 1

    # ---- agreement ---------------------------------------------------------
    deck_rows = len(re.findall(r"^\| `[a-z0-9-]+` \| B[1-5] \| [123] [123]", text, re.M))
    native_rows = len(re.findall(r"^\| `[a-z0-9-]+` \| B[1-5] \| (?![123] )", text, re.M))
    stated = dict(re.findall(r"## `agent:(deck|native)` — (\d+)", text))

    for lane, rows in (("deck", deck_rows), ("native", native_rows)):
        want = int(stated.get(lane, -1))
        if want != rows:
            print(f"count mismatch: agent:{lane} heading says {want}, table has {rows} rows")
            errors += 1

    # every other file in the pack must not quote a contradicting count
    for rel in QUOTING:
        f = PACK / rel
        if not f.exists():
            continue
        body = f.read_text(encoding="utf-8")
        for m in re.finditer(r"(\d+)\s+`?agent:deck`?", body):
            if int(m.group(1)) != deck_rows:
                print(f"{rel}: says {m.group(1)} agent:deck, SKILL-BANDS.md has {deck_rows}")
                errors += 1
        for m in re.finditer(r"the (\d+) `?agent:deck`? skills", body):
            if int(m.group(1)) != deck_rows:
                print(f"{rel}: says {m.group(1)} deck skills, SKILL-BANDS.md has {deck_rows}")
                errors += 1

    print(
        f"inventory={len(inv)} classified={len(inv & named)} unclassified={len(unclassified)} "
        f"deck={deck_rows} native={native_rows} errors={errors}"
    )
    return 1 if errors else 0


def _plausible_skill_names(text: str) -> set[str]:
    """Backticks in the doc also wrap code and paths; only treat a name as a
    classified skill if it appears at the start of a table row."""
    return set(re.findall(r"^\| `([a-z0-9][a-z0-9-]*)`", text, re.M))


if __name__ == "__main__":
    sys.exit(main())
