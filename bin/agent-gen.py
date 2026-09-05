#!/usr/bin/env python3
"""agent-gen.py — C10: generate .claude/agents/<name>.md from the pack's tables.

The declarative-agent-file half of the design: files carrying their own model
and effort, versioned on disk, applied by a command -- with the deck supplying
the values a human would otherwise type. The framework this shape was taken
from is not named here on purpose: C11 forbids its name in bin/ and agents/,
because that framework's login silently switches Claude Code to API billing.

Inputs, none of them typed here:
  SKILL-BANDS.md     agent:deck and agent:native rows, with band, Q-scores, reason
  JOB-CATALOGUE.md   the four deck-routable F11 routine shapes C13 names in its
                     deny messages: read-extract, repo-search, read-many,
                     mechanical-edit. A missing one wedges a session.
  deck.json          model = bands[band].ranked[0].router_model; floor = bands[band].floor

STOP conditions, enforced before any file is written:
  a band a generated agent needs has NO ranked card -- generating with a guessed
     model puts an estimate on a production path (R21). On week 0 this refuses,
     and that is correct sequencing, not a defect.
  an agent:deck row carries IRR -- never valid in the lane; refused here rather
     than left for C11.
  a target file exists and was not written by this generator -- a hand-written
     agent is never overwritten; the collision is reported and the file left alone.

Idempotent: a second run over the same inputs writes byte-identical files. Never
writes an agent for a skill-disposition entry, which has no band.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from mcp_brain_router.deck import BAND_FLOORS  # noqa: E402

PACK = Path.home() / "code workshop" / "specs" / "001-agent-capability-routing"
DEFAULT_DECK = Path.home() / ".local" / "state" / "brain-router" / "deck.json"
DEFAULT_AGENTS = Path.home() / ".claude" / "agents"
DELEGATE_TOOL = "mcp__brain-router__delegate"
MARKER = "generated-by: agent-gen (specs/001-agent-capability-routing C10)"
NATIVE_TOOLS = "Read, Grep, Glob, Bash"   # skills declare none; the house default

# The four routine shapes, keyed to their JOB-CATALOGUE F11 rows verbatim.
ROUTINE = [
    ("read-extract",    "B2", [1, 1, 2, 2], None,
     "Read a document, answer a bounded question / extract fields into a fixed shape"),
    ("repo-search",     "B2", [1, 2, 2, 1], None,
     "Find where a symbol is defined or used"),
    ("read-many",       "B3", [2, 2, 2, 2], None,
     "Read many files, summarise into one answer"),
    ("mechanical-edit", "B1", [1, 1, 1, 1], "manufacture harness",
     "Apply a mechanical edit from an exact instruction (Q3 2->1 collapse, executor named)"),
]


class Refused(SystemExit):
    def __init__(self, msg: str):
        print(f"REFUSED: {msg}", file=sys.stderr)
        super().__init__(3)


# ---------------------------------------------------------------- parse the tables

def _rows(section_heading: str, text: str) -> list[list[str]]:
    i = text.index(section_heading)
    j = text.find("\n## ", i + 1)
    block = text[i:j if j > 0 else None]
    out = []
    for line in block.splitlines():
        if line.startswith("| `"):
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            out.append(cells)
    return out


def parse_bands() -> tuple[list[dict], list[dict]]:
    text = (PACK / "SKILL-BANDS.md").read_text(encoding="utf-8")
    deck, native = [], []
    for name, band, q, why in _rows("## `agent:deck`", text):
        deck.append({"name": name.strip("`"), "band": band,
                     "q": [int(x) for x in q.split()], "why": why,
                     "irr": "`irr`" in why.lower() or " irr" in why.lower()})
    for name, band, why in _rows("## `agent:native`", text):
        native.append({"name": name.strip("`"), "band": band, "why": why})
    return deck, native


def load_deck(path: Path) -> dict[str, dict]:
    doc = json.loads(path.read_text(encoding="utf-8"))
    return {b["band"]: b for b in doc.get("bands") or []}


# ---------------------------------------------------------------- render

def _desc(s: str) -> str:
    return s.replace("**", "").replace('"', "'")


def render_deck(row: dict, band_row: dict, executor: str | None = None) -> str:
    card = band_row["ranked"][0]
    q = row["q"]
    fm = [
        "---",
        f"name: {row['name']}",
        f"description: \"{_desc(row['why'])} [agent:deck {row['band']}]\"",
        f"tools: {DELEGATE_TOOL}",
        f"model: {card['router_model']}",
        f"effort: {card.get('effort') or 'base'}",
        f"min_capability: {band_row['floor']}",
        f"band: {row['band']}",
        "lane: deck",
        "mode: agentic",     # C11 checks this key, not a sentence in the body
        "cwd: caller",       # the caller's real working directory, passed per call
        f"tier_why: \"{row['band']} because {_desc(row['why'])}\"",
        f"q_scores: [{q[0]}, {q[1]}, {q[2]}, {q[3]}]",
        "irr: false",
    ]
    if executor:
        fm += ["collapsed: true", f"executor: \"{executor}\""]
    fm += [f"{MARKER}", "---"]
    body = f"""
You are the `{row['name']}` executor. You hold exactly ONE tool: `{DELEGATE_TOOL}`.
You do not do this work yourself; you route it and return the backend's answer.

Call `{DELEGATE_TOOL}` once with role for band {row['band']}, `mode="agentic"`, and the
caller's real `cwd` -- in chat mode the same call returns text and the work never happens.
Pass the task verbatim. Return only the answer.

Why this band: {row['band']} -- {_desc(row['why'])}
Why one tool: a delegate-lane agent cannot do the work natively because it holds no tool
that could. That is a mechanism, not an instruction.
"""
    return "\n".join(fm) + "\n" + body


def render_native(row: dict, band_row: dict) -> str:
    card = band_row["ranked"][0]
    fm = [
        "---",
        f"name: {row['name']}",
        f"description: \"{_desc(row['why'])} [agent:native {row['band']}]\"",
        f"tools: {NATIVE_TOOLS}",
        f"model: {card['router_model']}",
        f"effort: {card.get('effort') or 'base'}",
        f"min_capability: {band_row['floor']}",
        f"band: {row['band']}",
        "lane: native",
        f"tier_why: \"{row['band']} because {_desc(row['why'])}\"",
        f"{MARKER}",
        "---",
    ]
    body = f"""
You are the `{row['name']}` agent, isolated for context and parallelism but running on Claude.

Kept native -- named exception: {_desc(row['why'])}

Band {row['band']} sets the model and effort above. This file names its exception so the
route can be checked (C11); "it'd be faster" is not one.
"""
    return "\n".join(fm) + "\n" + body


# ---------------------------------------------------------------- plan + write

def plan(deck_path: Path) -> list[tuple[str, str]]:
    """[(filename, content)] or raise Refused. Nothing written."""
    deck_rows, native_rows = parse_bands()
    bands = load_deck(deck_path)
    needed = sorted({r["band"] for r in deck_rows} | {r["band"] for r in native_rows}
                    | {b for _, b, *_ in ROUTINE})
    empty = [b for b in needed if not (bands.get(b) or {}).get("ranked")]
    if empty:
        raise Refused(f"bands {empty} have no ranked card. A generated agent would carry a "
                      "guessed model on a production path (R21). On week 0 this is correct "
                      "sequencing: build the deck's rankings first.")
    # A ranked list that is non-empty but whose winner carries no router_model, or
    # a band with no floor, would have written the literal 'None' into a production
    # agent file (refuter finding). Refuse before rendering.
    broken = []
    for b in needed:
        row = bands[b]; top = row["ranked"][0]
        if not top.get("router_model"):
            broken.append(f"{b}: ranked[0] {top.get('slug')!r} has no router_model")
        if row.get("floor") is None:
            broken.append(f"{b}: floor is null")
    if broken:
        raise Refused("deck rows unusable: " + "; ".join(broken))
    irr = [r["name"] for r in deck_rows if r["irr"]]
    if irr:
        raise Refused(f"agent:deck rows marked IRR: {irr}. Irreversible work never enters the "
                      "delegate lane (R24).")
    out: list[tuple[str, str]] = []
    for r in deck_rows:
        out.append((f"{r['name']}.md", render_deck(r, bands[r["band"]])))
    for r in native_rows:
        out.append((f"{r['name']}.md", render_native(r, bands[r["band"]])))
    for name, band, q, executor, why in ROUTINE:
        row = {"name": name, "band": band, "q": q, "why": why, "irr": False}
        out.append((f"{name}.md", render_deck(row, bands[band], executor=executor)))
    return out


def write(files: list[tuple[str, str]], agents_dir: Path) -> tuple[int, int]:
    agents_dir.mkdir(parents=True, exist_ok=True)
    written = unchanged = 0
    collisions = []
    for fname, content in files:
        target = agents_dir / fname
        if target.exists():
            existing = target.read_text(encoding="utf-8")
            fm = existing.split("\n---\n", 1)[0] if existing.startswith("---\n") else ""
            if f"\n{MARKER}" not in fm:   # marker as a frontmatter line, not prose
                collisions.append(fname)
                continue
            if existing == content:
                unchanged += 1
                continue
        fd, tmp = tempfile.mkstemp(dir=str(agents_dir), prefix=".agent.")
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(content)
        os.replace(tmp, target)
        written += 1
    if collisions:
        print(f"COLLISION (left alone, hand-written): {collisions}", file=sys.stderr)
    return written, unchanged


def main() -> int:
    ap = argparse.ArgumentParser(description="C10 agent generator")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--list-deck-agents", action="store_true",
                    help="print the paths of every agent:deck file this would write")
    ap.add_argument("--deck", default=str(DEFAULT_DECK))
    ap.add_argument("--agents-dir", default=str(DEFAULT_AGENTS))
    a = ap.parse_args()
    try:
        files = plan(Path(a.deck))
    except Refused as e:
        return int(e.code)
    if a.list_deck_agents:
        for fname, content in files:
            if f"tools: {DELEGATE_TOOL}\n" in content:
                print(Path(a.agents_dir) / fname)
        return 0
    n_deck = sum(1 for _, c in files if f"tools: {DELEGATE_TOOL}\n" in c)
    print(f"plan: {len(files)} agent files ({n_deck} deck, {len(files) - n_deck} native) -> {a.agents_dir}")
    if a.dry_run:
        print("dry-run: nothing written")
        return 0
    w, u = write(files, Path(a.agents_dir))
    print(f"wrote {w}, unchanged {u}")
    # Collisions were reported on stderr and left alone; exit 1 so a script cannot
    # read 'wrote 0, unchanged 0' as a clean no-op (refuter finding).
    return 1 if (w + u) < len(files) else 0


if __name__ == "__main__":
    sys.exit(main())
