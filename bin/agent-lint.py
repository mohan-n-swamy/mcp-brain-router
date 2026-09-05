#!/usr/bin/env python3
"""agent-lint.py — C11: the rules the pack states in prose, enforced in code.

Read-only over ~/.claude/agents/*.md and bin/*.py. Reports, never repairs.
Exit 1 on any failure. Each check maps to a rule that would otherwise depend
on someone remembering it:

  1  delegate lane holds exactly ONE tool -- mcp__brain-router__delegate.
     More than one means the agent can do the work natively, and the whole
     surface rests on it structurally not being able to.
  1b every agent:deck file names agentic mode and a real cwd. In chat mode
     the same agent returns text and the work never happens (SKILL-BANDS).
  2  no API billing path: no `ant`, ANTHROPIC_API_KEY or managed-agent
     reference in any agent or script. `ant auth login` silently switched
     Claude Code from the Max subscription to API billing -- the failure
     that shaped this project (SC10).
  3  no IRR job in the delegate lane. Keyed on `irr`, NEVER on q3: severity
     is not irreversibility, and the first draft of this check would have
     banned manuf-qa, manuf-design-qa and stress-test from the lane they
     belong in.
  3b a collapsed row (Q4=1 collapse declared) names an executor from the
     closed list. A one-tool agent cannot run its own postcondition.
  4  every agent:native names its exception. "It'd be faster" is not one.

A file is agent:deck when its frontmatter carries `lane: deck`, or its only
tool is the delegate tool, or its description says so. It is agent:native
when `lane: native`. Hand-written agents with neither marker are checked
only for rule 2.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

AGENTS = Path.home() / ".claude" / "agents"
SCRIPTS = Path(__file__).resolve().parent
DELEGATE_TOOL = "mcp__brain-router__delegate"
EXECUTORS = {"manufacture harness", "the spec's own check", "the rule's own unit test",
             "the check itself", "the query itself"}
# The accepted exceptions, in the vocabulary SKILL-BANDS.md and JOB-CATALOGUE.md
# actually use. Each phrase maps to one of the five reasons the delegate skill
# names: THIS session's state · THIS repo's files · live tools or side effects ·
# image generation · a judgement Mohan is paying Claude for. A row that merely
# points at another row ("same reason", "codex-side twin") matches none of these
# on purpose -- it has stated no exception of its own.
NATIVE_REASONS = (
    # this session / this machine
    "this session", "conversation's state", "this machine", "local state", "local telemetry",
    "launchd state", "durable state", "across sessions",
    # this repo
    "this repo", "repo's files",
    # live tools / side effects
    "live tool", "side effect", "live git", "real hardware", "orchestrates workflow",
    "long-running", "credential", "chrome profile",
    # image generation
    "image generation", "no delegate path", "wrong likeness", "real people",
    # judgement Mohan pays for
    "judgement mohan", "judgment mohan", "mohan in the loop", "live request",
    "chairman is native", "chairman synthesis native", "cannot be delegated",
)
BILLING = re.compile(r"(?<![\w-])ant(?![\w-])|ANTHROPIC_API_KEY|managed.agent", re.IGNORECASE)


def frontmatter(text: str) -> dict[str, str]:
    m = re.match(r"^---\n(.*?)\n---\n", text, re.S)
    if not m:
        return {}
    out: dict[str, str] = {}
    for line in m.group(1).splitlines():
        if ":" in line and not line.startswith(" "):
            k, v = line.split(":", 1)
            out[k.strip()] = v.strip()
    return out


def tools_of(fm: dict[str, str], text: str = "") -> list[str]:
    """Union of EVERY `tools:` line in the file, frontmatter or not.

    The C11 probe appends `tools: mcp__brain-router__delegate, Bash` with `>>`,
    which lands after the closing --- and never enters the frontmatter. A
    frontmatter-only read passed that probe -- an unenforcing lint, the STOP
    condition -- so a second tool declared ANYWHERE now counts."""
    found: list[str] = []
    for line in re.findall(r"(?m)^tools:\s*(.*)$", text) or [fm.get("tools", "")]:
        for t in line.split(","):
            t = t.strip()
            if t and t not in found:
                found.append(t)
    return found


def lane_of(fm: dict[str, str], text: str) -> str | None:
    lane = fm.get("lane", "").strip().lower()
    if lane in ("deck", "native"):
        return lane
    tools = tools_of(fm, text)
    if DELEGATE_TOOL in tools and fm.get("lane", "") == "":
        return "deck"   # holds the delegate tool: lane agent, however many others it grew
    if "agent:deck" in text:
        return "deck"
    if "agent:native" in text:
        return "native"
    return None


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser(description="C11 agent lint (read-only)")
    ap.add_argument("--agents-dir", default=str(AGENTS))
    agents = Path(ap.parse_args().agents_dir)
    failures: list[str] = []
    files = sorted(agents.glob("*.md")) if agents.is_dir() else []
    for f in files:
        text = f.read_text(encoding="utf-8")
        fm = frontmatter(text)
        name = fm.get("name") or f.stem
        lane = lane_of(fm, text)
        tools = tools_of(fm, text)

        if lane == "deck":
            # 1 -- exactly one tool, and it is the delegate tool
            if tools != [DELEGATE_TOOL]:
                extra = [t for t in tools if t != DELEGATE_TOOL]
                failures.append(f"{name}: delegate-lane agent holds {len(tools)} tools "
                                f"{tools}; must be exactly [{DELEGATE_TOOL}]"
                                + (f" -- extra: {extra}" if extra else ""))
            # 1b -- agentic mode + real cwd named
            if "agentic" not in text or "cwd" not in text:
                failures.append(f"{name}: agent:deck must name mode=agentic and a real cwd; "
                                "in chat mode the work never happens")
            # 3 -- IRR never in the lane. Keyed on irr, never on q3.
            if fm.get("irr", "false").strip().lower() == "true":
                failures.append(f"{name}: irr: true in the delegate lane -- an irreversible job "
                                "hard-floors to B4 and may not run behind a one-tool agent")
            # 3b -- a collapse needs a named executor from the closed list
            q = fm.get("q_scores", "")
            collapsed = fm.get("collapsed", "").strip().lower() == "true" or "→1" in q or "->1" in q
            if collapsed:
                ex = fm.get("executor", "").strip().strip("'\"")
                if ex not in EXECUTORS:
                    failures.append(f"{name}: Q4=1 collapse declared but executor {ex!r} is not "
                                    f"on the closed list {sorted(EXECUTORS)}")
        elif lane == "native":
            # 4 -- a named exception, from the accepted list
            low = text.lower()
            if not any(r in low for r in NATIVE_REASONS):
                failures.append(f"{name}: agent:native names no accepted exception "
                                "(this session's state / this repo's files / live tools or side "
                                "effects / image generation / a judgement Mohan is paying for)")

        # 2 -- no billing path, every file
        for m in BILLING.finditer(text):
            failures.append(f"{name}: billing reference {m.group(0)!r} (SC10)")
            break

    # 2 -- and every script in bin/
    for s in sorted(SCRIPTS.glob("*.py")):
        if s.name == Path(__file__).name:
            continue  # this file names the forbidden strings in order to forbid them
        for m in BILLING.finditer(s.read_text(encoding="utf-8")):
            failures.append(f"bin/{s.name}: billing reference {m.group(0)!r} (SC10)")
            break

    for line in failures:
        print("FAIL", line)
    print(f"agent-lint: {len(files)} agent files, {len(failures)} failure(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
