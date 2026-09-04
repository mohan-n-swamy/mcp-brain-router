#!/usr/bin/env python3
"""run-bench.py — C07, the benchmark harness (specs/001-agent-capability-routing).

Runs real jobs against real routing and records MEASURED cost and SCORED quality.
Owns bench/baseline.json and bench/results.json. Writes neither deck.json nor
cards.json -- C07's boundary: the benchmark feeds ranking, it must not perform it,
or the measurement and the thing measured become one artifact.

--baseline is the reason this exists and the reason it runs first. It measures the
8 frozen fixtures on TODAY's routing. prd.md R6 makes the headline result a delta,
and a delta needs a before; today's routing stops existing the moment C04 ships,
so this number is unrecoverable if skipped (execution-plan.md slice 0b).

Governed by specs/001-agent-capability-routing/.rigor.md. The must-not rows are
mechanised here, not merely intended:
  #1 no estimated cost      -> cost_basis has no default; unmeasurable writes "unmeasured"
  #3 no partial as whole    -> meta.planned set before the first call, meta.complete last
  #4 no rotating in a delta -> --baseline filters rotating and takes no job list
  #5 no similarity at B4/B5 -> refused at LOAD, before any spend
  #8 no log pollution       -> BRAIN_ROUTER_BENCH_RUN_ID stamps every record
 #11 no unapproved spend    -> --dry-run and --baseline are separate invocations
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import statistics
import subprocess
import sys
import pathlib
import shutil
import tempfile
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SPEC = Path.home() / "code workshop" / "specs" / "001-agent-capability-routing"
JOBS = SPEC / "bench" / "jobs"
ROUTING = SPEC / "bench" / "routing-today.json"
HEADROOM = Path.home() / ".local" / "state" / "headroom.json"
DELEG_LOG = Path.home() / ".local" / "state" / "brain-router-delegations.jsonl"
TRIALS = 3  # R30: median of 3. One run lets a 534s hang or a lucky 48s call decide a band.

sys.path.insert(0, str(REPO / "src"))


_JUDGE_SPEND: list[dict] = []  # scoring calls: real spend, never folded into job cost


class Refused(Exception):
    """A STOP condition from C07. Raised at load, before anything is spent."""


# ---------------------------------------------------------------- load + refuse

def load_fixtures(frozen_only: bool) -> list[dict]:
    fx = [json.loads(p.read_text()) for p in sorted(JOBS.glob("*.json"))]
    if not fx:
        raise Refused(f"no fixtures under {JOBS}")
    for f in fx:
        # C07 STOP: the runner must REFUSE a B4/B5 job scored by similarity. At those
        # bands the reference is an answer Opus produced, so similarity measures
        # Opus-likeness and marks a different-but-valid finding set wrong (R29).
        if f["band"] in ("B4", "B5") and f["scoring"] not in ("rubric",):
            raise Refused(
                f"{f['id']}: band {f['band']} scored by {f['scoring']!r}. "
                "B4/B5 score on rubric only (R29) -- refusing before any spend."
            )
        # C07 STOP: a job with no reference answer measures nothing and cannot be scored.
        if f["scoring"] == "reference" and not f.get("reference", {}).get("answer"):
            raise Refused(f"{f['id']}: scoring=reference but no reference answer.")
        if not any(a.get("p0") for a in f.get("acceptance", [])):
            raise Refused(f"{f['id']}: no P0 acceptance item (R28).")
    if frozen_only:
        fx = [f for f in fx if not f["rotating"]]
    return fx


def load_routing(fixtures: list[dict]) -> dict:
    """The 'before' routing table. Must-not #2: a job is never silently re-routed,
    so the table is a reviewed file, not a heuristic buried in this script."""
    if not ROUTING.exists():
        raise Refused(f"missing {ROUTING} -- the baseline cannot guess today's routing.")
    r = json.loads(ROUTING.read_text())
    missing = [f["id"] for f in fixtures if f["id"] not in r["jobs"]]
    if missing:
        raise Refused(f"routing-today.json has no row for: {missing}")
    return r


def fixture_set_hash(fixtures: list[dict]) -> str:
    """A changed prompt invalidates the series (bench/README.md). The hash makes
    that detectable rather than a thing someone has to remember."""
    blob = json.dumps([{"id": f["id"], "prompt": f["input"]["prompt"]} for f in fixtures],
                      sort_keys=True).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def read_headroom() -> dict:
    try:
        return json.loads(HEADROOM.read_text())
    except Exception as e:
        return {"error": str(e)}


def git_head() -> str:
    try:
        return subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception:
        return "unknown"


# ---------------------------------------------------------------- execution

def _attachment_text(fixture: dict, scratch: str) -> str:
    """Attachments are paths, resolved against the vault. Keep them as paths in the
    prompt where the executor has a cwd -- it Reads them itself."""
    # COPY each attachment into the trial's scratch dir and reference the COPY.
    # A scratch cwd alone does not contain these jobs: passing an absolute vault
    # path invites the worker to write back to it, and that is exactly what
    # happened -- b4-security-review was asked to REVIEW bin/fetch-provider-table.py
    # and instead rewrote it in place, 76 insertions into a production file, while
    # b3-write-test added a test to the repo. Both runs did it. The job stays real
    # because the content is identical; only the path it can reach changes.
    vault = pathlib.Path.home() / "code workshop"
    box = pathlib.Path(scratch)
    out = []
    for a in fixture["input"].get("attachments", []):
        q = pathlib.Path(a)
        src = q if q.is_absolute() else (vault / q)
        dst = box / src.name
        if src.is_dir():
            shutil.copytree(src, dst, dirs_exist_ok=True)
        elif src.is_file():
            shutil.copy2(src, dst)
        else:
            # A job whose source document is absent measures nothing, and it will
            # still score: b2-read-doc-answer pointed at a path that does not exist
            # and returned a P0 PASS in two consecutive runs, answering from the
            # model rather than from the document it was supposed to read.
            raise Refused(f"{fixture['id']}: attachment not found: {src}")
        out.append(f"- {dst}")
    return "\n".join(out)


def scratch_cwd(run_id: str) -> str:
    """Where a trial is allowed to WRITE. The jobs are agentic and they do write:
    b3-write-test put a real pytest file into the router repo, and both b4 reviews
    wrote their findings to disk instead of returning them. With the vault as cwd a
    trial can overwrite real work, and a benchmark that mutates what it measures is
    not a measurement. Reads still reach the vault -- attachments go in as absolute
    paths -- so the job stays real while its side effects stay contained."""
    d = pathlib.Path(tempfile.gettempdir()) / f"bench-{run_id}"
    d.mkdir(parents=True, exist_ok=True)
    return str(d)


def build_prompt(fixture: dict, scratch: str) -> str:
    body = fixture["input"]["prompt"]
    att = _attachment_text(fixture, scratch)
    return f"{body}\n\nFiles:\n{att}" if att else body


async def run_role(fixture: dict, role: str, cwd: str) -> dict:
    """One trial through the router, today's shards, unmodified."""
    from mcp_brain_router.server import _delegate_role_impl
    t0 = time.perf_counter()
    r = await _delegate_role_impl(
        role=role, prompt=build_prompt(fixture, cwd),
        orchestrator="claude", mode="agentic", cwd=cwd,
    )
    answer = (r.get("answer") or "").strip()
    # memory rig_brain_router_simple_lane_dead_on_glm: the lane returned rc 0 with
    # answer=null after 190s. An empty answer is a FAILED trial, never a cheap success.
    ok = bool(answer) and not r.get("error") and not r.get("exhausted")
    return {
        "ok": ok, "answer": answer, "backend": r.get("backend"), "model": r.get("model"),
        "tokens_in": r.get("tokens_in"), "tokens_out": r.get("tokens_out"),
        "cache_read_input_tokens": r.get("cache_read_input_tokens"),
        "usage_source": r.get("usage_source"), "cost_usd": r.get("cost_usd"),
        "elapsed_ms": r.get("elapsed_ms") or round((time.perf_counter() - t0) * 1000),
        "failure_kind": r.get("failure_kind"), "fell_back": bool(r.get("fell_back")),
    }


def run_native(fixture: dict, cwd: str) -> dict:
    """A job the rig does NOT route today. Measured the same way, through the
    Claude CLI's own JSON envelope, which carries total_cost_usd directly (C16).
    Recording this as a role would flatter the after-number (must-not #2)."""
    t0 = time.perf_counter()
    try:
        p = subprocess.run(
            ["claude", "-p", build_prompt(fixture, cwd), "--output-format", "json"],
            capture_output=True, text=True, cwd=cwd, timeout=900,
        )
        d = json.loads(p.stdout) if p.returncode == 0 else {}
    except Exception as e:
        return {"ok": False, "answer": "", "backend": "native", "model": None,
                "cost_usd": None, "usage_source": None, "tokens_in": None,
                "tokens_out": None, "cache_read_input_tokens": None,
                "elapsed_ms": round((time.perf_counter() - t0) * 1000),
                "failure_kind": type(e).__name__, "fell_back": False}
    u = d.get("usage") or {}
    answer = (d.get("result") or "").strip()
    return {
        "ok": bool(answer), "answer": answer, "backend": "native",
        "model": d.get("model") or "claude-cli",
        "tokens_in": u.get("input_tokens"), "tokens_out": u.get("output_tokens"),
        "cache_read_input_tokens": u.get("cache_read_input_tokens"),
        "usage_source": "cli-json" if u else None,
        "cost_usd": d.get("total_cost_usd"),
        "elapsed_ms": d.get("duration_ms") or round((time.perf_counter() - t0) * 1000),
        "failure_kind": None, "fell_back": False,
    }


# ---------------------------------------------------------------- scoring

def mechanically_scorable(fixture: dict) -> bool:
    """Mechanical scoring is legal ONLY where the reference is a list of strings and
    the acceptance items are set relations. b1 is that shape. The other reference
    fixtures carry PROSE checks -- "answers NO", "names the RTK proxy as the cause",
    "does not name a caller as the definition" -- which a subset test cannot evaluate.
    Scoring those mechanically would produce a number with nothing behind it, which is
    the thing must-not #9 exists to prevent, so they go to the judge instead."""
    return (fixture["scoring"] == "reference"
            and isinstance(fixture.get("reference", {}).get("answer"), list))


def score_reference(fixture: dict, answer: str) -> list[dict]:
    """Set comparison. Guarded by mechanically_scorable() -- never called otherwise."""
    ref = fixture["reference"]["answer"]
    want = [str(x).strip().lower() for x in (ref if isinstance(ref, list) else [ref])]
    got = [l.strip().lower() for l in answer.splitlines() if l.strip()]
    got_set, want_set = set(got), set(want)
    out = []
    for item in fixture["acceptance"]:
        cid = item["id"]
        if cid == "a1":
            passed = want_set.issubset(got_set)
        elif cid == "a2":
            passed = not (got_set - want_set)
        else:
            passed = got == sorted(set(got)) and len(got) == len(set(got))
        out.append({"id": cid, "p0": bool(item["p0"]), "check": item["check"],
                    "passed": bool(passed), "scored_by": "mechanical"})
    return out


async def score_rubric(fixture: dict, answer: str, cwd: str) -> list[dict]:
    """B3-B5: judged against the acceptance items, never against the reference text.
    A separate call, so its cost is never mixed into the job's cost."""
    from mcp_brain_router.server import _delegate_role_impl
    items = [{"id": a["id"], "check": a["check"]} for a in fixture["acceptance"]]
    prompt = (
        "Score an answer against binary acceptance items. Judge ONLY against the items.\n"
        "Do not reward resemblance to any particular style or author.\n\n"
        f"TASK GIVEN:\n{fixture['input']['prompt']}\n\n"
        f"ANSWER TO SCORE:\n{answer}\n\n"
        f"ACCEPTANCE ITEMS:\n{json.dumps(items, indent=2)}\n\n"
        'Reply with JSON only: {"results":[{"id":"a1","passed":true,"why":"..."}]}'
    )
    r = await _delegate_role_impl(role="adversary", prompt=prompt,
                                 orchestrator="claude", mode="agentic", cwd=cwd)
    raw = (r.get("answer") or "").strip()
    _JUDGE_SPEND.append({"backend": r.get("backend"), "cost_usd": r.get("cost_usd"),
                         "tokens_in": r.get("tokens_in")})
    verdicts = {}
    try:
        start, end = raw.find("{"), raw.rfind("}")
        for v in json.loads(raw[start:end + 1])["results"]:
            verdicts[v["id"]] = v
    except Exception:
        verdicts = {}
    out = []
    for item in fixture["acceptance"]:
        v = verdicts.get(item["id"])
        out.append({"id": item["id"], "p0": bool(item["p0"]), "check": item["check"],
                    # An unparseable judge is NOT a pass. It is an unscored item, and
                    # SC15 makes an unscored run unrankable rather than optimistic.
                    "passed": (None if v is None else bool(v.get("passed"))),
                    "why": (v or {}).get("why"),
                    "scored_by": f"judge:{r.get('backend')}"})
    return out


# ---------------------------------------------------------------- plan + run

def plan(fixtures: list[dict], routing: dict) -> dict:
    rows = []
    job_calls = judge_calls = 0
    for f in fixtures:
        route = routing["jobs"][f["id"]]["route"]
        jc = TRIALS
        kc = 0 if mechanically_scorable(f) else TRIALS
        job_calls += jc
        judge_calls += kc
        rows.append({"id": f["id"], "band": f["band"], "size_class": f["size_class"],
                     "scoring": f["scoring"],
                     "scored_by": "mechanical" if mechanically_scorable(f) else "judge",
                     "route": route,
                     "job_calls": jc, "judge_calls": kc,
                     "why": routing["jobs"][f["id"]]["why"]})
    return {"rows": rows, "job_calls": job_calls, "judge_calls": judge_calls,
            "total_calls": job_calls + judge_calls}


# The router and the deck name the same providers differently: the router's Provider
# enum says kimi/codex, while cards.json and headroom.json say moonshot/openai. C03 is
# unaffected (it looks up Card.provider against headroom, and both say moonshot), but
# anything crossing the two namespaces must translate or it silently reads nothing --
# which is how kimi at 68% first displayed here as "n/a". C04 owns the real fix.
ROUTER_TO_DECK = {"kimi": "moonshot", "codex": "openai"}


def first_candidate_load(p: dict) -> dict[str, int]:
    """Which provider absorbs each planned call, per the router's OWN resolution.
    resolve_role touches no backend, so this is free. Falls back to naming the role
    rather than guessing a provider if the config cannot be read."""
    from mcp_brain_router.config import Config
    from mcp_brain_router.router import Role, resolve_role
    try:
        cfg = Config.load()
    except Exception:
        cfg = None
    load: dict[str, int] = {}

    def first(role_name: str) -> str:
        if cfg is None:
            return f"?{role_name}"
        try:
            a = resolve_role(Role(role_name), "claude", cfg, mode="agentic")
            v = getattr(a.provider, "value", str(a.provider))
            return ROUTER_TO_DECK.get(v, v)
        except Exception:
            return f"?{role_name}"

    for r in p["rows"]:
        if r["job_calls"]:
            key = "native (claude -p)" if r["route"] == "native" else first(r["route"])
            load[key] = load.get(key, 0) + r["job_calls"]
        if r["judge_calls"]:
            key = first("adversary")
            load[key] = load.get(key, 0) + r["judge_calls"]
    return load


def print_plan(p: dict, fixtures: list[dict], routing: dict) -> None:
    print("\nPLANNED RUN — nothing has been spent.\n")
    print(f"{'job':24} {'band':5} {'size':6} {'scoring':9} {'scored by':10} {'route':7} {'calls':>5}")
    print("-" * 77)
    for r in p["rows"]:
        print(f"{r['id']:24} {r['band']:5} {r['size_class']:6} {r['scoring']:9} "
              f"{r['scored_by']:10} {r['route']:7} {r['job_calls'] + r['judge_calls']:>5}")
    print("-" * 77)
    print(f"{'job calls (n=3 each)':<52}{p['job_calls']:>6}")
    print(f"{'judge calls (every non-mechanical job, n=3 each)':<52}{p['judge_calls']:>6}")
    print(f"{'TOTAL PROVIDER CALLS':<52}{p['total_calls']:>6}\n")
    print("Routing today — must-not #2, review this against ~/.agents/skills/delegate/SKILL.md:")
    for r in p["rows"]:
        print(f"  {r['id']:24} -> {r['route']:7}  {r['why']}")
    # must-not #10: an exhaustion mid-run changes which shard answers, so the "before"
    # stops being today's routing. Printing headroom does not surface that -- what
    # matters is WHICH provider absorbs the calls, resolved by the router's own
    # skip-self logic (resolve_role calls no backend, so this spends nothing).
    h = read_headroom()
    load = first_candidate_load(p)
    print("\nWhere these calls actually land, and the quota they land on:")
    provs = h.get("providers") or {}
    for prov, n in sorted(load.items(), key=lambda kv: -kv[1]):
        uw = (provs.get(prov) or {}).get("used_week")
        shown = "n/a" if uw is None else f"{uw:.0%}"
        warn = "  <-- HIGH" if isinstance(uw, (int, float)) and uw >= 0.60 else ""
        print(f"  {prov:10} first-candidate for {n:>3} of {p['total_calls']} calls"
              f"   used_week={shown}{warn}")
    print("\nFull headroom:")
    for prov, v in provs.items():
        uw = v.get("used_week")
        print(f"  {prov:10} used_week={'n/a' if uw is None else f'{uw:.0%}'}")
    print(f"\nfixture-set hash: {fixture_set_hash(fixtures)}")
    print(f"reviewed_by: {routing.get('reviewed_by')!r}")
    print("\nNothing was spent. To run for real: python3 bin/run-bench.py --baseline\n")


async def execute(fixtures: list[dict], routing: dict, p: dict, cwd: str,
                  out_path: Path, run_id: str) -> int:
    meta = {
        "mode": "baseline", "run_id": run_id,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "router_head": git_head(), "fixture_set_hash": fixture_set_hash(fixtures),
        "trials_per_cell": TRIALS, "planned": len(fixtures),
        # `claude -p` has a smaller context and no conversation history, so a native
        # row is the honest measurable stand-in for in-session work -- not its cost.
        # Named here so nobody later reads these rows as the live session's spend.
        "native_proxy": "claude -p --output-format json",
        # Measured, not assumed: kimi is first candidate for worker AND simple but is
        # failing fast (process_error) as of this run, so calls advance to glm. That IS
        # today's routing and is recorded as such, not corrected for.
        "shard_note": "kimi first-candidate; failing process_error at run time",
        "scratch_cwd": scratch_cwd(run_id),
        "planned_calls": p["total_calls"],
        "headroom_at_start": read_headroom(),
        "routing_reviewed_by": routing.get("reviewed_by"),
        "complete": False,
    }
    def flush(rows: list, complete: bool) -> None:
        """Persist after every fixture. Money already spent must survive a crash on the
        next row -- and a partial file that SAYS it is partial is exactly what must-not
        #3 asks for. Same atomic temp+replace; a reader never sees a half-written file."""
        meta["complete"] = complete
        meta["judge_calls_spend"] = {
            "calls": len(_JUDGE_SPEND),
            "measured_usd": sum(x["cost_usd"] for x in _JUDGE_SPEND
                                if isinstance(x["cost_usd"], (int, float))),
            "unmeasured_calls": sum(1 for x in _JUDGE_SPEND if x["cost_usd"] is None),
            "note": "scoring spend, never folded into any job's cost",
        }
        meta["finished_at"] = datetime.now(timezone.utc).isoformat()
        meta["headroom_at_end"] = read_headroom()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=str(out_path.parent), prefix=".bench.")
        with os.fdopen(fd, "w") as fh:
            json.dump({"meta": meta, "results": rows}, fh, indent=2)
            fh.write("\n")
        os.replace(tmp, out_path)

    results = []
    for f in fixtures:
        route = routing["jobs"][f["id"]]["route"]
        trials = []
        for i in range(TRIALS):
            print(f"  {f['id']} trial {i + 1}/{TRIALS} via {route} ...", flush=True)
            box = scratch_cwd(run_id)
            t = run_native(f, box) if route == "native" else await run_role(f, route, box)
            if t["ok"]:
                t["acceptance"] = (score_reference(f, t["answer"])
                                   if mechanically_scorable(f)
                                   else await score_rubric(f, t["answer"], box))
            else:
                t["acceptance"] = None
            t.pop("answer", None)  # keep the file about cost and score, not transcripts
            trials.append(t)

        costs = [t["cost_usd"] for t in trials if t["ok"] and isinstance(t["cost_usd"], (int, float))]
        secs = [t["elapsed_ms"] for t in trials if isinstance(t["elapsed_ms"], (int, float))]
        scored = [t["acceptance"] for t in trials if t["acceptance"]]
        p0 = None
        if scored:
            # A card that misses any P0 item is unrankable in that band regardless of
            # cost (R28). An unscored item is not a pass.
            p0 = all(a["passed"] is True for tr in scored for a in tr if a["p0"])
        results.append({
            "id": f["id"], "band": f["band"], "size_class": f["size_class"],
            "scoring": f["scoring"], "rotating": f["rotating"], "route": route,
            "backend": next((t["backend"] for t in trials if t["ok"]), None),
            # The median blends three trials. If the shard advances mid-row -- kimi is
            # currently failing fast, so a recovery would send some trials to kimi and
            # some to glm -- the median stops describing one provider. Recording every
            # backend touched makes that visible instead of hiding it behind the first.
            "backends_all": sorted({t["backend"] for t in trials if t["ok"]}),
            "mixed_backends": len({t["backend"] for t in trials if t["ok"]}) > 1,
            "model": next((t["model"] for t in trials if t["ok"]), None),
            # must-not #1: no estimate, ever. Unmeasurable is a state, not a gap to fill.
            "total_cost_usd": statistics.median(costs) if costs else None,
            "cost_basis": "measured" if costs else "unmeasured",
            "cost_spread_usd": (max(costs) - min(costs)) if len(costs) > 1 else None,
            "wall_clock_seconds": round(statistics.median(secs) / 1000, 1) if secs else None,
            "trials": trials, "trials_ok": sum(1 for t in trials if t["ok"]),
            # Per-item pass COUNT across trials, not trial 1's verdict. Storing the
            # first trial made the row contradict itself: b1 showed every item passed
            # beside p0_pass=False, because trial 2 had failed and only the median
            # verdict knew. A summary that disagrees with its own verdict is worse
            # than no summary.
            "acceptance": ([{"id": a["id"], "p0": a["p0"], "check": a["check"],
                             "passed_trials": sum(1 for tr in scored for b in tr
                                                  if b["id"] == a["id"] and b["passed"] is True),
                             "of_trials": len(scored),
                             "scored_by": a["scored_by"]}
                            for a in scored[0]] if scored else None),
            "p0_pass": p0,
        })
        # must-not #3: complete is derived from the data, never set by hand.
        flush(results, complete=len(results) == meta["planned"])

    print(f"\nwrote {out_path}  complete={meta['complete']}  rows={len(results)}")
    return 0 if meta["complete"] else 1


async def probe(by_id: dict, routing: dict, cwd: str, run_id: str) -> int:
    """One role call and one native call, scored the way the real run scores them."""
    picks = [("b1-list-wikilinks", "role"), ("b2-find-symbol", "native")]
    ok_all = True
    for jid, kind in picks:
        f = by_id[jid]
        route = routing["jobs"][jid]["route"]
        print(f"\n=== {jid}  route={route}  ({kind} path)")
        box = scratch_cwd(run_id)
        t = run_native(f, box) if route == "native" else await run_role(f, route, box)
        print(f"  ok={t['ok']} backend={t['backend']} model={t['model']}")
        print(f"  tokens_in={t['tokens_in']} tokens_out={t['tokens_out']} "
              f"cache_read={t['cache_read_input_tokens']}")
        print(f"  usage_source={t['usage_source']} cost_usd={t['cost_usd']} "
              f"elapsed_ms={t['elapsed_ms']}")
        if not t["ok"]:
            print(f"  FAILED: failure_kind={t['failure_kind']}")
            ok_all = False
            continue
        print(f"  answer[:200]: {t['answer'][:200]!r}")
        acc = (score_reference(f, t["answer"]) if mechanically_scorable(f)
               else await score_rubric(f, t["answer"], box))
        for a in acc:
            print(f"    {a['id']} p0={a['p0']} passed={a['passed']} "
                  f"via={a['scored_by']}  ({a['check']})")
        if any(a["passed"] is None for a in acc):
            print("  NOTE: an item came back unscored — the judge did not parse.")
            ok_all = False
        if t["cost_usd"] is None:
            print("  NOTE: no cost measured on this path.")
            ok_all = False
    print(f"\nprobe {run_id}: {'all paths returned measurable, scored results'
                              if ok_all else 'SOMETHING DID NOT WORK — see NOTE/FAILED above'}")
    print("Nothing was written. baseline.json untouched.")
    return 0 if ok_all else 1


def main() -> int:
    ap = argparse.ArgumentParser(description="C07 benchmark harness")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--dry-run", action="store_true", help="print the plan, spend nothing")
    g.add_argument("--baseline", action="store_true",
                   help="run the 8 frozen fixtures on TODAY's routing")
    g.add_argument("--probe", action="store_true",
                   help="2 calls: one role trial, one native trial. Writes nothing.")
    ap.add_argument("--cwd", default=str(Path.home() / "code workshop"))
    args = ap.parse_args()

    try:
        # --baseline takes no job list, by construction: the delta series is the
        # frozen 8 and nothing else (must-not #4).
        fixtures = load_fixtures(frozen_only=True)
        routing = load_routing(fixtures)
    except Refused as e:
        print(f"REFUSED: {e}", file=sys.stderr)
        return 2

    if args.probe:
        # The guards all exit at load, so run_role / run_native / score_* have never
        # executed. Two calls through the REAL functions, before committing 45 to a
        # path nobody has seen work. C16's own lesson: probe before wiring.
        run_id = f"probe-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}"
        os.environ["BRAIN_ROUTER_BENCH_RUN_ID"] = run_id
        by_id = {f["id"]: f for f in fixtures}
        return asyncio.run(probe(by_id, routing, args.cwd, run_id))

    p = plan(fixtures, routing)
    if args.dry_run:
        print_plan(p, fixtures, routing)
        return 0

    if not routing.get("reviewed_by"):
        print("REFUSED: routing-today.json is unreviewed (reviewed_by is null).\n"
              "         Must-not #2 is HUMAN: no command settles whether these routes\n"
              "         are what the rig actually does today. Run --dry-run, review the\n"
              "         table, then set reviewed_by/reviewed_at.", file=sys.stderr)
        return 2

    run_id = f"bench-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{uuid.uuid4().hex[:6]}"
    os.environ["BRAIN_ROUTER_BENCH_RUN_ID"] = run_id  # must-not #8
    print(f"baseline run {run_id} — {p['total_calls']} planned calls")
    return asyncio.run(execute(fixtures, routing, p, args.cwd,
                               SPEC / "bench" / "baseline.json", run_id))


if __name__ == "__main__":
    sys.exit(main())
