"""Live smoke for the apple-fm simple tier: real ssh to the mini, real fall-through.

    uv run python specs/003-apple-fm-simple-tier/smoke.py <case>

Cases extract/oversize/unreachable/toolshaped use the live config with
simple = [apple-fm, haiku]; `live` uses the live config unmodified; `audit`
checks the rows those runs wrote. Rows carry BRAIN_ROUTER_BENCH_RUN_ID so they
stay out of the rig's real cost history (C07). Exit 0 = pass.
"""
import asyncio
import dataclasses
import json
import os
import sys
from pathlib import Path

RUN_ID = "fm-smoke-20261023"
os.environ["BRAIN_ROUTER_BENCH_RUN_ID"] = RUN_ID
os.environ["BRAIN_ROUTER_CALLER"] = "claude"

from mcp_brain_router import server  # noqa: E402
from mcp_brain_router.config import Config  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
MARKER = "Acme Labs"  # in the extract prompt; must never reach the audit log
EXTRACT = (
    "Extract the invoice number and the total due. Reply as `number, total`.\n\n"
    f"Invoice 4471 from {MARKER}, dated 12 March 2026. Billed to Ravi Kumar. "
    "Total due: INR 18,450, payable by 30 March 2026."
)


def run(prompt, config):
    server._config = config  # _load_config() returns this
    r = asyncio.run(server._delegate_role_impl("simple", prompt, "claude", cwd=str(REPO)))
    print(json.dumps({k: r.get(k) for k in ("backend", "tried", "failures", "error", "elapsed_ms")}))
    print("answer:", (r.get("answer") or "")[:200])
    return r


def override(**kw):
    base = Config.load()
    roles = dict(base.roles, simple=["apple-fm", "claude-haiku-4-5-20251001"])
    return dataclasses.replace(base, roles=roles, fm_host=kw.get("fm_host", "mini"))


def fell_through(r, kind):
    return (r.get("tried", [])[:1] == ["apple-fm"] and r.get("backend") == "anthropic-cli"
            and f"apple-fm:{kind}" in (r.get("failures") or []) and r.get("answer") and "error" not in r)


def main(case):
    if case == "extract":
        r = run(EXTRACT, override())
        return r.get("backend") == "apple-fm" and "4471" in (r.get("answer") or "")
    if case == "oversize":
        filler = "The quick brown fox jumps over the lazy dog. " * 600  # ~27k chars
        r = run("Reply with exactly: ok. Ignore the filler below.\n\n" + filler, override())
        return fell_through(r, "input_too_large")
    if case == "unreachable":
        r = run(EXTRACT, override(fm_host="fm-unreachable.invalid"))
        return fell_through(r, "unreachable") and r.get("tried") == ["apple-fm", "anthropic-cli"]
    if case == "toolshaped":
        r = run(f"Read {REPO}/STATUS.md and reply with its first heading only.", override())
        return fell_through(r, "needs_tools")
    if case == "live":
        config = Config.load()
        assert config.roles["simple"][0] == "apple-fm" and config.fm_host, config.roles["simple"]
        r = run(EXTRACT, config)
        return r.get("backend") == "apple-fm" and "4471" in (r.get("answer") or "")
    if case == "audit":
        log = server._DELEGATION_LOG
        text = log.read_text()
        rows = [json.loads(line) for line in text.splitlines() if RUN_ID in line]
        fm_rows = [r for r in rows if "apple-fm" in (r.get("tried") or [])]
        kinds = {f.split(":", 1)[1] for r in fm_rows for f in r.get("failures") or []}
        checks = {
            "fm rows exist": bool(fm_rows),
            "an fm answer was logged": any(r.get("backend") == "apple-fm" for r in fm_rows),
            "skip reasons logged": {"input_too_large", "unreachable", "needs_tools"} <= kinds,
            "all smoke rows carry the bench id": all(r.get("bench_run_id") == RUN_ID for r in rows),
            "no prompt body in the log": MARKER not in text,
            "no answer/prompt keys": not any({"answer", "prompt"} & r.keys() for r in rows),
        }
        for name, ok in checks.items():
            print("PASS" if ok else "FAIL", name)
        return all(checks.values())
    raise SystemExit(f"unknown case {case!r}")


if __name__ == "__main__":
    ok = main(sys.argv[1])
    print("PASS" if ok else "FAIL", sys.argv[1])
    sys.exit(0 if ok else 1)
