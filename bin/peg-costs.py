#!/usr/bin/env python3
"""peg-costs.py — give a card a cost basis when its CLI reports no usage.

Kimi returns no token counts, so a run routed to kimi comes back `unmeasured`
and, under R21, unrankable. That is the correct default -- but it is not the
only option, and treating it as one would route the deck around the provider
the rig leans on hardest for the wrong reason.

R38 already sanctions the mechanism: cost = tokens x fetched price. The tokens
here are MEASURED, on the same fixture, from a run that landed on a backend
which does report them; the price is FETCHED by C01, not typed. So a peg is an
arithmetic combination of two measurements, not a guess -- which is exactly the
distinction R38 draws when it allows a stand-in and forbids an estimate.

It is still weaker than a direct measurement, and it is labelled so:
  cost_basis "measured" -- the backend reported its own usage
  cost_basis "pegged"   -- tokens from a paired run, price from cards.json
  cost_basis "unmeasured" -- neither available
A pegged row carries the fixture, run and card it was derived from, so anyone
can recompute it or throw it away.

WHAT A PEG CANNOT TELL YOU. Input tokens are mostly a property of the job --
same prompt, same attachments, same harness context -- so pegging them across
backends is sound. Output tokens are a property of the MODEL: a terser model
genuinely costs less, and borrowing another model's output count hides exactly
the difference the benchmark exists to find. Both components are reported
separately so the borrowed half is visible rather than buried in one number.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

STATE = pathlib.Path.home() / ".local" / "state" / "brain-router"
SPEC = pathlib.Path.home() / "code workshop" / "specs" / "001-agent-capability-routing"

# The router calls the CLI's configured default. README + config name Kimi K3.
DEFAULT_CARD = {"kimi": "kimi-k3", "moonshot": "kimi-k3"}


def price_of(slug: str) -> tuple[float, float]:
    cards = json.loads((STATE / "cards.json").read_text())["cards"]
    hit = next((c for c in cards if c["slug"] == slug), None)
    if not hit:
        raise SystemExit(f"no card {slug!r} in cards.json")
    return float(hit["price_blended"]), float(hit.get("capability") or 0)


def peg(target: dict, donor: dict, slug: str) -> dict | None:
    """Cost for one target row, using the donor row's measured token counts."""
    dt = donor["trials"][0]
    ti, to = dt.get("tokens_in"), dt.get("tokens_out")
    if not isinstance(ti, (int, float)) or not isinstance(to, (int, float)):
        return None
    price, cap = price_of(slug)
    # price_blended is $/1M on a 3:1 in:out blend, which is how C01 fetched it.
    usd = (ti + to) / 1_000_000 * price
    return {
        "total_cost_usd": round(usd, 6),
        "cost_basis": "pegged",
        "peg": {
            "card": slug,
            "price_blended_per_1m": price,
            "capability": cap,
            "tokens_in": ti,
            "tokens_out": to,
            "tokens_from": {"run": donor.get("_run"), "backend": donor["backend"],
                            "fixture": donor["id"]},
            "caveat": "output tokens borrowed from a different model; input tokens are "
                      "a property of the job and travel across backends soundly",
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="peg unmeasured rows from a donor run")
    ap.add_argument("--target", default=str(SPEC / "bench" / "baseline.json"))
    ap.add_argument("--donor", required=True, help="a run whose rows carry token counts")
    ap.add_argument("--write", action="store_true", help="write back; default prints only")
    a = ap.parse_args()

    tgt = json.loads(pathlib.Path(a.target).read_text())
    don = json.loads(pathlib.Path(a.donor).read_text())
    donor_run = don["meta"]["run_id"]
    by_id = {r["id"]: dict(r, _run=donor_run) for r in don["results"]}

    print(f"{'job':22} {'backend':8} {'basis':10} {'cost $':>10}  source")
    print("-" * 78)
    pegged = 0
    for r in tgt["results"]:
        if r["cost_basis"] == "measured":
            print(f"{r['id']:22} {str(r['backend']):8} {'measured':10} {r['total_cost_usd']:10.6f}  own usage")
            continue
        slug = DEFAULT_CARD.get(str(r["backend"]))
        donor = by_id.get(r["id"])
        got = peg(r, donor, slug) if (slug and donor) else None
        if not got:
            print(f"{r['id']:22} {str(r['backend']):8} {'UNMEASURED':10} {'-':>10}  no donor tokens")
            continue
        r.update(got)
        pegged += 1
        p = got["peg"]
        print(f"{r['id']:22} {str(r['backend']):8} {'pegged':10} {got['total_cost_usd']:10.6f}  "
              f"{p['tokens_in']}+{p['tokens_out']} tok x ${p['price_blended_per_1m']}/1M ({p['card']})")
    print("-" * 78)
    tot = sum(x["total_cost_usd"] for x in tgt["results"] if isinstance(x["total_cost_usd"], (int, float)))
    meas = sum(x["total_cost_usd"] for x in tgt["results"] if x["cost_basis"] == "measured")
    print(f"  pegged {pegged} rows · total ${tot:.4f} (${meas:.4f} measured, ${tot-meas:.4f} pegged)")

    if a.write:
        tgt["meta"]["pegged_from"] = donor_run
        pathlib.Path(a.target).write_text(json.dumps(tgt, indent=2) + "\n")
        print(f"  wrote {a.target}")
    else:
        print("  (dry — pass --write to save)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
