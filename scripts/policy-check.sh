#!/usr/bin/env bash
# lean-rig invariant I22: the router's delegation policy holds.
#   class orange reaches only Claude and refuses rather than falls through;
#   the <=1,200 B preamble opens every prompt; one deadline per call
#   (simple 120 s, worker 15 min); the walk passes three failed backends;
#   gpt-oss skips when cold and takes one generation at a time; Grok returns
#   only its final message; Codex gets a closed stdin; the review sandbox
#   denies what it should. The body is tests/test_policy.py.
#
# Usage: bash scripts/policy-check.sh   (from anywhere; plain bash, Linux or Mac)
# Exit: 0 all hold; 1 a check failed; 2 infra: no test runner, or no bwrap to
#       prove the sandbox (gate 2 installs /usr/bin/bwrap; BRAIN_ROUTER_BWRAP
#       names another for a test run).
set -u
cd "$(dirname "$0")/.." || exit 2

if command -v uv >/dev/null 2>&1; then
  run="uv run --quiet --extra dev python -m pytest"
elif python3 -c 'import pytest, mcp_brain_router' >/dev/null 2>&1; then
  run="python3 -m pytest"
else
  echo "policy-check: no test runner (install uv, or pytest and this package)" >&2
  exit 2
fi

$run -q -p no:cacheprovider -rs tests/test_policy.py
rc=$?
if [ "$rc" -ne 0 ]; then
  echo "policy-check: FAIL (pytest exit $rc)" >&2
  exit 1
fi

bwrap=${BRAIN_ROUTER_BWRAP:-/usr/bin/bwrap}
if [ "$(uname -s)" != Linux ] || [ ! -x "$bwrap" ]; then
  echo "policy-check: routing policy holds; sandbox denials NOT proven: no bwrap at $bwrap (gate 2)" >&2
  exit 2
fi
echo "policy-check: pass (sandbox via $bwrap)"
