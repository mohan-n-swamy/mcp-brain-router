"""CI runs I22 for real: the fitness workflow installs and proves policy-check.sh's prerequisites.

scripts/policy-check.sh exits 2 (infra) without uv (or pytest plus this package) and without a working
/usr/bin/bwrap; the workflow turns exit 2 into a warning. The ubuntu-24.04 runner image has neither, so
without a prerequisite step ahead of fitness.sh, I22 stays untested and green on every run.
"""
import re
from pathlib import Path

WORKFLOW = Path(__file__).resolve().parent.parent / ".github" / "workflows" / "fitness.yml"


def _steps():
    text = WORKFLOW.read_text()
    parts = re.split(r"^      - (?=name: |uses: )", text, flags=re.M)
    return parts[1:]


def _index(steps, needle):
    hits = [i for i, s in enumerate(steps) if needle in s]
    assert hits, f"no step in {WORKFLOW.name} contains {needle!r}"
    return hits[0]


def test_policy_check_prerequisites_run_before_fitness():
    steps = _steps()
    fitness = _index(steps, "bash dotagents/checks/fitness.sh ci mcp-brain-router")
    prereq = _index(steps, "install -m 0755")
    step = steps[prereq]
    assert prereq < fitness
    assert "if: env.HAS_RIG_TOKEN == 'true'" in step
    # uv, pinned by checksum like gitleaks
    assert re.search(r"releases/download/[0-9.]+/uv-x86_64-unknown-linux-gnu\.tar\.gz", step)
    assert re.search(r'echo "[0-9a-f]{64}  \$RUNNER_TEMP/uv\.tgz" \| sha256sum -c -', step)
    assert "/usr/local/bin/uv" in step
    # bubblewrap at the path policy-check.sh reads, with user namespaces allowed, proven by a run
    assert "apt-get install -y -q bubblewrap" in step
    assert "kernel.apparmor_restrict_unprivileged_userns=0" in step
    assert re.search(r"^\s*/usr/bin/bwrap --unshare-all .* /bin/true$", step, flags=re.M)
    # a failed probe must fail the step: nothing in it may swallow an exit
    assert "|| true" not in step and "continue-on-error" not in step


def test_actions_pinned_by_commit_sha():
    for line in WORKFLOW.read_text().splitlines():
        m = re.search(r"uses: (\S+)", line)
        if m:
            assert re.fullmatch(r"[\w.-]+/[\w.-]+@[0-9a-f]{40}", m.group(1)), line
