"""lean-rig I18: no retired model ID anywhere in this clone.

The regex is lean-rig's fitness row I18 (dotagents checks/fitness.sh:54),
which greps the whole router clone except .git. Dated records count too;
they name a retired model in words and keep the literal ID in git history.
This test scans the tracked files, so a local .venv cannot make it flaky.
"""
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
# Same pattern as fitness.sh, with the 5 after gpt and the m in glm each
# written as a one-character class: it matches the same text, not itself.
I18_REGEX = r"gpt-[5](\.5|[^.0-9]|$)|\bo3\b|gpt-4(o|\.1)|gl[m]-|deepseek-(chat|reasoner)|(sonnet|opus|haiku)-4[-.]5"


@pytest.mark.skipif(shutil.which("git") is None or not (ROOT / ".git").exists(),
                    reason="needs a git checkout")
def test_no_retired_model_id_in_tracked_files():
    files = subprocess.run(["git", "-C", str(ROOT), "ls-files", "-z"],
                           capture_output=True, check=True).stdout.split(b"\0")
    paths = [str(ROOT / f.decode()) for f in files if f and (ROOT / f.decode()).is_file()]
    r = subprocess.run(["grep", "-nsIE", I18_REGEX, *paths], capture_output=True, text=True)
    assert r.returncode == 1, f"I18 hits:\n{r.stdout}{r.stderr}"
