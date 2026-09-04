"""Each CLI's --output-format value must be one THAT CLI accepts.

This file exists because of a regression it would have caught. C16 switched the
agentic kimi call from `text` to `json` to harvest the usage envelope the
Claude-family CLIs emit. Kimi's CLI accepts only `text` and `stream-json`; it
rejects `json` with exit 1 in about 450ms. Kimi is first candidate for BOTH the
worker and simple roles, so from that commit every agentic kimi call in the rig
failed -- and the fail-open cascade advanced to glm, so nothing appeared to break.
The work still got done, on a different provider, at a different cost.

Every other test in this suite mocks the subprocess, which is why 172 of them
passed while the binary was being handed an argument it refuses. These assert the
ARGUMENT VALUE against a per-CLI allowlist, so a family-wide edit cannot silently
apply a flag to a CLI that has no such mode.
"""
from __future__ import annotations

import inspect
import re

import pytest

from mcp_brain_router import backends

# What each binary actually accepts, read from its own error text:
#   kimi  -> "Allowed choices are text, stream-json."
#   claude/glm (Claude-Code family) and codex accept json.
ACCEPTED: dict[str, set[str]] = {
    "call_kimi_agentic": {"text", "stream-json"},
    "call_kimi": {"text", "stream-json"},
    "call_glm_agentic": {"json", "stream-json", "text"},
    "call_grok_agentic": {"json", "stream-json", "text"},
    "call_anthropic_agentic": {"json", "stream-json", "text"},
}


def _output_formats(fn_name: str) -> list[str]:
    """The literal that follows '--output-format' in the function's source."""
    src = inspect.getsource(getattr(backends, fn_name))
    return re.findall(r'"--output-format"\s*,\s*(?:#[^\n]*\n\s*)*"([^"]+)"', src)


@pytest.mark.parametrize("fn_name,allowed", sorted(ACCEPTED.items()))
def test_output_format_is_one_the_cli_accepts(fn_name: str, allowed: set[str]) -> None:
    found = _output_formats(fn_name)
    assert found, f"{fn_name}: no --output-format literal found; the regex needs updating"
    bad = [f for f in found if f not in allowed]
    assert not bad, (
        f"{fn_name} passes --output-format {bad} but that CLI accepts only "
        f"{sorted(allowed)}. The CLI exits non-zero on an unaccepted value, and the "
        f"fail-open cascade will hide it by advancing to the next provider."
    )


def test_kimi_agentic_is_not_json() -> None:
    """Pins the exact regression. Kept separate from the table so the failure
    message names the incident rather than a parametrised case id."""
    assert "json" not in _output_formats("call_kimi_agentic"), (
        "kimi -p --output-format json exits 1: 'Allowed choices are text, stream-json'. "
        "This broke every agentic kimi call in the rig once before."
    )
