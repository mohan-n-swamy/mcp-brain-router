"""Each CLI's --output-format value must be one THAT CLI accepts.

This file exists because of a regression it would have caught. C16 switched the
agentic kimi call from `text` to `json`; kimi's CLI accepts only `text` and
`stream-json` and exited 1 in about 450ms, so every agentic kimi call failed and
the fail-open cascade hid it by advancing. (Kimi no longer spawns at all: it has
no verified tools-off mode, lean-rig §6.)

Every other test in this suite mocks the subprocess. These assert the ARGUMENT
VALUE against a per-CLI allowlist, so a family-wide edit cannot silently apply a
flag to a CLI that has no such mode.
"""
from __future__ import annotations

import inspect
import re

import pytest

from mcp_brain_router import backends

# What each binary accepts:
#   grok   -> `grok --help` 1.0.77: plain, json, streaming-json, streaming-messages-json
#   claude -> json, stream-json, text
ACCEPTED: dict[str, set[str]] = {
    "call_grok": {"plain", "json", "streaming-json", "streaming-messages-json"},
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


def test_grok_reads_json_so_only_the_final_message_returns() -> None:
    """lean-rig G3: plain output cannot tell narration from the answer; json can."""
    assert _output_formats("call_grok") == ["json"]
