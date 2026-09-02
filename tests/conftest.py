"""Shared pytest config for mcp-brain-router tests."""

# Register the `slow` marker so live-subprocess tests (spec 002 SC-1/SC-5) don't
# trip pytest's unknown-marker warning. They are SKIPPED by default — opt in
# with RUN_AGENTIC_LIVE=1.
def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "slow: live subprocess test (skipped unless RUN_AGENTIC_LIVE=1)",
    )


import pytest


@pytest.fixture(autouse=True)
def _delegation_log_in_tmp(tmp_path, monkeypatch):
    """Never let a test append to the LIVE audit log. 274 rows with
    caller=unknown / elapsed_ms=0 in ~/.local/state/brain-router-delegations.jsonl
    were pytest (found 2026-09-02, rig-consolidation D-R-01); the ops watchdog
    and brain-router-check read that file."""
    from mcp_brain_router import server

    monkeypatch.setattr(server, "_DELEGATION_LOG", tmp_path / "delegations.jsonl")
