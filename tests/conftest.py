"""Shared pytest config for mcp-brain-router tests."""

# Register the `slow` marker so live-subprocess tests (spec 002 SC-1/SC-5) don't
# trip pytest's unknown-marker warning. They are SKIPPED by default — opt in
# with RUN_AGENTIC_LIVE=1.
def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "slow: live subprocess test (skipped unless RUN_AGENTIC_LIVE=1)",
    )
