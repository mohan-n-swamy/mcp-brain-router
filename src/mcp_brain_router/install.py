#!/usr/bin/env python3
"""
mcp-brain-router: Zero-friction interactive installer.

Console script: mcp-brain-router-install
Entry point: main()

Flow:
1. Banner + purpose
2. Prompt for GLM (z.ai) and DeepSeek API keys (required, using getpass)
3. Detect Codex on PATH; optionally enable + configure
4. Detect headroom proxy (env var or localhost); optional configuration
5. Write ~/.config/mcp-brain-router/config.toml with chmod 0600 (idempotent)
6. Self-register in Claude Code via 'claude mcp add'
7. Post-install smoke test (ping each backend)
8. Summary + next steps
"""

import getpass
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Optional

# Import actual config module
from mcp_brain_router.config import (
    CONFIG_FILE,
    DEFAULT_ROLE_MODES,
    DEFAULT_ROLES,
    Config,
    ConfigError,
    ensure_config_dir,
)


# Color codes for terminal output
class Color:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BLUE = "\033[94m"


def print_banner():
    """Print installer banner."""
    print(f"\n{Color.BOLD}{Color.BLUE}╔════════════════════════════════════════════════════════════╗{Color.RESET}")
    print(f"{Color.BOLD}{Color.BLUE}║{Color.RESET}   {Color.BOLD}mcp-brain-router: Interactive Installer{Color.RESET}")
    print(f"{Color.BOLD}{Color.BLUE}╚════════════════════════════════════════════════════════════╝{Color.RESET}")
    print()
    print("This installer will:")
    print("  1. Collect API keys for GLM (z.ai) and DeepSeek")
    print("  2. Optionally enable Codex adversarial backend")
    print("  3. Configure headroom proxy (if available)")
    print("  4. Store secrets in ~/.config/mcp-brain-router/config.toml (mode 0600)")
    print("  5. Register the MCP server in Claude Code")
    print("  6. Run a smoke test to verify backends are reachable")
    print()


def prompt_for_key(label: str, required: bool = True) -> Optional[str]:
    """
    Prompt for an API key using getpass (hidden input).

    Args:
        label: Descriptive label for the key (e.g., "GLM (z.ai) API Key")
        required: If True, re-prompt until a non-empty value is provided

    Returns:
        The API key string, or None if optional and user left blank
    """
    while True:
        try:
            key = getpass.getpass(
                f"{Color.YELLOW}Enter {label}: {Color.RESET}",
                stream=sys.stderr
            )
        except (KeyboardInterrupt, EOFError):
            print("\n" + Color.RED + "Installation cancelled." + Color.RESET)
            sys.exit(1)

        if key.strip():
            return key.strip()
        elif required:
            print(f"{Color.RED}This key is required. Please try again.{Color.RESET}")
        else:
            return None


def detect_codex_on_path() -> bool:
    """Check if 'codex' CLI is available on PATH."""
    return shutil.which("codex") is not None


def ask_enable_codex() -> bool:
    """Prompt user to enable Codex backend (only if available on PATH)."""
    if not detect_codex_on_path():
        print(f"{Color.YELLOW}Note:{Color.RESET} Codex CLI not found on PATH. Adversarial backend disabled.")
        return False

    try:
        response = input(
            f"{Color.YELLOW}Enable Codex adversarial backend (gpt-5.5 default; GPT-5.6 candidates require eval)? [y/N]: {Color.RESET}"
        ).strip().lower()
    except (KeyboardInterrupt, EOFError):
        print("\n" + Color.RED + "Installation cancelled." + Color.RESET)
        sys.exit(1)

    return response == "y"


def prompt_codex_model() -> str:
    """Prompt for Codex model name (default: gpt-5.5)."""
    try:
        model = input(
            f"{Color.YELLOW}Codex model name [gpt-5.5]: {Color.RESET}"
        ).strip()
    except (KeyboardInterrupt, EOFError):
        print("\n" + Color.RED + "Installation cancelled." + Color.RESET)
        sys.exit(1)

    return model or "gpt-5.5"


def detect_headroom_proxy() -> Optional[str]:
    """
    Check for headroom proxy availability.

    Priority:
    1. Environment variable HEADROOM_BASE_URL
    2. Localhost proxy at http://localhost:8282 (or standard headroom port)

    Returns:
        Base URL if found, else None
    """
    # Check environment variable
    env_url = os.environ.get("HEADROOM_BASE_URL", "").strip()
    if env_url:
        return env_url

    # Check localhost (common headroom default port is 8282)
    localhost_urls = [
        "http://localhost:8282",
        "http://127.0.0.1:8282",
    ]
    for url in localhost_urls:
        try:
            # Lightweight check: try to connect with short timeout
            import socket
            host, port = "localhost", 8282
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(1)
            result = sock.connect_ex((host, port))
            sock.close()
            if result == 0:
                return url
        except Exception:
            pass

    return None


def ask_headroom_config() -> Optional[str]:
    """
    Prompt user for headroom proxy URL.

    Checks for auto-detection first; if found, confirms with user.
    Otherwise, prompts for manual entry (optional).

    Returns:
        Headroom base URL, or None if not configured
    """
    detected = detect_headroom_proxy()

    if detected:
        try:
            response = input(
                f"{Color.YELLOW}Headroom proxy detected at {detected}. Use it? [Y/n]: {Color.RESET}"
            ).strip().lower()
        except (KeyboardInterrupt, EOFError):
            print("\n" + Color.RED + "Installation cancelled." + Color.RESET)
            sys.exit(1)

        if response != "n":
            return detected

    # Manual entry (optional)
    try:
        url = input(
            f"{Color.YELLOW}Headroom proxy base URL (optional, press Enter to skip): {Color.RESET}"
        ).strip()
    except (KeyboardInterrupt, EOFError):
        print("\n" + Color.RED + "Installation cancelled." + Color.RESET)
        sys.exit(1)

    return url if url else None


def write_config(
    glm_key: str,
    deepseek_key: str,
    codex_enabled: bool,
    codex_model: Optional[str],
    headroom_url: Optional[str],
) -> Path:
    """
    Write configuration to ~/.config/mcp-brain-router/config.toml via Config.save().

    Creates directory if needed. Sets permissions to 0600 (user read/write only).
    Idempotent: if file exists, loads existing values and updates only changed fields.

    Returns:
        Path to the written config file
    """
    ensure_config_dir()
    config_file = CONFIG_FILE

    # Build Config object
    model_overrides = None
    if codex_enabled and codex_model:
        model_overrides = {"adversarial": codex_model}

    existing = None
    if config_file.exists():
        try:
            existing = Config.load()
        except ConfigError:
            existing = None

    config = Config(
        deepseek_key=deepseek_key,
        glm_key=glm_key,
        codex_enabled=codex_enabled,
        grok_enabled=(
            existing.grok_enabled
            if existing is not None
            else shutil.which("grok") is not None
        ),
        headroom_base_url=headroom_url,
        model_overrides=model_overrides,
        roles=existing.roles if existing is not None else DEFAULT_ROLES.copy(),
        role_modes=DEFAULT_ROLE_MODES.copy(),
    )

    # Use Config.save() (handles TOML format + 0600 permissions)
    config.save()

    print(f"{Color.GREEN}✓{Color.RESET} Configuration written to {config_file}")
    print(f"  {Color.BOLD}Permissions: 0600 (user read/write only){Color.RESET}")

    return config_file


def is_claude_registered() -> bool:
    """Check if 'brain-router' MCP is already registered in Claude Code."""
    try:
        result = subprocess.run(
            ["claude", "mcp", "list"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return "brain-router" in result.stdout
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return False


def register_in_claude_code() -> bool:
    """Backward-compatible single-client registration entrypoint."""
    return _register_client(*_registration_specs(sys.executable)[0])


def _registration_specs(python_exe: str) -> tuple[tuple, ...]:
    return (
        (
            "Claude Code", "claude", ["claude", "mcp", "list"],
            ["claude", "mcp", "add", "--scope", "user", "brain-router", "-e",
             "BRAIN_ROUTER_CALLER=claude", "--", python_exe, "-m",
             "mcp_brain_router.server"],
        ),
        (
            "Codex", "codex", ["codex", "mcp", "list"],
            ["codex", "mcp", "add", "brain-router", "--env",
             "BRAIN_ROUTER_CALLER=codex", "--", python_exe, "-m",
             "mcp_brain_router.server"],
        ),
        (
            "Grok", "grok", ["grok", "mcp", "list"],
            ["grok", "mcp", "add", "--scope", "user", "brain-router", "-e",
             "BRAIN_ROUTER_CALLER=grok", "--", python_exe, "-m",
             "mcp_brain_router.server"],
        ),
    )


def _register_client(
    label: str, binary: str, list_cmd: list[str], add_cmd: list[str]
) -> bool:
    if shutil.which(binary) is None:
        return True
    try:
        listed = subprocess.run(list_cmd, capture_output=True, text=True, timeout=10)
        already_listed = listed.returncode == 0 and "brain-router" in listed.stdout
        if already_listed and binary != "grok":
            detail = subprocess.run(
                [binary, "mcp", "get", "brain-router"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            caller_marker = (
                "BRAIN_ROUTER_CALLER=claude"
                if binary == "claude"
                else "BRAIN_ROUTER_CALLER="
            )
            python_marker = add_cmd[-3]
            if (
                detail.returncode == 0
                and caller_marker in detail.stdout
                and python_marker in detail.stdout
                and "mcp_brain_router.server" in detail.stdout
            ):
                print(f"{Color.GREEN}✓{Color.RESET} brain-router verified in {label}")
                return True

        # Grok's `mcp add` is explicitly add-or-update. For Claude/Codex this
        # also installs a missing entry; a stale existing entry fails loud
        # instead of being accepted by name alone.
        added = subprocess.run(add_cmd, capture_output=True, text=True, timeout=15)
        if added.returncode != 0:
            print(
                f"{Color.YELLOW}Warning:{Color.RESET} {label} registration is stale "
                "or could not be updated; remove brain-router and rerun installer"
            )
            return False
        verified = subprocess.run(list_cmd, capture_output=True, text=True, timeout=10)
        if verified.returncode == 0 and "brain-router" in verified.stdout:
            print(f"{Color.GREEN}✓{Color.RESET} brain-router verified in {label}")
            return True
        print(f"{Color.YELLOW}Warning:{Color.RESET} {label} verification failed")
        return False
    except (OSError, subprocess.TimeoutExpired) as exc:
        print(f"{Color.YELLOW}Warning:{Color.RESET} {label}: {exc}")
        return False


def register_in_supported_clients() -> bool:
    """Register and verify every installed orchestrator CLI."""
    results = [
        _register_client(*spec) for spec in _registration_specs(sys.executable)
    ]
    return all(results)


def run_smoke_test(
    glm_key: str,
    deepseek_key: str,
    codex_enabled: bool,
    codex_model: Optional[str],
    headroom_url: Optional[str],
) -> Dict[str, bool]:
    """
    Run minimal smoke tests for each configured backend.

    For each backend, sends a minimal prompt with small max_tokens.
    Records PASS/FAIL and latency.

    Returns:
        Dict mapping backend name to pass/fail boolean
    """
    results = {}

    print(f"\n{Color.BOLD}Running smoke tests...{Color.RESET}")

    # Test GLM (z.ai)
    results["GLM (z.ai)"] = _test_backend_glm(glm_key, headroom_url)

    # Test DeepSeek
    results["DeepSeek"] = _test_backend_deepseek(deepseek_key, headroom_url)

    # Test Codex (if enabled)
    if codex_enabled:
        results["Codex"] = _test_backend_codex(codex_model or "gpt-5.5")

    return results


def _test_backend_glm(api_key: str, headroom_url: Optional[str]) -> bool:
    """Test GLM (z.ai) backend connectivity."""
    try:
        import httpx
    except ImportError:
        print(f"  {Color.YELLOW}⊘{Color.RESET} GLM: httpx not available")
        return False

    try:
        start = time.time()
        headers = {
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }

        # Use headroom proxy if configured
        if headroom_url:
            url = f"{headroom_url}/anthropic/v1/messages"
        else:
            url = "https://api.z.ai/api/anthropic/v1/messages"

        payload = {
            "model": "glm-5.2",
            "max_tokens": 10,
            "messages": [{"role": "user", "content": "OK"}],
        }

        with httpx.Client(timeout=10) as client:
            response = client.post(url, json=payload, headers=headers)

        elapsed = time.time() - start
        if response.status_code in (200, 400):  # 400 = malformed, but endpoint is reachable
            print(f"  {Color.GREEN}✓{Color.RESET} GLM (z.ai): {elapsed:.2f}s")
            return True
        else:
            print(f"  {Color.RED}✗{Color.RESET} GLM (z.ai): HTTP {response.status_code}")
            return False
    except Exception as e:
        print(f"  {Color.RED}✗{Color.RESET} GLM (z.ai): {type(e).__name__}")
        return False


def _test_backend_deepseek(api_key: str, headroom_url: Optional[str]) -> bool:
    """Test DeepSeek backend connectivity."""
    try:
        import httpx
    except ImportError:
        print(f"  {Color.YELLOW}⊘{Color.RESET} DeepSeek: httpx not available")
        return False

    try:
        start = time.time()
        headers = {
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }

        # Use headroom proxy if configured
        if headroom_url:
            url = f"{headroom_url}/anthropic/v1/messages"
        else:
            url = "https://api.deepseek.com/anthropic/v1/messages"

        payload = {
            "model": "deepseek-v4-flash",
            "max_tokens": 10,
            "messages": [{"role": "user", "content": "OK"}],
        }

        with httpx.Client(timeout=10) as client:
            response = client.post(url, json=payload, headers=headers)

        elapsed = time.time() - start
        if response.status_code in (200, 400):
            print(f"  {Color.GREEN}✓{Color.RESET} DeepSeek: {elapsed:.2f}s")
            return True
        else:
            print(f"  {Color.RED}✗{Color.RESET} DeepSeek: HTTP {response.status_code}")
            return False
    except Exception as e:
        print(f"  {Color.RED}✗{Color.RESET} DeepSeek: {type(e).__name__}")
        return False


def _test_backend_codex(model: str = "gpt-5.5") -> bool:
    """Test Codex CLI availability."""
    try:
        start = time.time()
        from .backends import CODEX_EXEC_BASE
        result = subprocess.run(
            CODEX_EXEC_BASE + ["-m", model, "--", "reply: OK"],
            capture_output=True,
            text=True,
            # codex answers in ~20-30s even with MCP boot skipped; 10s was
            # guaranteed-fail once startup included any model round-trip.
            timeout=45,
        )
        elapsed = time.time() - start

        if result.returncode == 0:
            print(f"  {Color.GREEN}✓{Color.RESET} Codex ({model}): {elapsed:.2f}s")
            return True
        else:
            print(f"  {Color.RED}✗{Color.RESET} Codex: exit code {result.returncode}")
            return False
    except (FileNotFoundError, subprocess.TimeoutExpired) as e:
        print(f"  {Color.RED}✗{Color.RESET} Codex: {type(e).__name__}")
        return False
    except Exception as e:
        print(f"  {Color.RED}✗{Color.RESET} Codex: {type(e).__name__}")
        return False


def print_summary(
    config_file: Path,
    registered: bool,
    smoke_results: Dict[str, bool],
):
    """Print final summary and next steps."""
    print(f"\n{Color.BOLD}{Color.GREEN}╔════════════════════════════════════════════════════════════╗{Color.RESET}")
    print(f"{Color.BOLD}{Color.GREEN}║{Color.RESET}   Installation Complete!")
    print(f"{Color.BOLD}{Color.GREEN}╚════════════════════════════════════════════════════════════╝{Color.RESET}")

    print(f"\n{Color.BOLD}Configuration:{Color.RESET}")
    print(f"  Config file: {config_file}")

    print(f"\n{Color.BOLD}Backend status:{Color.RESET}")
    for backend_name, passed in smoke_results.items():
        status = f"{Color.GREEN}✓ PASS{Color.RESET}" if passed else f"{Color.RED}✗ FAIL{Color.RESET}"
        print(f"  {backend_name}: {status}")

    if registered:
        print(f"\n{Color.BOLD}Claude Code:{Color.RESET}")
        print(f"  {Color.GREEN}✓{Color.RESET} MCP registered as 'brain-router'")
    else:
        print(f"\n{Color.BOLD}Claude Code:{Color.RESET}")
        print(f"  {Color.YELLOW}⊘{Color.RESET} Manual registration may be required (see above)")

    print(f"\n{Color.BOLD}Next steps:{Color.RESET}")
    print("  1. Restart Claude Code or reload the MCP")
    print("  2. Test the tool in a conversation")
    print("  3. To update API keys, run: mcp-brain-router-install")

    print()


def main():
    """Main installer entry point."""
    try:
        print_banner()

        # Prompt for required keys
        print(f"{Color.BOLD}Step 1: API Keys (required){Color.RESET}")
        glm_key = prompt_for_key("GLM (z.ai) API Key", required=True)
        deepseek_key = prompt_for_key("DeepSeek API Key", required=True)

        # Optional: Codex
        print(f"\n{Color.BOLD}Step 2: Codex Backend (optional){Color.RESET}")
        codex_enabled = ask_enable_codex()
        codex_model = None
        if codex_enabled:
            codex_model = prompt_codex_model()

        # Optional: Headroom
        print(f"\n{Color.BOLD}Step 3: Headroom Proxy (optional){Color.RESET}")
        headroom_url = ask_headroom_config()
        if headroom_url:
            print(f"  {Color.GREEN}✓{Color.RESET} Headroom configured: {headroom_url}")
        else:
            print(f"  {Color.YELLOW}⊘{Color.RESET} Headroom disabled (direct API calls)")

        # Write config
        print(f"\n{Color.BOLD}Step 4: Writing Configuration{Color.RESET}")
        config_file = write_config(
            glm_key=glm_key,
            deepseek_key=deepseek_key,
            codex_enabled=codex_enabled,
            codex_model=codex_model,
            headroom_url=headroom_url,
        )

        # Register in every installed orchestrator CLI.
        print(f"\n{Color.BOLD}Step 5: Claude/Codex/Grok Registration{Color.RESET}")
        registered = register_in_supported_clients()

        # Smoke test
        print(f"\n{Color.BOLD}Step 6: Smoke Tests{Color.RESET}")
        smoke_results = run_smoke_test(
            glm_key=glm_key,
            deepseek_key=deepseek_key,
            codex_enabled=codex_enabled,
            codex_model=codex_model,
            headroom_url=headroom_url,
        )

        # Summary
        print_summary(config_file, registered, smoke_results)

        return 0

    except KeyboardInterrupt:
        print("\n" + Color.RED + "Installation cancelled by user." + Color.RESET)
        return 1
    except Exception as e:
        print(f"\n{Color.RED}Installation failed: {e}{Color.RESET}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
