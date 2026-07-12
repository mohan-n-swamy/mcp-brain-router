"""Configuration management for mcp-brain-router.

Loads from ~/.config/mcp-brain-router/config.toml with 0600 permissions.
"""

import os
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

try:
    import tomllib
except ImportError:
    import tomli as tomllib


class ConfigError(Exception):
    """Unified config exception (file missing, permissions, parse error)."""
    pass


CONFIG_DIR = Path.home() / ".config" / "mcp-brain-router"
CONFIG_FILE = CONFIG_DIR / "config.toml"

# Code default for per-role execution mode (spec 002). The worker role is
# agentic by default (shells to a CLI harness that writes files); every other
# role defaults to chat (text-only) so adversaries/refuters stay read-only.
# A config.toml [role_modes] section is merged ON TOP of this — only non-default
# roles need to be listed there. Worker is permanently agentic-for-workers so no
# orchestrator ever re-asks "should the worker do the work".
DEFAULT_ROLE_MODES: Dict[str, str] = {
    "worker": "agentic",
    "adversary": "chat",
    "thinker": "chat",
    "simple": "chat",
}


def ensure_config_dir() -> Path:
    """Create config directory with secure permissions if needed."""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    return CONFIG_DIR


@dataclass
class Config:
    """Configuration for mcp-brain-router."""
    deepseek_key: Optional[str] = None
    glm_key: Optional[str] = None
    codex_enabled: bool = False
    headroom_base_url: Optional[str] = None
    model_overrides: Optional[Dict[str, str]] = None
    roles: Optional[Dict[str, List[str]]] = None
    # Per-role default execution mode (spec 002): 'agentic' shells to a CLI
    # harness in the real cwd (writes files); 'chat' is the text-only path.
    # A code default is merged on load so a config.toml that omits the section
    # still resolves worker->agentic, every other role->chat.
    role_modes: Optional[Dict[str, str]] = None

    @classmethod
    def load(cls) -> "Config":
        """Load configuration from file. Raises ConfigError if file missing or invalid."""
        if not CONFIG_FILE.exists():
            raise ConfigError(
                f"Config file not found: {CONFIG_FILE}\n"
                f"Run 'mcp-brain-router-install' to set up."
            )

        # Check file permissions (should be 0600)
        file_stat = CONFIG_FILE.stat()
        file_mode = stat.filemode(file_stat.st_mode)
        if (file_stat.st_mode & 0o777) != 0o600:
            raise ConfigError(
                f"Config file has insecure permissions: {file_mode}\n"
                f"Run: chmod 600 {CONFIG_FILE}"
            )

        try:
            with open(CONFIG_FILE, "rb") as f:
                data = tomllib.load(f)
        except Exception as e:
            raise ConfigError(f"Failed to parse config: {e}")

        return cls(
            deepseek_key=data.get("deepseek_key"),
            glm_key=data.get("glm_key"),
            codex_enabled=data.get("codex_enabled", False),
            headroom_base_url=data.get("headroom_base_url"),
            model_overrides=data.get("model_overrides"),
            roles=data.get("roles") or {},
            role_modes=DEFAULT_ROLE_MODES | (data.get("role_modes") or {}),
        )

    def save(self) -> None:
        """Save configuration to file with 0600 permissions."""
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)

        # Build TOML content (avoid tomli_w dependency; simple string format)
        lines = []
        if self.deepseek_key:
            lines.append(f'deepseek_key = "{self._escape_toml(self.deepseek_key)}"')
        if self.glm_key:
            lines.append(f'glm_key = "{self._escape_toml(self.glm_key)}"')
        if self.codex_enabled:
            lines.append("codex_enabled = true")
        if self.headroom_base_url:
            lines.append(
                f'headroom_base_url = "{self._escape_toml(self.headroom_base_url)}"'
            )
        if self.model_overrides:
            lines.append("[model_overrides]")
            for complexity, model in self.model_overrides.items():
                lines.append(f'{complexity} = "{self._escape_toml(model)}"')
        if self.roles:
            lines.append("[roles]")
            for role, models in self.roles.items():
                encoded = ", ".join(
                    f'"{self._escape_toml(model)}"' for model in models
                )
                lines.append(f"{role} = [{encoded}]")
        if self.role_modes:
            lines.append("[role_modes]")
            for role, mode in self.role_modes.items():
                lines.append(f'{role} = "{self._escape_toml(mode)}"')

        content = "\n".join(lines)

        # Atomic 0600 write: create the temp file with 0600 BEFORE writing the
        # secret (os.open with O_CREAT|O_EXCL|O_WRONLY + mode), so the key is
        # never on disk world-readable — closes the chmod-after-write race.
        # Unique per-pid name + O_EXCL defeats the predictable-name/symlink attack.
        temp_file = CONFIG_FILE.with_suffix(f".tmp.{os.getpid()}")
        flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
        fd = os.open(temp_file, flags, 0o600)
        try:
            with os.fdopen(fd, "w") as f:
                f.write(content)
            # Belt-and-suspenders: enforce 0600 even if umask altered creation.
            os.chmod(temp_file, 0o600)
            os.replace(temp_file, CONFIG_FILE)
        except BaseException:
            # Never leave a stray temp file holding the key on any failure.
            try:
                os.unlink(temp_file)
            except FileNotFoundError:
                pass
            raise

    @staticmethod
    def _escape_toml(value: str) -> str:
        """Escape a string for TOML."""
        return value.replace("\\", "\\\\").replace('"', '\\"')
