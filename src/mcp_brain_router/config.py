"""Configuration management for mcp-brain-router.

Loads from ~/.config/mcp-brain-router/config.toml with 0600 permissions.
"""

import os
import stat
from pathlib import Path
from typing import Dict, Optional
from dataclasses import dataclass

try:
    import tomllib
except ImportError:
    import tomli as tomllib


class ConfigError(Exception):
    """Unified config exception (file missing, permissions, parse error)."""
    pass


CONFIG_DIR = Path.home() / ".config" / "mcp-brain-router"
CONFIG_FILE = CONFIG_DIR / "config.toml"


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
            lines.append(f"codex_enabled = true")
        if self.headroom_base_url:
            lines.append(
                f'headroom_base_url = "{self._escape_toml(self.headroom_base_url)}"'
            )
        if self.model_overrides:
            lines.append("[model_overrides]")
            for complexity, model in self.model_overrides.items():
                lines.append(f'{complexity} = "{self._escape_toml(model)}"')

        content = "\n".join(lines)

        # Write with secure temp file + rename
        temp_file = CONFIG_FILE.with_suffix(".tmp")
        with open(temp_file, "w") as f:
            f.write(content)

        # Set permissions to 0600 before moving
        temp_file.chmod(0o600)
        temp_file.replace(CONFIG_FILE)

    @staticmethod
    def _escape_toml(value: str) -> str:
        """Escape a string for TOML."""
        return value.replace("\\", "\\\\").replace('"', '\\"')
