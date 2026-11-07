"""Configuration management for mcp-brain-router.

Loads from ~/.config/mcp-brain-router/config.toml with 0600 permissions.
"""

import logging
import os
import shlex
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
COMPUTE_ENV = Path.home() / ".config" / "mohan" / "compute.env"


def compute_settings() -> Dict[str, str]:
    """Read shared inference settings without executing shell code; env wins."""
    keys = ("MOHAN_OLLAMA_HOST",)
    values = {}
    try:
        for line in COMPUTE_ENV.read_text().splitlines():
            key, sep, raw = line.strip().removeprefix("export ").partition("=")
            if sep and key.strip() in keys:
                try:
                    parts = shlex.split(raw, comments=True)
                except ValueError as exc:
                    raise ConfigError(f"Invalid compute.env setting: {key.strip()}") from exc
                if len(parts) > 1:
                    raise ConfigError(f"Invalid compute.env setting: {key.strip()}")
                values[key.strip()] = parts[0] if parts else ""
    except FileNotFoundError:
        pass
    except (OSError, UnicodeError, ConfigError) as exc:
        # Optional local inference must never disable the independent cloud lanes.
        logging.getLogger(__name__).warning("compute.env unavailable (%s); Ollama needs env overrides", type(exc).__name__)
        values = {}
    return values | {key: os.environ[key] for key in keys if key in os.environ}

# Code default for per-role execution mode (spec 002). The worker role is
# agentic by default (shells to a CLI harness that writes files). HTTP-only
# delegates were removed from the active path: every role is now a CLI worker.
# A config.toml [role_modes] section is merged ON TOP of this — only non-default
# roles need to be listed there. Worker is permanently agentic-for-workers so no
# orchestrator ever re-asks "should the worker do the work".
DEFAULT_ROLE_MODES: Dict[str, str] = {
    "worker": "agentic",
    "adversary": "agentic",
    "thinker": "agentic",
    "simple": "agentic",
}

DEFAULT_ROLES: Dict[str, List[str]] = {
    # D5 (Mohan, 2026-11-06), free first. The orders live here, in the repo,
    # where the retired-ID scan (lean-rig I18) sees them; a host config.toml
    # [roles] table is ignored (Config.load). A missing binary or key is a free
    # skip, so Kimi (not installed) and DeepSeek (unkeyed) cost nothing.
    # gpt-oss runs only when warm and only for `simple` (backends.call_ollama).
    "simple": ["apple-fm", "gpt-oss:20b", "claude-haiku-5-5"],
    "worker": ["kimi", "grok-4.5", "deepseek-v4-pro", "claude-sonnet-5", "gpt-6.1-sol"],
    "adversary": ["gpt-6.1-sol", "claude-opus-5-5", "kimi"],
    # Not in D5; only its Codex entry moves to the one model the ChatGPT login takes.
    "thinker": ["grok-4.5", "claude-fable-5", "gpt-6.1-sol"],
}

# The Codex models a host [model_overrides] adversarial entry may name: the
# ChatGPT login refuses the rest (D5, tested 2026-11-06).
CODEX_MODELS = ("gpt-6.1-sol",)


def ensure_config_dir() -> Path:
    """Create config directory with secure permissions if needed."""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    return CONFIG_DIR


# C04: which band each role draws from when routing_mode == "deck".
DEFAULT_ROLE_BANDS: Dict[str, str] = {
    "simple": "B1", "worker": "B3", "adversary": "B4", "thinker": "B5",
}


@dataclass
class Config:
    """Configuration for mcp-brain-router."""
    deepseek_key: Optional[str] = None
    codex_enabled: bool = False
    grok_enabled: bool = False
    kimi_enabled: bool = False
    fm_host: Optional[str] = None  # ssh host running Apple's fm CLI; unset = no apple-fm tier
    jev_key_file: Optional[str] = None
    jev_key_files: Optional[List[str]] = None  # JEV-LOAD pool, tried in order
    decide_backends: Optional[List[Dict[str, str]]] = None  # DECIDE-SWITCH: {name, url, provider, key_file}
    headroom_base_url: Optional[str] = None
    ollama_host: Optional[str] = None  # shared compute.env, not duplicated in router TOML
    model_overrides: Optional[Dict[str, str]] = None
    roles: Optional[Dict[str, List[str]]] = None
    # Data class a delegate call gets when the caller names none. "orange"
    # (Orange work; the Orange host's default) reaches Claude only.
    default_class: Optional[str] = None
    # Per-role default execution mode (spec 002): 'agentic' shells to a CLI
    # harness in the real cwd (writes files); 'chat' is the text-only path.
    # A code default is merged on load so a config.toml that omits the section
    # still resolves worker->agentic, every other role->chat.
    role_modes: Optional[Dict[str, str]] = None
    # C04 (specs/001-agent-capability-routing). "legacy" walks [roles] exactly as
    # before; "deck" resolves the role's band to a card by deck lookup and falls
    # through to [roles] when the deck offers nothing. Legacy is the default and
    # stays so until every band's top-3 carries a measured cost (SC5, R26). The
    # rollback is this one value, not a redeploy (R16). [roles] is never deleted.
    routing_mode: str = "legacy"
    role_bands: Optional[Dict[str, str]] = None
    # R8: per-provider daily call ceiling (deck provider names). Absent = off.
    daily_call_cap: Optional[Dict[str, int]] = None
    # R26 says a band routes only when its top-3 carries measured cost. Week 0
    # (Mohan, 2026-09-05): one ranked card is enough; the spec default stays 3.
    deck_min_ranked: int = 3

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

        log = logging.getLogger(__name__)
        if data.get("roles"):
            # D5 orders live in the repo (DEFAULT_ROLES). A host table would
            # hide them from the retired-ID scan, so it no longer routes.
            log.warning("config.toml [roles] ignored: role orders come from the router repo")
        if data.get("routing_mode") not in (None, "legacy"):
            # Deck mode picks cards from a state file outside the repo, the
            # same hole as a host [roles] table, so the D5 walk is the only one.
            log.warning("config.toml routing_mode=%r ignored: the D5 role orders route",
                        data.get("routing_mode"))
        overrides = dict(data.get("model_overrides") or {})
        if overrides.get("adversarial") not in (None, *CODEX_MODELS):
            # An older install wrote the model the ChatGPT login now refuses.
            log.warning("config.toml adversarial override %r ignored: Codex runs %s",
                        overrides.pop("adversarial"), CODEX_MODELS[0])
        compute = compute_settings()
        return cls(
            deepseek_key=data.get("deepseek_key"),
            codex_enabled=data.get("codex_enabled", False),
            grok_enabled=data.get("grok_enabled", False),
            kimi_enabled=data.get("kimi_enabled", False),
            fm_host=data.get("fm_host"),
            jev_key_file=data.get("jev_key_file"),
            jev_key_files=data.get("jev_key_files"),
            decide_backends=data.get("decide_backends"),
            headroom_base_url=data.get("headroom_base_url"),
            ollama_host=compute.get("MOHAN_OLLAMA_HOST"),
            model_overrides=overrides or None,
            roles={role: list(order) for role, order in DEFAULT_ROLES.items()},
            default_class=data.get("default_class"),
            # Role delegation is agentic-only. Older config files may still
            # contain chat overrides; normalize them during the shard migration.
            role_modes=DEFAULT_ROLE_MODES.copy(),
            routing_mode="legacy",
            role_bands=DEFAULT_ROLE_BANDS | (data.get("role_bands") or {}),
            daily_call_cap=data.get("daily_call_cap") or None,
            deck_min_ranked=int(data.get("deck_min_ranked") or 3),
        )

    def save(self) -> None:
        """Save configuration to file with 0600 permissions."""
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)

        # Build TOML content (avoid tomli_w dependency; simple string format)
        lines = []
        if self.deepseek_key:
            lines.append(f'deepseek_key = "{self._escape_toml(self.deepseek_key)}"')
        if self.codex_enabled:
            lines.append("codex_enabled = true")
        if self.grok_enabled:
            lines.append("grok_enabled = true")
        if self.kimi_enabled:
            lines.append("kimi_enabled = true")
        if self.fm_host:
            lines.append(f'fm_host = "{self._escape_toml(self.fm_host)}"')
        if self.jev_key_file:
            lines.append(f'jev_key_file = "{self._escape_toml(self.jev_key_file)}"')
        if self.jev_key_files:
            lines.append("jev_key_files = [" + ", ".join(f'"{self._escape_toml(p)}"' for p in self.jev_key_files) + "]")
        if self.decide_backends:
            lines.append("decide_backends = [" + ", ".join("{" + ", ".join(f'{k} = "{self._escape_toml(str(v))}"' for k, v in b.items()) + "}" for b in self.decide_backends) + "]")
        if self.headroom_base_url:
            lines.append(
                f'headroom_base_url = "{self._escape_toml(self.headroom_base_url)}"'
            )
        if self.default_class:
            lines.append(f'default_class = "{self._escape_toml(self.default_class)}"')
        if self.model_overrides:
            lines.append("[model_overrides]")
            for complexity, model in self.model_overrides.items():
                lines.append(f'{complexity} = "{self._escape_toml(model)}"')
        # C04: a load -> save -> load cycle must not silently revert deck mode to
        # legacy or drop the bands (refuter finding). Emitted before the tables so
        # TOML keeps them top-level.
        if self.routing_mode and self.routing_mode != "legacy":
            lines.append(f'routing_mode = "{self._escape_toml(self.routing_mode)}"')
        if self.role_bands:
            lines.append("[role_bands]")
            for role, band in self.role_bands.items():
                lines.append(f'{role} = "{self._escape_toml(band)}"')
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
