"""Pure routing layer for mcp-brain-router.

Routes requests to backends based on complexity, resolves models, validates
credentials, and returns unified result dicts. Framework-free, unit-testable.
"""

import asyncio
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional

from . import backends
from .backends import BackendQuotaError
from .config import Config


class Complexity(str, Enum):
    """Routing complexity tiers (legacy axis; kept as a back-compat alias)."""

    CHEAP = "cheap"
    CODE = "code"
    ADVERSARIAL = "adversarial"


class Role(str, Enum):
    """The five orchestration roles (spec 001-role-based-routing).

    Each role is a distinct job in an orchestrated build loop, filled from a
    config-ordered candidate list. ORCHESTRATOR is NOT resolved by the router —
    it is whoever the human launched (an input), used to enforce the
    adversary-differs-in-provider rule.
    """

    ORCHESTRATOR = "orchestrator"
    THINKER = "thinker"
    ADVERSARY = "adversary"
    WORKER = "worker"
    SIMPLE = "simple"


class Provider(str, Enum):
    """Model provider — used to enforce 'adversary differs from orchestrator'
    and to decide whether a resolved role is handed back for native (Anthropic)
    execution vs called directly by the router."""

    ANTHROPIC = "anthropic"   # opus / sonnet / haiku / fable — NEVER called by the router
    CODEX = "codex"           # every gpt-* / sol / terra / luna id (OpenAI Codex CLI)
    ZHIPU = "zhipu"           # glm-4.7 / glm-5.2
    DEEPSEEK = "deepseek"


# Prefix → Provider. First matching prefix wins; order matters only where one
# prefix could shadow another (none currently do). Extend here when a new model
# family appears — this is the single source of truth for id→provider.
_MODEL_PROVIDER_PREFIXES = (
    ("opus", Provider.ANTHROPIC),
    ("sonnet", Provider.ANTHROPIC),
    ("haiku", Provider.ANTHROPIC),
    ("fable", Provider.ANTHROPIC),
    ("claude", Provider.ANTHROPIC),   # any claude-* id
    ("gpt-", Provider.CODEX),
    ("sol", Provider.CODEX),
    ("terra", Provider.CODEX),
    ("luna", Provider.CODEX),
    ("o3", Provider.CODEX),
    ("glm", Provider.ZHIPU),
    ("deepseek", Provider.DEEPSEEK),
)


def provider_for_model(model_id: str) -> Provider:
    """Map a model-id string to its Provider via the prefix table.

    Case-insensitive, prefix-anywhere (so 'gpt-5.6-sol-terra' matches 'sol'
    and 'terra' both → CODEX; the first table hit wins). Raises ValueError on
    an unknown id so a typo is loud, not silently misrouted.
    """
    if not model_id:
        raise ValueError("empty model id")
    low = model_id.lower()
    for prefix, provider in _MODEL_PROVIDER_PREFIXES:
        if low.startswith(prefix) or prefix in low:
            return provider
    raise ValueError(f"unknown model id (no provider prefix match): {model_id!r}")


# Pure tier ownership. The router selects exactly one backend and never crosses
# tiers. The calling orchestrator owns any cascade (for example GLM -> Codex ->
# native Claude). This keeps caller-specific fallback policy out of the MCP.
_TIER_BACKENDS = {
    Complexity.CHEAP: "deepseek",
    Complexity.CODE: "glm",
    Complexity.ADVERSARIAL: "codex",
}


@dataclass
class RouteResult:
    """Unified result from any backend."""

    content: str
    model: str
    backend: str
    complexity: Complexity
    headroom_used: bool
    usage: Optional[Dict[str, int]] = None  # token counts if available
    # Only a genuine provider quota response sets exhausted=True. Timeouts,
    # provider 5xx responses, and hard errors raise so the server can return a
    # distinct error result instead of mislabelling them as quota exhaustion.
    exhausted: bool = False
    tried: Optional[list] = None
    reset_at: Optional[str] = None
    failure_kind: Optional[str] = None
    failure_reason: Optional[str] = None


class BackendError(backends.BackendError):
    """Base exception for backend errors (credential/availability issues).

    Subclasses backends.BackendError so `except backends.BackendError`
    catches the WHOLE family — before 2026-07-02 these were two unrelated
    classes and codex call errors slipped past server.py's handler into
    the generic "Unexpected error" branch."""

    def __init__(self, backend: str, reason: str):
        self.backend = backend
        self.reason = reason
        super().__init__(
            f"Backend {backend}: {reason}",
            backend=backend,
            failure_kind="configuration_error",
        )


class MissingCredentialError(BackendError):
    """Raised when a required credential is missing."""

    def __init__(self, key_name: str, backend: str):
        self.key_name = key_name
        super().__init__(backend, f"Missing credential: {key_name}")


class BackendUnavailableError(BackendError):
    """Raised when a backend is not available."""

    def __init__(self, backend: str, reason: str):
        super().__init__(backend, f"Unavailable: {reason}")


async def route(
    complexity: Complexity,
    prompt: str,
    model_override: Optional[str] = None,
    config: Optional[Config] = None,
) -> RouteResult:
    """
    Route a request to the appropriate backend.

    Args:
        complexity: Routing tier (CHEAP, CODE, ADVERSARIAL).
        prompt: The prompt/message to send.
        model_override: Optional model override. If provided, uses this model
                       for the selected backend. If None, uses config mapping
                       or backend default.
        config: Configuration object. If None, loads from ~/.config/mcp-brain-router/config.toml.

    Returns:
        RouteResult containing response, metadata, and backend info.

    Raises:
        MissingCredentialError: If a required key/config is missing.
        BackendUnavailableError: If the backend cannot be reached or is disabled.
        ValueError: If complexity or model_override is invalid.
    """
    if config is None:
        config = Config.load()

    backend_name, model = _resolve_backend_and_model(complexity, model_override, config)
    _validate_credentials(backend_name, config)

    try:
        if backend_name == "deepseek":
            result = await _route_deepseek(prompt, model, config)
        elif backend_name == "glm":
            result = await _route_glm(prompt, model, config)
        elif backend_name == "codex":
            result = await _route_codex(prompt, model, config)
        else:  # pragma: no cover - _TIER_BACKENDS is the closed set
            raise ValueError(f"Unknown backend: {backend_name}")
    except BackendQuotaError as e:
        reset_hint = f" Earliest reset: {e.reset_at}" if e.reset_at else ""
        return RouteResult(
            content=(
                f"{backend_name} quota exhausted for tier {complexity.value}. "
                f"Call another tier or handle this task natively.{reset_hint}"
            ),
            model="",
            backend="none",
            complexity=complexity,
            headroom_used=False,
            usage=None,
            exhausted=True,
            tried=[backend_name],
            reset_at=e.reset_at,
            failure_kind="quota_exhausted",
            failure_reason=str(e),
        )

    result.complexity = complexity
    result.backend = backend_name
    result.exhausted = False
    result.tried = [backend_name]
    return result


def _resolve_backend_and_model(
    complexity: Complexity,
    model_override: Optional[str],
    config: Config,
) -> tuple[str, str]:
    """
    Resolve target backend and model.

    Priority: model_override > config mapping > backend default.

    Returns:
        Tuple of (backend_name, model).

    Raises:
        ValueError: If complexity is invalid.
    """
    if complexity not in _TIER_BACKENDS:
        raise ValueError(f"Unknown complexity: {complexity}")

    backend_name = _TIER_BACKENDS[complexity]

    # Resolve model
    if model_override:
        model = model_override
    elif config.model_overrides and complexity.value in config.model_overrides:
        model = config.model_overrides[complexity.value]
    else:
        # Use backend default
        model = _get_backend_default_model(backend_name, config)

    return backend_name, model


def _get_backend_default_model(backend_name: str, config: Config) -> str:
    """Get the default model for a backend."""
    defaults = {
        "deepseek": "deepseek-v4-flash",
        "glm": "glm-5.2",
        "codex": "gpt-5.5",
    }
    return defaults.get(backend_name, "unknown")


def _validate_credentials(backend_name: str, config: Config) -> None:
    """
    Validate that required credentials are present for the backend.

    Raises:
        MissingCredentialError: If a required key is missing.
    """
    if backend_name == "deepseek":
        if not config.deepseek_key:
            raise MissingCredentialError("DEEPSEEK_KEY", "deepseek")
    elif backend_name == "glm":
        if not config.glm_key:
            raise MissingCredentialError("GLM_KEY", "glm")
    elif backend_name == "codex":
        if not config.codex_enabled:
            raise BackendUnavailableError("codex", "Codex not enabled or not available on PATH")


async def _route_deepseek(
    prompt: str,
    model: str,
    config: Config,
) -> RouteResult:
    """Route to DeepSeek backend."""
    headroom_used = False

    # Check if headroom proxy is available
    if config.headroom_base_url:
        headroom_used = True
        result = await backends.call_deepseek_via_headroom(
            prompt=prompt,
            model=model,
            api_key=config.deepseek_key,
            headroom_url=config.headroom_base_url,
        )
    else:
        result = await backends.call_deepseek(
            prompt=prompt,
            model=model,
            api_key=config.deepseek_key,
        )

    return RouteResult(
        content=result["content"],
        model=model,
        backend="deepseek",
        complexity=Complexity.CHEAP,
        headroom_used=headroom_used,
        usage=result.get("usage"),
    )


async def _route_glm(
    prompt: str,
    model: str,
    config: Config,
) -> RouteResult:
    """Route to GLM backend."""
    headroom_used = False

    if config.headroom_base_url:
        headroom_used = True
        result = await backends.call_glm_via_headroom(
            prompt=prompt,
            model=model,
            api_key=config.glm_key,
            headroom_url=config.headroom_base_url,
        )
    else:
        result = await backends.call_glm(
            prompt=prompt,
            model=model,
            api_key=config.glm_key,
        )

    return RouteResult(
        content=result["content"],
        model=model,
        backend="glm",
        complexity=Complexity.CODE,
        headroom_used=headroom_used,
        usage=result.get("usage"),
    )


async def _route_codex(
    prompt: str,
    model: str,
    config: Config,
) -> RouteResult:
    """Route to Codex backend (subprocess-based)."""
    # Codex does not use headroom (it's not HTTP-based).
    # call_codex uses BLOCKING subprocess.run (~15-19s). Under the async MCP
    # stdio server this would freeze the event loop for the whole codex run →
    # the transport can't answer keepalive pings → client kills the connection
    # ("-32000 Connection closed"). Offload to a worker thread so the loop stays
    # responsive. (Root-caused 2026-07-06: this, not quota, was the real
    # "exhausted"/crash on the adversarial tier. DeepSeek/GLM are async-HTTP so
    # never hit it; only Codex is subprocess-based.)
    result = await asyncio.to_thread(
        backends.call_codex,
        prompt,
        model,
    )

    return RouteResult(
        content=result["content"],
        model=model,
        backend="codex",
        complexity=Complexity.ADVERSARIAL,
        headroom_used=False,
        usage=result.get("usage"),
    )


# Synchronous wrapper for testing and non-async contexts
def route_sync(
    complexity: Complexity,
    prompt: str,
    model_override: Optional[str] = None,
    config: Optional[Config] = None,
) -> RouteResult:
    """Synchronous wrapper around route() for non-async contexts."""
    return asyncio.run(route(complexity, prompt, model_override, config))
