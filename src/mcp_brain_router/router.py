"""Pure routing layer for mcp-brain-router.

Routes requests to backends based on complexity, resolves models, validates
credentials, and returns unified result dicts. Framework-free, unit-testable.
"""

import asyncio
import os
from enum import Enum
from typing import Any, Dict, Literal, Optional
from dataclasses import dataclass

from . import backends
from .backends import BackendQuotaError
from .config import Config


class Complexity(str, Enum):
    """Routing complexity tiers."""
    CHEAP = "cheap"
    CODE = "code"
    ADVERSARIAL = "adversarial"


# Per-tier fallback chain. chain[0] is the tier's primary backend; the rest are
# fallbacks tried (in order) when the primary is quota-exhausted (429/5xx).
# NO Anthropic backend appears here by design — when the whole chain is
# exhausted, route() returns exhausted=True and the orchestrator (Claude)
# handles the task natively (COMPLIANCE.md: the MCP never calls Anthropic).
#   - cheap:       DeepSeek -> GLM        (both cheap-ish chat backends)
#   - code:        GLM      -> DeepSeek   (GLM best for code; DeepSeek still capable)
#   - adversarial: Codex                  (local subprocess, no remote quota to exhaust)
_FALLBACK_CHAINS = {
    Complexity.CHEAP: ["deepseek", "glm"],
    Complexity.CODE: ["glm", "deepseek"],
    Complexity.ADVERSARIAL: ["codex"],
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
    # Quota-exhaustion signal. When every backend in the tier's fallback chain
    # is quota-exhausted, route() returns a RouteResult with exhausted=True
    # instead of raising — a clean "sorry, can't" the orchestrator (Claude)
    # reads to decide to handle the task NATIVELY. The MCP never calls Anthropic.
    exhausted: bool = False
    tried: Optional[list] = None        # backends attempted, in order
    reset_at: Optional[str] = None      # earliest provider reset time, if known


class BackendError(Exception):
    """Base exception for backend errors (credential/availability issues)."""
    def __init__(self, backend: str, reason: str):
        self.backend = backend
        self.reason = reason
        super().__init__(f"Backend {backend}: {reason}")


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

    # Walk the tier's fallback chain. Each entry is a non-Anthropic backend;
    # on quota exhaustion (429/5xx) we fall through to the next. The chain is
    # ordered "best-fit first" — see _FALLBACK_CHAINS. A model_override only
    # applies to the PRIMARY backend (the one the tier maps to); fallbacks use
    # their own defaults, since an override model id is provider-specific.
    chain = _FALLBACK_CHAINS[complexity]
    primary = chain[0]

    tried = []
    last_quota_err = None
    known_reset = None  # earliest reset time seen across quota errors, if any
    for backend_name in chain:
        # Skip a fallback backend that lacks credentials / isn't enabled —
        # it's not a failure, just unavailable. (The PRIMARY still surfaces a
        # missing-credential error so misconfiguration is loud, not silent.)
        try:
            _validate_credentials(backend_name, config)
        except (MissingCredentialError, BackendUnavailableError):
            if backend_name == primary:
                raise
            continue

        model = _resolve_model_for_backend(
            backend_name,
            model_override if backend_name == primary else None,
            config,
        )
        tried.append(backend_name)
        try:
            if backend_name == "deepseek":
                result = await _route_deepseek(prompt, model, config)
            elif backend_name == "glm":
                result = await _route_glm(prompt, model, config)
            elif backend_name == "codex":
                result = await _route_codex(prompt, model, config)
            else:
                raise ValueError(f"Unknown backend: {backend_name}")
        except BackendQuotaError as e:
            # Quota / transient — try the next backend in the chain.
            last_quota_err = e
            # Keep the earliest reset time any backend reported (string compare
            # is fine for the "YYYY-MM-DD HH:MM:SS" shape we lift).
            if e.reset_at and (known_reset is None or e.reset_at < known_reset):
                known_reset = e.reset_at
            continue

        # Success — augment with routing metadata and return.
        result.complexity = complexity
        result.backend = backend_name
        result.exhausted = False
        result.tried = tried
        return result

    # Every backend in the chain was quota-exhausted (or unavailable). Return a
    # structured "sorry, can't" signal — NOT an exception — so the orchestrator
    # handles the task natively. The MCP never falls back to Anthropic itself.
    reset_at = known_reset
    return RouteResult(
        content=(
            f"All {complexity.value if hasattr(complexity, 'value') else complexity} "
            f"backends exhausted ({', '.join(tried) or 'none available'}). "
            f"Handle this task natively. {('Earliest reset: ' + reset_at) if reset_at else ''}".strip()
        ),
        model="",
        backend="none",
        complexity=complexity,
        headroom_used=False,
        usage=None,
        exhausted=True,
        tried=tried,
        reset_at=reset_at,
    )


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
    # Map complexity to default backend
    backend_map = {
        Complexity.CHEAP: "deepseek",
        Complexity.CODE: "glm",
        Complexity.ADVERSARIAL: "codex",
    }

    if complexity not in backend_map:
        raise ValueError(f"Unknown complexity: {complexity}")

    backend_name = backend_map[complexity]

    # Resolve model
    if model_override:
        model = model_override
    elif config.model_overrides and complexity in config.model_overrides:
        model = config.model_overrides[complexity]
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


def _resolve_model_for_backend(
    backend_name: str, model_override: Optional[str], config: Config
) -> str:
    """Resolve the model id for a specific backend in the fallback chain.

    A caller-supplied model_override is provider-specific, so it is honored
    ONLY for the primary backend (the chain caller passes None for fallbacks).
    Otherwise: a matching [model_overrides] config entry, else the backend
    default.
    """
    if model_override:
        return model_override
    overrides = config.model_overrides or {}
    if backend_name in overrides:
        return overrides[backend_name]
    return _get_backend_default_model(backend_name, config)


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
            raise BackendUnavailableError(
                "codex",
                "Codex not enabled or not available on PATH"
            )


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
    """Route to Codex backend (subprocess-based, no async wrapper needed)."""
    # Codex does not use headroom (it's not HTTP-based)
    result = backends.call_codex(
        prompt=prompt,
        model=model,
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
