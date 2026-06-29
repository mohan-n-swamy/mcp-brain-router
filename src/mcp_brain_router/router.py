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
from .config import Config


class Complexity(str, Enum):
    """Routing complexity tiers."""
    CHEAP = "cheap"
    CODE = "code"
    ADVERSARIAL = "adversarial"


@dataclass
class RouteResult:
    """Unified result from any backend."""
    content: str
    model: str
    backend: str
    complexity: Complexity
    headroom_used: bool
    usage: Optional[Dict[str, int]] = None  # token counts if available


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

    # Resolve target backend and model
    backend_name, model = _resolve_backend_and_model(
        complexity, model_override, config
    )

    # Validate credentials
    _validate_credentials(backend_name, config)

    # Route to backend
    if backend_name == "deepseek":
        result = await _route_deepseek(prompt, model, config)
    elif backend_name == "glm":
        result = await _route_glm(prompt, model, config)
    elif backend_name == "codex":
        result = await _route_codex(prompt, model, config)
    else:
        raise ValueError(f"Unknown backend: {backend_name}")

    # Augment with routing metadata
    result.complexity = complexity
    result.backend = backend_name

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
