"""Pure routing layer for mcp-brain-router.

Routes requests to backends based on complexity, resolves models, validates
credentials, and returns unified result dicts. Framework-free, unit-testable.
"""

import asyncio
from dataclasses import dataclass
import logging
from enum import Enum
from typing import Dict, Iterable, Optional

from . import backends
from .backends import BackendQuotaError, BackendTransientError
from .config import Config


class Complexity(str, Enum):
    """Routing complexity tiers (legacy axis; kept as a back-compat alias)."""

    CHEAP = "cheap"
    CODE = "code"
    ADVERSARIAL = "adversarial"


logger = logging.getLogger(__name__)


class Role(str, Enum):
    """The five orchestration roles (spec 001-role-based-routing).

    Each role is a distinct job in an orchestrated build loop, filled from a
    config-ordered candidate list. ORCHESTRATOR is NOT resolved by the router —
    it is whoever the human launched (an input), used to enforce the
    skip-self rule (the adversary never resolves to the orchestrator's own
    provider; other roles may).
    """

    ORCHESTRATOR = "orchestrator"
    THINKER = "thinker"
    ADVERSARY = "adversary"
    WORKER = "worker"
    SIMPLE = "simple"


class Provider(str, Enum):
    """Model provider — used to enforce the skip-self rule (the adversary
    never resolves to the orchestrator's own provider; adversary only, since
    2026-09-07) and to decide whether a resolved role is handed back for native
    (Anthropic) execution vs called directly by the router."""

    ANTHROPIC = "anthropic"   # opus / sonnet / haiku / fable — NEVER called by the router
    CODEX = "codex"           # every gpt-* / sol / terra / luna id (OpenAI Codex CLI)
    XAI = "xai"               # grok-4.5 / grok-composer-* (xAI Grok CLI)
    KIMI = "kimi"             # kimi (Kimi Code CLI, OAuth login)
    DEEPSEEK = "deepseek"     # deepseek-v4-pro / deepseek-flash (Anthropic-shape HTTP)
    APPLE = "apple"           # apple-fm: on-device Apple Foundation Model on the mini (fm CLI over ssh)
    OLLAMA = "ollama"         # gpt-oss:20b on the host's Ollama, simple role only


# Prefix → Provider. First matching prefix wins; order matters where one prefix
# shadows another: "gpt-oss" must precede "gpt-" (an open-weight model on
# Ollama, not Codex). Extend here when a new model family appears — this is the
# single source of truth for id→provider.
_MODEL_PROVIDER_PREFIXES = (
    ("gpt-oss", Provider.OLLAMA),
    ("ollama", Provider.OLLAMA),
    ("opus", Provider.ANTHROPIC),
    ("sonnet", Provider.ANTHROPIC),
    ("haiku", Provider.ANTHROPIC),
    ("fable", Provider.ANTHROPIC),
    ("claude", Provider.ANTHROPIC),   # any claude-* id
    ("anthropic", Provider.ANTHROPIC),
    ("codex", Provider.CODEX),
    ("gpt-", Provider.CODEX),
    ("sol", Provider.CODEX),
    ("terra", Provider.CODEX),
    ("luna", Provider.CODEX),
    ("grok", Provider.XAI),
    ("kimi", Provider.KIMI),
    ("deepseek", Provider.DEEPSEEK),
    ("apple", Provider.APPLE),
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


@dataclass(frozen=True)
class Assignment:
    """Resolved execution target for one orchestration role."""

    role: Role
    model: str
    provider: Provider
    backend: Optional[str]
    execute_natively: bool
    reason: str


_PROVIDER_TARGETS = {
    Provider.DEEPSEEK: ("deepseek", Complexity.CHEAP),
    Provider.XAI: ("grok", Complexity.CODE),
    Provider.KIMI: ("kimi", Complexity.CODE),
    Provider.CODEX: ("codex", Complexity.ADVERSARIAL),
    Provider.APPLE: ("apple-fm", Complexity.CHEAP),
    Provider.OLLAMA: ("ollama", Complexity.CHEAP),
}


def default_mode_for_role(role: Role) -> str:
    """Code default for a role's execution mode (spec 002 SC-3).

    Every role is a CLI worker. This mirrors Config.DEFAULT_ROLE_MODES but lives
    in the router so resolve_role works without a config section.
    """
    return "agentic"


def orchestrator_provider(value: str | Provider) -> Provider:
    """Resolve an orchestrator provider name or model id."""
    if isinstance(value, Provider):
        return value
    normalized = value.strip().lower()
    try:
        return Provider(normalized)
    except ValueError:
        return provider_for_model(normalized)


def resolve_role(
    role: Role,
    orchestrator: str | Provider,
    config: Config,
    exhausted_providers: Iterable[Provider] = (),
    mode: str = "chat",
    claude_only: bool = False,
) -> Assignment:
    """Resolve one configured role without calling any backend.

    claude_only (data class `orange`): every non-Anthropic candidate is out, in
    the deck path and the legacy walk alike, so the walk ends in a refusal
    instead of falling through to another provider.

    mode='agentic' (spec 002) changes ONE thing: an Anthropic candidate is NOT
    handed back for native execution — it targets the anthropic-cli agentic
    harness (cc-brain claude). This is the codex-orchestrator adversary case
    AND the GLM+codex-exhausted final fallback worker, where there may be no
    native Anthropic loop to hand back to. The skip-self rule (the adversary
    never resolves to the orchestrator's own provider) applies either way."""
    if role is Role.ORCHESTRATOR:
        raise ValueError("orchestrator is selected by the human, not the router")

    orchestrator_owner = orchestrator_provider(orchestrator)
    exhausted = set(exhausted_providers)
    skip_self = role is Role.ADVERSARY

    # C04 (specs/001-agent-capability-routing): the deck path. It changes WHICH
    # candidates are offered and in what order; it does not change who may receive
    # work -- skip-self, skip-exhausted and the agentic Anthropic rule below are
    # applied to a deck card exactly as to a [roles] entry. Anything the deck
    # cannot answer falls through to the legacy walk, which is never deleted (R16).
    fallback_reason = None  # ROUTER: why the legacy walk ran when the deck was asked first
    if getattr(config, "routing_mode", "legacy") == "deck":
        from .deck import COST_ORDER, gated, load_deck, measured, read_daily_calls, read_quota
        band = (getattr(config, "role_bands", None) or {}).get(role.value)
        row = load_deck().get(band) if band else None
        n_measured = sum(measured(c, band) for c in (row.ranked + row.unranked)) if row else 0
        min_measured = getattr(config, "deck_min_measured", 2)
        if row is None:
            logger.warning("routing_mode=deck but role %r has no band (role_bands=%r); legacy walk",
                           role.value, getattr(config, "role_bands", None))
        elif n_measured < min_measured:
            # ROUTER (replaces R26's ranked-count rule): under relative eligibility one
            # ranked card is legitimate, but fewer than two MEASURED cards means there is
            # no comparison behind "best" -- the 2026-09-07 one-card-band failure.
            fallback_reason = "reference-thin-evidence (band %s: %d measured, need %d)" % (
                band, n_measured, min_measured)
            logger.warning("routing_mode=deck: %s; legacy walk", fallback_reason)
        else:
            survivors = gated(row, read_quota(),
                              daily_calls=read_daily_calls(),
                              daily_cap=getattr(config, "daily_call_cap", None))
            if not survivors:
                # C03 STOP: a gate that empties a band is a routing hole, and the
                # legacy walk must not then hand the work to the very provider the
                # gate removed. Exclude every gated-out provider from the walk.
                gated_out = {provider_for_model(c.router_model) for c in row.ranked
                             if _router_provider_ok(c.router_model)}
                exhausted |= gated_out
                fallback_reason = "fallback-quota (band %s gated: %s)" % (
                    band, ",".join(sorted(p.value for p in gated_out)))
                logger.warning("routing hole: band %s fully gated (%s); legacy walk excludes them",
                               band, sorted(p.value for p in gated_out))
            for card in survivors:
                if card.provider not in COST_ORDER:
                    continue  # a discontinued provider in an old deck file never routes
                # provider_for_model on the ROUTER model id gives the router's own
                # Provider, so the skip rules compare like with like. The deck's
                # Card.provider (moonshot/openai) is only ever compared against
                # headroom.json, inside gated(), where both sides use deck names.
                try:
                    provider = provider_for_model(card.router_model)
                except ValueError:
                    # A card the router cannot name cannot be routed. Found live:
                    # o1 and o4-mini match no prefix in _MODEL_PROVIDER_PREFIXES.
                    # Skipping is the only safe move -- raising here would take
                    # the whole role down for a card that was never going to win,
                    # and guessing a provider would send work to the wrong CLI.
                    logger.warning("deck %s: card %r has no router provider; skipped",
                                   band, card.router_model)
                    continue
                if provider in exhausted or (skip_self and provider is orchestrator_owner):
                    continue
                if not _eligible(role, provider, claude_only):
                    continue
                return _assignment_for(role, card.router_model, provider, mode,
                                       reason=f"deck {band}: eligible-cheapest (within the band's best, cost order)")

    candidates = (config.roles or {}).get(role.value, [])
    if not candidates:
        raise ValueError(f"no configured candidates for role: {role.value}")

    for model in candidates:
        provider = provider_for_model(model)
        if provider in exhausted:
            continue
        # Skip-self applies to ADVERSARY only (Mohan, 2026-09-07): a reviewer
        # must come from a different brain than the orchestrator. Every other
        # role walks its configured list in order, own provider included.
        if skip_self and provider is orchestrator_owner:
            continue
        if not _eligible(role, provider, claude_only):
            continue
        return _assignment_for(role, model, provider, mode,
                               reason=fallback_reason or "first eligible configured candidate")

    if role is Role.ADVERSARY:
        raise ValueError(
            "no eligible adversary differs from the orchestrator provider"
        )
    raise ValueError(
        f"no eligible candidate for role: {role.value} (all providers exhausted)"
    )


def _eligible(role: Role, provider: Provider, claude_only: bool) -> bool:
    """Class orange reaches Claude only; Ollama answers short `simple` text only."""
    if claude_only and provider is not Provider.ANTHROPIC:
        return False
    return provider is not Provider.OLLAMA or role is Role.SIMPLE


def _router_provider_ok(model_id: str) -> bool:
    try:
        provider_for_model(model_id); return True
    except ValueError:
        return False


def _assignment_for(role: Role, model: str, provider: Provider, mode: str,
                    reason: str) -> Assignment:
    """The one place a (model, provider) becomes an Assignment. Both the deck path
    and the legacy walk return through here, so the agentic-Anthropic rule cannot
    apply to one and not the other."""
    if provider is Provider.ANTHROPIC:
        if mode == "agentic":
            # 002: no native hand-back — shell to the Anthropic CLI worker.
            return Assignment(
                role=role, model=model, provider=provider,
                backend="anthropic-cli", execute_natively=False,
                reason=(
                    f"{reason}; Anthropic agentic worker — shells to cc-brain claude in "
                    "the real cwd"
                ),
            )
        # chat mode floor: the MCP never calls Anthropic. Native orchestrator
        # executes it.
        return Assignment(
            role=role, model=model, provider=provider,
            backend=None, execute_natively=True,
            reason="Anthropic candidate must execute in the native orchestrator",
        )
    backend, _ = _PROVIDER_TARGETS[provider]
    return Assignment(role=role, model=model, provider=provider,
                      backend=backend, execute_natively=False, reason=reason)


async def route_assignment(
    assignment: Assignment,
    prompt: str,
    config: Config,
    mode: str = "chat",
    cwd: Optional[str] = None,
    effort: Optional[str] = None,
) -> "RouteResult":
    """Execute a non-native assignment through existing tier machinery.

    cwd is the orchestrator-supplied working directory for agentic subprocesses
    (see route()); threaded through so the worker writes into the caller's repo.
    """
    if assignment.execute_natively or assignment.backend is None:
        raise ValueError("native assignment cannot be executed by the router")
    # 002: the anthropic-cli agentic harness has no Complexity tier
    # (it's the codex-orchestrator adversary / GLM+codex-exhausted fallback).
    # Dispatch straight to the agentic router by backend name.
    if assignment.backend == "anthropic-cli":
        try:
            return await _route_agentic(
                "anthropic-cli", prompt, assignment.model, config, cwd, effort=effort
            )
        except BackendQuotaError as e:
            return RouteResult(
                content="anthropic quota exhausted; advance to the next role candidate",
                model="",
                backend="none",
                complexity=Complexity.CODE,
                headroom_used=False,
                exhausted=True,
                tried=["anthropic-cli"],
                reset_at=e.reset_at,
                failure_kind="quota_exhausted",
                failure_reason=str(e),
            )
        except BackendTransientError as e:
            # A transient 5xx/timeout is NOT quota exhaustion, but the role
            # cascade must still ADVANCE (exhausted=True) so a blip on this
            # provider fails over to the next candidate instead of aborting the
            # whole shard. failure_kind stays honest — not quota_exhausted.
            return RouteResult(
                content="anthropic transient error; advance to the next role candidate",
                model="",
                backend="none",
                complexity=Complexity.CODE,
                headroom_used=False,
                exhausted=True,
                tried=["anthropic-cli"],
                failure_kind="transient_error",
                failure_reason=str(e),
            )
    # Non-Claude workers run tools-off with the prompt inline (lean-rig §6): one
    # text answer, no file or shell access, whatever `mode` says. A worker that
    # needs tools runs in the review sandbox (scripts/sandbox.sh), not here.
    provider = assignment.provider
    if provider is Provider.APPLE:
        # apple-fm answers text only (no tools, no cwd). Its failures raise
        # BackendError and the role walk moves on.
        _validate_credentials("apple-fm", config)
        result = await asyncio.to_thread(backends.call_fm, prompt, config.fm_host)
        return RouteResult(
            content=result["content"], model=assignment.model, backend="apple-fm",
            complexity=Complexity.CHEAP, headroom_used=False, usage=None, tried=["apple-fm"],
        )
    if provider is Provider.OLLAMA:
        _validate_credentials("ollama", config)
        result = await backends.call_ollama(prompt, config.ollama_host, assignment.model)
        return RouteResult(
            content=result["content"], model=assignment.model, backend="ollama",
            complexity=Complexity.CHEAP, headroom_used=False, usage=result.get("usage"),
            tried=["ollama"],
        )
    name = _PROVIDER_TARGETS[provider][0]
    _validate_credentials(name, config)
    try:
        if provider is Provider.XAI:
            result = await _route_grok(prompt, assignment.model, config, effort=effort)
        elif provider is Provider.KIMI:
            result = await asyncio.to_thread(backends.call_kimi, prompt, assignment.model)
        elif provider is Provider.DEEPSEEK:
            result = await _route_deepseek(prompt, assignment.model, config)
        else:
            result = await _route_codex(prompt, assignment.model, config, effort=effort)
    except (BackendQuotaError, BackendTransientError) as e:
        return _advance_result(name, e)
    result.backend = name
    result.tried = [name]
    return result


def _advance_result(name: str, e: Exception) -> "RouteResult":
    """Quota or transient failure: the role walk advances to the next candidate.
    failure_kind stays honest (a 5xx or timeout is never labelled quota)."""
    quota = isinstance(e, BackendQuotaError)
    return RouteResult(
        content=f"{name} {'quota exhausted' if quota else 'transient error'}; "
                "advance to the next role candidate",
        model="",
        backend="none",
        complexity=Complexity.CODE,
        headroom_used=False,
        exhausted=True,
        tried=[name],
        reset_at=getattr(e, "reset_at", None),
        failure_kind="quota_exhausted" if quota else "transient_error",
        failure_reason=str(e),
    )


# Legacy tier ownership. `adversarial` is the one single-backend tier left
# (Codex, tools-off). GLM expired 2026-10-25 and left the rig (D5), so `cheap`
# and `code` walk the `simple` and `worker` roles (server._delegate_impl).
_TIER_BACKENDS = {
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
    mode: str = "chat",
    cwd: Optional[str] = None,
    effort: Optional[str] = None,
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
        mode: Execution mode — 'chat' (text-only, the original path) or
              'agentic' (spec 002: shells to the per-provider CLI harness in the
              REAL cwd so the worker writes files / runs checks itself).
        cwd: Working directory for agentic-mode subprocesses. REQUIRED for
             correct agentic behaviour — the MCP server is a long-lived process
             whose os.getcwd() is fixed at launch (whichever session first
             spawned it), NOT the caller's repo. The orchestrator MUST pass its
             own cwd so the worker writes files into the caller's directory, not
             the server's launch dir. Ignored in chat mode (no subprocess cwd).

    Returns:
        RouteResult containing response, metadata, and backend info.

    Raises:
        MissingCredentialError: If a required key/config is missing.
        BackendUnavailableError: If the backend cannot be reached or is disabled.
        ValueError: If complexity or model_override is invalid.
    """
    if config is None:
        config = Config.load()

    if mode == "agentic" and not cwd:
        raise ValueError("cwd is required for agentic routing")

    backend_name, model = _resolve_backend_and_model(complexity, model_override, config)
    _validate_credentials(backend_name, config)

    try:
        # Codex is the one tier backend; it runs tools-off whatever `mode` says
        # (lean-rig §6). C06: effort rides the call.
        result = await _route_codex(prompt, model, config, effort=effort)
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
    except BackendTransientError as e:
        # Transient provider error (5xx / timeout) is not quota exhaustion, but
        # the role cascade must still advance (exhausted=True) so a blip fails
        # over to the next candidate rather than aborting the shard. Kept honest
        # via failure_kind="transient_error" (never labeled quota_exhausted).
        return RouteResult(
            content=(
                f"{backend_name} transient error for tier {complexity.value}. "
                f"Advance to the next role candidate or retry."
            ),
            model="",
            backend="none",
            complexity=complexity,
            headroom_used=False,
            usage=None,
            exhausted=True,
            tried=[backend_name],
            failure_kind="transient_error",
            failure_reason=str(e),
        )

    result.complexity = complexity
    result.backend = backend_name
    result.exhausted = False
    result.tried = [backend_name]
    return result


async def _route_agentic(
    backend_name: str,
    prompt: str,
    model: str,
    config: Config,
    cwd: Optional[str] = None,
    effort: Optional[str] = None,
) -> RouteResult:
    """Agentic dispatch — the Claude worker only (`cc-brain claude` in the REAL
    cwd, tools on). Non-Claude workers run tools-off (route_assignment). The
    call is a blocking subprocess.run, so it is offloaded to a worker thread.

    cwd is the orchestrator-supplied working directory (see route()); None
    falls back to the server's launch dir, so callers must pass it."""
    if backend_name != "anthropic-cli":  # pragma: no cover - closed set
        raise ValueError(f"No agentic harness for backend: {backend_name}")
    result = await asyncio.to_thread(
        backends.call_anthropic_agentic, prompt, model, cwd
    )
    return RouteResult(
        content=result["content"],
        model=model,
        backend="anthropic-cli",
        complexity=None,
        headroom_used=False,
        usage=result.get("usage"),
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
    if complexity not in _TIER_BACKENDS:
        raise ValueError(f"Unknown complexity: {complexity}")

    backend_name = _TIER_BACKENDS[complexity]

    # Resolve model
    if model_override:
        model = model_override
    elif config.model_overrides and complexity.value in config.model_overrides:
        model = config.model_overrides[complexity.value]
    else:
        model = _get_backend_default_model(backend_name, config)

    return backend_name, model


def _get_backend_default_model(backend_name: str, config: Config) -> str:
    """Get the default model for a backend."""
    defaults = {
        # D5 names: deepseek-flash (bulk reads) and deepseek-v4-pro (worker).
        "deepseek": "deepseek-flash",
        # xAI Grok agentic CLI worker (grok.com OAuth, no API key). It is a
        # coding provider selected by role routing, not a public complexity tier.
        "grok": "grok-4.5",
        # Kimi Code CLI agentic worker (OAuth login, no API key). It is a
        # coding provider selected by role routing, not a public complexity
        # tier. No -m flag is passed — the CLI runs its configured default
        # model, so the router model id is just "kimi".
        "kimi": "kimi",
        # D5: the only model the ChatGPT login accepts (tested 2026-11-06).
        "codex": "gpt-6.1-sol",
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
    elif backend_name == "grok":
        # Grok is a CLI worker (grok.com OAuth login, no API key). Availability
        # is a boolean enable flag like Codex, not a stored credential.
        if not config.grok_enabled:
            raise BackendUnavailableError("grok", "Grok not enabled or not available on PATH")
    elif backend_name == "kimi":
        # Kimi is a CLI worker (Kimi Code CLI OAuth login, no API key).
        # Availability is a boolean enable flag like Grok/Codex, not a stored
        # credential.
        if not config.kimi_enabled:
            raise BackendUnavailableError("kimi", "Kimi not enabled or not available on PATH")
    elif backend_name == "codex":
        if not config.codex_enabled:
            raise BackendUnavailableError("codex", "Codex not enabled or not available on PATH")
    elif backend_name == "apple-fm":
        # The ssh host doubles as the enable flag: no fm_host, no fm tier.
        if not config.fm_host:
            raise BackendUnavailableError("apple-fm", "fm_host not set in config.toml")
    elif backend_name == "ollama":
        if not getattr(config, "ollama_host", None):
            raise BackendUnavailableError("ollama", "MOHAN_OLLAMA_HOST not set in compute.env")


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


async def _route_grok(
    prompt: str,
    model: str,
    config: Config,
    effort: Optional[str] = None,
) -> RouteResult:
    """Route to Grok (subprocess, tools-off, final message only).

    The Grok CLI is subprocess-based (BLOCKING subprocess.run), so offload to a
    worker thread to keep the async MCP stdio event loop responsive (same
    freeze/keepalive reasoning as _route_codex). Never uses headroom.
    """
    result = await asyncio.to_thread(
        backends.call_grok,
        prompt,
        model, effort=effort)

    return RouteResult(
        content=result["content"],
        model=model,
        backend="grok",
        complexity=Complexity.CODE,
        headroom_used=False,
        usage=result.get("usage"),
    )


async def _route_codex(
    prompt: str,
    model: str,
    config: Config,
    effort: Optional[str] = None,
) -> RouteResult:
    """Route to Codex backend (subprocess-based, tools-off)."""
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
        effort,
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
    mode: str = "chat",
) -> RouteResult:
    """Synchronous wrapper around route() for non-async contexts."""
    return asyncio.run(route(complexity, prompt, model_override, config, mode=mode))
