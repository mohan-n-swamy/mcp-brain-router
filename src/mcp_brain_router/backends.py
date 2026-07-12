"""Backend implementations for DeepSeek, GLM, and Codex.

Each backend is isolated; router.py calls these functions.
"""

import os
import shutil
import subprocess
import time
from typing import Any, Dict, Optional

import httpx


class BackendError(Exception):
    """Base exception for backend errors."""

    def __init__(
        self,
        message: str,
        *,
        backend: Optional[str] = None,
        failure_kind: str = "backend_error",
        elapsed_ms: Optional[int] = None,
    ):
        if backend is not None:
            self.backend = backend
        self.failure_kind = failure_kind
        self.elapsed_ms = elapsed_ms
        super().__init__(message)


class BackendQuotaError(BackendError):
    """Raised only when a backend returns HTTP 429 quota/rate exhaustion.

    Carries the HTTP status and a sanitized message. For rate-limit bodies the
    provider message (e.g. GLM "Usage limit reached for 5 hour … reset at …")
    holds NO secret — only the reset time — so it is safe to surface and is
    what lets the orchestrator decide whether to wait.
    """

    def __init__(
        self, provider: str, status_code: int, message: str = "", reset_at: Optional[str] = None
    ):
        self.provider = provider
        self.backend = provider.lower()
        self.status_code = status_code
        self.message = message
        self.reset_at = reset_at
        detail = f" — {message}" if message else ""
        super().__init__(
            f"{provider} quota error {status_code}{detail}",
            backend=provider.lower(),
            failure_kind="quota_exhausted",
        )


# Resolve the codex binary to an ABSOLUTE path at import time. The MCP server
# is spawned by Claude Code with a minimal/empty env (`"env": {}`), so the
# subprocess PATH does NOT include `~/.npm-global/bin` where codex lives —
# bare "codex" → FileNotFoundError → adversarial tier fails instantly
# (root-caused 2026-07-05: 4/4 adversarial calls exhausted, 0 tokens).
# shutil.which honors PATH when present; the ~/.npm-global fallback covers the
# empty-PATH MCP case. Falls back to bare "codex" so a missing binary still
# raises the explicit "not found on PATH" BackendError, not an opaque one.
_CODEX_BIN = shutil.which("codex") or (
    os.path.expanduser("~/.npm-global/bin/codex")
    if os.path.exists(os.path.expanduser("~/.npm-global/bin/codex"))
    else "codex"
)

# Canonical lean Codex worker invocation — shared with install.py's smoke test
# so the two can never drift. Live A/B (2026-07-11): the full user rig used
# 23,145 tokens / ~69s for PONG; these flags used 14,907 / 10.6s. Delegated
# prompts are self-contained workers: no user config, rules, plugins, hooks,
# memories, apps, multi-agent fan-out, persisted session, MCPs, or repo context.
CODEX_EXEC_BASE = [
    _CODEX_BIN,
    "exec",
    "--skip-git-repo-check",
    "--ignore-user-config",
    "--ignore-rules",
    "--ephemeral",
    "--disable",
    "plugins",
    "--disable",
    "hooks",
    "--disable",
    "memories",
    "--disable",
    "apps",
    "--disable",
    "multi_agent",
    "-c",
    "mcp_servers={}",
    "-c",
    'model_reasoning_effort="low"',
    "-C",
    "/private/tmp",
]

CODEX_TIMEOUT_SECONDS = 360  # 004 C410: 180s starved long adversarial prompts (2026-07-12)
# 004 C410: 30s starved GLM on >1k-token prompts (died at exactly the cap, reported
# as network_error). One constant, all HTTP backends — DRY.
HTTP_TIMEOUT_SECONDS = 120.0

# Agentic worker timeout — agentic mode does REAL file + check work (a full
# build loop step), so it runs longer than a text-only chat reply. Same as the
# codex adversarial cap: one constant, every agentic backend.
AGENTIC_TIMEOUT_SECONDS = 360


# ============================================================================
# CLI harness binaries for the agentic worker mode (spec 002).
# Resolved at import time so the empty-PATH MCP-process case still finds them.
# Each is the per-provider headless CLI the router shells to in the REAL cwd so
# the worker reads the spec, writes files, and runs checks itself.
# ============================================================================
def _resolve_bin(primary: str, fallback_dirs: tuple[str, ...] = ("~/.local/bin",)) -> str:
    """Resolve a CLI binary to an absolute path (honors shutil.which, then
    ~/.local/bin fallbacks for the empty-PATH MCP-process case). Falls back to
    the bare name so a missing binary raises the explicit not-found error."""
    resolved = shutil.which(primary)
    if resolved:
        return resolved
    for d in fallback_dirs:
        cand = os.path.expanduser(os.path.join(d, primary))
        if os.path.exists(cand):
            return cand
    return primary


_CC_GLM_BIN = _resolve_bin("cc-glm")
_CC_BRAIN_BIN = _resolve_bin("cc-brain")


# Codex agentic base flags — SAME lean worker profile as CODEX_EXEC_BASE
# (skip-git-repo-check, ignore-user-config, ephemeral, all --disable, no MCPs,
# low reasoning) EXCEPT two flags dropped so the agentic worker can edit the
# caller's repo:
#   - `-C /private/tmp`  → run in the REAL cwd (files must land in the repo)
#   - `--ignore-rules`  → removed so codex's edit/permission rules can apply
# (spec 002 SC-6). The cwd is set per-call via subprocess.run(cwd=...).
CODEX_EXEC_BASE_AGENTIC = [
    _CODEX_BIN,
    "exec",
    "--skip-git-repo-check",
    "--ignore-user-config",
    "--ephemeral",
    "--disable",
    "plugins",
    "--disable",
    "hooks",
    "--disable",
    "memories",
    "--disable",
    "apps",
    "--disable",
    "multi_agent",
    "-c",
    "mcp_servers={}",
    "-c",
    'model_reasoning_effort="low"',
]


def _find_mise_node() -> str:
    """Locate the mise-managed node binary when PATH is empty (MCP-process
    case). Prefer the `lts` alias; fall back to any installed version. Returns
    "" if none found (caller then relies on whatever PATH provides)."""
    import glob

    base = os.path.expanduser("~/.local/share/mise/installs/node")
    for cand in [os.path.join(base, "lts", "bin", "node")] + sorted(
        glob.glob(os.path.join(base, "*", "bin", "node")), reverse=True
    ):
        if os.path.exists(cand):
            return cand
    return ""


def _codex_env() -> Dict[str, str]:
    """Env for the codex subprocess with a PATH that resolves BOTH codex and
    its `#!/usr/bin/env node` shebang's `node`. The MCP server runs with an
    empty env (`"env": {}`), so codex's bin dir AND the node bin dir must be
    prepended — otherwise the shim fails with `env: node: No such file` and
    codex exits non-zero (root-caused 2026-07-05, second half of the same
    PATH bug). Existing PATH entries are preserved; only prepended, never
    replaced, so a normal shell launch is unaffected."""
    extra = []
    node_path = shutil.which("node") or _find_mise_node()
    node_bin = os.path.dirname(node_path) if node_path else ""
    codex_bin = os.path.dirname(_CODEX_BIN) if os.path.sep in _CODEX_BIN else ""
    for d in (codex_bin, node_bin):
        if d and d not in extra:
            extra.append(d)
    env = dict(os.environ)
    current = env.get("PATH", "")
    env["PATH"] = os.pathsep.join([*extra, current]) if current else os.pathsep.join(extra)
    return env


# Caveman-ultra system directive prepended to EVERY delegated backend call
# (DeepSeek/GLM via the `system` field; Codex prepended to the prompt). Shapes
# how backends REPLY (terse, no filler) — cuts output tokens on every delegate
# without touching the task instruction itself, so answer quality is unaffected.
# One constant, all backends: DRY single source of truth.
CAVEMAN_SYSTEM = (
    "Reply caveman ultra: drop articles, filler, pleasantries, hedging. "
    "Fragments OK. Keep ALL technical substance exact — code, numbers, error "
    "strings, identifiers unchanged. Answer only what was asked; no preamble, "
    "no restating the question, no meta-commentary. Terse."
)


class BackendTransientError(BackendError):
    """Raised on timeout/provider failure that is not quota exhaustion."""

    def __init__(
        self,
        provider: str,
        reason: str,
        *,
        failure_kind: str = "transient_error",
        status_code: Optional[int] = None,
        elapsed_ms: Optional[int] = None,
    ):
        self.provider = provider
        self.status_code = status_code
        super().__init__(
            reason,
            backend=provider.lower(),
            failure_kind=failure_kind,
            elapsed_ms=elapsed_ms,
        )


# Provider-side transient failures are errors, not quota exhaustion. The
# orchestrator may choose another tier, but the router never labels these 5xx
# responses as exhausted quota.
_TRANSIENT_STATUS = frozenset({500, 502, 503, 504})


def _classify_http_error(provider: str, status_code: int, body: str) -> BackendError:
    """Map a non-200 HTTP response to the right exception type.

    429  -> BackendQuotaError (genuine quota exhaustion).
    5xx  -> BackendTransientError (provider failure, not quota).
    Other -> BackendError (hard stop, surfaces to caller).

    The body is parsed ONLY to lift a rate-limit message + reset time, which
    carry no secret. The raw body is never echoed wholesale.
    """
    message = ""
    reset_at = None
    try:
        import json as _json

        data = _json.loads(body)
        err = data.get("error", data) if isinstance(data, dict) else {}
        if isinstance(err, dict):
            message = str(err.get("message", ""))[:300]
    except Exception:
        message = ""
    # Best-effort reset-time lift from common "reset at YYYY-MM-DD HH:MM:SS" shape.
    if message:
        import re as _re

        m = _re.search(r"reset(?:s)?\s+at\s+([0-9:\- ]+)", message)
        if m:
            reset_at = m.group(1).strip()

    if status_code == 429:
        return BackendQuotaError(provider, status_code, message, reset_at)
    if status_code in _TRANSIENT_STATUS:
        detail = f": {message}" if message else ""
        return BackendTransientError(
            provider,
            f"{provider} provider error {status_code}{detail}",
            failure_kind="provider_error",
            status_code=status_code,
        )
    # Non-retryable: do NOT echo the body (may carry request echoes); message
    # from the parsed error field is safe and useful.
    safe = f": {message}" if message else " (body withheld)"
    return BackendError(
        f"{provider} API error {status_code}{safe}",
        backend=provider.lower(),
    )


def _extract_text(data: Dict[str, Any], provider: str) -> str:
    """Pull the assistant text from an Anthropic-shape response.

    Robust to reasoning models: scans content[] for the FIRST block whose
    type == 'text' (skipping 'thinking'/'redacted_thinking' blocks that
    DeepSeek/GLM reasoning models emit). Raises a clear BackendError if the
    response carried no text block (e.g. max_tokens exhausted while thinking).
    """
    blocks = data.get("content")
    if not isinstance(blocks, list) or not blocks:
        raise BackendError(f"{provider} response had no content blocks")
    for block in blocks:
        if isinstance(block, dict) and block.get("type") == "text" and "text" in block:
            return block["text"]
    # No text block — most commonly a reasoning model that spent the whole
    # token budget on 'thinking'. Surface it actionably.
    types = [b.get("type") for b in blocks if isinstance(b, dict)]
    stop = data.get("stop_reason")
    raise BackendError(
        f"{provider} returned no text block (block types={types}, "
        f"stop_reason={stop}). If a reasoning model hit max_tokens while "
        f"thinking, raise max_tokens or use a non-reasoning model."
    )


# ============================================================================
# DeepSeek (HTTP, Anthropic-compatible endpoint)
# ============================================================================


async def call_deepseek(
    prompt: str,
    model: str,
    api_key: str,
) -> Dict[str, Any]:
    """
    Call DeepSeek API directly.

    Returns:
        {
            "content": "response text",
            "usage": {"input_tokens": N, "output_tokens": M}
        }

    Raises:
        BackendError: On HTTP error or parse failure (never logs the key).
    """
    url = "https://api.deepseek.com/anthropic/v1/messages"
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    payload = {
        "model": model,
        "max_tokens": 4096,
        "system": CAVEMAN_SYSTEM,
        "messages": [{"role": "user", "content": prompt}],
    }

    try:
        async with httpx.AsyncClient(timeout=HTTP_TIMEOUT_SECONDS) as client:
            response = await client.post(url, json=payload, headers=headers)

            if response.status_code != 200:
                raise _classify_http_error("DeepSeek", response.status_code, response.text)

            data = response.json()

        # Extract content (robust to reasoning 'thinking' blocks) and usage
        content = _extract_text(data, data.get("model", "backend"))
        usage = data.get("usage", {})

        return {
            "content": content,
            "usage": usage,
        }
    except httpx.RequestError as e:
        raise BackendError(
            f"DeepSeek request failed: {e}",
            backend="deepseek",
            failure_kind="network_error",
        )
    except (KeyError, IndexError, ValueError) as e:
        raise BackendError(
            f"DeepSeek response parse error: {e}",
            backend="deepseek",
            failure_kind="response_error",
        )


async def call_deepseek_via_headroom(
    prompt: str,
    model: str,
    api_key: str,
    headroom_url: str,
) -> Dict[str, Any]:
    """
    Call DeepSeek through headroom proxy.

    Raises:
        BackendError: On HTTP error or parse failure (never logs the key).
    """
    url = f"{headroom_url}/anthropic/v1/messages"
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    payload = {
        "model": model,
        "max_tokens": 4096,
        "system": CAVEMAN_SYSTEM,
        "messages": [{"role": "user", "content": prompt}],
    }

    try:
        async with httpx.AsyncClient(timeout=HTTP_TIMEOUT_SECONDS) as client:
            response = await client.post(url, json=payload, headers=headers)

            if response.status_code != 200:
                raise _classify_http_error("DeepSeek-headroom", response.status_code, response.text)

            data = response.json()

        content = _extract_text(data, data.get("model", "DeepSeek-headroom"))
        usage = data.get("usage", {})

        return {
            "content": content,
            "usage": usage,
        }
    except httpx.RequestError as e:
        raise BackendError(
            f"DeepSeek headroom request failed: {e}",
            backend="deepseek",
            failure_kind="network_error",
        )
    except (KeyError, IndexError, ValueError) as e:
        raise BackendError(
            f"DeepSeek headroom response parse error: {e}",
            backend="deepseek",
            failure_kind="response_error",
        )


# ============================================================================
# GLM (HTTP, Anthropic-compatible endpoint)
# ============================================================================


async def call_glm(
    prompt: str,
    model: str,
    api_key: str,
) -> Dict[str, Any]:
    """
    Call GLM API directly.

    Returns:
        {
            "content": "response text",
            "usage": {"input_tokens": N, "output_tokens": M}
        }

    Raises:
        BackendError: On HTTP error or parse failure (never logs the key).
    """
    url = "https://api.z.ai/api/anthropic/v1/messages"
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    payload = {
        "model": model,
        "max_tokens": 4096,
        "system": CAVEMAN_SYSTEM,
        "messages": [{"role": "user", "content": prompt}],
    }

    try:
        async with httpx.AsyncClient(timeout=HTTP_TIMEOUT_SECONDS) as client:
            response = await client.post(url, json=payload, headers=headers)

            if response.status_code != 200:
                raise _classify_http_error("GLM", response.status_code, response.text)

            data = response.json()

        content = _extract_text(data, data.get("model", "GLM"))
        usage = data.get("usage", {})

        return {
            "content": content,
            "usage": usage,
        }
    except httpx.RequestError as e:
        raise BackendError(
            f"GLM request failed: {e}",
            backend="glm",
            failure_kind="network_error",
        )
    except (KeyError, IndexError, ValueError) as e:
        raise BackendError(
            f"GLM response parse error: {e}",
            backend="glm",
            failure_kind="response_error",
        )


async def call_glm_via_headroom(
    prompt: str,
    model: str,
    api_key: str,
    headroom_url: str,
) -> Dict[str, Any]:
    """
    Call GLM through headroom proxy.

    Raises:
        BackendError: On HTTP error or parse failure (never logs the key).
    """
    url = f"{headroom_url}/anthropic/v1/messages"
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    payload = {
        "model": model,
        "max_tokens": 4096,
        "system": CAVEMAN_SYSTEM,
        "messages": [{"role": "user", "content": prompt}],
    }

    try:
        async with httpx.AsyncClient(timeout=HTTP_TIMEOUT_SECONDS) as client:
            response = await client.post(url, json=payload, headers=headers)

            if response.status_code != 200:
                raise _classify_http_error("GLM-headroom", response.status_code, response.text)

            data = response.json()

        content = _extract_text(data, data.get("model", "GLM-headroom"))
        usage = data.get("usage", {})

        return {
            "content": content,
            "usage": usage,
        }
    except httpx.RequestError as e:
        raise BackendError(
            f"GLM headroom request failed: {e}",
            backend="glm",
            failure_kind="network_error",
        )
    except (KeyError, IndexError, ValueError) as e:
        raise BackendError(
            f"GLM headroom response parse error: {e}",
            backend="glm",
            failure_kind="response_error",
        )


# ============================================================================
# Codex (subprocess-based, adversarial)
# ============================================================================


def _validate_model_name(model: str) -> None:
    """
    Validate model name against argument injection attacks.

    Args:
        model: Model string to validate

    Raises:
        BackendError: If model contains disallowed characters.
    """
    import re

    if not re.match(r"^[A-Za-z0-9._-]+$", model):
        raise BackendError(
            "Invalid model name. Must contain only alphanumeric, dot, underscore, or hyphen.",
            backend="codex",
            failure_kind="validation_error",
        )


def call_codex(
    prompt: str,
    model: str,
) -> Dict[str, Any]:
    """
    Call Codex CLI via subprocess: codex exec -m <model> <prompt>.

    Args:
        model: Codex model string (e.g., "gpt-5.6-sol")
        prompt: The prompt/code to send

    Returns:
        {
            "content": "response text",
            "usage": None (Codex CLI doesn't expose token counts)
        }

    Raises:
        BackendError: If model name is invalid or subprocess fails (never logs secrets).
    """
    # Validate model name to prevent argument injection via model string
    _validate_model_name(model)

    started = time.perf_counter()
    try:
        result = subprocess.run(
            # "--" ends option parsing so a prompt starting with "-" can't be
            # read as a codex flag (live-verified: without it, prompt="-h"
            # prints CLI help instead of delegating).
            CODEX_EXEC_BASE + ["-m", model, "--", f"{CAVEMAN_SYSTEM}\n\n{prompt}"],
            capture_output=True,
            text=True,
            timeout=CODEX_TIMEOUT_SECONDS,
            check=False,
            env=_codex_env(),
        )

        if result.returncode != 0:
            raise BackendError(
                "Codex subprocess failed",
                backend="codex",
                failure_kind="process_error",
                elapsed_ms=round((time.perf_counter() - started) * 1000),
            )

        return {
            "content": result.stdout.strip(),
            "usage": None,
        }

    except subprocess.TimeoutExpired:
        elapsed_ms = round((time.perf_counter() - started) * 1000)
        raise BackendTransientError(
            "codex",
            f"Codex subprocess timed out after {CODEX_TIMEOUT_SECONDS}s",
            failure_kind="timeout",
            elapsed_ms=elapsed_ms,
        )
    except FileNotFoundError:
        raise BackendError(
            "Codex binary not found on PATH",
            backend="codex",
            failure_kind="configuration_error",
            elapsed_ms=round((time.perf_counter() - started) * 1000),
        )


# ============================================================================
# Agentic worker backends (spec 002) — shell to a per-provider CLI harness in
# the REAL working directory so the worker reads the spec, writes files, and
# runs checks itself. Each returns the {content, ...} dict shape the chat
# backends return; the prompt is passed as an argv token (cc-glm / claude -p)
# or after a `--` separator (codex), the same secure pattern as call_codex —
# never echo a secret and never let an untrusted prompt be parsed as a CLI flag.
# ============================================================================


def call_glm_agentic(prompt: str, model: str, cwd: Optional[str] = None) -> Dict[str, Any]:
    """Agentic GLM worker: `cc-glm -p <prompt> --model <model>` in the REAL cwd.

    cc-glm is the rig's headless GLM CLI; it has file + shell tools and writes
    into the caller's working directory. The resolved model (the tier's glm id,
    e.g. glm-5.2 / glm-4.7) is passed via --model so the caller decides which
    GLM variant runs.
    """
    _validate_model_name(model)
    started = time.perf_counter()
    argv = [_CC_GLM_BIN, "-p", f"{CAVEMAN_SYSTEM}\n\n{prompt}", "--model", model]
    try:
        result = subprocess.run(
            argv,
            capture_output=True,
            text=True,
            timeout=AGENTIC_TIMEOUT_SECONDS,
            check=False,
            cwd=cwd or os.getcwd(),
        )
        if result.returncode != 0:
            raise BackendError(
                "cc-glm agentic subprocess failed",
                backend="glm",
                failure_kind="process_error",
                elapsed_ms=round((time.perf_counter() - started) * 1000),
            )
        return {"content": result.stdout.strip(), "usage": None}
    except subprocess.TimeoutExpired:
        elapsed_ms = round((time.perf_counter() - started) * 1000)
        raise BackendTransientError(
            "glm",
            f"cc-glm agentic subprocess timed out after {AGENTIC_TIMEOUT_SECONDS}s",
            failure_kind="timeout",
            elapsed_ms=elapsed_ms,
        )
    except FileNotFoundError:
        raise BackendError(
            "cc-glm binary not found on PATH",
            backend="glm",
            failure_kind="configuration_error",
            elapsed_ms=round((time.perf_counter() - started) * 1000),
        )


def call_codex_agentic(prompt: str, model: str, cwd: Optional[str] = None) -> Dict[str, Any]:
    """Agentic Codex worker: `codex exec` in the REAL cwd (NOT /private/tmp).

    Reuses CODEX_EXEC_BASE_AGENTIC — the lean worker flags (no MCPs, low
    reasoning, no plugins/hooks/memories/apps/multi_agent) EXCEPT the
    sandbox-cwd flag (`-C /private/tmp`) and `--ignore-rules` are dropped so
    the worker can edit the repo (spec 002 SC-6).
    """
    _validate_model_name(model)
    started = time.perf_counter()
    try:
        result = subprocess.run(
            # "--" ends option parsing so a prompt starting with "-" can't be
            # read as a codex flag (same protection as call_codex).
            CODEX_EXEC_BASE_AGENTIC + ["-m", model, "--", f"{CAVEMAN_SYSTEM}\n\n{prompt}"],
            capture_output=True,
            text=True,
            timeout=AGENTIC_TIMEOUT_SECONDS,
            check=False,
            cwd=cwd or os.getcwd(),
            env=_codex_env(),
        )
        if result.returncode != 0:
            raise BackendError(
                "Codex agentic subprocess failed",
                backend="codex",
                failure_kind="process_error",
                elapsed_ms=round((time.perf_counter() - started) * 1000),
            )
        return {"content": result.stdout.strip(), "usage": None}
    except subprocess.TimeoutExpired:
        elapsed_ms = round((time.perf_counter() - started) * 1000)
        raise BackendTransientError(
            "codex",
            f"Codex agentic subprocess timed out after {AGENTIC_TIMEOUT_SECONDS}s",
            failure_kind="timeout",
            elapsed_ms=elapsed_ms,
        )
    except FileNotFoundError:
        raise BackendError(
            "Codex binary not found on PATH",
            backend="codex",
            failure_kind="configuration_error",
            elapsed_ms=round((time.perf_counter() - started) * 1000),
        )


def call_anthropic_agentic(prompt: str, model: str, cwd: Optional[str] = None) -> Dict[str, Any]:
    """Agentic Anthropic (Opus/Sonnet/Haiku) worker: `cc-brain claude -p` in
    the REAL cwd. This is the agentic-CLI fallback for: the codex-orchestrator
    adversary case (codex can't be its own adversary) AND the GLM+codex-both-
    exhausted final fallback (spec 002 SC-4, SC-8). The resolved candidate's
    model id (e.g. claude-sonnet-5 / claude-opus-4-8) is passed via --model."""
    _validate_model_name(model)
    started = time.perf_counter()
    argv = [
        _CC_BRAIN_BIN,
        "claude",
        "-p",
        f"{CAVEMAN_SYSTEM}\n\n{prompt}",
        "--model",
        model,
    ]
    try:
        result = subprocess.run(
            argv,
            capture_output=True,
            text=True,
            timeout=AGENTIC_TIMEOUT_SECONDS,
            check=False,
            cwd=cwd or os.getcwd(),
        )
        if result.returncode != 0:
            raise BackendError(
                "cc-brain claude agentic subprocess failed",
                backend="anthropic-cli",
                failure_kind="process_error",
                elapsed_ms=round((time.perf_counter() - started) * 1000),
            )
        return {"content": result.stdout.strip(), "usage": None}
    except subprocess.TimeoutExpired:
        elapsed_ms = round((time.perf_counter() - started) * 1000)
        raise BackendTransientError(
            "anthropic-cli",
            f"cc-brain claude agentic subprocess timed out after {AGENTIC_TIMEOUT_SECONDS}s",
            failure_kind="timeout",
            elapsed_ms=elapsed_ms,
        )
    except FileNotFoundError:
        raise BackendError(
            "cc-brain binary not found on PATH",
            backend="anthropic-cli",
            failure_kind="configuration_error",
            elapsed_ms=round((time.perf_counter() - started) * 1000),
        )
