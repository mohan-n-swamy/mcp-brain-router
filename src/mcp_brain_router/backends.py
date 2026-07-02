"""Backend implementations for DeepSeek, GLM, and Codex.

Each backend is isolated; router.py calls these functions.
"""

import asyncio
import subprocess
from typing import Any, Dict, Optional
import httpx


class BackendError(Exception):
    """Base exception for backend errors."""
    pass


class BackendQuotaError(BackendError):
    """Raised when a backend rejects the call for a transient/quota reason
    (HTTP 429 rate limit, or 5xx server error) — i.e. a DIFFERENT backend
    might succeed, so the router should fall through to the next in the chain.

    Carries the HTTP status and a sanitized message. For rate-limit bodies the
    provider message (e.g. GLM "Usage limit reached for 5 hour … reset at …")
    holds NO secret — only the reset time — so it is safe to surface and is
    what lets the orchestrator decide whether to wait.
    """
    def __init__(self, provider: str, status_code: int, message: str = "",
                 reset_at: Optional[str] = None):
        self.provider = provider
        self.status_code = status_code
        self.message = message
        self.reset_at = reset_at
        detail = f" — {message}" if message else ""
        super().__init__(f"{provider} quota/transient error {status_code}{detail}")


class BackendTransientError(BackendError):
    """Raised on a transient hard failure with no HTTP status — subprocess
    timeout or death (Codex CLI). Like BackendQuotaError, a DIFFERENT backend
    might succeed, so the router falls through the chain; unlike config
    errors (401/bad model), it must NOT propagate loud."""
    pass


# HTTP status codes that mean "try a different backend" rather than "fix config".
# 429 = rate limit / quota; 5xx = provider-side transient failure.
# 4xx (auth, bad request) are NOT here — those are config errors, no fallback.
_RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})


def _classify_http_error(provider: str, status_code: int, body: str) -> BackendError:
    """Map a non-200 HTTP response to the right exception type.

    Retryable (429/5xx) -> BackendQuotaError (router falls through).
    Anything else        -> BackendError (hard stop, surfaces to caller).

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

    if status_code in _RETRYABLE_STATUS:
        return BackendQuotaError(provider, status_code, message, reset_at)
    # Non-retryable: do NOT echo the body (may carry request echoes); message
    # from the parsed error field is safe and useful.
    safe = f": {message}" if message else " (body withheld)"
    return BackendError(f"{provider} API error {status_code}{safe}")


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
        "messages": [{"role": "user", "content": prompt}],
    }

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(url, json=payload, headers=headers)

            if response.status_code != 200:
                raise _classify_http_error(
                    "DeepSeek", response.status_code, response.text
                )

            data = response.json()

        # Extract content (robust to reasoning 'thinking' blocks) and usage
        content = _extract_text(data, data.get("model", "backend"))
        usage = data.get("usage", {})

        return {
            "content": content,
            "usage": usage,
        }
    except httpx.RequestError as e:
        raise BackendError(f"DeepSeek request failed: {e}")
    except (KeyError, IndexError, ValueError) as e:
        raise BackendError(f"DeepSeek response parse error: {e}")


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
        "messages": [{"role": "user", "content": prompt}],
    }

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(url, json=payload, headers=headers)

            if response.status_code != 200:
                raise _classify_http_error(
                    "DeepSeek-headroom", response.status_code, response.text
                )

            data = response.json()

        content = data["content"][0]["text"]
        usage = data.get("usage", {})

        return {
            "content": content,
            "usage": usage,
        }
    except httpx.RequestError as e:
        raise BackendError(f"DeepSeek headroom request failed: {e}")
    except (KeyError, IndexError, ValueError) as e:
        raise BackendError(f"DeepSeek headroom response parse error: {e}")


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
        "messages": [{"role": "user", "content": prompt}],
    }

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(url, json=payload, headers=headers)

            if response.status_code != 200:
                raise _classify_http_error(
                    "GLM", response.status_code, response.text
                )

            data = response.json()

        content = data["content"][0]["text"]
        usage = data.get("usage", {})

        return {
            "content": content,
            "usage": usage,
        }
    except httpx.RequestError as e:
        raise BackendError(f"GLM request failed: {e}")
    except (KeyError, IndexError, ValueError) as e:
        raise BackendError(f"GLM response parse error: {e}")


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
        "messages": [{"role": "user", "content": prompt}],
    }

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(url, json=payload, headers=headers)

            if response.status_code != 200:
                raise _classify_http_error(
                    "GLM-headroom", response.status_code, response.text
                )

            data = response.json()

        content = data["content"][0]["text"]
        usage = data.get("usage", {})

        return {
            "content": content,
            "usage": usage,
        }
    except httpx.RequestError as e:
        raise BackendError(f"GLM headroom request failed: {e}")
    except (KeyError, IndexError, ValueError) as e:
        raise BackendError(f"GLM headroom response parse error: {e}")


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
            f"Invalid model name. Must contain only alphanumeric, dot, underscore, or hyphen."
        )


def call_codex(
    prompt: str,
    model: str,
) -> Dict[str, Any]:
    """
    Call Codex CLI via subprocess: codex exec -m <model> <prompt>.

    Args:
        model: Codex model string (e.g., "gpt-5.5")
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

    try:
        result = subprocess.run(
            # mcp_servers={} — codex otherwise boots every MCP server in
            # ~/.codex/config.toml (12+, incl. auth-blocked ones) and blows
            # the timeout before answering. Delegated prompts never need MCP.
            [
                "codex", "exec",
                "--skip-git-repo-check",
                "-c", "mcp_servers={}",
                "-m", model,
                prompt,
            ],
            capture_output=True,
            text=True,
            timeout=90,
            check=False,
        )

        if result.returncode != 0:
            raise BackendError("Codex subprocess failed")

        return {
            "content": result.stdout.strip(),
            "usage": None,
        }

    except subprocess.TimeoutExpired:
        # Transient (slow model / cold start), not config — router may fall
        # through the chain or return exhausted=True instead of a raw error.
        raise BackendTransientError("Codex subprocess timed out (90s)")
    except FileNotFoundError:
        raise BackendError("Codex binary not found on PATH")
