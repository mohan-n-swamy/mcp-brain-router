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
                raise BackendError(
                    f"DeepSeek API error {response.status_code}: "
                    f"(body hidden for security)"
                )

            data = response.json()

        # Extract content and usage
        content = data["content"][0]["text"]
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
                raise BackendError(
                    f"DeepSeek via headroom error {response.status_code}: "
                    f"(body hidden for security)"
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
                raise BackendError(
                    f"GLM API error {response.status_code}: "
                    f"(body hidden for security)"
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
                raise BackendError(
                    f"GLM via headroom error {response.status_code}: "
                    f"(body hidden for security)"
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
            ["codex", "exec", "-m", model, prompt],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )

        if result.returncode != 0:
            raise BackendError("Codex subprocess failed")

        return {
            "content": result.stdout.strip(),
            "usage": None,
        }

    except subprocess.TimeoutExpired:
        raise BackendError("Codex subprocess timed out (60s)")
    except FileNotFoundError:
        raise BackendError("Codex binary not found on PATH")
