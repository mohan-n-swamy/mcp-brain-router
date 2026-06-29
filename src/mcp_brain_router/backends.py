"""Backend implementations for DeepSeek, GLM, and Codex.

Each backend is isolated; router.py calls these functions.
"""

import asyncio
import subprocess
from typing import Any, Dict, Optional


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
    """
    import aiohttp

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

    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=payload, headers=headers) as resp:
            if resp.status != 200:
                text = await resp.text()
                raise RuntimeError(f"DeepSeek API error {resp.status}: {text}")
            data = await resp.json()

    # Extract content and usage
    content = data["content"][0]["text"]
    usage = data.get("usage", {})

    return {
        "content": content,
        "usage": usage,
    }


async def call_deepseek_via_headroom(
    prompt: str,
    model: str,
    api_key: str,
    headroom_url: str,
) -> Dict[str, Any]:
    """Call DeepSeek through headroom proxy."""
    import aiohttp

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

    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=payload, headers=headers) as resp:
            if resp.status != 200:
                text = await resp.text()
                raise RuntimeError(
                    f"DeepSeek via headroom error {resp.status}: {text}"
                )
            data = await resp.json()

    content = data["content"][0]["text"]
    usage = data.get("usage", {})

    return {
        "content": content,
        "usage": usage,
    }


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
    """
    import aiohttp

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

    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=payload, headers=headers) as resp:
            if resp.status != 200:
                text = await resp.text()
                raise RuntimeError(f"GLM API error {resp.status}: {text}")
            data = await resp.json()

    content = data["content"][0]["text"]
    usage = data.get("usage", {})

    return {
        "content": content,
        "usage": usage,
    }


async def call_glm_via_headroom(
    prompt: str,
    model: str,
    api_key: str,
    headroom_url: str,
) -> Dict[str, Any]:
    """Call GLM through headroom proxy."""
    import aiohttp

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

    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=payload, headers=headers) as resp:
            if resp.status != 200:
                text = await resp.text()
                raise RuntimeError(f"GLM via headroom error {resp.status}: {text}")
            data = await resp.json()

    content = data["content"][0]["text"]
    usage = data.get("usage", {})

    return {
        "content": content,
        "usage": usage,
    }


# ============================================================================
# Codex (subprocess-based, no HTTP)
# ============================================================================

def call_codex(
    prompt: str,
    model: str,
) -> Dict[str, Any]:
    """
    Call Codex via subprocess: codex exec -m gpt-5.5 "<prompt>".

    Returns:
        {
            "content": "response text",
            "usage": None
        }
    """
    try:
        result = subprocess.run(
            ["codex", "exec", "-m", model, prompt],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )

        if result.returncode != 0:
            raise RuntimeError(
                f"Codex subprocess failed: {result.stderr}"
            )

        return {
            "content": result.stdout.strip(),
            "usage": None,  # Codex CLI doesn't expose token counts
        }

    except FileNotFoundError:
        raise RuntimeError(
            "Codex binary not found on PATH. Is codex-cli installed?"
        )
