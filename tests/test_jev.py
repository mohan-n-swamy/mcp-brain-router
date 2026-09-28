"""Tests for the bounded Jev HTTP adapter (src/mcp_brain_router/jev.py).

Mocks httpx.AsyncClient.post the same way test_effort_plumbing.py does —
no live network calls, no real credentials.
"""
from __future__ import annotations

import httpx
import pytest

from mcp_brain_router import jev

QUESTIONS_NOUL = {
    "q1": {"type": "noul", "instructions": "Is this true?"},
}
QUESTIONS_CHOICE = {
    "q1": {"type": "choice", "instructions": "Pick one", "criteria": {"a": "Option A", "b": "Option B"}},
}
QUESTIONS_SCORE = {
    "q1": {"type": "score", "instructions": "Rate it", "criteria": ["low", "medium", "high"]},
}


class _Post:
    def __init__(self, response: httpx.Response):
        self.response = response
        self.calls = []

    async def __call__(self, url, json=None, headers=None, **kw):
        self.calls.append({"url": url, "json": json, "headers": headers})
        return self.response


def _json_response(status_code: int, body: dict, headers: dict | None = None) -> httpx.Response:
    req = httpx.Request("POST", jev.JEV_ENDPOINT)
    return httpx.Response(status_code, request=req, json=body, headers=headers or {})


# ---------------------------------------------------------------- success, 3 types


@pytest.mark.asyncio
async def test_evaluate_noul_success(monkeypatch):
    body = {
        "model": "jev-1.13.0",
        "answers": {"q1": {"type": "noul", "noul": 0.7}},
        "usage": {"input_tokens": 5, "output_tokens": 2},
    }
    post = _Post(_json_response(200, body))
    monkeypatch.setattr(httpx.AsyncClient, "post", post)

    result = await jev.evaluate("some state", QUESTIONS_NOUL, "sk-test")

    assert result == body
    assert post.calls[0]["url"] == jev.JEV_ENDPOINT
    assert post.calls[0]["headers"]["Authorization"] == "Bearer sk-test"
    assert post.calls[0]["json"]["model"] == jev.JEV_DEFAULT_MODEL


@pytest.mark.asyncio
async def test_evaluate_choice_success(monkeypatch):
    body = {
        "model": "jev-1.13.0",
        "answers": {
            "q1": {
                "type": "choice",
                "choice": "a",
                "probabilities": {"a": 0.9, "b": 0.1},
                "confidence": 0.8,
            }
        },
        "usage": {"input_tokens": 5, "output_tokens": 2},
    }
    post = _Post(_json_response(200, body))
    monkeypatch.setattr(httpx.AsyncClient, "post", post)

    result = await jev.evaluate({"k": "v"}, QUESTIONS_CHOICE, "sk-test")

    assert result == body


@pytest.mark.asyncio
async def test_evaluate_score_success(monkeypatch):
    body = {
        "model": "jev-1.13.0",
        "answers": {
            "q1": {
                "type": "score",
                "score": 1.6,
                "legend": {"0": "low", "1": "medium", "2": "high"},
                "probabilities": {"0": 0.1, "1": 0.2, "2": 0.7},
                "confidence": 0.6,
            }
        },
        "usage": {"input_tokens": 5, "output_tokens": 2},
    }
    post = _Post(_json_response(200, body))
    monkeypatch.setattr(httpx.AsyncClient, "post", post)

    result = await jev.evaluate(["a", "b"], QUESTIONS_SCORE, "sk-test", model="jev-2.0.0")

    assert result == body
    assert post.calls[0]["json"]["model"] == "jev-2.0.0"


# ---------------------------------------------------------------- invalid request (rejected before call)


@pytest.mark.asyncio
async def test_evaluate_rejects_bad_state(monkeypatch):
    post = _Post(_json_response(200, {}))
    monkeypatch.setattr(httpx.AsyncClient, "post", post)

    with pytest.raises(jev.JevError) as exc:
        await jev.evaluate(123, QUESTIONS_NOUL, "sk-test")

    assert exc.value.failure_kind == "validation_error"
    assert not post.calls


@pytest.mark.asyncio
async def test_evaluate_rejects_bad_questions_missing_choice_map(monkeypatch):
    post = _Post(_json_response(200, {}))
    monkeypatch.setattr(httpx.AsyncClient, "post", post)

    bad_questions = {"q1": {"type": "choice", "instructions": "x", "criteria": {"only_one": "a"}}}
    with pytest.raises(jev.JevError) as exc:
        await jev.evaluate("s", bad_questions, "sk-test")

    assert exc.value.failure_kind == "validation_error"
    assert not post.calls


@pytest.mark.asyncio
async def test_evaluate_rejects_missing_credential(monkeypatch):
    post = _Post(_json_response(200, {}))
    monkeypatch.setattr(httpx.AsyncClient, "post", post)

    with pytest.raises(jev.JevError) as exc:
        await jev.evaluate("s", QUESTIONS_NOUL, "")

    assert exc.value.failure_kind == "validation_error"
    assert not post.calls


# ---------------------------------------------------------------- invalid response


@pytest.mark.asyncio
async def test_evaluate_rejects_malformed_response(monkeypatch):
    body = {
        "model": "jev-1.13.0",
        "answers": {"q1": {"type": "noul", "noul": 1.5}},  # out of 0..1 range
        "usage": {"input_tokens": 5, "output_tokens": 2},
    }
    post = _Post(_json_response(200, body))
    monkeypatch.setattr(httpx.AsyncClient, "post", post)

    with pytest.raises(jev.JevError) as exc:
        await jev.evaluate("s", QUESTIONS_NOUL, "sk-test")

    assert exc.value.failure_kind == "response_error"


@pytest.mark.asyncio
async def test_evaluate_rejects_negative_usage(monkeypatch):
    body = {
        "model": "jev-1.13.0",
        "answers": {"q1": {"type": "noul", "noul": 0.5}},
        "usage": {"input_tokens": -1, "output_tokens": 2},
    }
    post = _Post(_json_response(200, body))
    monkeypatch.setattr(httpx.AsyncClient, "post", post)

    with pytest.raises(jev.JevError) as exc:
        await jev.evaluate("s", QUESTIONS_NOUL, "sk-test")

    assert exc.value.failure_kind == "response_error"


# ---------------------------------------------------------------- HTTP status mapping


@pytest.mark.asyncio
async def test_evaluate_401_authentication_error(monkeypatch):
    post = _Post(_json_response(401, {"error": "bad key"}))
    monkeypatch.setattr(httpx.AsyncClient, "post", post)

    with pytest.raises(jev.JevError) as exc:
        await jev.evaluate("s", QUESTIONS_NOUL, "sk-bad")

    assert exc.value.failure_kind == "authentication_error"
    assert exc.value.status_code == 401
    assert "sk-bad" not in str(exc.value)


@pytest.mark.asyncio
async def test_evaluate_429_rate_limited(monkeypatch):
    post = _Post(_json_response(429, {"error": "slow down"}, headers={"retry-after": "12"}))
    monkeypatch.setattr(httpx.AsyncClient, "post", post)

    with pytest.raises(jev.JevError) as exc:
        await jev.evaluate("s", QUESTIONS_NOUL, "sk-test")

    assert exc.value.failure_kind == "rate_limited"
    assert exc.value.retry_after == 12.0


@pytest.mark.asyncio
async def test_evaluate_529_unavailable(monkeypatch):
    post = _Post(_json_response(529, {"error": "overloaded"}))
    monkeypatch.setattr(httpx.AsyncClient, "post", post)

    with pytest.raises(jev.JevError) as exc:
        await jev.evaluate("s", QUESTIONS_NOUL, "sk-test")

    assert exc.value.failure_kind == "unavailable"


# ---------------------------------------------------------------- timeout


@pytest.mark.asyncio
async def test_evaluate_timeout(monkeypatch):
    async def _raise_timeout(self, url, json=None, headers=None, **kw):
        raise httpx.ReadTimeout("timed out")

    monkeypatch.setattr(httpx.AsyncClient, "post", _raise_timeout)

    with pytest.raises(jev.JevError) as exc:
        await jev.evaluate("s", QUESTIONS_NOUL, "sk-test")

    assert exc.value.failure_kind == "timeout"


@pytest.mark.asyncio
async def test_unhashable_question_type_is_validation_error(monkeypatch):
    with pytest.raises(jev.JevError, match="invalid type"):
        await jev.evaluate("s", {"q": {"type": [], "instructions": "x"}}, "key")


@pytest.mark.asyncio
async def test_nonfinite_state_rejected_before_request():
    with pytest.raises(jev.JevError, match="finite JSON"):
        await jev.evaluate({"value": float("nan")}, QUESTIONS_NOUL, "key")


def test_invalid_answer_does_not_echo_question_id():
    private_id = "private-text-in-question-id"
    with pytest.raises(jev.JevError) as caught:
        jev._validate_answer(private_id, {"type":"choice","criteria":{"a":None,"b":None}},
                             {"type":"choice","choice":"a","probabilities":{},"confidence":1})
    assert private_id not in str(caught.value)
