"""Jev HTTP adapter — bounded, single-purpose client for the TypeSafe API's
"systemone" evaluate endpoint (POST https://api.typesafe.ai/v1/systemone).

Original adapter written against the published contract at
https://docs.typesafe.ai/api.md — NOT copied from a vendor SDK. Contract:
Bearer auth, JSON body {state, questions, model}, JSON response
{model, answers, usage}. Three question/answer shapes: noul (0..1 float),
choice (pick one of >=2 named options, with a probability map), score (a
probability-weighted value across >=2 levels, with an index->description legend).

Deliberately bounded: no retries (callers see failure_kind + retry_after and
decide), a fixed HTTPS endpoint, a 20s timeout, and strict validation of both
the outgoing request and the incoming response before any value is trusted.
Never logs or surfaces the API key, request state/questions, or a raw
response body/exception string.
"""

from __future__ import annotations

import ipaddress
import json
import math
import re
from typing import Any, Dict, List, Optional
from urllib.parse import urlsplit

import httpx

JEV_ENDPOINT = "https://api.typesafe.ai/v1/systemone"
JEV_DEFAULT_MODEL = "jev-1.13.0"
JEV_TIMEOUT_SECONDS = 20.0

# DECIDE-SWITCH (2026-09-28): the same systemone contract is also served by a
# self-hosted laya-serve. "laya" relaxes only the response model-name check.
PROVIDERS = frozenset({"jev", "laya"})
_TAILNET = ipaddress.ip_network("100.64.0.0/10")

_QUESTION_TYPES = frozenset({"noul", "choice", "score"})


class JevError(Exception):
    """Raised for any Jev request/response failure.

    failure_kind is one of: validation_error, authentication_error,
    rate_limited, unavailable, network_error, timeout, response_error,
    backend_error. Never carries the request body, response body, API key,
    or a raw exception string in its message.
    """

    def __init__(
        self,
        message: str,
        *,
        failure_kind: str,
        status_code: Optional[int] = None,
        retry_after: Optional[float] = None,
    ):
        self.failure_kind = failure_kind
        self.status_code = status_code
        self.retry_after = retry_after
        super().__init__(message)


def _fail(message: str, failure_kind: str, **kw) -> None:
    raise JevError(message, failure_kind=failure_kind, **kw)


def _is_finite_number(value: Any) -> bool:
    try:
        return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
    except OverflowError:
        return False


def _validate_state(state: Any) -> None:
    if not isinstance(state, (str, dict, list)):
        _fail("state must be a string, object, or array", "validation_error")


def _validate_questions(questions: Any) -> Dict[str, Dict[str, Any]]:
    if not isinstance(questions, dict) or not questions:
        _fail("questions must be a non-empty mapping of id -> question", "validation_error")
    for qid, q in questions.items():
        if not isinstance(qid, str) or not qid:
            _fail("question id must be a non-empty string", "validation_error")
        if not isinstance(q, dict):
            _fail("question must be an object", "validation_error")
        qtype = q.get("type")
        if not isinstance(qtype, str) or qtype not in _QUESTION_TYPES:
            _fail("question has invalid type", "validation_error")
        if "instructions" not in q or not isinstance(q["instructions"], (str, dict, list)):
            _fail("question instructions must be a string, object, or array", "validation_error")
        criteria = q.get("criteria")
        if qtype == "noul" and criteria is not None:
            if not isinstance(criteria, dict) or not set(criteria) <= {"true", "false"}:
                _fail("noul criteria must contain only true/false descriptions", "validation_error")
            if not all(isinstance(v, str) for v in criteria.values()):
                _fail("noul criteria descriptions must be strings", "validation_error")
        if qtype == "choice":
            options = q.get("criteria")
            if not isinstance(options, dict) or len(options) < 2:
                _fail("question choice must be a map of >=2 options", "validation_error")
            for key, desc in options.items():
                if not isinstance(key, str) or not key:
                    _fail("question choice keys must be non-empty strings", "validation_error")
                if desc is not None and not isinstance(desc, str):
                    _fail("question choice descriptions must be a string or null", "validation_error")
        elif qtype == "score":
            legend = q.get("criteria")
            if not isinstance(legend, list) or len(legend) < 2:
                _fail("question score must be a list of >=2 strings", "validation_error")
            if not all(isinstance(v, str) for v in legend):
                _fail("question score entries must be strings", "validation_error")
    return questions


def _validate_credential(api_key: Any) -> str:
    if not isinstance(api_key, str) or not api_key.strip() or any(c.isspace() for c in api_key):
        _fail("api_key is required", "validation_error")
    return api_key


def _validate_model(model: Any) -> str:
    if not isinstance(model, str) or not re.fullmatch(r"jev-[a-zA-Z0-9.-]+", model):
        _fail("model must be a non-empty string", "validation_error")
    return model


def _validate_probabilities(probs: Any, keys: List[str], where: str) -> None:
    if not isinstance(probs, dict) or set(probs) != set(keys):
        _fail(f"{where} probabilities must map every option to a probability", "response_error")
    for v in probs.values():
        if not _is_finite_number(v) or not (0 <= v <= 1):
            _fail(f"{where} probabilities must be finite numbers in 0..1", "response_error")

    if not math.isclose(sum(probs.values()), 1.0, abs_tol=0.01):
        _fail("probabilities must sum to one", "response_error")


def _validate_confidence(conf: Any, where: str) -> None:
    if not _is_finite_number(conf) or not (0 <= conf <= 1):
        _fail(f"{where} confidence must be a finite number in 0..1", "response_error")


def _validate_answer(qid: str, question: Dict[str, Any], answer: Any) -> None:
    if not isinstance(answer, dict):
        _fail("answer must be an object", "response_error")
    qtype = question["type"]
    if answer.get("type") != qtype:
        _fail("answer type does not match question type", "response_error")

    if qtype == "noul":
        noul = answer.get("noul")
        if not _is_finite_number(noul) or not (0 <= noul <= 1):
            _fail("answer noul must be a finite number in 0..1", "response_error")

    elif qtype == "choice":
        options = list(question["criteria"].keys())
        choice = answer.get("choice")
        if choice not in options:
            _fail("answer choice is not one of the question's options", "response_error")
        _validate_probabilities(answer.get("probabilities"), options, "answer")
        _validate_confidence(answer.get("confidence"), "answer")

    elif qtype == "score":
        legend_list = question["criteria"]
        n = len(legend_list)
        score = answer.get("score")
        if not _is_finite_number(score) or not (0 <= score <= n - 1):
            _fail(f"answer score must be a number in 0..{n - 1}", "response_error")
        legend = answer.get("legend")
        expected_keys = [str(i) for i in range(n)]
        if not isinstance(legend, dict) or set(legend) != set(expected_keys):
            _fail(f"answer legend must map every index 0..{n - 1} to a description", "response_error")
        if not all(isinstance(v, str) for v in legend.values()):
            _fail("answer legend descriptions must be strings", "response_error")
        _validate_probabilities(answer.get("probabilities"), expected_keys, "answer")
        _validate_confidence(answer.get("confidence"), "answer")


def _validate_usage(usage: Any) -> None:
    if not isinstance(usage, dict):
        _fail("usage must be an object", "response_error")
    for field in ("input_tokens", "output_tokens"):
        value = usage.get(field)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            _fail(f"usage.{field} must be a nonnegative integer", "response_error")


def validate_endpoint(endpoint: Any) -> str:
    """https to any host; plain http only to loopback or a Tailscale 100.64.0.0/10 address."""
    parts = urlsplit(endpoint) if isinstance(endpoint, str) else None
    host = parts.hostname if parts else None
    if parts and host and parts.scheme == "https":
        return endpoint
    if parts and host and parts.scheme == "http":
        if host == "localhost":
            return endpoint
        try:
            ip = ipaddress.ip_address(host)
        except ValueError:
            ip = None
        if ip is not None and (ip.is_loopback or ip in _TAILNET):
            return endpoint
    _fail("endpoint must be https, or http to loopback or a tailnet address", "validation_error")


def _validate_response(data: Any, questions: Dict[str, Dict[str, Any]], provider: str = "jev") -> Dict[str, Any]:
    if not isinstance(data, dict):
        _fail("response must be an object", "response_error")
    model = data.get("model")
    if not isinstance(model, str) or not model.strip() or (
            provider == "jev" and not re.fullmatch(r"jev-[a-zA-Z0-9.-]+", model)):
        _fail("response model must be a non-empty string", "response_error")
    answers = data.get("answers")
    if not isinstance(answers, dict) or set(answers) != set(questions):
        _fail("response answers must match the requested question ids", "response_error")
    for qid, answer in answers.items():
        _validate_answer(qid, questions[qid], answer)
    _validate_usage(data.get("usage"))
    return data


def _classify_http_error(status_code: int, retry_after: Optional[float]) -> JevError:
    if status_code == 429:
        return JevError("Jev rate limit exceeded", failure_kind="rate_limited",
                         status_code=status_code, retry_after=retry_after)
    if status_code in (401, 403):
        return JevError("Jev authentication failed", failure_kind="authentication_error",
                         status_code=status_code)
    if status_code == 529 or status_code >= 500:
        return JevError("Jev backend unavailable", failure_kind="unavailable",
                         status_code=status_code, retry_after=retry_after)
    return JevError("Jev request rejected", failure_kind="validation_error", status_code=status_code)


def _parse_retry_after(response: httpx.Response) -> Optional[float]:
    raw = response.headers.get("retry-after")
    if raw is None:
        return None
    try:
        value = float(raw)
        return value if math.isfinite(value) and value >= 0 else None
    except (TypeError, ValueError):
        return None


async def evaluate(
    state: Any,
    questions: Dict[str, Dict[str, Any]],
    api_key: str,
    model: str = JEV_DEFAULT_MODEL,
    *,
    endpoint: str = JEV_ENDPOINT,
    provider: str = "jev",
) -> Dict[str, Any]:
    """Evaluate `state` against `questions` via the Jev systemone endpoint.

    Returns the validated response dict: {model, answers, usage}. Raises
    JevError (never a raw httpx/exception) on any validation, auth, rate
    limit, backend, network, or timeout failure. No implicit retries — a
    rate_limited/unavailable error may carry retry_after for the caller to
    respect.
    """
    _validate_state(state)
    _validate_questions(questions)
    _validate_credential(api_key)
    model = _validate_model(model)
    if provider not in PROVIDERS:
        _fail("provider must be jev or laya", "validation_error")
    endpoint = validate_endpoint(endpoint)

    payload = {"state": state, "questions": questions, "model": model}
    try:
        json.dumps(payload, allow_nan=False)
    except (TypeError, ValueError):
        _fail("request must contain only finite JSON values", "validation_error")
    headers = {"Authorization": f"Bearer {api_key}", "content-type": "application/json"}

    try:
        async with httpx.AsyncClient(timeout=JEV_TIMEOUT_SECONDS) as client:
            response = await client.post(endpoint, json=payload, headers=headers)
    except httpx.TimeoutException:
        _fail("Jev request timed out", "timeout")
    except httpx.RequestError:
        _fail("Jev request failed", "network_error")

    if response.status_code != 200:
        raise _classify_http_error(response.status_code, _parse_retry_after(response))

    try:
        data = response.json()
    except ValueError:
        _fail("Jev response was not valid JSON", "response_error")

    return _validate_response(data, questions, provider)
