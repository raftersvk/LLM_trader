"""Tests for the JEV System One client (no network: fake transport only)."""

from __future__ import annotations

import asyncio
import json
import logging
import random
from typing import Any, Dict, List, Optional, Tuple

import pytest

from src.signal_service.jev_client import (
    DEFAULT_ENDPOINT,
    DEFAULT_MODEL,
    OPENCODE_BASE_URL,
    OPENCODE_MODEL,
    JevAuthError,
    JevClient,
    JevError,
    JevRateLimitError,
    JevTimeoutError,
)

QUESTIONS: Dict[str, Dict[str, Any]] = {
    "tradeability": {"type": "noul", "instructions": "Is it tradeable?"},
}
STATE: Dict[str, Any] = {"symbol": "BTC/USDT", "last_close": 68000.0}


def _body(model: str = "vllm-sr/d3-flash", **answers: Dict[str, Any]) -> bytes:
    payload = {
        "model": model,
        "answers": answers,
        "usage": {"input_tokens": 42, "output_tokens": 0},
    }
    return json.dumps(payload).encode("utf-8")


def _decision_answers() -> Dict[str, Dict[str, Any]]:
    """A real Decision 3.0 answer set, including the 3-level score legend."""
    return {
        "tradeability": {"type": "noul", "noul": 0.87},
        "bias": {
            "type": "choice",
            "choice": "long",
            "probabilities": {"long": 0.9, "short": 0.05, "none": 0.05},
            "confidence": 0.9,
        },
        "regime": {
            "type": "choice",
            "choice": "trend",
            "probabilities": {"trend": 0.8, "range": 0.1, "high_vol": 0.05, "low_vol": 0.05},
            "confidence": 0.8,
        },
        "confidence": {
            "type": "score",
            "score": 1.69,
            "probabilities": {"0": 0.05, "1": 0.21, "2": 0.74},
            "confidence": 0.55,
            "legend": {"0": "low", "1": "medium", "2": "high"},
        },
    }


class FakeTransport:
    """Deterministic transport returning queued responses or raising errors."""

    def __init__(self, responses: List[Any]) -> None:
        self.responses = list(responses)
        self.calls: List[Dict[str, Any]] = []

    async def post(
        self,
        url: str,
        payload: Dict[str, Any],
        headers: Dict[str, str],
        timeout_seconds: float,
    ) -> Tuple[int, Dict[str, str], bytes]:
        self.calls.append(
            {"url": url, "payload": payload, "headers": headers, "timeout": timeout_seconds}
        )
        if not self.responses:
            raise AssertionError("FakeTransport received more calls than expected")
        item = self.responses.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item


def _client(transport: FakeTransport, delays: Optional[List[float]] = None, **kwargs: Any) -> JevClient:
    recorded: List[float] = delays if delays is not None else []

    async def fake_sleep(delay: float) -> None:
        recorded.append(delay)

    return JevClient(
        transport=transport,
        sleep=fake_sleep,
        jitter_ratio=0.0,
        rng=random.Random(0),
        **kwargs,
    )


@pytest.mark.asyncio
async def test_decision_payload_sends_state_as_json_text_and_parses_answers():
    """The client serializes state to JSON text and exposes every answer."""
    transport = FakeTransport([(200, {}, _body(**_decision_answers()))])
    client = _client(transport, api_key="sk-vsr-test-key")

    response = await client.ask(STATE, QUESTIONS)

    call = transport.calls[0]
    assert call["url"] == DEFAULT_ENDPOINT
    assert call["headers"]["Authorization"] == "Bearer sk-vsr-test-key"
    # Decision 3.0 requires a string state, not a JSON object.
    assert isinstance(call["payload"]["state"], str)
    assert json.loads(call["payload"]["state"]) == STATE
    assert call["payload"]["model"] == DEFAULT_MODEL
    assert call["payload"]["questions"] == QUESTIONS
    assert response.model == DEFAULT_MODEL
    assert response.answer("tradeability")["noul"] == 0.87
    assert response.answer("bias")["choice"] == "long"
    assert response.answer("confidence")["score"] == 1.69
    assert len(response.answer("confidence")["legend"]) == 3
    assert response.input_tokens == 42
    assert response.output_tokens == 0


@pytest.mark.asyncio
async def test_legacy_opencode_provider_needs_no_auth():
    """Overriding base_url/model still supports the old OpenCode endpoint."""
    transport = FakeTransport([(200, {}, _body(model=OPENCODE_MODEL))])
    client = _client(transport, base_url=OPENCODE_BASE_URL, model=OPENCODE_MODEL)

    response = await client.ask(STATE, QUESTIONS)

    assert transport.calls[0]["url"] == OPENCODE_BASE_URL + "/v1/systemone"
    assert transport.calls[0]["headers"] == {}
    assert response.model == OPENCODE_MODEL


@pytest.mark.asyncio
async def test_decision_key_is_not_forwarded_to_legacy_host():
    """The Decision token stays bound to decision.vllm-sr.ai, never OpenCode."""
    transport = FakeTransport([(200, {}, _body(model=OPENCODE_MODEL))])
    client = _client(
        transport,
        base_url=OPENCODE_BASE_URL,
        model=OPENCODE_MODEL,
        api_key="sk-vsr-secret-value",
    )

    await client.ask(STATE, QUESTIONS)

    assert transport.calls[0]["url"] == OPENCODE_BASE_URL + "/v1/systemone"
    assert transport.calls[0]["headers"] == {}
    assert client.api_key_host == "decision.vllm-sr.ai"


@pytest.mark.asyncio
async def test_api_key_and_state_are_never_logged(caplog):
    """Only model, latency and usage are logged; no key, header or state."""
    transport = FakeTransport([(200, {}, _body(**_decision_answers()))])
    client = JevClient(
        transport=transport,
        api_key="sk-vsr-secret-value",
        jitter_ratio=0.0,
        rng=random.Random(0),
    )
    secret_state = {"symbol": "BTC/USDT", "note": "do-not-log-this-state"}

    with caplog.at_level(logging.INFO, logger="src.signal_service.jev_client"):
        response = await client.ask(secret_state, QUESTIONS)

    assert response.model == DEFAULT_MODEL
    logs = "\n".join(record.getMessage() for record in caplog.records)
    assert "model=vllm-sr/d3-flash" in logs
    assert "input_tokens=42" in logs
    assert "do-not-log-this-state" not in logs
    assert "sk-vsr-secret-value" not in logs
    assert "authorization" not in logs.lower()
    assert "api_key" not in logs.lower()
    assert "headers" not in transport.calls[0]["payload"]


@pytest.mark.asyncio
async def test_rate_limit_honors_retry_after_then_succeeds():
    delays: List[float] = []
    transport = FakeTransport([
        (429, {"retry-after": "1"}, b'{"error":{"type":"rate_limit_exceeded"}}'),
        (200, {}, _body()),
    ])
    client = _client(transport, delays)

    response = await client.ask(STATE, QUESTIONS)

    assert response.model == DEFAULT_MODEL
    assert len(transport.calls) == 2
    assert delays == [1.0]


@pytest.mark.asyncio
async def test_quota_exceeded_surfaces_exact_error_and_retry_after():
    transport = FakeTransport([
        (
            429,
            {"retry-after": "30"},
            b'{"error":{"type":"quota_exceeded","code":"quota_exceeded",'
            b'"message":"daily quota reached"}}',
        ),
        (429, {"retry-after": "30"}, b'{"error":{"type":"quota_exceeded"}}'),
    ])
    client = _client(transport, max_retries=1)

    with pytest.raises(JevRateLimitError) as excinfo:
        await client.ask(STATE, QUESTIONS)

    assert excinfo.value.status == 429
    assert excinfo.value.retry_after == 30.0
    assert "quota_exceeded" in str(excinfo.value)
    assert len(transport.calls) == 2


@pytest.mark.asyncio
async def test_missing_api_key_is_not_retried():
    transport = FakeTransport([
        (401, {}, b'{"error":{"type":"missing_api_key","code":"missing_api_key"}}')
    ])
    client = _client(transport, max_retries=3)

    with pytest.raises(JevAuthError) as excinfo:
        await client.ask(STATE, QUESTIONS)

    assert excinfo.value.status == 401
    assert excinfo.value.retryable is False
    assert len(transport.calls) == 1


@pytest.mark.asyncio
async def test_timeout_retries_then_raises_timeout():
    transport = FakeTransport([asyncio.TimeoutError(), asyncio.TimeoutError()])
    client = _client(transport, max_retries=1)

    with pytest.raises(JevTimeoutError):
        await client.ask(STATE, QUESTIONS)

    assert len(transport.calls) == 2


@pytest.mark.asyncio
async def test_per_attempt_timeout_is_enforced():
    class SlowTransport(FakeTransport):
        async def post(self, url, payload, headers, timeout_seconds):
            self.calls.append({"url": url, "payload": payload, "headers": headers})
            await asyncio.sleep(1.0)
            return 200, {}, _body()

    transport = SlowTransport([])
    client = JevClient(
        transport=transport,
        timeout_seconds=0.05,
        max_retries=0,
        jitter_ratio=0.0,
        rng=random.Random(0),
    )

    with pytest.raises(JevTimeoutError):
        await client.ask(STATE, QUESTIONS)


@pytest.mark.asyncio
async def test_server_error_is_retried_with_backoff():
    delays: List[float] = []
    transport = FakeTransport([
        (503, {"retry-after": "2"}, b"upstream unavailable"),
        (200, {}, _body()),
    ])
    client = _client(transport, delays, backoff_base=0.25)

    await client.ask(STATE, QUESTIONS)

    assert len(transport.calls) == 2
    assert delays == [0.25]


@pytest.mark.asyncio
async def test_bad_request_is_not_retried():
    transport = FakeTransport([(400, {}, b'{"message":"invalid questions"}')])
    client = _client(transport, max_retries=3)

    with pytest.raises(JevError) as excinfo:
        await client.ask(STATE, QUESTIONS)

    assert excinfo.value.status == 400
    assert excinfo.value.retryable is False
    assert len(transport.calls) == 1


@pytest.mark.asyncio
async def test_malformed_json_and_missing_answers_raise():
    client = _client(FakeTransport([(200, {}, b"not-json")]))
    with pytest.raises(JevError, match="invalid JSON"):
        await client.ask(STATE, QUESTIONS)

    client = _client(FakeTransport([(200, {}, b'{"model":"vllm-sr/d3-flash"}')]))
    with pytest.raises(JevError, match="answers"):
        await client.ask(STATE, QUESTIONS)
