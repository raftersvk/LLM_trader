"""Tests for the JEV System One client (no network: fake transport only)."""

from __future__ import annotations

import asyncio
import logging
import random
from typing import Any, Dict, List, Optional, Tuple

import pytest

from src.signal_service.jev_client import (
    DEFAULT_ENDPOINT,
    DEFAULT_MODEL,
    JevClient,
    JevError,
    JevRateLimitError,
    JevTimeoutError,
)

QUESTIONS: Dict[str, Dict[str, Any]] = {
    "tradeability": {"type": "noul", "instructions": "Is it tradeable?"},
}
STATE: Dict[str, Any] = {"symbol": "BTC/USDT", "last_close": 68000.0}


def _body(model: str = "jev-1.13", **answers: Dict[str, Any]) -> bytes:
    import json

    payload = {
        "model": model,
        "answers": answers,
        "usage": {"input_tokens": 42, "output_tokens": 4},
    }
    return json.dumps(payload).encode("utf-8")


class FakeTransport:
    """Deterministic transport returning queued responses or raising errors."""

    def __init__(self, responses: List[Any]) -> None:
        self.responses = list(responses)
        self.calls: List[Dict[str, Any]] = []

    async def post(
        self,
        url: str,
        payload: Dict[str, Any],
        timeout_seconds: float,
    ) -> Tuple[int, Dict[str, str], bytes]:
        self.calls.append({"url": url, "payload": payload, "timeout": timeout_seconds})
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
async def test_payload_and_parsing_noul_choice_score():
    """The client sends state+questions and exposes every typed answer."""
    transport = FakeTransport([
        (
            200,
            {},
            _body(
                model="jev-1.13.0",
                tradeability={"type": "noul", "noul": 0.82},
                bias={"type": "choice", "choice": "long", "confidence": 0.7},
                confidence={"type": "score", "score": 3, "legend": {"0": "low", "4": "high"}},
            ),
        )
    ])
    client = _client(transport)

    response = await client.ask(STATE, QUESTIONS)

    assert transport.calls[0]["url"] == DEFAULT_ENDPOINT
    assert transport.calls[0]["payload"] == {
        "model": DEFAULT_MODEL,
        "state": STATE,
        "questions": QUESTIONS,
    }
    assert response.model == "jev-1.13.0"
    assert response.requested_model == DEFAULT_MODEL
    assert response.answer("tradeability")["noul"] == 0.82
    assert response.answer("bias")["choice"] == "long"
    assert response.answer("confidence")["score"] == 3
    assert response.input_tokens == 42
    assert response.output_tokens == 4


@pytest.mark.asyncio
async def test_model_actually_used_is_logged_without_payload(caplog):
    """Only response.model is logged; state and credentials never are."""
    transport = FakeTransport([(200, {}, _body(model="jev-1.13.5"))])
    client = JevClient(transport=transport, jitter_ratio=0.0, rng=random.Random(0))
    secret_state = {"symbol": "BTC/USDT", "note": "do-not-log-this-state"}

    with caplog.at_level(logging.INFO, logger="src.signal_service.jev_client"):
        response = await client.ask(secret_state, QUESTIONS)

    assert response.model == "jev-1.13.5"
    logs = "\n".join(record.getMessage() for record in caplog.records)
    assert "model=jev-1.13.5" in logs
    assert "do-not-log-this-state" not in logs
    assert "authorization" not in logs.lower()
    assert "api_key" not in logs.lower()
    # No auth header is ever part of the payload.
    assert "headers" not in transport.calls[0]["payload"]


@pytest.mark.asyncio
async def test_rate_limit_honors_retry_after_then_succeeds():
    delays: List[float] = []
    transport = FakeTransport([
        (429, {"retry-after": "1"}, b'{"error":{"type":"RateLimitError"}}'),
        (200, {}, _body()),
    ])
    client = _client(transport, delays)

    response = await client.ask(STATE, QUESTIONS)

    assert response.model == "jev-1.13"
    assert len(transport.calls) == 2
    assert delays == [1.0]


@pytest.mark.asyncio
async def test_free_usage_limit_surfaces_exact_error():
    transport = FakeTransport([
        (
            429,
            {},
            b'{"type":"error","error":{"type":"FreeUsageLimitError",'
            b'"message":"Free usage limit reached."}}',
        ),
        (429, {}, b'{"error":{"type":"FreeUsageLimitError","message":"still limited"}}'),
    ])
    client = _client(transport, max_retries=1)

    with pytest.raises(JevRateLimitError) as excinfo:
        await client.ask(STATE, QUESTIONS)

    assert excinfo.value.status == 429
    assert "FreeUsageLimitError" in str(excinfo.value)
    assert len(transport.calls) == 2


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
        async def post(self, url, payload, timeout_seconds):
            self.calls.append({"url": url, "payload": payload, "timeout": timeout_seconds})
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
        (503, {}, b"upstream unavailable"),
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

    client = _client(FakeTransport([(200, {}, b'{"model":"jev-1.13"}')]))
    with pytest.raises(JevError, match="answers"):
        await client.ask(STATE, QUESTIONS)
