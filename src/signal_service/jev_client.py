"""Async client for the JEV System One decision endpoint.

JEV returns typed answers (noul / choice / score) for one shared state. The
OpenCode-hosted endpoint serves the free ``jev-1.13-free`` model without any
authentication, so this client never sends credentials.

Only the model actually used by the server is logged, together with latency
and status. Payloads, prompts, headers and keys are never logged.
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, Tuple

import aiohttp

logger = logging.getLogger(__name__)

DEFAULT_ENDPOINT = "https://opencode.ai/inference/systemone/v1/systemone"
DEFAULT_MODEL = "jev-1.13-free"
DEFAULT_TIMEOUT_SECONDS = 8.0
DEFAULT_MAX_RETRIES = 2
DEFAULT_BACKOFF_BASE = 0.5
DEFAULT_MAX_BACKOFF = 4.0

RETRYABLE_STATUS = frozenset({408, 425, 429, 500, 502, 503, 504})


class JevError(RuntimeError):
    """Base error for a failed JEV request."""

    def __init__(
        self,
        message: str,
        status: Optional[int] = None,
        retryable: bool = False,
    ) -> None:
        """Store the HTTP status and whether a retry could help."""
        super().__init__(message)
        self.status = status
        self.retryable = retryable


class JevRateLimitError(JevError):
    """JEV refused the request because of a rate or free-usage limit."""

    def __init__(self, message: str, status: int = 429, retry_after: Optional[float] = None) -> None:
        """Keep the parsed Retry-After value for the caller."""
        super().__init__(message, status=status, retryable=True)
        self.retry_after = retry_after


class JevTimeoutError(JevError):
    """The JEV request exceeded the configured timeout."""


class JevTransport(Protocol):
    """Minimal HTTP transport contract, injectable for offline tests."""

    async def post(
        self,
        url: str,
        payload: Dict[str, Any],
        timeout_seconds: float,
    ) -> Tuple[int, Dict[str, str], bytes]:
        """POST a JSON payload and return (status, headers, body)."""


class AiohttpTransport:
    """Default aiohttp transport; no authentication headers are ever sent."""

    def __init__(self, session: Optional[aiohttp.ClientSession] = None) -> None:
        """Optionally reuse an existing aiohttp session."""
        self._session = session
        self._own_session = session is None

    async def post(
        self,
        url: str,
        payload: Dict[str, Any],
        timeout_seconds: float,
    ) -> Tuple[int, Dict[str, str], bytes]:
        """POST JSON and read the full response body."""
        session = self._session
        if session is None:
            session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=timeout_seconds))
            self._session = session
        response_headers: Dict[str, str] = {}
        async with session.post(
            url,
            json=payload,
            headers={"Content-Type": "application/json"},
        ) as response:
            response_headers = {key.lower(): value for key, value in response.headers.items()}
            body = await response.read()
            return response.status, response_headers, body

    async def close(self) -> None:
        """Close a session created by this transport."""
        if self._own_session and self._session is not None:
            await self._session.close()
            self._session = None


@dataclass
class JevResponse:
    """Parsed JEV response for one state."""

    model: str
    answers: Dict[str, Dict[str, Any]]
    latency_ms: int
    requested_model: str = DEFAULT_MODEL
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    credits_used: Optional[float] = None
    raw_model_field: Optional[str] = field(default=None)

    def answer(self, question_id: str) -> Dict[str, Any]:
        """Return one answer payload, or an empty dict when absent."""
        value = self.answers.get(question_id)
        return value if isinstance(value, dict) else {}


def _retry_after_seconds(headers: Dict[str, str]) -> Optional[float]:
    """Parse a Retry-After header expressed in seconds, if present."""
    raw = headers.get("retry-after")
    if raw is None:
        return None
    try:
        value = float(str(raw).strip())
    except ValueError:
        return None
    return value if value >= 0 else None


def _error_message(body: bytes) -> str:
    """Extract a compact error description without echoing the state."""
    text = body.decode("utf-8", errors="replace").strip()
    if not text:
        return "empty response body"
    try:
        import json

        payload = json.loads(text)
    except ValueError:
        return text[:300]
    if isinstance(payload, dict):
        error = payload.get("error")
        if isinstance(error, dict):
            parts: List[str] = []
            for key in ("type", "message", "code"):
                if error.get(key):
                    parts.append(str(error[key]))
            if parts:
                return ": ".join(parts)[:300]
        if isinstance(error, str):
            return error[:300]
        if payload.get("message"):
            return str(payload["message"])[:300]
    return text[:300]


class JevClient:
    """Bounded-retry async client for the JEV System One endpoint.

    Retries apply to timeouts, 429 responses, transient 5xx errors and network
    errors only. Other 4xx responses fail immediately: they signal a malformed
    request, which a retry cannot fix.
    """

    def __init__(
        self,
        endpoint: str = DEFAULT_ENDPOINT,
        model: str = DEFAULT_MODEL,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        max_retries: int = DEFAULT_MAX_RETRIES,
        backoff_base: float = DEFAULT_BACKOFF_BASE,
        max_backoff: float = DEFAULT_MAX_BACKOFF,
        transport: Optional[JevTransport] = None,
        sleep: Any = None,
        jitter_ratio: float = 0.25,
        rng: Optional[random.Random] = None,
    ) -> None:
        """Configure endpoint, model, timeout and retry policy.

        Args:
            endpoint: Full System One URL.
            model: Model id sent in the payload (``jev-1.13-free`` by default).
            timeout_seconds: Per-attempt wall-clock timeout.
            max_retries: Retries after the first attempt (bounded).
            backoff_base: First backoff delay in seconds.
            max_backoff: Backoff delay ceiling in seconds.
            transport: HTTP transport; defaults to aiohttp.
            sleep: Awaitable sleep function, injectable for tests.
            jitter_ratio: Relative jitter added to each backoff delay.
            rng: Random source for jitter, injectable for determinism.
        """
        if max_retries < 0:
            raise ValueError("max_retries must be >= 0")
        self.endpoint = endpoint
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.backoff_base = backoff_base
        self.max_backoff = max_backoff
        self.transport: JevTransport = transport or AiohttpTransport()
        self._sleep = sleep or asyncio.sleep
        self.jitter_ratio = jitter_ratio
        self._rng = rng or random.Random()
        self.last_response: Optional[JevResponse] = None

    # ------------------------------------------------------------------ public
    async def ask(
        self,
        state: Dict[str, Any],
        questions: Dict[str, Dict[str, Any]],
    ) -> JevResponse:
        """Send one state and its typed questions to JEV.

        Args:
            state: JSON-serializable state built from closed-candle data only.
            questions: Map of question id -> typed question object.

        Returns:
            The parsed JevResponse.

        Raises:
            JevRateLimitError: on an exhausted 429 / free-usage limit.
            JevTimeoutError: when every attempt timed out.
            JevError: on any other non-retryable or exhausted failure.
        """
        payload = {"model": self.model, "state": state, "questions": questions}
        attempt = 0
        last_timeout: Optional[BaseException] = None
        while attempt <= self.max_retries:
            attempt += 1
            started = time.monotonic()
            try:
                status, headers, body = await asyncio.wait_for(
                    self.transport.post(self.endpoint, payload, self.timeout_seconds),
                    timeout=self.timeout_seconds,
                )
            except asyncio.TimeoutError as exc:
                last_timeout = exc
                logger.warning(
                    "JEV attempt %d/%d timed out after %.1fs",
                    attempt, self.max_retries + 1, self.timeout_seconds,
                )
                if attempt > self.max_retries:
                    raise JevTimeoutError(
                        f"JEV request timed out after {self.max_retries + 1} attempt(s)"
                    ) from exc
                await self._sleep(self._backoff(attempt))
                continue
            except (aiohttp.ClientError, OSError) as exc:
                logger.warning("JEV attempt %d/%d network error: %s", attempt, self.max_retries + 1, exc)
                if attempt > self.max_retries:
                    raise JevError(
                        f"JEV network error after {self.max_retries + 1} attempt(s): {exc}",
                        retryable=True,
                    ) from exc
                await self._sleep(self._backoff(attempt))
                continue

            latency_ms = int((time.monotonic() - started) * 1000)
            if status == 200:
                return self._parse_response(body, latency_ms)

            message = _error_message(body)
            if status == 429:
                retry_after = _retry_after_seconds(headers)
                logger.warning(
                    "JEV attempt %d/%d rate limited (status=429, retry_after=%s)",
                    attempt, self.max_retries + 1, retry_after,
                )
                if attempt > self.max_retries:
                    raise JevRateLimitError(
                        f"JEV rate limited (429): {message}",
                        status=status,
                        retry_after=retry_after,
                    )
                await self._sleep(self._rate_limit_delay(attempt, retry_after))
                continue

            if status in RETRYABLE_STATUS:
                logger.warning(
                    "JEV attempt %d/%d failed with status=%d: %s",
                    attempt, self.max_retries + 1, status, message,
                )
                if attempt > self.max_retries:
                    raise JevError(
                        f"JEV request failed with status {status}: {message}",
                        status=status,
                        retryable=True,
                    )
                await self._sleep(self._backoff(attempt))
                continue

            raise JevError(
                f"JEV request rejected with status {status}: {message}",
                status=status,
                retryable=False,
            )

        raise JevError(
            "JEV request exhausted retries",
            retryable=True,
        ) from last_timeout

    async def close(self) -> None:
        """Release transport resources when supported."""
        close = getattr(self.transport, "close", None)
        if close is not None:
            await close()

    # ----------------------------------------------------------------- helpers
    def _backoff(self, attempt: int) -> float:
        """Exponential backoff with bounded jitter."""
        base = min(self.max_backoff, self.backoff_base * (2 ** (attempt - 1)))
        if self.jitter_ratio <= 0:
            return base
        return base + self._rng.uniform(0.0, base * self.jitter_ratio)

    def _rate_limit_delay(self, attempt: int, retry_after: Optional[float]) -> float:
        """Honor Retry-After when sane, otherwise use exponential backoff."""
        if retry_after is not None:
            return min(self.max_backoff, retry_after)
        return self._backoff(attempt)

    def _parse_response(self, body: bytes, latency_ms: int) -> JevResponse:
        """Parse and validate a 200 response; never logs the payload."""
        import json

        try:
            data = json.loads(body.decode("utf-8"))
        except (ValueError, UnicodeDecodeError) as exc:
            raise JevError("JEV returned invalid JSON", status=200) from exc
        if not isinstance(data, dict):
            raise JevError("JEV returned a non-object JSON body", status=200)
        answers = data.get("answers")
        if not isinstance(answers, dict):
            raise JevError("JEV response is missing the 'answers' object", status=200)

        raw_model = data.get("model")
        used_model = str(raw_model) if raw_model else self.model
        usage = data.get("usage") if isinstance(data.get("usage"), dict) else {}
        response = JevResponse(
            model=used_model,
            answers={key: value for key, value in answers.items() if isinstance(value, dict)},
            latency_ms=latency_ms,
            requested_model=self.model,
            input_tokens=_as_int(usage.get("input_tokens")),
            output_tokens=_as_int(usage.get("output_tokens")),
            credits_used=_as_float(data.get("credits_used")),
            raw_model_field=str(raw_model) if raw_model is not None else None,
        )
        self.last_response = response
        logger.info(
            "JEV response received: requested_model=%s model=%s latency_ms=%d answers=%d",
            self.model, response.model, response.latency_ms, len(response.answers),
        )
        return response


def _as_int(value: Any) -> Optional[int]:
    """Best-effort int coercion, returning None on non-numeric input."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _as_float(value: Any) -> Optional[float]:
    """Best-effort float coercion, returning None on non-numeric input."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
