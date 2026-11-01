"""Token-expiry behaviour of the read-only Freqtrade position-sync client.

Freqtrade returns 401/403 when the cached bearer token expired. The client must
re-login exactly once and replay the status request, then stop retrying so a
persistent auth failure can never turn into a request loop.
"""

from __future__ import annotations

import asyncio

import httpx
import pytest

from src.signal_service.position_sync import FreqtradeReadOnlyClient

TRADE = {
    "trade_id": 42,
    "pair": "BTC/USDT",
    "is_short": False,
    "open_rate": 67500.0,
    "amount": 0.1,
}


def _login_ok(counter: list[int], request: httpx.Request) -> httpx.Response:
    """Issue a unique token per login so callers can tell a refresh from a cache hit."""
    counter[0] += 1
    return httpx.Response(200, json={"access_token": f"tok-{counter[0]}"})


def _run(handler) -> tuple[list[dict[str, str]], list[httpx.Request]]:
    """Drive ``open_trades`` through a mock transport and record the traffic."""
    status_requests: list[dict[str, str]] = []
    requests: list[httpx.Request] = []

    def wrapped(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path == "/api/v1/status":
            status_requests.append(dict(request.headers))
        return handler(request)

    async def run() -> None:
        transport = httpx.MockTransport(wrapped)
        async with httpx.AsyncClient(transport=transport) as raw:
            client = FreqtradeReadOnlyClient("http://api", "u", "p", client=raw)
            result = await client.open_trades()
            assert result == [TRADE]

    asyncio.run(run())
    return status_requests, requests


def _count_login(requests: list[httpx.Request]) -> int:
    return sum(1 for request in requests if request.url.path == "/api/v1/token/login")


@pytest.mark.parametrize("auth_status", [401, 403])
def test_expired_token_triggers_one_relogin_and_one_retry(auth_status: int) -> None:
    """First status call is rejected, the refreshed token gets a 200."""
    logins: list[int] = [0]
    status_calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal status_calls
        if request.url.path == "/api/v1/token/login":
            return _login_ok(logins, request)
        status_calls += 1
        if status_calls == 1:
            return httpx.Response(auth_status, json={"detail": "token expired"})
        return httpx.Response(200, json=[TRADE])

    status_requests, requests = _run(handler)

    assert _count_login(requests) == 2  # initial login + one re-login
    assert status_calls == 2
    assert status_requests[0]["authorization"] == "Bearer tok-1"
    assert status_requests[1]["authorization"] == "Bearer tok-2"


def test_valid_token_never_relogins() -> None:
    """A 200 on the first attempt must not trigger a second login."""
    logins: list[int] = [0]
    status_calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal status_calls
        if request.url.path == "/api/v1/token/login":
            return _login_ok(logins, request)
        status_calls += 1
        return httpx.Response(200, json=[TRADE])

    _status_requests, requests = _run(handler)

    assert _count_login(requests) == 1
    assert status_calls == 1


@pytest.mark.parametrize("auth_status", [401, 403])
def test_persistent_auth_failure_is_retried_only_once(auth_status: int) -> None:
    """A second auth failure propagates instead of looping forever."""
    logins: list[int] = [0]
    status_calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal status_calls
        if request.url.path == "/api/v1/token/login":
            return _login_ok(logins, request)
        status_calls += 1
        return httpx.Response(auth_status, json={"detail": "still expired"})

    with pytest.raises(httpx.HTTPStatusError):
        _run(handler)

    assert logins[0] == 2  # initial login + the single re-login
    assert status_calls == 2  # the original request + exactly one replay
