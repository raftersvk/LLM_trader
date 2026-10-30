"""position_sync — poll Freqtrade's read-only REST API into positions.json.

Separate process from the signal service by design:

* it is the only component that talks to Freqtrade's REST API;
* it writes ``runtime/positions.json`` atomically;
* the producer reads that file and needs no Freqtrade credentials;
* it never places, changes or cancels an order.

Credentials come from the environment (never from CLI args or git)::

    FREQTRADE_API_URL=http://127.0.0.1:8080
    FREQTRADE_API_USER=...
    FREQTRADE_API_PASSWORD=...
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import logging
import os
import sys
from typing import Any, Dict, List, Optional

import httpx

from src.signal_service.positions import (
    PositionSnapshot,
    PositionsSnapshot,
    utc_now_iso,
    write_positions_atomic,
)

logger = logging.getLogger(__name__)

DEFAULT_API_URL = "http://127.0.0.1:8080"
TOKEN_TIMEOUT = 10.0


class FreqtradeReadOnlyClient:
    """Minimal authenticated read-only client for Freqtrade's REST API."""

    def __init__(
        self,
        base_url: str,
        username: str,
        password: str,
        client: Optional[httpx.AsyncClient] = None,
    ) -> None:
        """Store credentials and an optional injected client (for tests)."""
        self.base_url = base_url.rstrip("/")
        self._username = username
        self._password = password
        self._client = client
        self._owns_client = client is None
        self._token: Optional[str] = None

    async def __aenter__(self) -> "FreqtradeReadOnlyClient":
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=TOKEN_TIMEOUT)
        return self

    async def __aexit__(self, *_exc: Any) -> None:
        if self._owns_client and self._client is not None:
            await self._client.aclose()

    async def _ensure_token(self) -> str:
        if self._token:
            return self._token
        assert self._client is not None
        credential = base64.b64encode(
            f"{self._username}:{self._password}".encode("utf-8")
        ).decode("ascii")
        response = await self._client.post(
            f"{self.base_url}/api/v1/token/login",
            headers={"Authorization": f"Basic {credential}"},
        )
        response.raise_for_status()
        payload = response.json()
        token = payload.get("access_token")
        if not token:
            raise RuntimeError("Freqtrade token login returned no access_token")
        self._token = str(token)
        return self._token

    async def open_trades(self) -> List[Dict[str, Any]]:
        """Return the open trades (never logs prices, only a count)."""
        assert self._client is not None
        token = await self._ensure_token()
        response = await self._client.get(
            f"{self.base_url}/api/v1/status",
            headers={"Authorization": f"Bearer {token}"},
        )
        response.raise_for_status()
        data = response.json()
        if not isinstance(data, list):
            raise RuntimeError("Freqtrade status payload is not a list")
        return data


def snapshot_from_trades(trades: List[Dict[str, Any]]) -> PositionsSnapshot:
    """Convert Freqtrade open-trade dicts into a validated snapshot."""
    positions: List[PositionSnapshot] = []
    for trade in trades:
        if not isinstance(trade, dict):
            continue
        try:
            snapshot = PositionSnapshot(
                position_id=str(trade.get("trade_id", "")),
                symbol=str(trade.get("pair", "")),
                side="short" if trade.get("is_short") else "long",
                entry_price=float(trade.get("open_rate", 0.0) or 0.0),
                amount=float(trade.get("amount", 0.0) or 0.0),
                stake_amount=float(trade.get("stake_amount", 0.0) or 0.0),
                open_date=trade.get("open_date"),
                leverage=float(trade.get("leverage", 1.0) or 1.0),
                stop_loss=_optional_float(trade.get("stop_loss") or trade.get("stop_loss_abs")),
                take_profit=_optional_float(trade.get("take_profit")),
                unrealized_pnl_pct=_optional_float(trade.get("profit_pct")),
                current_price=_optional_float(
                    trade.get("current_rate") or trade.get("last") or trade.get("close_rate")
                ),
                enter_tag=trade.get("enter_tag"),
            )
            snapshot.validate()
        except (TypeError, ValueError) as exc:
            logger.warning("Skipping malformed Freqtrade trade: %s", exc)
            continue
        positions.append(snapshot)
    return PositionsSnapshot(updated_at=utc_now_iso(), positions=positions)


async def sync_once(client: FreqtradeReadOnlyClient, positions_file: str) -> int:
    """Poll once and write the snapshot; returns the number of positions."""
    trades = await client.open_trades()
    snapshot = snapshot_from_trades(trades)
    write_positions_atomic(positions_file, snapshot)
    logger.info("positions.json updated: %d open position(s)", len(snapshot.positions))
    return len(snapshot.positions)


def _optional_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def build_client_from_env() -> FreqtradeReadOnlyClient:
    """Build the client from FREQTRADE_API_* environment variables."""
    url = os.environ.get("FREQTRADE_API_URL", DEFAULT_API_URL)
    user = os.environ.get("FREQTRADE_API_USER", "")
    password = os.environ.get("FREQTRADE_API_PASSWORD", "")
    if not user or not password:
        raise RuntimeError(
            "FREQTRADE_API_USER and FREQTRADE_API_PASSWORD must be set in the environment"
        )
    return FreqtradeReadOnlyClient(url, user, password)


async def run_forever(positions_file: str, interval: float) -> None:
    """Poll Freqtrade every ``interval`` seconds and rewrite the snapshot."""
    async with build_client_from_env() as client:
        while True:
            try:
                await sync_once(client, positions_file)
            except Exception as exc:  # pylint: disable=broad-exception-caught
                # A sync failure must never crash the producer path: the service
                # keeps the previous snapshot and retries on the next tick.
                logger.warning("position_sync failed: %s", exc)
            await asyncio.sleep(max(interval, 1.0))


def main(argv: Optional[List[str]] = None) -> int:
    """CLI entry point for ``python -m src.signal_service.position_sync``."""
    parser = argparse.ArgumentParser(description="Sync Freqtrade open positions to positions.json")
    parser.add_argument("--output", required=True, help="Path to runtime/positions.json")
    parser.add_argument("--interval", type=float, default=15.0, help="Poll interval in seconds")
    parser.add_argument("--once", action="store_true", help="Sync once and exit")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.once:
        async def _once() -> None:
            async with build_client_from_env() as client:
                await sync_once(client, args.output)
        asyncio.run(_once())
        return 0
    asyncio.run(run_forever(args.output, args.interval))
    return 0


if __name__ == "__main__":
    sys.exit(main())
