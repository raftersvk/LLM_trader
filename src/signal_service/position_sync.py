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
    contract_symbol,
    utc_now_iso,
    write_positions_atomic,
)

logger = logging.getLogger(__name__)

DEFAULT_API_URL = "http://127.0.0.1:8080"
TOKEN_TIMEOUT = 10.0
VALID_MARKETS = frozenset({"spot", "futures"})


class FreqtradeReadOnlyClient:
    """Minimal authenticated read-only client for Freqtrade's REST API."""

    def __init__(
        self,
        base_url: str,
        username: str,
        password: str,
        client: Optional[httpx.AsyncClient] = None,
        market: str = "",
    ) -> None:
        """Store credentials and an optional injected client (for tests)."""
        self.base_url = base_url.rstrip("/")
        self._username = username
        self._password = password
        self.market = market
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


def snapshot_from_trades(
    trades: List[Dict[str, Any]], market: Optional[str] = None
) -> PositionsSnapshot:
    """Convert Freqtrade open-trade dicts into a validated snapshot.

    ``BTC/USDT:USDT`` is normalized to the contract symbol ``BTC/USDT`` so the
    producer's per-symbol lookup works for both spot and futures. ``market``
    records which API produced the position (informational, never a secret).
    """
    positions: List[PositionSnapshot] = []
    market_label = (market or "").strip().lower()
    if market_label and market_label not in VALID_MARKETS:
        raise ValueError(f"invalid market label: {market!r} (expected spot|futures)")
    for trade in trades:
        if not isinstance(trade, dict):
            continue
        trade_id = str(trade.get("trade_id", "") or "").strip()
        if not trade_id:
            logger.warning("Skipping Freqtrade trade without a trade_id")
            continue
        try:
            # Market-scoped id: spot and futures keep separate trade-id
            # sequences, so a bare id could collide across markets. Records
            # written before this change keep the bare id (legacy fallback).
            position_id = f"{market_label}:{trade_id}" if market_label else trade_id
            snapshot = PositionSnapshot(
                position_id=position_id,
                symbol=contract_symbol(str(trade.get("pair", ""))),
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
                market=market_label or None,
            )
            snapshot.validate()
        except (TypeError, ValueError) as exc:
            logger.warning("Skipping malformed Freqtrade trade: %s", exc)
            continue
        positions.append(snapshot)
    return PositionsSnapshot(updated_at=utc_now_iso(), positions=positions)


async def sync_once(client: FreqtradeReadOnlyClient, positions_file: str) -> int:
    """Poll one API once and write the snapshot."""
    return await sync_once_multi([client], positions_file)


async def sync_once_multi(clients: List[FreqtradeReadOnlyClient], positions_file: str) -> int:
    """Poll every API once, merge, and write a single atomic snapshot.

    Any API failure aborts the write so the previous snapshot is kept: dropping
    one market's positions would make the producer think it is flat there.
    """
    merged: List[PositionSnapshot] = []
    for client in clients:
        trades = await client.open_trades()
        merged.extend(snapshot_from_trades(trades, market=client.market or None).positions)
    snapshot = PositionsSnapshot(updated_at=utc_now_iso(), positions=merged)
    write_positions_atomic(positions_file, snapshot)
    logger.info(
        "positions.json updated: %d open position(s) from %d API(s)",
        len(snapshot.positions), len(clients),
    )
    return len(snapshot.positions)


def _optional_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def build_client_from_config(
    config_path: str, market: Optional[str] = None
) -> FreqtradeReadOnlyClient:
    """Build a read-only client from a Freqtrade config JSON.

    Credentials stay in the existing config files; they are never logged.
    ``market`` defaults to the config's ``trading_mode``.
    """
    import json

    with open(config_path, "r", encoding="utf-8") as handle:
        config = json.load(handle)
    api = config.get("api_server", {})
    port = int(api.get("listen_port", 8080))
    username = str(api.get("username", ""))
    password = str(api.get("password", ""))
    if not username or not password:
        raise RuntimeError(f"api_server credentials missing in {config_path}")
    raw_label = market or str(config.get("trading_mode", ""))
    label = raw_label.strip().lower()
    if label not in VALID_MARKETS:
        raise RuntimeError(
            f"config {config_path} has an invalid trading_mode/market {raw_label!r} "
            "(expected spot|futures)"
        )
    return FreqtradeReadOnlyClient(f"http://127.0.0.1:{port}", username, password, market=label)


def build_clients_from_configs(config_paths: List[str]) -> List[FreqtradeReadOnlyClient]:
    """Build one client per Freqtrade config (spot + futures).

    Two configs with the same market label are rejected: they would produce
    duplicate position ids and cross-apply updates.
    """
    clients = [build_client_from_config(path) for path in config_paths]
    labels = [client.market for client in clients]
    if len(labels) != len(set(labels)):
        raise RuntimeError(f"duplicate market labels in configs: {labels}")
    return clients


def build_client_from_env() -> FreqtradeReadOnlyClient:
    """Build the client from FREQTRADE_API_* environment variables."""
    url = os.environ.get("FREQTRADE_API_URL", DEFAULT_API_URL)
    user = os.environ.get("FREQTRADE_API_USER", "")
    password = os.environ.get("FREQTRADE_API_PASSWORD", "")
    if not user or not password:
        raise RuntimeError(
            "FREQTRADE_API_USER and FREQTRADE_API_PASSWORD must be set in the environment"
        )
    market = os.environ.get("FREQTRADE_API_MARKET", "spot").strip().lower()
    if market not in VALID_MARKETS:
        raise RuntimeError(
            f"FREQTRADE_API_MARKET must be spot|futures, got {market!r}"
        )
    return FreqtradeReadOnlyClient(url, user, password, market=market)


async def run_forever(
    clients: List[FreqtradeReadOnlyClient], positions_file: str, interval: float
) -> None:
    """Poll every API every ``interval`` seconds and rewrite the snapshot."""
    entered: List[FreqtradeReadOnlyClient] = []
    try:
        for client in clients:
            entered.append(await client.__aenter__())
        while True:
            try:
                await sync_once_multi(entered, positions_file)
            except Exception as exc:  # pylint: disable=broad-exception-caught
                # A sync failure must never crash the producer path: the service
                # keeps the previous snapshot and retries on the next tick.
                logger.warning("position_sync failed: %s", exc)
            await asyncio.sleep(max(interval, 1.0))
    finally:
        for client in entered:
            await client.__aexit__(None, None, None)


def main(argv: Optional[List[str]] = None) -> int:
    """CLI entry point for ``python -m src.signal_service.position_sync``."""
    parser = argparse.ArgumentParser(description="Sync Freqtrade open positions to positions.json")
    parser.add_argument("--output", required=True, help="Path to runtime/positions.json")
    parser.add_argument("--interval", type=float, default=15.0, help="Poll interval in seconds")
    parser.add_argument(
        "--config", action="append", default=[],
        help="Freqtrade config JSON (repeatable: spot + futures). Credentials are read "
             "from api_server; without --config, FREQTRADE_API_* env vars are used.",
    )
    parser.add_argument("--once", action="store_true", help="Sync once and exit")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    clients = (
        build_clients_from_configs(args.config) if args.config else [build_client_from_env()]
    )
    if args.once:
        async def _once() -> None:
            entered = []
            try:
                for client in clients:
                    entered.append(await client.__aenter__())
                await sync_once_multi(entered, args.output)
            finally:
                for client in entered:
                    await client.__aexit__(None, None, None)
        asyncio.run(_once())
        return 0
    asyncio.run(run_forever(clients, args.output, args.interval))
    return 0


if __name__ == "__main__":
    sys.exit(main())
