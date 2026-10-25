"""Fetch aligned hourly candles for a JEV calibration campaign.

Every symbol is fetched over the same calendar window: the script resolves one
last closed candle open time (or accepts ``--end``), then asks Binance spot for
that symbol's candles starting ``limit - 1`` hours earlier. Writing the CSVs
into the repository makes the campaign reproducible even after long API
outages.

    python scripts/fetch_jev_bench_candles.py \\
        --symbols BTC/USDT,SOL/USDT --limit 700 \\
        --output-dir runtime/jev_bench

USDT-M perpetual futures use ``--market futures``; spot symbols such as
``BTC/USDT`` are mapped to ccxt contracts such as ``BTC/USDT:USDT`` and the
CSV gets a ``_futures`` suffix:

    python scripts/fetch_jev_bench_candles.py --market futures \\
        --symbols BTC/USDT,ETH/USDT --limit 700 \\
        --output-dir runtime/jev_bench

The output format matches the calibration loader:
``time,open,high,low,close,volume`` with ISO-8601 UTC timestamps.
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timedelta, timezone
from typing import Iterable, List, Sequence

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import ccxt  # noqa: E402  (path setup must run first)

DEFAULT_SYMBOLS = (
    "BTC/USDT",
    "ETH/USDT",
    "SOL/USDT",
    "AXS/USDT",
    "AVAX/USDT",
    "NEXO/USDT",
    "ARB/USDT",
)
DEFAULT_LIMIT = 700
DEFAULT_TIMEFRAME = "1h"


def _parse_end(value: str) -> datetime:
    """Parse an ISO-8601 ``--end`` value into an aware UTC datetime."""
    text = value.strip().replace("Z", "+00:00")
    parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _last_closed_open(now: datetime, timeframe_seconds: int) -> datetime:
    """Return the open time of the last fully closed candle."""
    epoch = int(now.timestamp())
    current_bucket = epoch - (epoch % timeframe_seconds)
    return datetime.fromtimestamp(current_bucket - timeframe_seconds, tz=timezone.utc)


def _base_name(symbol: str) -> str:
    """Return ``SOL_USDT`` for ``SOL/USDT`` and ``SOL/USDT:USDT``."""
    return symbol.split(":")[0].replace("/", "_")


def _market_symbol(symbol: str, market: str) -> str:
    """Map a spot symbol to the ccxt contract symbol used by USDT-M futures."""
    if market == "futures" and ":" not in symbol:
        return f"{symbol}:USDT"
    return symbol


def _write_csv(path: str, rows: Sequence[Sequence[float]]) -> None:
    """Write ``time,open,high,low,close,volume`` rows atomically."""
    temporary = path + ".tmp"
    with open(temporary, "w", encoding="utf-8", newline="") as handle:
        handle.write("time,open,high,low,close,volume\n")
        for row in rows:
            opened = datetime.fromtimestamp(row[0] / 1000, tz=timezone.utc)
            handle.write(
                f"{opened.strftime('%Y-%m-%dT%H:%M:%SZ')},"
                f"{row[1]:.8f},{row[2]:.8f},{row[3]:.8f},{row[4]:.8f},{row[5]:.8f}\n"
            )
    os.replace(temporary, path)


def fetch_symbol(
    exchange: ccxt.Exchange,
    symbol: str,
    timeframe: str,
    end_open: datetime,
    limit: int,
) -> List[List[float]]:
    """Fetch exactly ``limit`` candles ending at ``end_open`` for one symbol."""
    period = timeframe_seconds(timeframe)
    since = end_open - timedelta(seconds=period * (limit - 1))
    since_ms = int(since.timestamp() * 1000)
    end_ms = int(end_open.timestamp() * 1000)
    rows = exchange.fetch_ohlcv(symbol, timeframe, since=since_ms, limit=limit + 5)
    rows = [row for row in rows if since_ms <= int(row[0]) <= end_ms]
    if len(rows) != limit:
        raise SystemExit(
            f"{symbol}: expected {limit} candles in the shared window, got {len(rows)}"
        )
    return rows


def timeframe_seconds(timeframe: str) -> int:
    """Convert a ccxt timeframe such as ``1h`` to seconds."""
    unit = timeframe[-1]
    amount = int(timeframe[:-1])
    multipliers = {"m": 60, "h": 3600, "d": 86400}
    if unit not in multipliers:
        raise SystemExit(f"unsupported timeframe: {timeframe}")
    return amount * multipliers[unit]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fetch aligned JEV benchmark candles")
    parser.add_argument("--symbols", default=",".join(DEFAULT_SYMBOLS))
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT)
    parser.add_argument("--timeframe", default=DEFAULT_TIMEFRAME)
    parser.add_argument("--output-dir", default="runtime/jev_bench")
    parser.add_argument(
        "--market",
        choices=("spot", "futures"),
        default="spot",
        help="Binance spot or USDT-M perpetual futures (default spot)",
    )
    parser.add_argument(
        "--end",
        default=None,
        help="Open time of the last wanted candle (ISO-8601; default: last closed candle)",
    )
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    """Fetch every symbol over one shared window and write one CSV per symbol."""
    args = _build_parser().parse_args(list(argv) if argv is not None else None)
    symbols = [item.strip() for item in args.symbols.split(",") if item.strip()]
    period = timeframe_seconds(args.timeframe)
    if args.end:
        end_open = _parse_end(args.end)
    else:
        end_open = _last_closed_open(datetime.now(timezone.utc), period)
    if end_open.timestamp() % period:
        raise SystemExit("--end must align to the start of a candle")

    exchange: ccxt.Exchange
    if args.market == "futures":
        exchange = ccxt.binanceusdm({"enableRateLimit": True})
        suffix = "_futures"
    else:
        exchange = ccxt.binance({"enableRateLimit": True})
        suffix = ""
    os.makedirs(args.output_dir, exist_ok=True)
    for symbol in symbols:
        market_symbol = _market_symbol(symbol, args.market)
        rows = fetch_symbol(exchange, market_symbol, args.timeframe, end_open, args.limit)
        path = os.path.join(
            args.output_dir, f"{_base_name(market_symbol)}_{args.timeframe}{suffix}.csv"
        )
        _write_csv(path, rows)
        first = datetime.fromtimestamp(rows[0][0] / 1000, tz=timezone.utc)
        print(f"{market_symbol}: {len(rows)} candles {first.isoformat()} -> "
              f"{end_open.isoformat()} -> {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
