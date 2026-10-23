"""Command-line entry point for the signal service.

Subcommands:
    run      candle-aligned scheduler (live/dry-run, no orders ever)
    replay   historical replay with strict no-look-ahead (as_of slicing)
    validate validate existing JSONL signal files against the contract

Example:
    python -m src.signal_service replay \
        --pairs BTC/USDT --timeframe 1h \
        --start 2026-01-01 --end 2026-04-01 \
        --candles data/BTC_USDT_1h.csv \
        --output /home/djelinux/TradingBots/trading-stack/runtime/signals_backtest \
        --no-news

Debug (replay the hourly analysis every minute, no need to wait an hour):
    python -m src.signal_service run \
        --pairs BTC/USDT,ETH/USDT --timeframe 1h \
        --debug-cycle-seconds 60 --no-dedup \
        --output /home/djelinux/TradingBots/trading-stack/runtime/signals_debug
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import logging
import os
import sys
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Dict, List, Tuple

from src.signal_service.analyzer_adapter import PairAnalyzer, ReplayDataInjector
from src.signal_service.contract import parse_iso, timeframe_seconds
from src.signal_service.service import SignalService
from src.signal_service.storage import SignalStorage, signal_file_path

logger = logging.getLogger(__name__)

DEFAULT_SIGNALS_DIR = "/home/djelinux/TradingBots/trading-stack/runtime/signals"


def _parse_bool_flag(value: str) -> bool:
    return value.lower() not in ("0", "false", "no")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m src.signal_service")
    parser.add_argument("--config", default="config/config.ini", help="Config file path")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="Run the candle-aligned scheduler")
    run.add_argument("--pairs", default="BTC/USDT,ETH/USDT")
    run.add_argument("--timeframe", default="1h")
    run.add_argument("--output", default=DEFAULT_SIGNALS_DIR)
    run.add_argument("--buffer-seconds", type=int, default=120)
    run.add_argument("--timeout-seconds", type=float, default=300.0)
    run.add_argument("--max-cycles", type=int, default=None)
    run.add_argument("--use-news", type=_parse_bool_flag, default="true")
    run.add_argument("--use-chart", type=_parse_bool_flag, default="true")
    run.add_argument(
        "--debug-cycle-seconds", type=float, default=0.0,
        help="Debug: replay the last closed candle every N seconds (e.g. 60) "
             "instead of waiting for the next candle close",
    )
    run.add_argument(
        "--no-dedup", action="store_true",
        help="Debug: also write repeated analyses of the same candle "
             "(use with a debug --output directory, never runtime/signals)",
    )

    replay = sub.add_parser("replay", help="Historical replay (no future data)")
    replay.add_argument("--pairs", default="BTC/USDT")
    replay.add_argument("--timeframe", default="1h")
    replay.add_argument("--start", required=True, help="e.g. 2026-01-01")
    replay.add_argument("--end", required=True, help="e.g. 2026-04-01")
    replay.add_argument("--as-of", default=None, help="Freeze 'now' at this ISO timestamp")
    replay.add_argument("--candles", required=True, help="OHLCV csv (time,open,high,low,close,volume)")
    replay.add_argument("--output", default="/home/djelinux/TradingBots/trading-stack/runtime/signals_backtest")
    replay.add_argument("--no-news", action="store_true")
    replay.add_argument("--use-chart", type=_parse_bool_flag, default="true")
    replay.add_argument("--timeout-seconds", type=float, default=300.0)

    validate = sub.add_parser("validate", help="Validate JSONL signal files")
    validate.add_argument("--input", default=DEFAULT_SIGNALS_DIR)
    validate.add_argument("--file", default=None, help="Validate a single JSONL file")
    return parser


def _load_engine_factory():
    """Build the analysis engine factory through the standard composition root.

    This reuse guarantees the same wiring as the live bot without duplicating
    providers, config or exchange logic. Signals only; no order placement.
    """
    from start import CompositionRoot

    return CompositionRoot()


async def _build_initialized_engines(
    pairs: List[str],
    engine_factory: Callable[[str], Any],
    exchange_manager: Any,
    timeframe: str,
) -> Dict[str, Any]:
    """Create and initialize one AnalysisEngine per pair.

    An AnalysisEngine must be initialized with its symbol, exchange and
    timeframe before analysis. Otherwise the data collector has no fetcher and
    every analysis falls back to HOLD.

    Args:
        pairs: Trading pairs to prepare.
        engine_factory: Callable symbol -> AnalysisEngine (composition root).
        exchange_manager: ExchangeManager used to resolve the pair's exchange.
        timeframe: Candle timeframe label.

    Returns:
        Map symbol -> initialized engine.

    Raises:
        RuntimeError: when no exchange supports one of the requested pairs.
    """
    engines: Dict[str, Any] = {}
    for symbol in pairs:
        exchange, exchange_id = await exchange_manager.find_symbol_exchange(symbol)
        if exchange is None:
            raise RuntimeError(
                f"No exchange supports {symbol!r}: analysis engine not initialized. "
                "Fix the pair list or the exchange configuration."
            )
        engine = engine_factory(symbol)
        engine.initialize_for_symbol(symbol=symbol, exchange=exchange, timeframe=timeframe)
        engines[symbol] = engine
        logger.info("Initialized analysis engine for %s on %s (%s)", symbol, exchange_id, timeframe)
    return engines


async def _run_command(args: argparse.Namespace) -> None:
    root = _load_engine_factory()
    deps = await root.build_dependencies()
    pairs = [pair.strip() for pair in args.pairs.split(",") if pair.strip()]
    engines = await _build_initialized_engines(
        pairs,
        deps["market_analyzer_factory"],
        deps["exchange_manager"],
        args.timeframe,
    )

    def analyzer_factory(symbol: str) -> PairAnalyzer:
        return PairAnalyzer(
            engine=engines[symbol],
            symbol=symbol,
            timeframe=args.timeframe,
            use_news=args.use_news,
            use_chart=args.use_chart,
        )

    service = SignalService(
        pairs=pairs,
        timeframe=args.timeframe,
        analyzer_factory=analyzer_factory,
        signals_dir=args.output,
        buffer_seconds=args.buffer_seconds,
        timeout_seconds=args.timeout_seconds,
        max_cycles=args.max_cycles,
        debug_cycle_seconds=args.debug_cycle_seconds or None,
        dedupe=not args.no_dedup,
    )
    await service.run()


def _load_candles_csv(path: str) -> Tuple[List[list], List[datetime]]:
    """Load an OHLCV csv (freqtrade-exported or generated) into candles+timestamps."""
    candles: List[list] = []
    timestamps: List[datetime] = []
    with open(path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader, None)
        first = header[0].strip().lower() if header else ""
        for row in reader:
            if not row or len(row) < 5:
                continue
            if first in ("time", "date", "timestamp"):
                tokens = row[0].replace("Z", "+00:00")
                open_at = datetime.fromisoformat(tokens)
                if open_at.tzinfo is None:
                    open_at = open_at.replace(tzinfo=timezone.utc)
                values = row[1:6]
            else:
                raise ValueError(f"Unsupported csv header: {header}")
            candles.append([float(v) for v in values])
            timestamps.append(open_at)
    return candles, timestamps



async def _replay_command(args: argparse.Namespace) -> None:
    pairs: List[str] = args.pairs.split(",")
    if len(pairs) != 1:
        # v1: one csv per pair; run one pair per invocation.
        raise ValueError("replay supports exactly one pair per invocation in v1")
    symbol = pairs[0]
    start = parse_iso(args.start if "T" in args.start else args.start + "T00:00:00Z")
    end = parse_iso(args.end if "T" in args.end else args.end + "T00:00:00Z")
    period = timedelta(seconds=timeframe_seconds(args.timeframe))

    root = _load_engine_factory()
    deps = await root.build_dependencies()
    engines = await _build_initialized_engines(
        [symbol],
        deps["market_analyzer_factory"],
        deps["exchange_manager"],
        args.timeframe,
    )
    engine = engines[symbol]
    analyzer = PairAnalyzer(
        engine=engine,
        symbol=symbol,
        timeframe=args.timeframe,
        use_news=not args.no_news,
        use_chart=args.use_chart,
    )
    storage = SignalStorage(signal_file_path(args.output, symbol, args.timeframe))

    candles, timestamps = _load_candles_csv(args.candles)
    injector = ReplayDataInjector(analyzer, candles, timestamps)
    injector.install()

    as_of_cap = parse_iso(args.as_of) if args.as_of else None
    candle = start
    produced = 0
    while candle < end:
        if as_of_cap is not None and candle + period > as_of_cap:
            break
        # No future data: only candles closed at or before `candle + period`.
        as_of = min(candle + period, datetime.now(timezone.utc))
        injector.set_as_of(as_of)
        try:
            signal = await asyncio.wait_for(
                analyzer.analyze(candle_open_time=candle.isoformat()),
                timeout=args.timeout_seconds,
            )
            if storage.append(signal):
                produced += 1
        except Exception as exc:  # pylint: disable=broad-exception-caught
            logger.warning("Replay analysis failed for candle %s: %s", candle, exc)
        candle = candle + period
    injector.restore()
    print(f"Replay complete for {symbol}: {produced} signals written to {storage.path}")


def _validate_file(path: str, schema_path: str) -> Tuple[int, int]:
    """Validate one JSONL file; returns (valid_signals, failures)."""
    from src.signal_service.contract import Signal

    schema = None
    with open(schema_path, "r", encoding="utf-8") as handle:
        schema = json.load(handle)
    validator = None
    try:
        import jsonschema
        validator = jsonschema.Draft202012Validator(schema)
    except ImportError:
        validator = None

    valid = 0
    failures = 0
    with open(path, "r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                data = json.loads(stripped)
                signal = Signal.from_dict(data)
                if validator is not None:
                    validator.validate(signal.to_dict())
                valid += 1
            except (json.JSONDecodeError, ValueError) as exc:
                print(f"{path}:{line_number}: INVALID: {exc}")
                failures += 1
    return valid, failures


def _validate_command(args: argparse.Namespace) -> int:
    import os

    schema_path = "/home/djelinux/TradingBots/trading-stack/contracts/signal.schema.json"
    files = [args.file] if args.file else []
    if not files:
        files = sorted(
            os.path.join(args.input, name)
            for name in os.listdir(args.input)
            if name.endswith(".jsonl")
        )
    checked = 0
    failures = 0
    for path in files:
        valid, errors = _validate_file(path, schema_path)
        checked += valid
        failures += errors
        print(f"{path}: ok={valid} errors={errors}")
    if args.file is None and not os.path.isdir(args.input):
        print(f"input directory not found: {args.input}")
        return 2
    if not files:
        print(f"no .jsonl files found in {args.input}")
        return 2
    print(f"checked={checked} failures={failures}")
    return 0 if failures == 0 else 1


def _force_exit(code: int) -> None:
    """Exit even when background libraries keep non-daemon threads alive.

    CompositionRoot opens an aiohttp session, ChromaDB and the embedding model;
    some of them leave threads running, so a normal return can hang after a
    bounded run (e.g. --max-cycles). Logs are flushed before the hard exit.
    """
    logging.shutdown()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)


def main(argv: List[str] = None) -> int:
    """CLI entry point."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    args = _build_parser().parse_args(argv)
    command = args.command
    if command == "validate":
        return _validate_command(args)
    if command == "run":
        asyncio.run(_run_command(args))
        _force_exit(0)
    if command == "replay":
        asyncio.run(_replay_command(args))
        _force_exit(0)
    print(f"Unknown command: {command}")
    return 2


if __name__ == "__main__":
    sys.exit(main())