"""Candle-aligned scheduler for the signal service.

One worker per pair (sequential in v1). For each closed candle the service
waits close + buffer seconds, runs the pair analyses with a timeout and a
deterministic HOLD fallback, appends results to JSONL and refreshes the
per-pair heartbeat.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Callable, Dict, List, Optional

from src.signal_service.analyzer_adapter import PairAnalyzer, analyze_and_store
from src.signal_service.contract import iso_z, timeframe_seconds, utc_now_iso
from src.signal_service.jev_gate import JevGate
from src.signal_service.storage import (
    SignalStorage,
    heartbeat_path,
    signal_file_path,
    write_heartbeat,
)

logger = logging.getLogger(__name__)

PairAnalyzerFactory = Callable[[str], PairAnalyzer]


class SignalService:
    """Schedules pair analyses aligned on candle boundaries."""

    def __init__(
        self,
        pairs: List[str],
        timeframe: str = "1h",
        analyzer_factory: PairAnalyzerFactory = None,  # type: ignore[assignment]
        signals_dir: str = "/home/djelinux/TradingBots/trading-stack/runtime/signals",
        buffer_seconds: int = 120,
        timeout_seconds: float = 300.0,
        max_cycles: Optional[int] = None,
        jev_gate: Optional[JevGate] = None,
        debug_cycle_seconds: Optional[float] = None,
        dedupe: bool = True,
    ) -> None:
        """Initialize the scheduler.

        Args:
            pairs: Trading pairs, e.g. ["BTC/USDT", "ETH/USDT"].
            timeframe: Candle timeframe label.
            analyzer_factory: Callable symbol -> PairAnalyzer.
            signals_dir: Output directory for JSONL signals.
            buffer_seconds: Extra wait after candle close before analysis.
            timeout_seconds: Hard timeout per analysis (HOLD fallback after).
            max_cycles: Optional stop after N candle cycles (for testing).
            jev_gate: Optional JEV gate attached to each analyzer that does not
                already carry one. Default mode is off (no JEV call).
            debug_cycle_seconds: Debug mode. When set, the service analyses the
                last closed candle every N seconds instead of waiting for the
                next candle close (lets you replay the 1h analysis every minute).
            dedupe: When False, repeated analyses of the same candle are written
                too. Only for debug output directories.
        """
        self.pairs = pairs
        self.timeframe = timeframe
        self.analyzer_factory = analyzer_factory
        self.signals_dir = signals_dir
        self.buffer_seconds = buffer_seconds
        self.timeout_seconds = timeout_seconds
        self.max_cycles = max_cycles
        self.jev_gate = jev_gate
        self.debug_cycle_seconds = debug_cycle_seconds
        self.dedupe = dedupe
        self._stop_requested = False
        self.heartbeats: Dict[str, str] = {}
        self.storages: Dict[str, SignalStorage] = {}
        self.analyzers: Dict[str, PairAnalyzer] = {}

    # ------------------------------------------------------------------ setup
    def _pair_state(self, symbol: str) -> tuple[PairAnalyzer, SignalStorage]:
        if symbol not in self.storages:
            self.storages[symbol] = SignalStorage(signal_file_path(self.signals_dir, symbol, self.timeframe))
        if symbol not in self.analyzers:
            analyzer = self.analyzer_factory(symbol)
            if self.jev_gate is not None and getattr(analyzer, "jev_gate", None) is None:
                analyzer.jev_gate = self.jev_gate
            self.analyzers[symbol] = analyzer
        return self.analyzers[symbol], self.storages[symbol]

    def request_stop(self) -> None:
        """Ask the scheduler to stop after the current cycle."""
        self._stop_requested = True

    # ----------------------------------------------------------------- timing
    def next_candle_boundary(self, now: Optional[datetime] = None) -> datetime:
        """Return the open time of the next candle after `now`."""
        period = timeframe_seconds(self.timeframe)
        ref = (now or datetime.now(timezone.utc)).timestamp()
        return datetime.fromtimestamp((ref // period + 1) * period, tz=timezone.utc)

    def seconds_until_close(self, candle_open: Optional[datetime] = None) -> float:
        """Seconds to wait until candle close + buffer.

        Args:
            candle_open: Open time of the candle to wait for; default: the
                candle currently running.
        """
        start = candle_open or self._current_candle_open()
        close_at = start + timedelta(seconds=timeframe_seconds(self.timeframe))
        wait = (close_at - datetime.now(timezone.utc)).total_seconds() + self.buffer_seconds
        return max(0.0, wait)

    def _current_candle_open(self) -> datetime:
        period = timeframe_seconds(self.timeframe)
        ref = datetime.now(timezone.utc).timestamp()
        return datetime.fromtimestamp(ref // period * period, tz=timezone.utc)

    # ------------------------------------------------------------------ cycle
    async def run_cycle(self, candle_open: Optional[datetime] = None) -> Dict[str, str]:
        """Analyze one closed candle for every pair (sequential in v1).

        Args:
            candle_open: Open time of the candle that just closed. Defaults to
                the candle currently running, whose close is in the future and
                therefore never analysed by the scheduler.

        Returns:
            Map symbol -> decision_id produced (or "duplicate"/"error").
        """
        candle_open_dt = candle_open if candle_open is not None else self._current_candle_open()
        candle_open_iso = iso_z(candle_open_dt)
        results: Dict[str, str] = {}
        for symbol in self.pairs:
            analyzer, storage = self._pair_state(symbol)
            try:
                signal = await analyze_and_store(
                    analyzer,
                    storage,
                    self.timeout_seconds,
                    candle_open_time=candle_open_iso,
                    dedupe=self.dedupe,
                )
                results[symbol] = signal.decision_id
                self.heartbeats[symbol] = utc_now_iso()
                write_heartbeat(self.signals_dir, symbol, self.timeframe, signal.candle_open_time)
            except Exception as exc:  # pylint: disable=broad-exception-caught
                logger.exception("Cycle failed for %s: %s", symbol, exc)
                results[symbol] = "error"
                # Keep liveness truthful: a failed pair is still running, so
                # TTL monitoring must not report it as dead.
                try:
                    write_heartbeat(self.signals_dir, symbol, self.timeframe, candle_open_iso)
                    self.heartbeats[symbol] = utc_now_iso()
                except Exception as heartbeat_exc:  # pylint: disable=broad-exception-caught
                    logger.warning("Heartbeat refresh failed for %s: %s", symbol, heartbeat_exc)
        try:
            write_heartbeat_summary(self.signals_dir, self.heartbeats)
        except Exception as summary_exc:  # pylint: disable=broad-exception-caught
            logger.warning("Heartbeat summary write failed: %s", summary_exc)
        logger.info("Cycle done for candle %s: %s", candle_open, results)
        return results

    def last_closed_candle_open(self, now: Optional[datetime] = None) -> datetime:
        """Return the open time of the most recent *closed* candle.

        Args:
            now: Reference time; defaults to current UTC time.
        """
        period = timeframe_seconds(self.timeframe)
        ref = (now or datetime.now(timezone.utc)).timestamp()
        return datetime.fromtimestamp((ref // period - 1) * period, tz=timezone.utc)

    async def run(self) -> None:
        """Run the scheduler loop until stopped.

        Normal mode waits for each candle close + buffer. Debug mode
        (`debug_cycle_seconds`) analyses the last closed candle every N
        seconds, so the hourly analysis can be replayed once a minute.
        """
        logger.info(
            "Signal service starting: pairs=%s timeframe=%s buffer=%ss jev_mode=%s debug_cycle=%s",
            self.pairs, self.timeframe, self.buffer_seconds,
            self.jev_gate.mode.value if self.jev_gate is not None else "off",
            self.debug_cycle_seconds,
        )
        cycles = 0
        if self.debug_cycle_seconds:
            logger.info(
                "DEBUG mode: replaying the last closed %s candle every %.0fs (dedupe=%s)",
                self.timeframe, self.debug_cycle_seconds, self.dedupe,
            )
            while not self._stop_requested:
                await self.run_cycle(self.last_closed_candle_open())
                cycles += 1
                if self.max_cycles is not None and cycles >= self.max_cycles:
                    break
                await asyncio.sleep(self.debug_cycle_seconds)
            logger.info("Signal service stopped after %d debug cycles", cycles)
            return

        while not self._stop_requested:
            candle_open = self._current_candle_open()
            wait = self.seconds_until_close(candle_open)
            logger.info("Sleeping %.0fs until candle %s closes (+buffer)", wait, candle_open)
            await asyncio.sleep(wait)
            await self.run_cycle(candle_open)
            cycles += 1
            if self.max_cycles is not None and cycles >= self.max_cycles:
                break
        logger.info("Signal service stopped after %d cycles", cycles)


def write_heartbeat_summary(signals_dir: str, heartbeats: Dict[str, str]) -> None:
    """Persist the aggregated per-pair heartbeats."""
    os.makedirs(signals_dir, exist_ok=True)
    payload = {"updated_at": utc_now_iso(), "pairs": dict(heartbeats)}
    path = os.path.join(signals_dir, os.path.basename(heartbeat_path(signals_dir)).replace(
        "heartbeat.json", "heartbeat_summary.json"
    ))
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, separators=(",", ":")) + "\n")
        handle.flush()
        os.fsync(handle.fileno())