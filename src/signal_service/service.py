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
from typing import Any, Callable, Dict, List, Optional

from src.signal_service.analyzer_adapter import (
    PairAnalyzer,
    UpdateContext,
    analyze_and_store,
)
from src.signal_service.contract import iso_z, parse_iso, timeframe_seconds, utc_now_iso
from src.signal_service.jev_gate import JevGate
from src.signal_service.positions import (
    PositionSnapshot,
    PositionsSnapshot,
    format_position_context,
    load_positions,
)
from src.signal_service.storage import (
    SignalStorage,
    UpdateStorage,
    default_signals_dir,
    default_updates_dir,
    heartbeat_path,
    signal_file_path,
    update_file_path,
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
        signals_dir: str = default_signals_dir(),
        buffer_seconds: int = 120,
        timeout_seconds: float = 300.0,
        max_cycles: Optional[int] = None,
        jev_gate: Optional[JevGate] = None,
        debug_cycle_seconds: Optional[float] = None,
        dedupe: bool = True,
        position_aware: bool = False,
        positions_file: Optional[str] = None,
        updates_dir: Optional[str] = None,
        max_snapshot_age_seconds: float = 300.0,
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
            position_aware: When True and a position snapshot lists open trades,
                the pair's analysis produces position UPDATE records instead of
                entry signals. Off by default: without positions the behaviour
                is exactly the stateless one.
            positions_file: Path to the snapshot written by position_sync.
                Defaults to ``<signals parent>/positions.json``.
            updates_dir: Directory for the v1.1 update JSONL files. Defaults to
                the ``updates`` sibling of ``signals_dir``.
            max_snapshot_age_seconds: A positions.json older than this is
                treated as stale. The service then fails OPEN to entry signals
                (the stateless path) instead of freezing a pair forever, and
                records ``metrics["stale_snapshots"]``.
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
        self.position_aware = position_aware
        self.positions_file = positions_file or os.path.join(
            os.path.dirname(os.path.normpath(signals_dir)), "positions.json"
        )
        self.updates_dir = updates_dir or default_updates_dir(signals_dir)
        self.max_snapshot_age_seconds = max(float(max_snapshot_age_seconds), 1.0)
        self.metrics: Dict[str, int] = {"stale_snapshots": 0, "stale_positions": 0}
        self._missing_snapshot_logged = False
        self._stop_requested = False
        self.heartbeats: Dict[str, str] = {}
        self.storages: Dict[str, SignalStorage] = {}
        self.update_storages: Dict[str, UpdateStorage] = {}
        self.analyzers: Dict[str, PairAnalyzer] = {}

    # ------------------------------------------------------------------ setup
    def _pair_state(self, symbol: str) -> tuple[PairAnalyzer, SignalStorage]:
        if symbol not in self.storages:
            self.storages[symbol] = SignalStorage(signal_file_path(self.signals_dir, symbol, self.timeframe))
        if self.position_aware and symbol not in self.update_storages:
            self.update_storages[symbol] = UpdateStorage(
                update_file_path(self.updates_dir, symbol, self.timeframe)
            )
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
        snapshot = self._load_fresh_snapshot() if self.position_aware else None
        for symbol in self.pairs:
            analyzer, storage = self._pair_state(symbol)
            try:
                open_positions = snapshot.for_symbol(symbol) if snapshot is not None else []
                if open_positions:
                    results[symbol] = await self._run_position_updates(
                        analyzer, symbol, open_positions, candle_open_iso
                    )
                    self.heartbeats[symbol] = utc_now_iso()
                    write_heartbeat(self.signals_dir, symbol, self.timeframe, candle_open_iso)
                    continue
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

    def _load_fresh_snapshot(self) -> PositionsSnapshot:
        """Load positions.json, failing open to entries when it is stale.

        A stale snapshot means position_sync likely died: keeping the pair
        "position open" forever would freeze all its entry signals. The service
        therefore treats it as empty, logs a warning and increments metrics so
        supervision can alert. Updates are then not emitted either.
        """
        if not os.path.exists(self.positions_file):
            if not self._missing_snapshot_logged:
                logger.warning(
                    "position_aware is enabled but %s does not exist; running stateless",
                    self.positions_file,
                )
                self._missing_snapshot_logged = True
            return PositionsSnapshot()
        snapshot = load_positions(self.positions_file)
        age = self._snapshot_age_seconds(snapshot)
        if age is None:
            if snapshot.positions:
                self.metrics["stale_snapshots"] += 1
                self.metrics["stale_positions"] += len(snapshot.positions)
                logger.warning(
                    "positions.json has no parseable updated_at; failing open to entry signals",
                )
                return PositionsSnapshot()
            return snapshot
        if age > self.max_snapshot_age_seconds:
            if snapshot.positions:
                self.metrics["stale_snapshots"] += 1
                self.metrics["stale_positions"] += len(snapshot.positions)
                logger.warning(
                    "positions.json is stale (%.0fs > %.0fs); failing open to entry "
                    "signals for %d position(s)",
                    age, self.max_snapshot_age_seconds, len(snapshot.positions),
                )
            return PositionsSnapshot()
        logger.debug("positions.json age %.0fs", age)
        return snapshot

    def _snapshot_age_seconds(self, snapshot: PositionsSnapshot) -> Optional[float]:
        """Age of the snapshot in seconds, or None when it has no timestamp."""
        if not snapshot.updated_at:
            return None
        try:
            updated_at = parse_iso(snapshot.updated_at)
        except (TypeError, ValueError):
            return None
        return max(0.0, (datetime.now(timezone.utc) - updated_at).total_seconds())

    async def _run_position_updates(
        self,
        analyzer: PairAnalyzer,
        symbol: str,
        positions: List[PositionSnapshot],
        candle_open_iso: str,
    ) -> str:
        """Produce v1.1 UPDATE records for the symbol's open positions.

        Entry analysis is skipped while a position is open: no new entry may be
        opened from a candle that should manage an existing trade.
        """
        updates = self.update_storages[symbol]
        latest_by_position: Dict[str, Any] = {}
        for update in updates.load_all():
            position_id = getattr(update, "position_id", None)
            if not position_id:
                continue
            previous = latest_by_position.get(position_id)
            if previous is None or (update.revision or 0) > (previous.revision or 0):
                latest_by_position[position_id] = update
        produced = 0
        for position in positions:
            entry_sl, entry_tp = self._entry_levels(symbol, position)
            stop_loss, take_profit = self._current_levels(
                position, entry_sl, entry_tp, latest_by_position.get(position.position_id)
            )
            if stop_loss is None:
                logger.warning(
                    "Position %s %s has no known stop-loss; update skipped",
                    symbol, position.position_id,
                )
                continue
            latest = latest_by_position.get(position.position_id)
            revision = (latest.revision or 0) + 1 if latest is not None else 1
            as_of = parse_iso(candle_open_iso) + timedelta(
                seconds=timeframe_seconds(self.timeframe)
            )
            try:
                metrics = analyzer.position_metrics(position, as_of=as_of)
            except Exception as exc:  # pylint: disable=broad-exception-caught
                logger.debug("Position metrics unavailable for %s: %s", symbol, exc)
                metrics = {}
            ctx = UpdateContext(
                position_id=position.position_id,
                position_side=position.side,
                entry_price=position.entry_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                revision=revision,
                based_on_entry=position.enter_tag or f"position-{position.position_id}",
                current_price=position.current_price,
                entry_sl=entry_sl,
                entry_tp=entry_tp,
            )
            try:
                signal = await asyncio.wait_for(
                    analyzer.analyze(
                        candle_open_time=candle_open_iso,
                        position_context=self._position_context_text(
                            position, stop_loss, take_profit, metrics
                        ),
                        update_context=ctx,
                    ),
                    timeout=self.timeout_seconds,
                )
            except Exception as exc:  # pylint: disable=broad-exception-caught
                logger.warning("Position update analysis failed for %s %s: %s", symbol, position.position_id, exc)
                continue
            if signal.action not in ("UPDATE", "CLOSE"):
                logger.info(
                    "No position update for %s %s: %s",
                    symbol, position.position_id, signal.reasoning[:120],
                )
                continue
            if updates.append(signal, dedupe=self.dedupe):
                produced += 1
                logger.info(
                    "Position %s %s: %s revision %s",
                    symbol, position.position_id, signal.action, signal.revision,
                )
        return f"updates:{produced}"

    def _entry_levels(
        self, symbol: str, position: PositionSnapshot
    ) -> tuple[Optional[float], Optional[float]]:
        """Original entry-signal SL/TP (the widening base), if resolvable."""
        if not position.enter_tag:
            return None, None
        for entry in self.storages[symbol].load_all():
            if entry.decision_id == position.enter_tag:
                return entry.sl, entry.tp
        return None, None

    @staticmethod
    def _current_levels(
        position: PositionSnapshot,
        entry_sl: Optional[float],
        entry_tp: Optional[float],
        latest_update: Any,
    ) -> tuple[Optional[float], Optional[float]]:
        """Current SL/TP: snapshot, then entry signal, then the latest update."""
        stop_loss = position.stop_loss if position.stop_loss is not None else entry_sl
        take_profit = position.take_profit if position.take_profit is not None else entry_tp
        if latest_update is not None:
            stop_loss = latest_update.sl if latest_update.sl is not None else stop_loss
            take_profit = latest_update.tp if latest_update.tp is not None else take_profit
        return stop_loss, take_profit

    @staticmethod
    def _position_context_text(
        position: PositionSnapshot,
        stop_loss: float,
        take_profit: Optional[float],
        metrics: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Return the shared position-management prompt block."""
        return format_position_context(position, stop_loss, take_profit, metrics)

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