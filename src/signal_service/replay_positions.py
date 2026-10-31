"""Position-aware replay: reuse entry signals, manage open trades only.

This is the missing half of ``replay``. ``replay`` analyses every candle
statelessly and never produces v1.1 UPDATE/CLOSE records, so a Freqtrade
backtest can only run the entry signal's static SL/TP. This runner keeps the
existing entry signals untouched and calls the LLM **only on candles where a
simulated position is open**, writing UPDATE/CLOSE records for the
``signal_updates_dir`` channel.

Fills mirror SignalStrategy as closely as possible: entry at the next candle
open, stoploss intrabar, take-profit and TTL on the candle close/open. The
caller mounts the updates directory into Freqtrade; no position, API key or
exchange call is involved here.

Identifiers are namespaced with ``replay:`` so they can never collide with the
market-scoped live ids written by ``position_sync`` (``spot:<trade_id>`` /
``futures:<trade_id>``). When the caller knows which market it replays, pass
``market`` and the position id becomes ``replay:<market>:<entry>``.
``based_on_entry`` stays the real entry decision id so the consumer's
``update_matches_entry`` linkage keeps working. The update ``decision_id`` is
``<based_on_entry>-u<revision>``; a spot and a futures replay of the same pair
must therefore write to separate ``--output`` directories to avoid
deduplicating each other.
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from src.signal_service.analyzer_adapter import PairAnalyzer, ReplayDataInjector, UpdateContext
from src.signal_service.contract import iso_z, parse_iso, timeframe_seconds
from src.signal_service.positions import PositionSnapshot, format_position_context
from src.signal_service.storage import UpdateStorage

logger = logging.getLogger(__name__)


@dataclass
class SimPosition:
    """One simulated open position."""

    enter_tag: str
    side: str  # "long" | "short"
    entry: float
    sl: float
    tp: Optional[float]
    opened_at: datetime
    entry_sl: float
    entry_tp: Optional[float]
    revision: int = 0
    bars: int = 0
    mfe: float = 0.0
    mae: float = 0.0

    @property
    def is_short(self) -> bool:
        return self.side == "short"

    def snapshot(self, symbol: str, current_price: float) -> PositionSnapshot:
        return PositionSnapshot(
            position_id=self.enter_tag,
            symbol=symbol,
            side=self.side,
            entry_price=self.entry,
            current_price=current_price,
            stop_loss=self.sl,
            take_profit=self.tp,
            open_date=iso_z(self.opened_at),
            enter_tag=self.enter_tag,
        )


@dataclass
class ReplayPositionStats:
    """Counters reported at the end of a position-aware replay."""

    entries_reused: int = 0
    candles_seen: int = 0
    candles_with_position: int = 0
    analysis_calls: int = 0
    updates: int = 0
    closes: int = 0
    no_update: int = 0
    failed: int = 0
    closed_sim: int = 0
    details: List[Dict[str, Any]] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        data = {k: v for k, v in self.__dict__.items() if k != "details"}
        data["positions"] = self.details
        return data


def _effective_end(
    end: datetime,
    as_of_cap: Optional[datetime] = None,
    now: Optional[datetime] = None,
) -> datetime:
    """Latest candle close the replay may use: never in the future."""
    effective_end = min(end, now or datetime.now(timezone.utc))
    if as_of_cap is not None:
        effective_end = min(effective_end, as_of_cap)
    return effective_end


def _entry_index(
    entries: Iterable[Dict[str, Any]],
    symbol: Optional[str] = None,
) -> Dict[datetime, Dict[str, Any]]:
    """Map signal candle close time to an executable BUY/SELL entry.

    Only approved BUY/SELL rows are kept: HOLD rows, unapproved proposals and
    entries whose stop-loss is missing cannot open a simulated position. When
    ``symbol`` is given, rows that name a different symbol are ignored so a
    mixed or hand-edited JSONL cannot open a position on the wrong pair.
    """
    index: Dict[datetime, Dict[str, Any]] = {}
    for entry in entries:
        if entry.get("action") not in ("BUY", "SELL"):
            continue
        if entry.get("risk_approved") is not True:
            continue
        entry_symbol = entry.get("symbol")
        if symbol is not None and entry_symbol is not None and entry_symbol != symbol:
            continue
        close_at = parse_iso(entry.get("candle_close_time"))
        sl = entry.get("sl")
        if close_at is None or not isinstance(sl, (int, float)):
            continue
        index[close_at] = entry
    return index


def _apply_levels(position: SimPosition, update: Any) -> None:
    """Apply a validated UPDATE record to the simulated position."""
    if isinstance(getattr(update, "sl", None), (int, float)):
        position.sl = float(update.sl)
    if isinstance(getattr(update, "tp", None), (int, float)):
        position.tp = float(update.tp)
    position.revision = int(getattr(update, "revision", position.revision + 1) or position.revision + 1)


def _exit_on_candle(position: SimPosition, candle: Sequence[Any]) -> Optional[Tuple[str, float]]:
    """Return (reason, exit_price) when the candle closes the position.

    Stop is intrabar; take-profit is close-based, like ``custom_exit``.
    Candles are [timestamp_ms, open, high, low, close, volume].
    """
    high, low, close = float(candle[2]), float(candle[3]), float(candle[4])
    if position.is_short:
        if high >= position.sl:
            return "sl", max(position.sl, float(candle[1]))
        if position.tp is not None and close <= position.tp:
            return "tp", close
    else:
        if low <= position.sl:
            return "sl", min(position.sl, float(candle[1]))
        if position.tp is not None and close >= position.tp:
            return "tp", close
    return None


async def replay_positions_for_pair(
    analyzer: PairAnalyzer,
    symbol: str,
    timeframe: str,
    candles: Sequence[Sequence[float]],
    timestamps: List[datetime],
    entries: Iterable[Dict[str, Any]],
    start: datetime,
    end: datetime,
    update_storage: UpdateStorage,
    max_hold_hours: float = 24.0,
    timeout_seconds: float = 300.0,
    as_of_cap: Optional[datetime] = None,
    now: Optional[datetime] = None,
    injector: Optional[Any] = None,
    market: Optional[str] = None,
) -> ReplayPositionStats:
    """Replay one pair's open positions and write UPDATE/CLOSE records.

    Entry analysis is never re-run: the existing entry JSONL is reused. The LLM
    is called only when a simulated position is still open at a candle close,
    so a 7-day backtest costs one call per open-position candle, not one per
    candle.

    Candles are taken from the loaded series itself (not a synthetic grid), so
    an off-grid ``start`` or a gap in the CSV skips nothing silently. ``market``
    optionally scopes the emitted position id with ``spot``/``futures``.
    """
    if market:
        market = market.strip().lower()
        if market not in ("spot", "futures"):
            raise ValueError(f"invalid market label: {market!r} (expected spot|futures)")
    else:
        market = None

    stats = ReplayPositionStats()
    by_close = _entry_index(entries, symbol=symbol)
    stats.entries_reused = len(by_close)
    period = timedelta(seconds=timeframe_seconds(timeframe))
    effective_end = _effective_end(end, as_of_cap, now)
    owns_injector = injector is None
    if injector is None:
        injector = ReplayDataInjector(analyzer, candles, timestamps)
        injector.install()
    position: Optional[SimPosition] = None
    pending: Optional[Any] = None
    try:
        for idx, candle_open in enumerate(timestamps):
            as_of = candle_open + period
            # `timestamps` is strictly ascending (the CSV loader enforces it),
            # so the first close past the window ends the replay.
            if as_of > effective_end:
                break
            if candle_open < start:
                continue
            stats.candles_seen += 1
            row = candles[idx]
            injector.set_as_of(as_of)

            # A decision taken at the previous close is executable at this open.
            # Apply it even when it was deduplicated on a rerun: otherwise the
            # in-memory levels would drift from the persisted history and every
            # later revision would be deduplicated too.
            if pending is not None and position is not None:
                if pending.action == "CLOSE":
                    stats.closed_sim += 1
                    position = None
                else:
                    _apply_levels(position, pending)
                pending = None

            # Entry: a signal whose candle closed at this open.
            if position is None:
                entry = by_close.get(candle_open)
                if entry is not None:
                    position = SimPosition(
                        enter_tag=str(entry.get("decision_id")),
                        side="short" if entry.get("action") == "SELL" else "long",
                        entry=float(row[1]),
                        sl=float(entry["sl"]),
                        tp=float(entry["tp"]) if isinstance(entry.get("tp"), (int, float)) else None,
                        opened_at=candle_open,
                        entry_sl=float(entry["sl"]),
                        entry_tp=float(entry["tp"]) if isinstance(entry.get("tp"), (int, float)) else None,
                    )

            if position is None:
                continue
            stats.candles_with_position += 1

            # TTL is evaluated at the candle open, like custom_exit.
            if max_hold_hours and max_hold_hours > 0:
                if candle_open - position.opened_at >= timedelta(hours=max_hold_hours):
                    position = None
                    continue

            exit_hit = _exit_on_candle(position, row)
            if exit_hit is not None:
                position = None
                continue

            # Still open at the close: update MFE/MAE and ask the LLM to manage it.
            high, low, close = float(row[2]), float(row[3]), float(row[4])
            if position.is_short:
                position.mfe = max(position.mfe, position.entry - low)
                position.mae = max(position.mae, high - position.entry)
            else:
                position.mfe = max(position.mfe, high - position.entry)
                position.mae = max(position.mae, position.entry - low)
            position.bars += 1

            # Market-scoped position id: never collide with the live
            # ``spot:``/``futures:`` ids. ``based_on_entry`` stays the real entry
            # decision id so the consumer's update/entry linkage keeps working.
            scoped_position_id = (
                f"replay:{market}:{position.enter_tag}" if market
                else f"replay:{position.enter_tag}"
            )
            snapshot = position.snapshot(symbol, close)
            try:
                metrics = analyzer.position_metrics(snapshot, as_of=as_of)
            except Exception as exc:  # pylint: disable=broad-exception-caught
                logger.debug("Position metrics unavailable for %s: %s", symbol, exc)
                metrics = {}
            ctx = UpdateContext(
                position_id=scoped_position_id,
                position_side=position.side,
                entry_price=position.entry,
                stop_loss=position.sl,
                take_profit=position.tp,
                revision=position.revision + 1,
                based_on_entry=position.enter_tag,
                current_price=close,
                entry_sl=position.entry_sl,
                entry_tp=position.entry_tp,
            )
            try:
                coro = analyzer.analyze(
                    candle_open_time=iso_z(candle_open),
                    position_context=format_position_context(
                        snapshot, position.sl, position.tp, metrics
                    ),
                    update_context=ctx,
                )
                if timeout_seconds and timeout_seconds > 0:
                    signal = await asyncio.wait_for(coro, timeout=timeout_seconds)
                else:
                    signal = await coro
                stats.analysis_calls += 1
            except Exception as exc:  # pylint: disable=broad-exception-caught
                stats.failed += 1
                logger.warning("Position replay analysis failed for %s %s: %s", symbol, position.enter_tag, exc)
                continue

            if signal.action not in ("UPDATE", "CLOSE"):
                stats.no_update += 1
                continue
            # The producer stamps wall-clock now; replay must stamp the simulated
            # candle close or the backtest would treat the update as future data.
            signal.created_at = iso_z(as_of)
            stored = update_storage.append(signal)
            if stored:
                stats.updates += 1
                if signal.action == "CLOSE":
                    stats.closes += 1
            # Always advance the simulation, even on a deduplicated rerun.
            pending = signal
            stats.details.append(
                {
                    "position": scoped_position_id,
                    "candle": iso_z(candle_open),
                    "action": signal.action,
                    "sl": signal.sl,
                    "tp": signal.tp,
                    "stored": stored,
                    "reasoning": (signal.reasoning or "")[:200],
                }
            )
    finally:
        if owns_injector:
            injector.restore()
    return stats


def load_entries(path: str) -> List[Dict[str, Any]]:
    """Load an entry JSONL file (HOLD rows are kept but ignored by the index)."""
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                rows.append(json.loads(stripped))
            except json.JSONDecodeError:
                logger.warning("Skipping invalid JSONL line %d in %s", line_number, path)
    return rows
