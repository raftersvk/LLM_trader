"""Position-aware replay: reuse entry signals, manage open trades only.

This is the missing half of ``replay``. ``replay`` analyses every candle
statelessly and never produces v1.1 UPDATE/CLOSE records, so a Freqtrade
backtest can only run the entry signal's static SL/TP. This runner keeps the
existing entry signals untouched and calls the LLM **only on candles where a
simulated position is open**, writing UPDATE/CLOSE records for the
``signal_updates_dir`` channel.

Fills mirror SignalStrategy as closely as possible: entry only when the
consumer's own gate accepts it (signal window/expiry, side capability,
:func:`entry_price_gate` at the real open, i.e. bracket + RR), stoploss intrabar,
take-profit on the close, TTL on the open. A TTL exit also writes a CLOSE record
(``signal_ttl``) so the simulated holding horizon is explicit; SL/TP exits are
left to the consumer, which computes them from the same levels.

No order is ever placed and no position is read or mutated. The simulation
itself makes no exchange call; engine initialization inherited from ``replay``
may still perform a public ``load_markets`` request.

Identifiers are namespaced with ``replay:`` so they can never collide with the
market-scoped live ids written by ``position_sync`` (``spot:<trade_id>`` /
``futures:<trade_id>``). When the caller knows which market it replays, pass
``market`` and the position id becomes ``replay:<entry>`` or
``<market>:replay:<entry>``; the consumer's entry alias
(``entry:<based_on_entry>``) then attaches these updates to the backtest trade
whose ``enter_tag`` is the entry decision id. ``based_on_entry`` therefore stays
the real entry decision id. The update ``decision_id`` is
``<based_on_entry>-u<revision>``; a spot and a futures replay of the same pair
must still write to separate ``--output`` directories to avoid deduplicating
each other.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from src.signal_service.analyzer_adapter import PairAnalyzer, ReplayDataInjector, UpdateContext
from src.signal_service.contract import (
    Signal,
    iso_z,
    parse_iso,
    timeframe_seconds,
)
from src.signal_service.positions import PositionSnapshot, format_position_context
from src.signal_service.storage import UpdateStorage

logger = logging.getLogger(__name__)

#: Mirrors ``signal_logic.SIGNAL_MIN_RR`` (the consumer's real-fill RR gate).
DEFAULT_MIN_RR = 1.5
#: Mirrors ``SignalStrategy._max_hold_delta``: the consumer never holds < 1h.
MIN_HOLD_HOURS = 1.0


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

    @property
    def is_short(self) -> bool:
        return self.side == "short"

    @property
    def decision_id(self) -> str:
        return self.enter_tag

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

    def metrics_snapshot(self, symbol: str, current_price: float) -> PositionSnapshot:
        """Snapshot whose stop is the *entry* SL, the fixed R denominator."""
        return replace(self.snapshot(symbol, current_price), stop_loss=self.entry_sl)


@dataclass
class ReplayPositionStats:
    """Counters reported at the end of a position-aware replay."""

    entries_reused: int = 0
    entries_entered: int = 0
    entries_refused: int = 0
    entries_expired: int = 0
    entries_dropped: int = 0
    candles_seen: int = 0
    candles_with_position: int = 0
    analysis_calls: int = 0
    updates: int = 0
    closes: int = 0
    no_update: int = 0
    failed: int = 0
    sl_exits: int = 0
    tp_exits: int = 0
    ttl_exits: int = 0
    llm_closes: int = 0
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


def _is_positive_finite(value: Any) -> bool:
    """True for a usable absolute price (positive finite, not a bool)."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    return math.isfinite(value) and value > 0


def _sl_tp_coherent(entry: Dict[str, Any]) -> bool:
    """Mirror the consumer's bracket check: BUY needs sl < tp, SELL tp < sl."""
    sl = entry.get("sl")
    tp = entry.get("tp")
    if not _is_positive_finite(sl) or not _is_positive_finite(tp):
        return False
    if entry.get("action") == "BUY":
        return sl < tp
    if entry.get("action") == "SELL":
        return tp < sl
    return False


def entry_price_gate(
    entry: Dict[str, Any],
    side: str,
    rate: Any,
    min_rr: float = DEFAULT_MIN_RR,
) -> Tuple[bool, str]:
    """Mirror ``signal_logic.entry_price_gate`` at the real fill price.

    The producer approves an entry from its own reference price; by the time the
    replay fills it at the next open the market may be outside the SL/TP
    bracket or the reward may have shrunk below the RR threshold. Such an entry
    would be refused by the consumer, so the replay must not open it either.
    """
    sl = entry.get("sl")
    tp = entry.get("tp")
    if not _is_positive_finite(sl) or not _is_positive_finite(tp):
        return False, "missing or invalid SL/TP"
    try:
        rate_value = float(rate)
    except (TypeError, ValueError):
        return False, "no usable entry rate"
    if not math.isfinite(rate_value) or rate_value <= 0:
        return False, "no usable entry rate"
    sl_value = float(sl)
    tp_value = float(tp)
    if str(side).lower() == "short":
        if not (tp_value < rate_value < sl_value):
            return False, (
                f"entry rate {rate_value:g} outside SELL bracket "
                f"(tp {tp_value:g} < rate < sl {sl_value:g})"
            )
        risk = sl_value - rate_value
        reward = rate_value - tp_value
    else:
        if not (sl_value < rate_value < tp_value):
            return False, (
                f"entry rate {rate_value:g} outside BUY bracket "
                f"(sl {sl_value:g} < rate < tp {tp_value:g})"
            )
        risk = rate_value - sl_value
        reward = tp_value - rate_value
    if risk <= 0 or reward <= 0:
        return False, "non-positive risk or reward at entry rate"
    rr = reward / risk
    if rr < min_rr:
        return False, f"RR {rr:.2f} below threshold {min_rr:g} at entry rate"
    return True, "ok"


def _entry_index(
    entries: Iterable[Dict[str, Any]],
    symbol: Optional[str] = None,
    timeframe: Optional[str] = None,
) -> Dict[datetime, Dict[str, Any]]:
    """Map signal candle close time to an executable BUY/SELL entry.

    Only rows the consumer could actually trade are kept: BUY/SELL,
    ``risk_approved``, a non-empty ``decision_id`` (the alias anchor), a
    coherent positive SL/TP bracket, a parseable close time and expiry, and a
    matching symbol/timeframe when the caller knows them.
    """
    index: Dict[datetime, Dict[str, Any]] = {}
    for entry in entries:
        if entry.get("action") not in ("BUY", "SELL"):
            continue
        if entry.get("risk_approved") is not True:
            continue
        decision_id = entry.get("decision_id")
        if not isinstance(decision_id, str) or not decision_id.strip():
            logger.warning("Skipping entry without a decision_id")
            continue
        entry_symbol = entry.get("symbol")
        if symbol is not None and entry_symbol is not None and entry_symbol != symbol:
            continue
        entry_timeframe = entry.get("timeframe")
        if timeframe is not None and entry_timeframe is not None and entry_timeframe != timeframe:
            logger.warning(
                "Skipping %s entry %s: timeframe %r does not match %r",
                entry_symbol or "?", decision_id, entry_timeframe, timeframe,
            )
            continue
        raw_close = entry.get("candle_close_time")
        raw_expires = entry.get("expires_at")
        if not isinstance(raw_close, str) or not isinstance(raw_expires, str):
            continue
        try:
            close_at = parse_iso(raw_close)
            expires_at = parse_iso(raw_expires)
        except ValueError:
            continue
        if expires_at <= close_at:
            continue
        if not _sl_tp_coherent(entry):
            logger.warning("Skipping entry %s: incoherent SL/TP", decision_id)
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

    Stop is intrabar (gap-filled at the open); take-profit is close-based, like
    ``custom_exit``. Candles are [timestamp_ms, open, high, low, close, volume].
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


def _scoped_position_id(position: SimPosition, market: Optional[str]) -> str:
    """``replay:<entry>`` or ``<market>:replay:<entry>`` (synthetic for C)."""
    return f"{market}:replay:{position.enter_tag}" if market else f"replay:{position.enter_tag}"


def _simulated_close(
    position: SimPosition,
    symbol: str,
    timeframe: str,
    period: timedelta,
    candle_open: datetime,
    market: Optional[str],
    reason: str,
    entry: float,
) -> Signal:
    """Build a v1.1 CLOSE for a producer-only simulated exit (TTL)."""
    revision = position.revision + 1
    return Signal(
        schema_version="1.1",
        decision_id=f"{position.enter_tag}-u{revision}",
        source="llm_trader",
        symbol=symbol,
        timeframe=timeframe,
        candle_open_time=iso_z(candle_open),
        candle_close_time=iso_z(candle_open + period),
        created_at=iso_z(candle_open),
        expires_at=iso_z(candle_open + 2 * period),
        action="CLOSE",
        confidence=0.0,
        risk_approved=False,
        sl=None,
        tp=None,
        position_size_pct=0.0,
        reasoning=f"simulated exit: {reason}",
        model="replay",
        provider="replay",
        prompt_version="replay",
        position_id=_scoped_position_id(position, market),
        position_side=position.side,
        revision=revision,
        based_on_entry=position.enter_tag,
        close_reason=reason,
        source_price=entry,
    )


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
    min_rr: float = DEFAULT_MIN_RR,
) -> ReplayPositionStats:
    """Replay one pair's open positions and write UPDATE/CLOSE records.

    Entry analysis is never re-run. The LLM is called only when a simulated
    position is open after the consumer's own entry gate accepted it, so a
    7-day backtest costs one call per open-position candle, not one per candle.

    Candles are taken from the loaded series itself (not a synthetic grid), and
    an entry is filled on the first candle opening at or after its close, before
    ``expires_at`` — a CSV gap delays the fill instead of silently dropping the
    entry. ``market`` scopes the emitted position id and enables the
    spot-is-long-only rule. ``min_rr`` mirrors the consumer's ``signal_min_rr``.
    """
    if market:
        market = market.strip().lower()
        if market not in ("spot", "futures"):
            raise ValueError(f"invalid market label: {market!r} (expected spot|futures)")
    else:
        market = None
    try:
        min_rr = float(min_rr)
    except (TypeError, ValueError):
        min_rr = DEFAULT_MIN_RR
    if not math.isfinite(min_rr) or min_rr <= 0:
        min_rr = DEFAULT_MIN_RR

    stats = ReplayPositionStats()
    by_close = _entry_index(entries, symbol=symbol, timeframe=timeframe)
    stats.entries_reused = len(by_close)
    scheduled = sorted(
        by_close.values(),
        key=lambda item: parse_iso(item.get("candle_close_time")),
    )
    period = timedelta(seconds=timeframe_seconds(timeframe))
    effective_end = _effective_end(end, as_of_cap, now)
    hold_delta: Optional[timedelta] = None
    if max_hold_hours and max_hold_hours > 0:
        hold_delta = timedelta(hours=max(float(max_hold_hours), MIN_HOLD_HOURS))

    owns_injector = injector is None
    if injector is None:
        injector = ReplayDataInjector(analyzer, candles, timestamps)
        injector.install()
    # Persisted stream: on a rerun a deduplicated proposal must be replaced by
    # the revision that is actually stored, or the simulation diverges from the
    # history it will be replayed against.
    persisted = {
        signal.decision_id: signal
        for signal in update_storage.load_all()
        if getattr(signal, "decision_id", None)
    }
    position: Optional[SimPosition] = None
    pending: Optional[Any] = None
    cursor = 0
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
            if pending is not None and position is not None:
                if pending.action == "CLOSE":
                    stats.llm_closes += 1
                    stats.closed_sim += 1
                    position = None
                else:
                    _apply_levels(position, pending)
                pending = None

            # Entry: first signal whose window [close, expires) contains this
            # open. Expired candidates are dropped and counted, never silently.
            if position is None:
                while cursor < len(scheduled):
                    candidate = scheduled[cursor]
                    close_at = parse_iso(candidate.get("candle_close_time"))
                    expires_at = parse_iso(candidate.get("expires_at"))
                    if expires_at <= candle_open:
                        stats.entries_expired += 1
                        logger.info(
                            "Entry %s expired before a fill (window ended %s)",
                            candidate.get("decision_id"), iso_z(expires_at),
                        )
                        cursor += 1
                        continue
                    if close_at > candle_open:
                        break
                    cursor += 1
                    side = "short" if candidate.get("action") == "SELL" else "long"
                    if side == "short" and market == "spot":
                        stats.entries_refused += 1
                        logger.info(
                            "Entry %s refused: shorting disabled on spot",
                            candidate.get("decision_id"),
                        )
                        continue
                    allowed, reason = entry_price_gate(candidate, side, float(row[1]), min_rr)
                    if not allowed:
                        stats.entries_refused += 1
                        logger.info(
                            "Entry %s refused at %.8g: %s",
                            candidate.get("decision_id"), float(row[1]), reason,
                        )
                        continue
                    position = SimPosition(
                        enter_tag=str(candidate["decision_id"]),
                        side=side,
                        entry=float(row[1]),
                        sl=float(candidate["sl"]),
                        tp=float(candidate["tp"]),
                        opened_at=candle_open,
                        entry_sl=float(candidate["sl"]),
                        entry_tp=float(candidate["tp"]),
                    )
                    stats.entries_entered += 1
                    break

            if position is None:
                continue
            stats.candles_with_position += 1

            # TTL is evaluated at the candle open, like custom_exit.
            if hold_delta is not None and candle_open - position.opened_at >= hold_delta:
                stats.ttl_exits += 1
                stats.closed_sim += 1
                close_signal = _simulated_close(
                    position, symbol, timeframe, period, candle_open, market,
                    "signal_ttl", position.entry,
                )
                if update_storage.append(close_signal):
                    stats.updates += 1
                    stats.closes += 1
                position = None
                continue

            exit_hit = _exit_on_candle(position, row)
            if exit_hit is not None:
                reason, _exit_price = exit_hit
                if reason == "sl":
                    stats.sl_exits += 1
                else:
                    stats.tp_exits += 1
                stats.closed_sim += 1
                position = None
                continue

            # Still open at the close: ask the LLM to manage it.
            close = float(row[4])
            position_id = _scoped_position_id(position, market)
            snapshot = position.snapshot(symbol, close)
            try:
                # The R denominator is the fixed entry SL, not a trailed stop.
                metrics = analyzer.position_metrics(
                    position.metrics_snapshot(symbol, close), as_of=as_of,
                )
            except Exception as exc:  # pylint: disable=broad-exception-caught
                logger.debug("Position metrics unavailable for %s: %s", symbol, exc)
                metrics = {}
            ctx = UpdateContext(
                position_id=position_id,
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
            applied = signal
            if stored:
                stats.updates += 1
                if signal.action == "CLOSE":
                    stats.closes += 1
                persisted[signal.decision_id] = signal
            else:
                # Rerun: drive the simulation from the persisted revision, not
                # the fresh (and dropped) proposal, so later revisions are
                # rebuilt on the levels the consumer will actually apply.
                applied = persisted.get(signal.decision_id, signal)
                if applied is signal:
                    logger.warning(
                        "Deduplicated update %s is absent from storage; using the new proposal",
                        signal.decision_id,
                    )
            pending = applied
            stats.details.append(
                {
                    "position": position_id,
                    "candle": iso_z(candle_open),
                    "action": applied.action,
                    "sl": applied.sl,
                    "tp": applied.tp,
                    "stored": stored,
                    "reasoning": (applied.reasoning or "")[:200],
                }
            )

        stats.entries_dropped += max(0, len(scheduled) - cursor)
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
