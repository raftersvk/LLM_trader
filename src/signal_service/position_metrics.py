"""Entry context and live position metrics for the position-aware prompt.

Everything here is advisory prompt material. It never changes the signal
contract: the adapter renders these blocks in front of the existing open
position block, and the model still answers UPDATE/CLOSE.

Risk multiples (``pnl_R``/``mfe_R``/``mae_R``) are always measured against the
ORIGINAL entry-signal stop-loss (``entry_sl``), never against the current stop
that may already have been tightened by a previous UPDATE. When the entry
signal cannot be resolved there is no R basis, so the R fields stay ``None``
instead of silently falling back to the current stop.

MFE/MAE use only candle extrema that are fully closed at the analysis time.
The still-forming candle is excluded, so the model can never see a high/low
that happened after the decision point.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Mapping, Optional, Sequence

from src.config.regime_taxonomy import (
    RegimeTaxonomyConfig,
    get_taxonomy_config,
)
from src.signal_service.contract import parse_iso, timeframe_seconds


@dataclass(frozen=True)
class EntryContext:  # pylint: disable=too-many-instance-attributes
    """The entry signal that opened the live position, joined by decision id."""

    setup: Optional[str] = None
    regime: Optional[str] = None
    htf_alignment: Optional[str] = None
    confidence: Optional[float] = None
    reasoning: Optional[str] = None
    invalidation: Optional[str] = None
    sl: Optional[float] = None
    tp: Optional[float] = None
    created_at: Optional[str] = None

    @classmethod
    def from_signal(
        cls,
        signal: Any,
        taxonomy_config: Optional[RegimeTaxonomyConfig] = None,
    ) -> "EntryContext":
        """Build the entry context from a stored entry ``Signal``.

        ``invalidation`` is the taxonomy wording for the entry setup; it is
        omitted when the setup is unknown or the config has no definition.
        """
        setup = getattr(signal, "setup", None)
        invalidation = None
        if setup:
            try:
                config = taxonomy_config or get_taxonomy_config()
                definition = config.setups.get(setup)
                if definition is not None:
                    invalidation = definition.invalidation
            except Exception:  # pylint: disable=broad-exception-caught
                # The entry thesis must never block position management: a
                # broken taxonomy document only drops the optional wording.
                invalidation = None
        return cls(
            setup=setup,
            regime=getattr(signal, "regime", None),
            htf_alignment=getattr(signal, "htf_alignment", None),
            confidence=_finite(getattr(signal, "confidence", None)),
            reasoning=getattr(signal, "reasoning", None),
            invalidation=invalidation,
            sl=_finite(getattr(signal, "sl", None)),
            tp=_finite(getattr(signal, "tp", None)),
            created_at=getattr(signal, "created_at", None),
        )


@dataclass(frozen=True)
class PositionMetrics:  # pylint: disable=too-many-instance-attributes
    """Live position measurements rendered into the prompt."""

    current_price: Optional[float] = None
    pnl_r: Optional[float] = None
    mfe_r: Optional[float] = None
    mae_r: Optional[float] = None
    mfe_pct: Optional[float] = None
    mae_pct: Optional[float] = None
    atr: Optional[float] = None
    atr_pct: Optional[float] = None
    hours_in_trade: Optional[float] = None
    risk_basis: Optional[float] = None


def build_position_metrics(  # pylint: disable=too-many-arguments
    *,
    entry_price: Optional[float],
    entry_sl: Optional[float],
    side: str,
    current_price: Optional[float] = None,
    candles: Any = None,
    timestamps: Optional[Sequence[Any]] = None,
    opened_at: Optional[str] = None,
    as_of: Optional[datetime] = None,
    timeframe: str = "1h",
    atr: Optional[float] = None,
    atr_pct: Optional[float] = None,
) -> PositionMetrics:
    """Compute live metrics from closed candles and the original entry SL.

    Args:
        entry_price: Position entry price.
        entry_sl: Stop-loss of the entry signal. This is the R basis; the
            current stop-loss is deliberately not used.
        side: ``long`` or ``short``.
        current_price: Latest price reported by the position snapshot.
        candles: OHLCV rows (sequence or NumPy array).
        timestamps: Optional candle open times aligned with ``candles``.
        opened_at: Position open time; candles opened before it are ignored.
        as_of: Analysis time. Candles not fully closed at ``as_of`` are ignored.
        timeframe: Candle timeframe label.
        atr: ATR of the decision timeframe, when known.
        atr_pct: ATR percent supplied by the collector, when known.

    Returns:
        A ``PositionMetrics`` instance; missing inputs degrade to ``None``.
    """
    entry = _finite(entry_price)
    risk_price = _finite(entry_sl)
    price = _finite(current_price)
    highs, lows = _closed_highs_lows(candles, timestamps, opened_at, as_of, timeframe)
    if price is None:
        price = _last_close(candles, timestamps, as_of, timeframe)
    if price is None:
        price = entry
    if atr is None:
        atr = None

    risk = None
    if entry is not None and risk_price is not None:
        risk = abs(entry - risk_price)
        if risk <= 0 or not math.isfinite(risk):
            risk = None

    if entry is None or price is None:
        return PositionMetrics(
            current_price=price,
            atr=_finite(atr),
            atr_pct=_atr_pct(atr, atr_pct, price),
            hours_in_trade=_hours_in_trade(opened_at, as_of),
            risk_basis=None,
        )

    if side == "short":
        pnl_pct = (entry - price) / entry * 100.0
        mfe_pct = _pct_from_move(entry, min([entry, *lows] + [price]) if lows else min(entry, price), short=True)
        mae_pct = _pct_from_move(entry, max([entry, *highs] + [price]) if highs else max(entry, price), short=True)
    else:
        pnl_pct = (price - entry) / entry * 100.0
        mfe_pct = _pct_from_move(entry, max([entry, *highs] + [price]) if highs else max(entry, price), short=False)
        mae_pct = _pct_from_move(entry, min([entry, *lows] + [price]) if lows else min(entry, price), short=False)

    pnl_r = _r_multiple(pnl_pct, entry, risk)
    mfe_r = _r_multiple(mfe_pct, entry, risk)
    mae_r = _r_multiple(mae_pct, entry, risk)
    return PositionMetrics(
        current_price=price,
        pnl_r=pnl_r,
        mfe_r=mfe_r,
        mae_r=abs(mae_r) if mae_r is not None else None,
        mfe_pct=mfe_pct,
        mae_pct=mae_pct,
        atr=_finite(atr),
        atr_pct=_atr_pct(atr, atr_pct, price),
        hours_in_trade=_hours_in_trade(opened_at, as_of),
        risk_basis=risk_price,
    )


def format_entry_context_block(entry: Optional[EntryContext]) -> str:
    """Render the ``ENTRY CONTEXT`` prompt block.

    Returns an explicit ``unavailable`` block when the entry signal could not
    be joined, so the model knows the thesis is missing instead of inferring
    one from the current position.
    """
    if entry is None:
        return (
            "## ENTRY CONTEXT\n"
            "unavailable: no entry signal matched this position's enter_tag/decision_id\n"
            "Treat setup/regime/invalidation as unknown; manage from live price action only."
        )
    return (
        "## ENTRY CONTEXT (entry signal, read-only)\n"
        f"setup={_text(entry.setup)} entry_regime={_text(entry.regime)} "
        f"entry_htf_alignment={_text(entry.htf_alignment)} "
        f"entry_confidence={_fmt(entry.confidence, 3)}\n"
        f"entry_sl={_num(entry.sl, 8)} entry_tp={_num(entry.tp, 8)} "
        f"entry_created_at={_text(entry.created_at)}\n"
        f"invalidation={_text(entry.invalidation)}\n"
        f"entry_reasoning={_text(_clip(entry.reasoning, 800))}"
    )


def format_position_metrics_block(metrics: PositionMetrics) -> str:
    """Render the live ``POSITION METRICS`` prompt block."""
    basis = _num(metrics.risk_basis, 8)
    return (
        "## POSITION METRICS (live)\n"
        f"current_price={_num(metrics.current_price, 8)} pnl_R={_fmt(metrics.pnl_r, 2)} "
        f"mfe_R={_fmt(metrics.mfe_r, 2)} mae_R={_fmt(metrics.mae_r, 2)}\n"
        f"mfe_pct={_fmt(metrics.mfe_pct, 2)} mae_pct={_fmt(metrics.mae_pct, 2)} "
        f"atr={_num(metrics.atr, 8)} atr_pct={_fmt(metrics.atr_pct, 2)} "
        f"hours_in_trade={_fmt(metrics.hours_in_trade, 1)}\n"
        f"R basis=entry_sl={basis} (original stop, not the current stop)"
    )


def format_regime_comparison_block(
    entry_regime: Optional[str],
    snapshot: Optional[Mapping[str, Any]],
) -> str:
    """Render only the entry/current regime comparison.

    The prompt already contains the single ``REGIME SNAPSHOT (deterministic)``
    block with the current regime and its allowed setups, rendered from the
    freshly collected context. This block must not repeat those values: it
    only reports whether the entry regime still matches that snapshot.
    """
    current_regime = snapshot.get("regime") if snapshot else None
    if entry_regime and current_regime:
        match = "yes" if entry_regime == current_regime else "no"
    else:
        match = "unknown"
    return (
        "## POSITION REGIME CHECK\n"
        f"entry_regime={_text(entry_regime)} regime_match={match}\n"
        "Current regime and allowed_setups are in the REGIME SNAPSHOT block above."
    )


# ------------------------------------------------------------------ internals
def _finite(value: Any) -> Optional[float]:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _as_datetime(value: Any) -> Optional[datetime]:
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if isinstance(value, (int, float)):
        seconds = float(value) / 1000.0 if float(value) > 1e11 else float(value)
        try:
            return datetime.fromtimestamp(seconds, tz=timezone.utc)
        except (OSError, OverflowError, ValueError):
            return None
    try:
        return parse_iso(str(value))
    except (TypeError, ValueError):
        return None


def _row_values(row: Any) -> Optional[list[Any]]:
    if isinstance(row, Mapping):
        return [row.get(key) for key in ("timestamp", "open", "high", "low", "close", "volume")]
    try:
        return list(row)
    except TypeError:
        return None


def _row_open_time(
    row: Any,
    timestamps: Optional[Sequence[Any]],
    index: int,
) -> Optional[datetime]:
    if timestamps is not None:
        try:
            if index < len(timestamps):
                return _as_datetime(timestamps[index])
        except TypeError:
            pass
    values = _row_values(row)
    if values and values[0] is not None:
        return _as_datetime(values[0])
    return None


def _row_high_low(row: Any) -> tuple[Optional[float], Optional[float]]:
    values = _row_values(row)
    if not values or len(values) < 4:
        return None, None
    return _finite(values[2]), _finite(values[3])


def _row_close(row: Any) -> Optional[float]:
    values = _row_values(row)
    if not values:
        return None
    if len(values) >= 5:
        return _finite(values[4])
    return _finite(values[-1])


def _closed_highs_lows(
    candles: Any,
    timestamps: Optional[Sequence[Any]],
    opened_at: Optional[str],
    as_of: Optional[datetime],
    timeframe: str,
) -> tuple[list[float], list[float]]:
    """Return highs/lows of closed, post-entry candles.

    A candle that opened before the entry is dropped even when it closed after
    it: its high/low may predate the fill, so excluding it keeps MFE/MAE
    conservative instead of crediting pre-entry movement. Likewise a candle
    that is not fully closed at ``as_of`` is excluded (no look-ahead).
    """
    highs: list[float] = []
    lows: list[float] = []
    if candles is None:
        return highs, lows
    try:
        rows = list(candles)
    except TypeError:
        return highs, lows
    horizon = timedelta(seconds=timeframe_seconds(timeframe))
    opened = _as_datetime(opened_at) if opened_at else None
    for index, row in enumerate(rows):
        open_at = _row_open_time(row, timestamps, index)
        if open_at is not None and as_of is not None and open_at + horizon > as_of:
            # Still forming at the decision time: exclude (no look-ahead).
            continue
        if open_at is not None and opened is not None and open_at < opened:
            # The whole candle opened before the position: pre-entry excursion.
            continue
        high, low = _row_high_low(row)
        if high is not None:
            highs.append(high)
        if low is not None:
            lows.append(low)
    return highs, lows


def _last_close(
    candles: Any,
    timestamps: Optional[Sequence[Any]],
    as_of: Optional[datetime],
    timeframe: str,
) -> Optional[float]:
    if candles is None:
        return None
    try:
        rows = list(candles)
    except TypeError:
        return None
    horizon = timedelta(seconds=timeframe_seconds(timeframe))
    for index in range(len(rows) - 1, -1, -1):
        open_at = _row_open_time(rows[index], timestamps, index)
        if open_at is not None and as_of is not None and open_at + horizon > as_of:
            continue
        return _row_close(rows[index])
    return None


def _pct_from_move(entry: float, extreme: Optional[float], short: bool) -> Optional[float]:
    if extreme is None or entry == 0.0:
        return None
    if short:
        return (entry - extreme) / entry * 100.0
    return (extreme - entry) / entry * 100.0


def _r_multiple(pct: Optional[float], entry: Optional[float], risk: Optional[float]) -> Optional[float]:
    if pct is None or entry is None or not risk:
        return None
    return (pct / 100.0 * entry) / risk


def _atr_pct(atr: Any, supplied: Any, price: Optional[float]) -> Optional[float]:
    value = _finite(supplied)
    if value is not None:
        return value
    atr_value = _finite(atr)
    if atr_value is None or not price:
        return None
    return atr_value / price * 100.0


def _hours_in_trade(opened_at: Optional[str], as_of: Optional[datetime]) -> Optional[float]:
    opened = _as_datetime(opened_at) if opened_at else None
    if opened is None or as_of is None:
        return None
    return max(0.0, (as_of - opened).total_seconds() / 3600.0)


def _clip(value: Any, limit: int) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _text(value: Any) -> str:
    text = _clip(value, 800)
    return text if text else "unknown"


def _num(value: Optional[float], digits: int) -> str:
    if value is None:
        return "unknown"
    return f"{value:.{digits}f}".rstrip("0").rstrip(".")


def _fmt(value: Optional[float], digits: int) -> str:
    if value is None:
        return "na"
    return f"{value:.{digits}f}"
