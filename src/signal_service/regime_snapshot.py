"""Deterministic per-pair regime snapshot for the signal service.

The snapshot is a pure derivation from data the analyzer already collected:
the current technical snapshot (ADX / choppiness / ATR / supertrend), the
daily long-term block, the weekly macro block and any close series the caller
already has. It never fetches data, never calls a model and never looks at a
future candle.

Output contract (internal, not part of the public signal schema):

* ``regime``          ``trend_up|trend_down|range|low_vol|high_vol|unclear``
* ``confidence``      0..1 agreement score for the selected label
* ``htf_alignment``   ``aligned|counter|neutral`` against the daily/weekly trend
* ``extension_atr``   signed distance of price from SMA20, in ATR units
* ``rs_btc_7d`` / ``rs_btc_30d``  relative strength vs BTC, in percent
* ``allowed_setups``  setup families the regime allows (P0 lists families only)

The setup taxonomy and the contract fields are deliberately out of scope for
this phase.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

from src.config.regime_taxonomy import (
    RegimeTaxonomyConfig,
    RegimeThresholds,
    get_taxonomy_config,
)


def _finite(value: Any) -> float | None:
    """Return a finite float or None (NumPy scalars included)."""
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _last(values: Sequence[float] | None) -> float | None:
    """Return the last finite value of a close series."""
    if not values:
        return None
    for value in reversed(values):
        number = _finite(value)
        if number is not None:
            return number
    return None


def _mean(values: Sequence[float]) -> float | None:
    """Arithmetic mean of the finite values, or None."""
    finite = [value for value in (_finite(item) for item in values) if value is not None]
    if not finite:
        return None
    return sum(finite) / len(finite)


def _return_pct(closes: Sequence[float] | None, periods: int) -> float | None:
    """Percent return over ``periods`` bars of a close series."""
    if not closes or len(closes) <= periods:
        return None
    start = _finite(closes[-1 - periods])
    end = _finite(closes[-1])
    if start is None or end is None or start == 0.0:
        return None
    return (end / start - 1.0) * 100.0


def _relative_strength(
    pair_hourly: Sequence[float] | None,
    pair_daily: Sequence[float] | None,
    pair_weekly: Sequence[float] | None,
    btc_hourly: Sequence[float] | None,
    btc_daily: Sequence[float] | None,
    btc_weekly: Sequence[float] | None,
    candidates: Sequence[tuple[Sequence[float] | None, Sequence[float] | None, int]],
) -> float | None:
    """Pair return minus BTC return over the first candidate series that fits."""
    for pair, btc, periods in candidates:
        pair_return = _return_pct(pair, periods)
        btc_return = _return_pct(btc, periods)
        if pair_return is not None and btc_return is not None:
            return pair_return - btc_return
    return None


def _supertrend_label(value: Any) -> str | None:
    """Normalize the numeric supertrend direction to bullish/bearish."""
    number = _finite(value)
    if number is None:
        return None
    if number > 0:
        return "bullish"
    if number < 0:
        return "bearish"
    return "neutral"


def _nested_sma(long_term_data: Mapping[str, Any], period: int) -> float | None:
    """Read a daily SMA from either the nested or the flattened layout."""
    direct = _finite(long_term_data.get(f"sma_{period}"))
    if direct is not None:
        return direct
    sma_values = long_term_data.get("sma_values")
    if isinstance(sma_values, Mapping):
        return _finite(sma_values.get(period))
    return None


def _daily_direction(
    long_term_data: Mapping[str, Any] | None,
    thresholds: RegimeThresholds,
) -> str | None:
    """Direction of the daily block: bullish, bearish or None."""
    if not long_term_data:
        return None
    plus_di = _finite(long_term_data.get("daily_plus_di"))
    minus_di = _finite(long_term_data.get("daily_minus_di"))
    adx = _finite(long_term_data.get("daily_adx"))
    if plus_di is not None and minus_di is not None and (adx is None or adx >= thresholds.adx_weak):
        if plus_di > minus_di:
            return "bullish"
        if minus_di > plus_di:
            return "bearish"
    sma_50 = _nested_sma(long_term_data, 50)
    sma_200 = _nested_sma(long_term_data, 200)
    if sma_50 is not None and sma_200 is not None and sma_50 != sma_200:
        return "bullish" if sma_50 > sma_200 else "bearish"
    macro = long_term_data.get("macro_trend")
    if isinstance(macro, Mapping):
        trend = str(macro.get("trend_direction", "")).strip().lower()
        if trend in ("bullish", "bearish"):
            return trend
    return None


def _weekly_direction(weekly_macro: Mapping[str, Any] | None) -> str | None:
    """Direction of the weekly macro block: bullish, bearish or None."""
    if not weekly_macro:
        return None
    trend = weekly_macro.get("weekly_macro_trend")
    if not isinstance(trend, Mapping):
        trend = weekly_macro
    value = str(trend.get("trend_direction", "")).strip().lower()
    if value in ("bullish", "bearish"):
        return value
    return None


def _htf_alignment(regime: str, daily: str | None, weekly: str | None) -> str:
    """Compare the higher-timeframe trend with the 1h regime direction."""
    if regime not in ("trend_up", "trend_down"):
        return "neutral"
    votes = [direction for direction in (daily, weekly) if direction in ("bullish", "bearish")]
    if not votes:
        return "neutral"
    bullish = votes.count("bullish")
    bearish = votes.count("bearish")
    if bullish == bearish:
        return "neutral"
    wanted = "bullish" if regime == "trend_up" else "bearish"
    return "aligned" if (bullish > bearish) == (wanted == "bullish") else "counter"


def _classify(
    adx: float | None,
    choppiness: float | None,
    plus_di: float | None,
    minus_di: float | None,
    last_close: float | None,
    sma20: float | None,
    atr_pct: float | None,
    atr_pct_median_90d: float | None,
    thresholds: RegimeThresholds,
) -> tuple[str, float]:
    """Return the regime label and its agreement confidence.

    Priority is fixed and ex-ante: high_vol, trend_up, trend_down, low_vol,
    range, unclear. Only closed data supplied by the caller is used. The
    numeric cut-offs come from the taxonomy config (defaults match P0-B).
    """
    high_vol_reference = thresholds.high_vol_atr_pct_floor
    if atr_pct_median_90d is not None:
        high_vol_reference = max(
            thresholds.high_vol_atr_pct_floor,
            thresholds.high_vol_atr_multiple * atr_pct_median_90d,
        )
    if (
        atr_pct is not None
        and atr_pct > high_vol_reference
        and (adx is None or adx < thresholds.adx_trend)
    ):
        return "high_vol", 0.8

    directional = plus_di is not None and minus_di is not None and plus_di != minus_di
    if (
        adx is not None
        and adx >= thresholds.adx_trend
        and directional
        and last_close is not None
        and sma20 is not None
    ):
        if plus_di > minus_di and last_close > sma20:
            return "trend_up", 1.0
        if minus_di > plus_di and last_close < sma20:
            return "trend_down", 1.0

    if (
        atr_pct is not None
        and atr_pct_median_90d is not None
        and atr_pct < thresholds.low_vol_atr_multiple * atr_pct_median_90d
        and choppiness is not None
        and choppiness > thresholds.low_vol_choppiness
    ):
        return "low_vol", 0.8

    chop_range = choppiness is not None and choppiness > thresholds.range_choppiness
    weak_adx = adx is not None and adx < thresholds.adx_weak
    if chop_range or weak_adx:
        return "range", 1.0 if (chop_range and weak_adx) else 0.7

    if adx is None and choppiness is None and atr_pct is None:
        return "unclear", 0.0
    return "unclear", 0.0


def build_regime_snapshot(
    technical_data: Mapping[str, Any] | None = None,
    long_term_data: Mapping[str, Any] | None = None,
    weekly_macro: Mapping[str, Any] | None = None,
    closes_1h: Sequence[float] | None = None,
    closes_1d: Sequence[float] | None = None,
    closes_1w: Sequence[float] | None = None,
    btc_closes_1h: Sequence[float] | None = None,
    btc_closes_1d: Sequence[float] | None = None,
    btc_closes_1w: Sequence[float] | None = None,
    atr_pct_median_90d: float | None = None,
    timeframe: str = "1h",
    symbol: str | None = None,
    taxonomy_config: RegimeTaxonomyConfig | None = None,
) -> dict[str, Any]:
    """Build a deterministic regime snapshot from already-collected data.

    Missing inputs degrade to ``None`` fields; they never raise. A snapshot is
    always returned, even when every input is missing (``regime=unclear``).
    Thresholds and allowed setups come from the taxonomy config, with optional
    per-pair overrides selected by ``symbol``. A disabled timeframe exposes no
    allowed setup and is marked no-trade.
    """
    config = (taxonomy_config or get_taxonomy_config()).for_symbol(symbol)
    thresholds = config.regime_thresholds
    technical = technical_data or {}
    adx = _finite(technical.get("adx"))
    choppiness = _finite(technical.get("choppiness"))
    atr = _finite(technical.get("atr"))
    atr_pct = _finite(technical.get("atr_percent"))
    plus_di = _finite(technical.get("plus_di"))
    minus_di = _finite(technical.get("minus_di"))
    supertrend = _supertrend_label(technical.get("supertrend_direction"))

    last_close = _last(closes_1h) or _last(closes_1d) or _last(closes_1w)
    sma20 = _finite(technical.get("sma_20"))
    if sma20 is None and closes_1h and len(closes_1h) >= 20:
        sma20 = _mean(list(closes_1h)[-20:])
    if atr_pct is None and atr is not None and last_close not in (None, 0.0):
        atr_pct = atr / last_close * 100.0

    extension_atr = None
    if last_close is not None and sma20 is not None and atr not in (None, 0.0):
        extension_atr = (last_close - sma20) / atr

    regime, confidence = _classify(
        adx,
        choppiness,
        plus_di,
        minus_di,
        last_close,
        sma20,
        atr_pct,
        _finite(atr_pct_median_90d),
        thresholds,
    )
    daily = _daily_direction(long_term_data, thresholds)
    weekly = _weekly_direction(weekly_macro)
    htf_alignment = _htf_alignment(regime, daily, weekly)

    rs_7d = _relative_strength(
        closes_1h,
        closes_1d,
        closes_1w,
        btc_closes_1h,
        btc_closes_1d,
        btc_closes_1w,
        (
            (closes_1d, btc_closes_1d, 7),
            (closes_1h, btc_closes_1h, 168),
            (closes_1w, btc_closes_1w, 1),
        ),
    )
    rs_30d = _relative_strength(
        closes_1h,
        closes_1d,
        closes_1w,
        btc_closes_1h,
        btc_closes_1d,
        btc_closes_1w,
        (
            (closes_1d, btc_closes_1d, 30),
            (closes_1h, btc_closes_1h, 720),
            (closes_1w, btc_closes_1w, 4),
        ),
    )

    allowed_setups = (
        list(config.allowed_setups(regime)) if config.timeframe_enabled(timeframe) else []
    )
    no_trade = not allowed_setups
    return {
        "timeframe": timeframe,
        "regime": regime,
        "confidence": confidence,
        "no_trade": no_trade,
        "last_close": _round(last_close, 8),
        "adx": _round(adx, 2),
        "choppiness": _round(choppiness, 2),
        "atr_pct": _round(atr_pct, 3),
        "atr_pct_median_90d": _round(_finite(atr_pct_median_90d), 3),
        "plus_di": _round(plus_di, 2),
        "minus_di": _round(minus_di, 2),
        "supertrend": supertrend,
        "daily_trend": daily,
        "weekly_trend": weekly,
        "htf_alignment": htf_alignment,
        "extension_atr": _round(extension_atr, 3),
        "rs_btc_7d": _round(rs_7d, 2),
        "rs_btc_30d": _round(rs_30d, 2),
        "allowed_setups": allowed_setups,
    }


def _round(value: float | None, digits: int) -> float | None:
    """Round a finite float, preserving None."""
    if value is None:
        return None
    return round(value, digits)


def _fmt(value: Any, digits: int = 2) -> str:
    """Format an optional number for the compact prompt block."""
    number = _finite(value)
    if number is None:
        return "na"
    return f"{number:.{digits}f}"


def format_regime_snapshot_block(snapshot: Mapping[str, Any] | None) -> str:
    """Render the compact ``REGIME SNAPSHOT`` prompt block.

    Returns an empty string when there is no snapshot so a caller can append
    the result unconditionally.
    """
    if not snapshot:
        return ""
    regime = str(snapshot.get("regime") or "unclear")
    confidence = _fmt(snapshot.get("confidence"), 2)
    adx = _fmt(snapshot.get("adx"), 1)
    chop = _fmt(snapshot.get("choppiness"), 1)
    atr_pct = _fmt(snapshot.get("atr_pct"), 2)
    htf = str(snapshot.get("htf_alignment") or "neutral")
    extension = _fmt(snapshot.get("extension_atr"), 2)
    rs7 = _fmt(snapshot.get("rs_btc_7d"), 2)
    rs30 = _fmt(snapshot.get("rs_btc_30d"), 2)
    setups = ",".join(snapshot.get("allowed_setups") or []) or "none"
    tradeable = "no" if snapshot.get("no_trade") else "yes"
    return (
        "## REGIME SNAPSHOT (deterministic)\n"
        f"regime={regime} confidence={confidence} tradeable={tradeable} "
        f"adx={adx} chop={chop} atr%={atr_pct} htf={htf}\n"
        f"extension_atr={extension} rs_btc_7d={rs7} rs_btc_30d={rs30}\n"
        f"allowed_setups={setups}"
    )
