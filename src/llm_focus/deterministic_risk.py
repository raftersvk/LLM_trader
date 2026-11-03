"""Deterministic regime, sizing and target-validation primitives for LLM-focus v2.

Policy C: the LLM proposes absolute stop-loss/take-profit levels and this layer
validates them (correct side, distance bounds, R/R, entry present). Invalid
levels reject the signal; they are never silently replaced by ATR targets.
The target-geometry resolver is a documented fallback for signals that carry no
levels at all. These helpers stay pure so tests can assert the boundary.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from src.config.regime_taxonomy import DEFAULT_REGIME_SETUPS
from src.llm_focus.target_geometry import resolve_target_geometry, validate_target

TRENDING_UP = "TRENDING_UP"
TRENDING_DOWN = "TRENDING_DOWN"
RANGING = "RANGING"
NO_TRADE = "NO_TRADE"
UNKNOWN = "UNKNOWN"

# Canonical P0-A labels (src/config/regime_taxonomy.py REGIMES) plus legacy
# aliases kept for older prompts/archives. Low/high volatility and unclear are
# hard no-trade regimes, not unknown ones.
_REGIME_ALIASES = {
    "TREND_UP": TRENDING_UP,
    "TREND_DOWN": TRENDING_DOWN,
    "RANGE": RANGING,
    "LOW_VOL": NO_TRADE,
    "HIGH_VOL": NO_TRADE,
    "UNCLEAR": NO_TRADE,
    "BULLISH": TRENDING_UP,
    "UP": TRENDING_UP,
    "UPTREND": TRENDING_UP,
    "BEARISH": TRENDING_DOWN,
    "DOWN": TRENDING_DOWN,
    "DOWNTREND": TRENDING_DOWN,
    "NEUTRAL": RANGING,
    "SIDEWAYS": RANGING,
}

_DEFAULT_SETUPS_FROM_TAXONOMY = {
    TRENDING_UP: DEFAULT_REGIME_SETUPS["trend_up"],
    TRENDING_DOWN: DEFAULT_REGIME_SETUPS["trend_down"],
    RANGING: DEFAULT_REGIME_SETUPS["range"],
}

DEFAULT_ALLOWED_SETUPS: dict[str, tuple[str, ...]] = {
    regime: tuple(setup.upper() for setup in setups)
    for regime, setups in _DEFAULT_SETUPS_FROM_TAXONOMY.items()
}
DEFAULT_ALLOWED_SETUPS.update({NO_TRADE: (), UNKNOWN: ()})

_CONFIDENCE_SIZE = {"HIGH": 0.03, "MEDIUM": 0.02, "LOW": 0.01}

TARGET_SOURCE_LLM = "llm"
TARGET_SOURCE_FALLBACK = "fallback_geometry"


@dataclass(frozen=True)
class RegimeAssessment:
    """Normalized P0-A regime snapshot used by the v2 layers."""

    regime: str
    allowed_setups: tuple[str, ...]
    atr_pct: float
    volatility: str


@dataclass(frozen=True)
class DeterministicRiskPlan:
    """Validated policy-C risk plan.

    ``stop_loss``/``take_profit`` are the LLM levels verbatim when provided.
    ``risk_approved=False`` means execution must be refused and
    ``rejection_reason`` explains why; no level is ever invented in that case.
    """

    direction: str
    entry_price: float | None
    stop_loss: float | None
    take_profit: float | None
    size_pct: float
    sl_distance_pct: float | None
    tp_distance_pct: float | None
    rr_ratio: float | None
    volatility_level: str
    risk_approved: bool = False
    rejection_reason: str | None = None
    target_source: str = TARGET_SOURCE_LLM


def normalize_regime(regime: Any) -> str:
    """Normalize a regime label to one of the v2 constants."""
    if regime is None:
        return UNKNOWN
    normalized = str(regime).strip().upper().replace("-", "_").replace(" ", "_")
    normalized = _REGIME_ALIASES.get(normalized, normalized)
    if normalized in DEFAULT_ALLOWED_SETUPS:
        return normalized
    return UNKNOWN


def normalize_setups(setups: Iterable[Any] | None) -> tuple[str, ...]:
    """Normalize setup names and remove duplicates while preserving order."""
    if not setups:
        return ()
    if isinstance(setups, str):
        setups = setups.split(",")
    normalized: list[str] = []
    for setup in setups:
        name = str(setup).strip().upper().replace("-", "_").replace(" ", "_")
        if name and name not in normalized:
            normalized.append(name)
    return tuple(normalized)


def classify_volatility(atr_pct: float) -> str:
    """Classify volatility with the same thresholds used by RiskManager."""
    if atr_pct > 3:
        return "HIGH"
    if atr_pct < 1.5:
        return "LOW"
    return "MEDIUM"


def assess_regime(
    regime: Any = None,
    allowed_setups: Iterable[Any] | None = None,
    atr_pct: float | None = None,
) -> RegimeAssessment:
    """Build a normalized regime assessment from P0-A inputs.

    ``allowed_setups=None`` means the field was omitted, so the defaults for the
    normalized regime are used. An explicit empty list means P0-A declared a
    hard no-trade (no-trade regime or disabled timeframe) and must stay empty
    so the shortlist rejects the candidate. Unknown regimes have no allowed
    setups, which makes the shortlist fail closed.
    """
    normalized_regime = normalize_regime(regime)
    if allowed_setups is None:
        setups = DEFAULT_ALLOWED_SETUPS[normalized_regime]
    else:
        setups = normalize_setups(allowed_setups)
    safe_atr_pct = float(atr_pct) if atr_pct is not None else 0.0
    return RegimeAssessment(
        regime=normalized_regime,
        allowed_setups=setups,
        atr_pct=safe_atr_pct,
        volatility=classify_volatility(safe_atr_pct),
    )


def assess_regime_from_mapping(p0a: Mapping[str, Any] | None) -> RegimeAssessment:
    """Build a regime assessment from a P0-A style mapping."""
    source = p0a or {}
    atr_pct = source.get("atr_pct", source.get("atr_percentage"))
    return assess_regime(
        regime=source.get("regime"),
        allowed_setups=source.get("allowed_setups"),
        atr_pct=atr_pct,
    )


def build_atr_stops(
    direction: str,
    entry_price: float,
    atr: float,
    sl_atr_multiple: float = 2.0,
    tp_atr_multiple: float = 4.0,
) -> tuple[float, float]:
    """Return deterministic ATR-based (stop_loss, take_profit)."""
    normalized_direction = str(direction).strip().upper()
    if normalized_direction not in ("LONG", "SHORT"):
        raise ValueError("direction must be LONG or SHORT")
    if entry_price <= 0:
        raise ValueError("entry_price must be positive")
    if atr <= 0:
        raise ValueError("atr must be positive")
    if sl_atr_multiple <= 0 or tp_atr_multiple <= 0:
        raise ValueError("ATR multiples must be positive")

    if normalized_direction == "LONG":
        stop_loss = entry_price - atr * sl_atr_multiple
        take_profit = entry_price + atr * tp_atr_multiple
    else:
        stop_loss = entry_price + atr * sl_atr_multiple
        take_profit = entry_price - atr * tp_atr_multiple
    return stop_loss, take_profit


def compute_position_size(confidence: Any) -> float:
    """Return deterministic size as a fraction of capital."""
    if isinstance(confidence, (int, float)):
        if confidence >= 70:
            level = "HIGH"
        elif confidence >= 50:
            level = "MEDIUM"
        else:
            level = "LOW"
    else:
        level = str(confidence).strip().upper()
    return _CONFIDENCE_SIZE.get(level, _CONFIDENCE_SIZE["MEDIUM"])


def _is_positive_finite(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and value > 0
    )


def _effective_atr_pct(entry_price: Any, atr: Any, atr_pct: Any) -> float:
    if atr_pct is not None:
        try:
            value = float(atr_pct)
        except (TypeError, ValueError):
            return 0.0
        return value if math.isfinite(value) else 0.0
    if _is_positive_finite(entry_price) and _is_positive_finite(atr):
        return float(atr) / float(entry_price) * 100
    return 0.0


def _make_risk_plan(
    *,
    direction: str,
    entry_price: float | None,
    stop_loss: Any,
    take_profit: Any,
    confidence: Any,
    atr: Any,
    atr_pct: Any,
    risk_approved: bool,
    rejection_reason: str | None,
    target_source: str,
    validation: Any = None,
) -> DeterministicRiskPlan:
    return DeterministicRiskPlan(
        direction=direction,
        entry_price=entry_price,
        stop_loss=stop_loss,
        take_profit=take_profit,
        size_pct=compute_position_size(confidence),
        sl_distance_pct=validation.sl_distance_pct if validation else None,
        tp_distance_pct=validation.tp_distance_pct if validation else None,
        rr_ratio=validation.rr_ratio if validation else None,
        volatility_level=classify_volatility(_effective_atr_pct(entry_price, atr, atr_pct)),
        risk_approved=risk_approved,
        rejection_reason=rejection_reason,
        target_source=target_source,
    )


def build_risk_plan(
    direction: str,
    entry_price: float | None,
    atr: float | None,
    confidence: Any,
    atr_pct: float | None = None,
    sl_atr_multiple: float = 2.0,
    tp_atr_multiple: float = 4.0,
    stop_loss: float | None = None,
    take_profit: float | None = None,
    min_rr: float = 1.0,
    min_distance_pct: float = 0.5,
    max_distance_pct: float = 10.0,
    setup: Any = None,
    range_high: float | None = None,
    range_low: float | None = None,
    vwap: float | None = None,
    measured_move_target: float | None = None,
) -> DeterministicRiskPlan:
    """Validate policy-C levels or build a documented fallback plan.

    When the LLM provides ``stop_loss``/``take_profit`` they are authoritative:
    they are returned verbatim when valid, and the plan is rejected with an
    explicit reason when they are invalid. They are never replaced by ATR
    values. The target-geometry/ATR fallback is used only when both levels are
    absent.
    """
    normalized_direction = str(direction or "").strip().upper()

    def rejected(reason: str, *, source: str, validation: Any = None) -> DeterministicRiskPlan:
        return _make_risk_plan(
            direction=normalized_direction or "UNKNOWN",
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            confidence=confidence,
            atr=atr,
            atr_pct=atr_pct,
            risk_approved=False,
            rejection_reason=reason,
            target_source=source,
            validation=validation,
        )

    if normalized_direction not in ("LONG", "SHORT"):
        return rejected("direction must be LONG or SHORT", source=TARGET_SOURCE_LLM)
    if not _is_positive_finite(entry_price):
        return rejected("entry price unavailable", source=TARGET_SOURCE_LLM)

    has_stop_loss = stop_loss is not None
    has_take_profit = take_profit is not None
    if has_stop_loss or has_take_profit:
        if not (has_stop_loss and has_take_profit):
            return rejected(
                "LLM stop_loss and take_profit must both be provided",
                source=TARGET_SOURCE_LLM,
            )
        validation = validate_target(
            normalized_direction,
            float(entry_price),
            stop_loss,
            take_profit,
            min_rr=min_rr,
            min_distance_pct=min_distance_pct,
            max_distance_pct=max_distance_pct,
        )
        if not validation.valid:
            return rejected("; ".join(validation.issues), source=TARGET_SOURCE_LLM, validation=validation)
        return _make_risk_plan(
            direction=normalized_direction,
            entry_price=float(entry_price),
            stop_loss=stop_loss,
            take_profit=take_profit,
            confidence=confidence,
            atr=atr,
            atr_pct=atr_pct,
            risk_approved=True,
            rejection_reason=None,
            target_source=TARGET_SOURCE_LLM,
            validation=validation,
        )

    # No LLM levels at all: documented fallback only (should not occur in
    # practice). The structure/ATR geometry module is never consulted when the
    # LLM supplied levels.
    if not _is_positive_finite(atr):
        return rejected("fallback target requires a positive ATR", source=TARGET_SOURCE_FALLBACK)

    entry = float(entry_price)
    atr_value = float(atr)
    fallback_take_profit: float | None = None
    if setup is not None:
        geometry = resolve_target_geometry(
            setup,
            direction=normalized_direction,
            entry_price=entry,
            atr=atr_value,
            range_high=range_high,
            range_low=range_low,
            vwap=vwap,
            measured_move_target=measured_move_target,
        )
        fallback_take_profit = geometry.target_price
    if fallback_take_profit is None:
        fallback_take_profit = (
            entry + atr_value * tp_atr_multiple
            if normalized_direction == "LONG"
            else entry - atr_value * tp_atr_multiple
        )
    fallback_stop_loss = (
        entry - atr_value * sl_atr_multiple
        if normalized_direction == "LONG"
        else entry + atr_value * sl_atr_multiple
    )
    validation = validate_target(
        normalized_direction,
        entry,
        fallback_stop_loss,
        fallback_take_profit,
        min_rr=min_rr,
        min_distance_pct=min_distance_pct,
        max_distance_pct=max_distance_pct,
    )
    if not validation.valid:
        return _make_risk_plan(
            direction=normalized_direction,
            entry_price=entry,
            stop_loss=fallback_stop_loss,
            take_profit=fallback_take_profit,
            confidence=confidence,
            atr=atr,
            atr_pct=atr_pct,
            risk_approved=False,
            rejection_reason="fallback target invalid: " + "; ".join(validation.issues),
            target_source=TARGET_SOURCE_FALLBACK,
            validation=validation,
        )
    return _make_risk_plan(
        direction=normalized_direction,
        entry_price=entry,
        stop_loss=fallback_stop_loss,
        take_profit=fallback_take_profit,
        confidence=confidence,
        atr=atr,
        atr_pct=atr_pct,
        risk_approved=True,
        rejection_reason=None,
        target_source=TARGET_SOURCE_FALLBACK,
        validation=validation,
    )
