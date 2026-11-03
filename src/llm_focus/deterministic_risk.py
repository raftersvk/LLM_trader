"""Deterministic regime, sizing and ATR risk primitives for LLM-focus v2.

In v2 the LLM no longer supplies position size or absolute SL/TP. These pure
helpers define the deterministic side of the contract so P0-A/JEV/strategy can
own those values and tests can assert the boundary.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from src.config.regime_taxonomy import DEFAULT_REGIME_SETUPS

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


@dataclass(frozen=True)
class RegimeAssessment:
    """Normalized P0-A regime snapshot used by the v2 layers."""

    regime: str
    allowed_setups: tuple[str, ...]
    atr_pct: float
    volatility: str


@dataclass(frozen=True)
class DeterministicRiskPlan:
    """Risk parameters owned by the deterministic layer."""

    direction: str
    entry_price: float
    stop_loss: float
    take_profit: float
    size_pct: float
    sl_distance_pct: float
    tp_distance_pct: float
    rr_ratio: float
    volatility_level: str


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


def build_risk_plan(
    direction: str,
    entry_price: float,
    atr: float,
    confidence: Any,
    atr_pct: float | None = None,
    sl_atr_multiple: float = 2.0,
    tp_atr_multiple: float = 4.0,
) -> DeterministicRiskPlan:
    """Build the full deterministic plan for a v2 qualitative signal."""
    normalized_direction = str(direction).strip().upper()
    stop_loss, take_profit = build_atr_stops(
        normalized_direction,
        entry_price,
        atr,
        sl_atr_multiple=sl_atr_multiple,
        tp_atr_multiple=tp_atr_multiple,
    )

    # Reuse the circuit-breaker bounds from the live RiskManager so the v2
    # deterministic layer cannot produce degenerate stops.
    sl_distance = abs(entry_price - stop_loss) / entry_price
    if sl_distance > 0.10:
        stop_loss = entry_price * (0.90 if normalized_direction == "LONG" else 1.10)
    elif sl_distance < 0.005:
        stop_loss = entry_price * (0.995 if normalized_direction == "LONG" else 1.005)

    if normalized_direction == "LONG":
        if stop_loss >= entry_price:
            stop_loss = entry_price - atr * sl_atr_multiple
        if take_profit <= entry_price:
            take_profit = entry_price + atr * tp_atr_multiple
    else:
        if stop_loss <= entry_price:
            stop_loss = entry_price + atr * sl_atr_multiple
        if take_profit >= entry_price:
            take_profit = entry_price - atr * tp_atr_multiple

    sl_distance_pct = abs(entry_price - stop_loss) / entry_price
    tp_distance_pct = abs(take_profit - entry_price) / entry_price
    rr_ratio = tp_distance_pct / sl_distance_pct if sl_distance_pct > 0 else 0.0
    effective_atr_pct = float(atr_pct) if atr_pct is not None else (atr / entry_price) * 100

    return DeterministicRiskPlan(
        direction=normalized_direction,
        entry_price=entry_price,
        stop_loss=stop_loss,
        take_profit=take_profit,
        size_pct=compute_position_size(confidence),
        sl_distance_pct=sl_distance_pct,
        tp_distance_pct=tp_distance_pct,
        rr_ratio=rr_ratio,
        volatility_level=classify_volatility(effective_atr_pct),
    )
