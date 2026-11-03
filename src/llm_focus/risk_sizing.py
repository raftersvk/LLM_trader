"""Risk/volatility-based position sizing for the LLM-focus v2 layer.

The v2 pipeline derives the position size from a fixed risk budget and the
validated stop distance instead of trusting the LLM's confidence:

.. code-block:: text

    size_pct = risk_per_trade_pct / sl_distance_pct

Both inputs are percentages, so ``0.25 / 2.0 = 0.125`` means 12.5% of capital.
There is **no fixed per-trade percentage cap** by default: the size is only
braked by

* an optional canonical per-pair ``max_stake`` (quote currency),
* the portfolio exposure ceiling (``max_portfolio_exposure_pct``), either by
  scaling the new position down or by rejecting it,
* the account limits: spot rejects above 100% of capital and futures rejects
  above ``max_leverage`` (``leverage_exceeded``),
* the ``min_size_pct`` floor (``size_below_min``) and an invalid-stop rejection.

Everything here is pure and flag-gated by the caller: the legacy
confidence-based size stays in use while ``llm_focus_v2`` is off.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any

from src.config.regime_taxonomy import (
    EXPOSURE_MODE_REJECT,
    RegimeTaxonomyConfig,
    RiskSettings,
)
from src.llm_focus.flags import llm_focus_v2_enabled

LOGGER = logging.getLogger(__name__)

TRADING_MODE_SPOT = "spot"
TRADING_MODE_FUTURES = "futures"
VALID_TRADING_MODES = (TRADING_MODE_SPOT, TRADING_MODE_FUTURES)

#: The validated SL distance is absent, zero or negative: no size is derived.
REASON_INVALID_STOP_DISTANCE = "invalid_stop_distance"
#: Risk sizing produced less than the configured minimum tradable size.
REASON_SIZE_BELOW_MIN = "size_below_min"
#: The new position would breach the portfolio exposure ceiling (reject mode).
REASON_PORTFOLIO_EXPOSURE_EXCEEDED = "portfolio_exposure_exceeded"
#: Futures sizing needs more leverage than the configured ceiling.
REASON_LEVERAGE_EXCEEDED = "leverage_exceeded"
#: Spot sizing needs more than 100% of capital.
REASON_SPOT_EXPOSURE_EXCEEDED = "spot_exposure_exceeded"

CAP_PAIR_STAKE = "pair_max_stake"
CAP_PORTFOLIO_SCALE_DOWN = "portfolio_scale_down"


@dataclass(frozen=True)
class PositionSizeDecision:
    """Deterministic outcome of risk/volatility position sizing."""

    approved: bool
    size_pct: float
    requested_size_pct: float
    reason: str | None
    risk_per_trade_pct: float
    base_risk_per_trade_pct: float
    regime_multiplier: float
    sl_distance_pct: float
    min_size_pct: float
    trading_mode: str
    max_portfolio_exposure_pct: float
    exposure_mode: str
    current_exposure_pct: float = 0.0
    resulting_exposure_pct: float = 0.0
    leverage: float = 1.0
    capped_by: str | None = None
    risk_amount_pct: float = 0.0
    risk_amount_currency: float | None = None


def risk_sizing_enabled(config: Any) -> bool:
    """Return whether the existing ``llm_focus_v2`` flag enables this sizing."""
    return llm_focus_v2_enabled(config)


def resolve_pair_max_stake(
    taxonomy_config: RegimeTaxonomyConfig | None,
    symbol: str | None,
) -> float | None:
    """Return the canonical per-pair absolute stake cap, when one is set.

    The W3 ``pair_overrides.max_stake`` value is a quote-currency amount, so
    callers must pass ``capital`` to :func:`compute_risk_based_size` for it to
    act as a cap.
    """
    if taxonomy_config is None or not symbol:
        return None
    value = taxonomy_config.pair_override(symbol).get("max_stake")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) and number > 0 else None


def _normalize_trading_mode(trading_mode: str | None) -> str:
    mode = str(trading_mode or TRADING_MODE_SPOT).strip().lower()
    if mode not in VALID_TRADING_MODES:
        raise ValueError("trading_mode must be 'spot' or 'futures'")
    return mode


def _is_positive_finite(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and value > 0
    )


def _exposure_pct(value: Any) -> float:
    if value is None:
        return 0.0
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("current_exposure_pct must be a non-negative finite number")
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise ValueError("current_exposure_pct must be a non-negative finite number")
    return number


def compute_risk_based_size(
    sl_distance_pct: Any,
    settings: RiskSettings | None = None,
    *,
    capital: float | None = None,
    trading_mode: str | None = TRADING_MODE_SPOT,
    pair_max_stake: float | None = None,
    current_exposure_pct: float = 0.0,
    symbol: str | None = None,
    regime: str | None = None,
) -> PositionSizeDecision:
    """Derive a position size from the risk budget and the stop distance.

    Args:
        sl_distance_pct: Validated stop distance in percent of entry price.
        settings: Effective risk policy (built-in defaults when omitted).
        capital: Current capital in quote currency, used for the optional
            absolute per-pair stake cap and the reported dollar risk.
        trading_mode: ``spot`` rejects sizes above 100% of capital; ``futures``
            allows leverage only up to ``settings.max_leverage``.
        pair_max_stake: Optional absolute per-pair stake cap in quote currency
            (the canonical W3 ``pair_overrides.max_stake``). Ignored when
            ``capital`` is unknown because it cannot be converted to a fraction.
        current_exposure_pct: Sum of existing open position sizes, in percent of
            capital. The new size must fit within the portfolio ceiling.
        symbol: Signal symbol, used to resolve an optional per-pair risk level.
        regime: Regime label, used to resolve an optional risk multiplier.

    Returns:
        A :class:`PositionSizeDecision`. ``approved`` is false with an explicit
        ``reason`` when the trade must be skipped.
    """
    effective = settings or RiskSettings()
    mode = _normalize_trading_mode(trading_mode)
    current_exposure = _exposure_pct(current_exposure_pct)
    ceiling_pct = effective.portfolio_exposure_pct(mode)
    min_fraction = effective.min_size_fraction()
    base_risk = effective.risk_for_symbol(symbol)
    regime_multiplier = effective.regime_multiplier(regime)
    risk_level = base_risk * regime_multiplier

    def decision(
        *,
        approved: bool,
        size: float,
        requested: float,
        reason: str | None,
        stop_distance: float,
        capped_by: str | None = None,
        leverage: float = 1.0,
    ) -> PositionSizeDecision:
        risk_pct = max(0.0, size) * stop_distance
        return PositionSizeDecision(
            approved=approved,
            size_pct=max(0.0, size),
            requested_size_pct=requested,
            reason=reason,
            risk_per_trade_pct=risk_level,
            base_risk_per_trade_pct=base_risk,
            regime_multiplier=regime_multiplier,
            sl_distance_pct=stop_distance,
            min_size_pct=effective.min_size_pct,
            trading_mode=mode,
            max_portfolio_exposure_pct=ceiling_pct,
            exposure_mode=effective.exposure_mode,
            current_exposure_pct=current_exposure,
            resulting_exposure_pct=current_exposure + max(0.0, size) * 100.0,
            leverage=leverage,
            capped_by=capped_by,
            risk_amount_pct=risk_pct,
            risk_amount_currency=_currency_risk(capital, risk_pct),
        )

    if not _is_positive_finite(sl_distance_pct):
        return decision(
            approved=False,
            size=0.0,
            requested=0.0,
            reason=REASON_INVALID_STOP_DISTANCE,
            stop_distance=0.0,
        )

    stop_distance = float(sl_distance_pct)
    requested = risk_level / stop_distance
    size = requested
    capped_by: str | None = None

    # Optional canonical per-pair absolute stake cap.
    if _is_positive_finite(pair_max_stake) and _is_positive_finite(capital):
        pair_fraction = float(pair_max_stake) / float(capital)
        if size > pair_fraction:
            size = pair_fraction
            capped_by = CAP_PAIR_STAKE

    # Per-trade account sanity: never invent leverage on spot, never exceed the
    # futures leverage ceiling.
    if mode == TRADING_MODE_SPOT:
        if size > 1.0:
            return decision(
                approved=False,
                size=size,
                requested=requested,
                reason=REASON_SPOT_EXPOSURE_EXCEEDED,
                stop_distance=stop_distance,
                capped_by=capped_by,
                leverage=1.0,
            )
    elif size > effective.max_leverage:
        return decision(
            approved=False,
            size=size,
            requested=requested,
            reason=REASON_LEVERAGE_EXCEEDED,
            stop_distance=stop_distance,
            capped_by=capped_by,
            leverage=size,
        )

    # Portfolio-level guard: the real brake in the absence of a per-trade cap.
    room_pct = ceiling_pct - current_exposure
    size_pct = size * 100.0
    if size_pct > room_pct:
        if effective.exposure_mode == EXPOSURE_MODE_REJECT:
            return decision(
                approved=False,
                size=size,
                requested=requested,
                reason=REASON_PORTFOLIO_EXPOSURE_EXCEEDED,
                stop_distance=stop_distance,
                capped_by=capped_by,
                leverage=max(1.0, size),
            )
        scaled = max(0.0, room_pct) / 100.0
        LOGGER.info(
            "Portfolio exposure scale-down: requested=%.2f%% open=%.2f%% "
            "ceiling=%.2f%% scaled=%.2f%%",
            size_pct,
            current_exposure,
            ceiling_pct,
            scaled * 100.0,
        )
        size = scaled
        capped_by = CAP_PORTFOLIO_SCALE_DOWN

    if size < min_fraction:
        return decision(
            approved=False,
            size=size,
            requested=requested,
            reason=REASON_SIZE_BELOW_MIN,
            stop_distance=stop_distance,
            capped_by=capped_by,
            leverage=max(1.0, size),
        )

    leverage = max(1.0, size) if mode == TRADING_MODE_FUTURES else 1.0
    return decision(
        approved=True,
        size=size,
        requested=requested,
        reason=None,
        stop_distance=stop_distance,
        capped_by=capped_by,
        leverage=leverage,
    )


def _currency_risk(capital: float | None, risk_pct: float) -> float | None:
    if not _is_positive_finite(capital):
        return None
    return float(capital) * risk_pct / 100.0
