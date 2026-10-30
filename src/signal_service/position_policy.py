"""Position SL/TP policy for the signal producer's position-aware mode.

This module is the bridge between one open position and one proposed UPDATE:

* the tightening gate is the core's :class:`StopLossTighteningPolicy`, so a
  premature tightening is refused exactly as the historical bot would;
* widening is allowed only within ``MAX_WIDENING_RATIO`` (150%) of the original
  entry-to-SL distance, matching ``PositionManagementMixin``;
* take-profit moves are accepted only in the profit direction (monotonic);
* the dynamic TP multiplier comes from the core's regime selector.

It deliberately does not import ``TradingStrategy`` or ``PersistenceManager``:
the producer only needs the pure policy, never the core's execution state.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from src.trading.regime_risk_profile import RegimeRiskProfileSelector
from src.trading.stop_loss_tightening_policy import StopLossTighteningPolicy

#: Same widening cap as the core's PositionManagementMixin (150% of the
#: original entry-to-SL distance).
MAX_WIDENING_RATIO = 1.5


@dataclass
class PositionView:
    """Minimal position shape needed by the tightening policy (duck-typed)."""

    entry_price: float
    stop_loss: float
    take_profit: float
    direction: str  # "LONG" | "SHORT"


@dataclass
class UpdateDecision:
    """Outcome of evaluating a proposed UPDATE."""

    allowed: bool
    stop_loss: Optional[float]
    take_profit: Optional[float]
    reason: str
    is_tightening: bool = False
    widening_capped: bool = False

    @property
    def changed(self) -> bool:
        """True when at least one level is proposed for a change."""
        return self.stop_loss is not None or self.take_profit is not None


def direction_for_side(side: str) -> str:
    """Map a contract side to the core direction label."""
    return "SHORT" if str(side).lower() == "short" else "LONG"


def is_tightening(direction: str, current_sl: float, proposed_sl: float) -> bool:
    """True when ``proposed_sl`` moves toward the entry (profit direction)."""
    if direction == "LONG":
        return proposed_sl > current_sl
    return proposed_sl < current_sl


def is_tp_improvement(direction: str, current_tp: float, proposed_tp: float) -> bool:
    """True when ``proposed_tp`` moves in the profit direction."""
    if direction == "LONG":
        return proposed_tp > current_tp
    return proposed_tp < current_tp


def widening_allowed(position: PositionView, proposed_sl: float) -> bool:
    """Allow a widening stop only within the core's 150% distance cap."""
    original_distance = abs(position.entry_price - position.stop_loss)
    proposed_distance = abs(position.entry_price - proposed_sl)
    if original_distance <= 0:
        return False
    return proposed_distance <= original_distance * MAX_WIDENING_RATIO


def evaluate_update(
    position: PositionView,
    proposed_sl: Optional[float],
    proposed_tp: Optional[float],
    current_price: Optional[float],
    tf_minutes: int,
    policy: Optional[StopLossTighteningPolicy] = None,
    brain_thresholds: Optional[dict[str, Any]] = None,
) -> UpdateDecision:
    """Decide whether a proposed SL/TP update is compliant.

    Args:
        position: Current position levels and direction.
        proposed_sl: Requested stop-loss, or None to leave it unchanged.
        proposed_tp: Requested take-profit, or None to leave it unchanged.
        current_price: Real price used to measure progress toward TP.
        tf_minutes: Position timeframe in minutes (tightening thresholds).
        policy: Injected tightening policy (defaults to the core defaults).
        brain_thresholds: Optional learned thresholds passed to the policy.

    Returns:
        An :class:`UpdateDecision`; ``allowed=False`` means "do not emit".
    """
    policy = policy or StopLossTighteningPolicy()
    direction = position.direction

    new_sl: Optional[float] = None
    new_tp: Optional[float] = None
    is_tight = False
    widening_capped = False
    reasons: list[str] = []

    if proposed_sl is not None and proposed_sl != position.stop_loss:
        if proposed_sl <= 0:
            return UpdateDecision(False, None, None, "stop_loss must be positive")
        if is_tightening(direction, position.stop_loss, proposed_sl):
            evaluation = policy.evaluate_update(
                position=position,
                proposed_sl=proposed_sl,
                current_price=current_price or 0.0,
                tf_minutes=tf_minutes,
                brain_thresholds=brain_thresholds,
            )
            if not evaluation.allowed:
                return UpdateDecision(False, None, None, evaluation.reason)
            is_tight = True
            new_sl = proposed_sl
            reasons.append(evaluation.reason)
        else:
            # Widening: allowed by the core, but never beyond 150% of the
            # original distance. This is the only non-monotonic SL move.
            if widening_allowed(position, proposed_sl):
                new_sl = proposed_sl
                widening_capped = True
                reasons.append("widening allowed within 150% of the original SL distance")
            else:
                return UpdateDecision(
                    False,
                    None,
                    None,
                    "widening exceeds 150% of the original SL distance",
                )

    if proposed_tp is not None and proposed_tp != position.take_profit:
        if proposed_tp <= 0:
            return UpdateDecision(False, new_sl, None, "take_profit must be positive")
        if is_tp_improvement(direction, position.take_profit, proposed_tp):
            new_tp = proposed_tp
            reasons.append("take-profit improved in the profit direction")
        else:
            # Monotonic TP: a move against the position is refused.
            reasons.append("take-profit move is not an improvement; ignored")

    if new_sl is None and new_tp is None:
        reason = "; ".join(reasons) if reasons else "no change proposed"
        return UpdateDecision(False, None, None, reason, is_tight, widening_capped)
    return UpdateDecision(
        True, new_sl, new_tp, "; ".join(reasons), is_tight, widening_capped
    )


def dynamic_take_profit(
    direction: str,
    entry_price: float,
    atr: float,
    atr_percentage: Optional[float] = None,
    choppiness: Optional[float] = None,
    selector: Optional[RegimeRiskProfileSelector] = None,
) -> float:
    """Dynamic TP from the core regime selector (ATR * profile multiplier).

    Used when the model does not propose a take-profit but ATR is available.
    """
    if atr <= 0 or entry_price <= 0:
        raise ValueError("entry_price and atr must be positive")
    selector = selector or RegimeRiskProfileSelector()
    profile = selector.select_profile(
        choppiness=choppiness,
        atr_percentage=atr_percentage if atr_percentage is not None else (atr / entry_price) * 100,
    )
    distance = atr * selector.get_tp_multiplier(profile)
    if direction == "LONG":
        return entry_price + distance
    return entry_price - distance
