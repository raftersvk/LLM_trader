"""Position SL/TP policy for the signal producer's position-aware mode.

This module is the bridge between one open position and one proposed UPDATE:

* the tightening gate is the core's :class:`StopLossTighteningPolicy`, so a
  premature tightening is refused exactly as the historical bot would;
* a hard price guard refuses a stop on the wrong side of the current price,
  which would otherwise trigger an immediate exit;
* widening is allowed only within ``MAX_WIDENING_RATIO`` (150%) of the
  *original* entry-to-SL distance, matching ``PositionManagementMixin`` and the
  Freqtrade consumer;
* take-profit moves are accepted only in the profit direction (monotonic).

It deliberately does not import ``TradingStrategy`` or ``PersistenceManager``:
the producer only needs the pure policy, never the core's execution state.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Optional

from src.trading.stop_loss_tightening_policy import StopLossTighteningPolicy

#: Same widening cap as the core's PositionManagementMixin (150% of the
#: original entry-to-SL distance).
MAX_WIDENING_RATIO = 1.5


@dataclass
class PositionView:
    """Minimal position shape needed by the tightening policy (duck-typed)."""

    entry_price: float
    stop_loss: float
    take_profit: Optional[float]
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


def is_tp_improvement(
    direction: str, current_tp: Optional[float], proposed_tp: float
) -> bool:
    """True when ``proposed_tp`` moves in the profit direction.

    With no known current TP there is nothing to improve on: the price guard
    and final-coherence check still apply.
    """
    if current_tp is None:
        return True
    if direction == "LONG":
        return proposed_tp > current_tp
    return proposed_tp < current_tp


def widening_allowed(
    position: PositionView, proposed_sl: float, original_sl: Optional[float] = None
) -> bool:
    """Allow a widening stop only within 150% of the *original* distance.

    ``original_sl`` must be the entry signal's SL, not the current (possibly
    already tightened) SL; otherwise a first widening changes the base and the
    producer and the Freqtrade consumer disagree.
    """
    base_sl = original_sl if original_sl is not None else position.stop_loss
    original_distance = abs(position.entry_price - base_sl)
    proposed_distance = abs(position.entry_price - proposed_sl)
    if original_distance <= 0:
        return False
    return proposed_distance <= original_distance * MAX_WIDENING_RATIO


def _valid_price(value: Any) -> bool:
    """True for a finite positive number, rejecting bools and NaN/Infinity."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    return math.isfinite(value) and value > 0


def _price_guard(
    direction: str, sl: Optional[float], tp: Optional[float], current_price: Optional[float]
) -> Optional[str]:
    """Return a refusal reason when a level is on the wrong side of the price."""
    if current_price is None or current_price <= 0:
        return "no current price for the SL/TP price guard"
    if sl is not None:
        if direction == "LONG" and sl >= current_price:
            return f"stop-loss {sl:g} is at or above the current price {current_price:g}"
        if direction == "SHORT" and sl <= current_price:
            return f"stop-loss {sl:g} is at or below the current price {current_price:g}"
    if tp is not None and tp > 0:
        if direction == "LONG" and tp <= current_price:
            return f"take-profit {tp:g} is at or below the current price {current_price:g}"
        if direction == "SHORT" and tp >= current_price:
            return f"take-profit {tp:g} is at or above the current price {current_price:g}"
    return None


def _coherence_guard(
    direction: str, sl: Optional[float], tp: Optional[float]
) -> Optional[str]:
    """Return a refusal reason when the final SL/TP are inverted or equal."""
    if sl is None or tp is None:
        return None
    if direction == "LONG" and sl >= tp:
        return f"incoherent levels: stop-loss {sl:g} >= take-profit {tp:g} for a LONG"
    if direction == "SHORT" and sl <= tp:
        return f"incoherent levels: stop-loss {sl:g} <= take-profit {tp:g} for a SHORT"
    return None


def evaluate_update(
    position: PositionView,
    proposed_sl: Optional[float],
    proposed_tp: Optional[float],
    current_price: Optional[float],
    tf_minutes: int,
    policy: Optional[StopLossTighteningPolicy] = None,
    brain_thresholds: Optional[dict[str, Any]] = None,
    original_sl: Optional[float] = None,
) -> UpdateDecision:
    """Decide whether a proposed SL/TP update is compliant.

    Args:
        position: Current position levels and direction.
        proposed_sl: Requested stop-loss, or None to leave it unchanged.
        proposed_tp: Requested take-profit, or None to leave it unchanged.
        current_price: Real price used for the guard and progress.
        tf_minutes: Position timeframe in minutes (tightening thresholds).
        policy: Injected tightening policy (defaults to the core defaults).
        brain_thresholds: Optional learned thresholds passed to the policy.
        original_sl: Entry-signal SL used as the widening base.

    Returns:
        An :class:`UpdateDecision`; ``allowed=False`` means "do not emit".
    """
    policy = policy or StopLossTighteningPolicy()
    direction = position.direction

    sl_change = proposed_sl is not None and proposed_sl != position.stop_loss
    tp_change = proposed_tp is not None and proposed_tp != position.take_profit

    if sl_change and not _valid_price(proposed_sl):
        return UpdateDecision(False, None, None, "stop_loss must be a positive finite number")
    if tp_change and not _valid_price(proposed_tp):
        return UpdateDecision(False, None, None, "take_profit must be a positive finite number")

    # Hard price guard: never move a stop/TP to the wrong side of the market.
    guard = _price_guard(
        direction, proposed_sl if sl_change else None,
        proposed_tp if tp_change else None, current_price,
    )
    if guard is not None:
        return UpdateDecision(False, None, None, guard)

    new_sl: Optional[float] = None
    new_tp: Optional[float] = None
    is_tight = False
    widening_capped = False
    reasons: list[str] = []

    if sl_change:
        if is_tightening(direction, position.stop_loss, proposed_sl):
            if position.take_profit is None or position.take_profit <= 0:
                # No TP to measure progress against: keep the price guard and
                # the widening cap, but the progress gate cannot run.
                new_sl = proposed_sl
                is_tight = True
                reasons.append("progress gate skipped (no known take-profit)")
            else:
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
            if widening_allowed(position, proposed_sl, original_sl):
                new_sl = proposed_sl
                widening_capped = True
                reasons.append("widening allowed within 150% of the original SL distance")
            else:
                return UpdateDecision(
                    False, None, None,
                    "widening exceeds 150% of the original SL distance",
                )

    if tp_change:
        if is_tp_improvement(direction, position.take_profit, proposed_tp):
            new_tp = proposed_tp
            reasons.append("take-profit improved in the profit direction")
        else:
            reasons.append("take-profit move is not an improvement; ignored")

    final_sl = new_sl if new_sl is not None else position.stop_loss
    final_tp = new_tp if new_tp is not None else position.take_profit
    coherence = _coherence_guard(direction, final_sl, final_tp)
    if coherence is not None and (sl_change or tp_change):
        return UpdateDecision(False, None, None, coherence)

    if new_sl is None and new_tp is None:
        reason = "; ".join(reasons) if reasons else "no change proposed"
        return UpdateDecision(False, None, None, reason, is_tight, widening_capped)
    return UpdateDecision(
        True, new_sl, new_tp, "; ".join(reasons), is_tight, widening_capped
    )


