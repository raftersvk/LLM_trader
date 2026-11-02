"""Pure event triggers for v2 position review.

The strategy's existing trailing/TP/TTL logic remains the default. This module
only answers one question: given the current position and market snapshot, is
an LLM position review actually worth its cost?
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

from src.llm_focus.deterministic_risk import UNKNOWN, normalize_regime
from src.llm_focus.flags import event_positions_v2_enabled

TRIGGER_REGIME_FLIP = "regime_flip"
TRIGGER_NEAR_STOP_LOSS = "near_stop_loss"
TRIGGER_NEAR_TAKE_PROFIT = "near_take_profit"
TRIGGER_ATR_SHOCK = "atr_shock"
TRIGGER_STALE_POSITION = "stale_position"

DEFAULT_NEAR_LEVEL_PCT = 0.25
DEFAULT_ATR_SHOCK_MULTIPLIER = 1.5
DEFAULT_STALE_AFTER_MINUTES = 24 * 60


@dataclass(frozen=True)
class PositionSnapshot:
    """Everything the trigger logic needs about an open position."""

    symbol: str
    direction: str
    entry_price: float
    stop_loss: float
    take_profit: float
    opened_at: datetime
    current_price: float
    current_atr_pct: float = 0.0
    entry_atr_pct: float = 0.0
    regime: str | None = None
    entry_regime: str | None = None


@dataclass(frozen=True)
class PositionTrigger:
    """One reason an LLM position review may be needed."""

    kind: str
    reason: str


@dataclass(frozen=True)
class PositionReviewDecision:
    """Aggregate trigger decision."""

    needs_review: bool
    triggers: tuple[PositionTrigger, ...]

    @property
    def kinds(self) -> tuple[str, ...]:
        return tuple(trigger.kind for trigger in self.triggers)


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _near_level(current_price: float, level: float, threshold_pct: float) -> bool:
    if level <= 0:
        return False
    return abs(current_price - level) / level * 100 <= threshold_pct


def evaluate_position_events(
    position: PositionSnapshot,
    now: datetime | None = None,
    near_level_pct: float = DEFAULT_NEAR_LEVEL_PCT,
    atr_shock_multiplier: float = DEFAULT_ATR_SHOCK_MULTIPLIER,
    stale_after_minutes: int = DEFAULT_STALE_AFTER_MINUTES,
) -> PositionReviewDecision:
    """Evaluate all v2 review triggers without side effects."""
    if near_level_pct < 0:
        raise ValueError("near_level_pct must be non-negative")
    if atr_shock_multiplier <= 0:
        raise ValueError("atr_shock_multiplier must be positive")
    if stale_after_minutes < 0:
        raise ValueError("stale_after_minutes must be non-negative")

    current_time = _as_utc(now or datetime.now(timezone.utc))
    triggers: list[PositionTrigger] = []

    entry_regime = normalize_regime(position.entry_regime)
    current_regime = normalize_regime(position.regime)
    if entry_regime != UNKNOWN and current_regime != UNKNOWN and entry_regime != current_regime:
        triggers.append(
            PositionTrigger(
                kind=TRIGGER_REGIME_FLIP,
                reason=f"regime changed from {entry_regime} to {current_regime}",
            )
        )

    if _near_level(position.current_price, position.stop_loss, near_level_pct):
        triggers.append(
            PositionTrigger(
                kind=TRIGGER_NEAR_STOP_LOSS,
                reason=f"price within {near_level_pct:g}% of stop loss",
            )
        )

    if _near_level(position.current_price, position.take_profit, near_level_pct):
        triggers.append(
            PositionTrigger(
                kind=TRIGGER_NEAR_TAKE_PROFIT,
                reason=f"price within {near_level_pct:g}% of take profit",
            )
        )

    if position.entry_atr_pct > 0 and position.current_atr_pct >= (
        position.entry_atr_pct * atr_shock_multiplier
    ):
        triggers.append(
            PositionTrigger(
                kind=TRIGGER_ATR_SHOCK,
                reason=(
                    f"ATR% rose from {position.entry_atr_pct:g} to "
                    f"{position.current_atr_pct:g} (multiplier {atr_shock_multiplier:g})"
                ),
            )
        )

    age = current_time - _as_utc(position.opened_at)
    if age >= timedelta(minutes=stale_after_minutes):
        triggers.append(
            PositionTrigger(
                kind=TRIGGER_STALE_POSITION,
                reason=f"position age {age} exceeds {stale_after_minutes} minutes",
            )
        )

    return PositionReviewDecision(needs_review=bool(triggers), triggers=tuple(triggers))


def maybe_review_position(
    config: Any,
    position: PositionSnapshot,
    now: datetime | None = None,
    near_level_pct: float = DEFAULT_NEAR_LEVEL_PCT,
    atr_shock_multiplier: float = DEFAULT_ATR_SHOCK_MULTIPLIER,
    stale_after_minutes: int = DEFAULT_STALE_AFTER_MINUTES,
) -> PositionReviewDecision | None:
    """Evaluate triggers only when ``event_positions_v2`` is enabled.

    ``None`` means the flag is off and the existing strategy-owned exits are the
    only path, preserving default behavior.
    """
    if not event_positions_v2_enabled(config):
        return None
    return evaluate_position_events(
        position,
        now=now,
        near_level_pct=near_level_pct,
        atr_shock_multiplier=atr_shock_multiplier,
        stale_after_minutes=stale_after_minutes,
    )
