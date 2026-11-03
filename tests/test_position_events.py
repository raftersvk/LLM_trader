"""Tests for pure v2 position-event triggers and default-off isolation."""

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from src.llm_focus.position_events import (
    TRIGGER_ATR_SHOCK,
    TRIGGER_NEAR_STOP_LOSS,
    TRIGGER_NEAR_TAKE_PROFIT,
    TRIGGER_REGIME_FLIP,
    TRIGGER_STALE_POSITION,
    PositionSnapshot,
    evaluate_position_events,
    maybe_review_position,
)

NOW = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)


def _position(**overrides):
    fields = {
        "symbol": "BTC/USDT",
        "direction": "LONG",
        "entry_price": 100_000.0,
        "stop_loss": 98_000.0,
        "take_profit": 104_000.0,
        "opened_at": NOW - timedelta(hours=1),
        "current_price": 101_000.0,
        "current_atr_pct": 2.0,
        "entry_atr_pct": 2.0,
        "regime": "TRENDING_UP",
        "entry_regime": "TRENDING_UP",
    }
    fields.update(overrides)
    return PositionSnapshot(**fields)


class TestTriggers:
    def test_no_trigger_keeps_default_path(self):
        decision = evaluate_position_events(_position(), now=NOW)
        assert not decision.needs_review
        assert decision.triggers == ()

    def test_regime_flip(self):
        decision = evaluate_position_events(_position(regime="RANGING"), now=NOW)
        assert decision.needs_review
        assert TRIGGER_REGIME_FLIP in decision.kinds

    def test_unknown_regime_does_not_trigger(self):
        decision = evaluate_position_events(_position(regime="UNKNOWN"), now=NOW)
        assert TRIGGER_REGIME_FLIP not in decision.kinds

    def test_near_stop_loss(self):
        decision = evaluate_position_events(_position(current_price=98_100.0), now=NOW)
        assert TRIGGER_NEAR_STOP_LOSS in decision.kinds

    def test_near_take_profit(self):
        decision = evaluate_position_events(_position(current_price=103_900.0), now=NOW)
        assert TRIGGER_NEAR_TAKE_PROFIT in decision.kinds

    def test_atr_shock(self):
        decision = evaluate_position_events(
            _position(current_atr_pct=3.2, entry_atr_pct=2.0),
            now=NOW,
            atr_shock_multiplier=1.5,
        )
        assert TRIGGER_ATR_SHOCK in decision.kinds

    def test_stale_position(self):
        decision = evaluate_position_events(
            _position(opened_at=NOW - timedelta(days=2)),
            now=NOW,
            stale_after_minutes=60,
        )
        assert TRIGGER_STALE_POSITION in decision.kinds

    def test_multiple_triggers_are_reported(self):
        decision = evaluate_position_events(
            _position(regime="RANGING", current_price=98_050.0, current_atr_pct=4.0),
            now=NOW,
        )
        assert set(decision.kinds) >= {
            TRIGGER_REGIME_FLIP,
            TRIGGER_NEAR_STOP_LOSS,
            TRIGGER_ATR_SHOCK,
        }

    def test_naive_timestamps_are_treated_as_utc(self):
        naive_now = NOW.replace(tzinfo=None)
        decision = evaluate_position_events(
            _position(opened_at=(NOW - timedelta(hours=2)).replace(tzinfo=None)),
            now=naive_now,
            stale_after_minutes=60,
        )
        assert TRIGGER_STALE_POSITION in decision.kinds

    def test_invalid_parameters(self):
        with pytest.raises(ValueError):
            evaluate_position_events(_position(), now=NOW, near_level_pct=-1)
        with pytest.raises(ValueError):
            evaluate_position_events(_position(), now=NOW, atr_shock_multiplier=0)
        with pytest.raises(ValueError):
            evaluate_position_events(_position(), now=NOW, stale_after_minutes=-1)


class TestFlagIsolation:
    def test_disabled_flag_returns_none(self):
        assert maybe_review_position(SimpleNamespace(EVENT_POSITIONS_V2=False), _position(), now=NOW) is None
        assert maybe_review_position({}, _position(), now=NOW) is None

    def test_enabled_flag_returns_decision(self):
        decision = maybe_review_position(
            SimpleNamespace(EVENT_POSITIONS_V2=True),
            _position(regime="RANGING"),
            now=NOW,
        )
        assert decision is not None
        assert decision.needs_review
