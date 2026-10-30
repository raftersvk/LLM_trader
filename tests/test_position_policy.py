"""Tests for the producer's position SL/TP policy (core reuse)."""

import pytest

from src.signal_service.position_policy import (
    MAX_WIDENING_RATIO,
    PositionView,
    dynamic_take_profit,
    evaluate_update,
    is_tp_improvement,
)


def long_position(sl=66500.0, tp=72000.0, entry=67500.0):
    return PositionView(entry_price=entry, stop_loss=sl, take_profit=tp, direction="LONG")


def short_position(sl=69500.0, tp=64000.0, entry=68000.0):
    return PositionView(entry_price=entry, stop_loss=sl, take_profit=tp, direction="SHORT")


def test_tightening_allowed_after_progress():
    # price 69000 = 33% of the 4500 entry-to-TP distance (intraday 20%).
    decision = evaluate_update(long_position(), 68000.0, None, 69000.0, tf_minutes=60)
    assert decision.allowed is True
    assert decision.stop_loss == 68000.0
    assert decision.is_tightening is True


def test_premature_tightening_refused():
    # price 67800 = 6.7% progress < 20% intraday threshold.
    decision = evaluate_update(long_position(), 67000.0, None, 67800.0, tf_minutes=60)
    assert decision.allowed is False
    assert "Premature tightening" in decision.reason


def test_widening_within_cap_allowed():
    # original distance 1000; proposed distance 1200 <= 1500.
    decision = evaluate_update(long_position(), 66300.0, None, 69000.0, tf_minutes=60)
    assert decision.allowed is True
    assert decision.stop_loss == 66300.0
    assert decision.widening_capped is True


def test_widening_beyond_cap_refused():
    # original distance 1000; proposed distance 2000 > 1500.
    decision = evaluate_update(long_position(), 65500.0, None, 69000.0, tf_minutes=60)
    assert decision.allowed is False
    assert "150%" in decision.reason
    assert MAX_WIDENING_RATIO == 1.5


def test_take_profit_improvement_applied():
    decision = evaluate_update(long_position(), None, 73000.0, 69000.0, tf_minutes=60)
    assert decision.allowed is True
    assert decision.take_profit == 73000.0


def test_take_profit_degradation_ignored():
    decision = evaluate_update(long_position(), None, 70000.0, 69000.0, tf_minutes=60)
    assert decision.allowed is False
    assert decision.take_profit is None
    assert "not an improvement" in decision.reason


def test_short_tightening_and_tp_moves():
    # SHORT: tightening lowers the stop; TP improvement lowers the target.
    decision = evaluate_update(short_position(), 68500.0, 63000.0, 66000.0, tf_minutes=60)
    assert decision.allowed is True
    assert decision.stop_loss == 68500.0
    assert decision.take_profit == 63000.0
    assert is_tp_improvement("SHORT", 64000.0, 63000.0) is True


def test_dynamic_take_profit_long_and_short():
    long_tp = dynamic_take_profit("LONG", 100.0, atr=2.0, atr_percentage=2.0)
    short_tp = dynamic_take_profit("SHORT", 100.0, atr=2.0, atr_percentage=2.0)
    assert long_tp > 100.0
    assert short_tp < 100.0
    assert 100.0 - short_tp == pytest.approx(long_tp - 100.0)
