"""Tests for the producer's position SL/TP policy (core reuse)."""

from src.signal_service.position_policy import (
    MAX_WIDENING_RATIO,
    PositionView,
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


def test_long_stop_at_or_beyond_current_price_refused():
    # entry 67500, current 69000: a stop at 69500 is above the market.
    decision = evaluate_update(long_position(), 69500.0, None, 69000.0, tf_minutes=60)
    assert decision.allowed is False
    assert "at or above the current price" in decision.reason


def test_short_stop_at_or_below_current_price_refused():
    # SHORT entry 68000, current 66000: a stop at 65500 is below the market.
    decision = evaluate_update(short_position(), 65500.0, None, 66000.0, tf_minutes=60)
    assert decision.allowed is False
    assert "at or below the current price" in decision.reason


def test_long_tp_at_or_below_current_price_refused():
    decision = evaluate_update(long_position(), None, 68800.0, 69000.0, tf_minutes=60)
    assert decision.allowed is False
    assert "take-profit" in decision.reason


def test_incoherent_final_levels_refused():
    # A tightening SL above the existing TP must be refused, not emitted.
    decision = evaluate_update(long_position(tp=69000.0), 69500.0, None, 68000.0, tf_minutes=60)
    assert decision.allowed is False
    assert "incoherent" in decision.reason or "above the current price" in decision.reason


def test_widening_base_is_the_original_entry_sl():
    # Current SL already tightened to 67000; base must stay 66500.
    position = PositionView(entry_price=67500.0, stop_loss=67000.0, take_profit=72000.0, direction="LONG")
    # 150% of |67500-66500| = 1500 -> 66000 exactly is allowed.
    assert evaluate_update(position, 66000.0, None, 70000.0, 60, original_sl=66500.0).allowed is True
    # Using the current SL as the base would compute only 750*1.5=1125 and reject; prove the original wins.
    assert evaluate_update(position, 65900.0, None, 70000.0, 60, original_sl=66500.0).allowed is False


def test_sl_only_update_when_tp_unknown():
    # No TP: the progress gate cannot run, but a safe tightening is still allowed.
    position = PositionView(entry_price=67500.0, stop_loss=66500.0, take_profit=None, direction="LONG")
    decision = evaluate_update(position, 68000.0, None, 69000.0, tf_minutes=60)
    assert decision.allowed is True
    assert decision.stop_loss == 68000.0
    assert "no known take-profit" in decision.reason


def test_sl_update_refused_without_current_price():
    position = PositionView(entry_price=67500.0, stop_loss=66500.0, take_profit=72000.0, direction="LONG")
    decision = evaluate_update(position, 68000.0, None, None, tf_minutes=60)
    assert decision.allowed is False
    assert "no current price" in decision.reason


def test_non_finite_or_bool_proposals_are_refused():
    for bad in (float("inf"), float("nan"), float("-inf"), True):
        decision = evaluate_update(long_position(), bad, None, 69000.0, tf_minutes=60)
        assert decision.allowed is False
        assert "finite" in decision.reason
        decision = evaluate_update(long_position(), None, bad, 69000.0, tf_minutes=60)
        assert decision.allowed is False
        assert "finite" in decision.reason
