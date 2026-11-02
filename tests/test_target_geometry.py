"""Tests for v2 target geometry and target validation."""

import pytest

from src.llm_focus.target_geometry import (
    STRUCTURE_MEASURED_MOVE,
    STRUCTURE_MID_RANGE,
    STRUCTURE_VWAP,
    TP_MODE_ATR_MULTIPLE,
    TP_MODE_LLM_LEVEL,
    TP_MODE_STRUCTURE,
    read_tp_mode,
    resolve_target_geometry,
    validate_target,
)


class TestReadTpMode:
    def test_absent_defaults_to_atr(self):
        mode, submode, note = read_tp_mode({"name": "trend_pullback", "tp_atr": 2.0})
        assert mode == TP_MODE_ATR_MULTIPLE
        assert submode is None
        assert "absent" in note

    def test_invalid_mode_falls_back(self):
        mode, submode, note = read_tp_mode({"name": "x", "tp_mode": "weird"})
        assert mode == TP_MODE_ATR_MULTIPLE
        assert "invalid" in note

    def test_structure_submode(self):
        mode, submode, note = read_tp_mode(
            {"name": "range_mean_reversion", "tp_mode": "structure", "tp_submode": "vwap"}
        )
        assert mode == TP_MODE_STRUCTURE
        assert submode == STRUCTURE_VWAP
        assert note is None

    def test_invalid_structure_submode_falls_back(self):
        mode, submode, note = read_tp_mode(
            {"name": "range_mean_reversion", "tp_mode": "structure", "tp_submode": "fib"}
        )
        assert mode == TP_MODE_ATR_MULTIPLE
        assert "invalid structure mode" in note


class TestResolveTarget:
    def test_atr_multiple(self):
        geometry = resolve_target_geometry(
            {"name": "trend_pullback", "tp_mode": "atr_multiple", "tp_atr": 2.0},
            direction="LONG",
            entry_price=100.0,
            atr=1.0,
        )
        assert geometry.effective_mode == TP_MODE_ATR_MULTIPLE
        assert geometry.target_price == pytest.approx(102.0)
        assert not geometry.fallback_used

    def test_mid_range_primary(self):
        geometry = resolve_target_geometry(
            {"name": "range_mean_reversion", "tp_mode": "structure", "tp_atr": 4.5},
            direction="LONG",
            entry_price=96.0,
            atr=1.0,
            range_low=95.0,
            range_high=105.0,
        )
        assert geometry.effective_mode == TP_MODE_STRUCTURE
        assert geometry.structure_mode == STRUCTURE_MID_RANGE
        assert geometry.target_price == pytest.approx(100.0)

    def test_mid_range_wrong_side_falls_back_to_vwap(self):
        geometry = resolve_target_geometry(
            {"name": "range_mean_reversion", "tp_mode": "structure", "tp_atr": 4.5},
            direction="LONG",
            entry_price=101.0,
            atr=1.0,
            range_low=95.0,
            range_high=105.0,
            vwap=103.0,
        )
        assert geometry.effective_mode == TP_MODE_STRUCTURE
        assert geometry.structure_mode == STRUCTURE_VWAP
        assert geometry.target_price == pytest.approx(103.0)

    def test_structure_unavailable_falls_back_to_atr(self):
        geometry = resolve_target_geometry(
            {"name": "range_fade_extreme", "tp_mode": "structure", "tp_atr": 4.5},
            direction="SHORT",
            entry_price=100.0,
            atr=2.0,
        )
        assert geometry.effective_mode == TP_MODE_ATR_MULTIPLE
        assert geometry.fallback_used
        assert "unavailable" in geometry.fallback_reason
        assert geometry.target_price == pytest.approx(91.0)

    def test_no_tp_atr_keeps_llm_level(self):
        geometry = resolve_target_geometry(
            {"name": "breakout_retest", "tp_mode": "atr_multiple"},
            direction="LONG",
            entry_price=100.0,
            atr=1.0,
        )
        assert geometry.effective_mode == TP_MODE_LLM_LEVEL
        assert geometry.target_price is None
        assert geometry.fallback_used
        assert "keep LLM target" in geometry.fallback_reason

    def test_measured_move_is_explicit_and_unvalidated(self):
        geometry = resolve_target_geometry(
            {
                "name": "trend_pullback",
                "tp_mode": "structure",
                "tp_submode": "measured_move",
                "tp_atr": 2.0,
            },
            direction="LONG",
            entry_price=100.0,
            atr=1.0,
            measured_move_target=110.0,
        )
        assert geometry.effective_mode == TP_MODE_STRUCTURE
        assert geometry.structure_mode == STRUCTURE_MEASURED_MOVE
        assert geometry.target_price == pytest.approx(110.0)
        assert any("unvalidated" in note for note in geometry.notes)

    def test_invalid_entry_or_atr(self):
        with pytest.raises(ValueError):
            resolve_target_geometry({"name": "x"}, direction="LONG", entry_price=0, atr=1)
        with pytest.raises(ValueError):
            resolve_target_geometry({"name": "x"}, direction="LONG", entry_price=100, atr=0)


class TestValidateTarget:
    def test_valid_long(self):
        result = validate_target("LONG", 100.0, 98.0, 104.0)
        assert result.valid
        assert result.rr_ratio == pytest.approx(2.0)
        assert result.sl_distance_pct == pytest.approx(2.0)
        assert result.tp_distance_pct == pytest.approx(4.0)

    def test_valid_short(self):
        result = validate_target("SHORT", 100.0, 102.0, 96.0)
        assert result.valid
        assert result.rr_ratio == pytest.approx(2.0)

    def test_wrong_side(self):
        result = validate_target("LONG", 100.0, 102.0, 96.0)
        assert not result.valid
        assert any("LONG stop_loss" in issue for issue in result.issues)
        assert any("LONG take_profit" in issue for issue in result.issues)

    def test_bounds(self):
        tight = validate_target("LONG", 100.0, 99.9, 100.2)
        wide = validate_target("LONG", 100.0, 80.0, 130.0)
        assert not tight.valid
        assert any("outside 0.5-10%" in issue for issue in tight.issues)
        assert not wide.valid
        assert sum("outside 0.5-10%" in issue for issue in wide.issues) == 2

    def test_rr_floor(self):
        result = validate_target("LONG", 100.0, 98.0, 101.5, min_rr=1.5)
        assert not result.valid
        assert any("R/R" in issue for issue in result.issues)

    def test_invalid_direction_and_prices(self):
        assert not validate_target("SIDEWAYS", 100.0, 98.0, 104.0).valid
        assert not validate_target("LONG", 0, 98.0, 104.0).valid
