"""Tests for deterministic regime, size and policy-C target validation."""

import pytest

from src.llm_focus.deterministic_risk import (
    NO_TRADE,
    RANGING,
    TARGET_SOURCE_FALLBACK,
    TARGET_SOURCE_LLM,
    TRENDING_DOWN,
    TRENDING_UP,
    UNKNOWN,
    assess_regime,
    assess_regime_from_mapping,
    build_atr_stops,
    build_risk_plan,
    classify_volatility,
    compute_position_size,
    normalize_regime,
)


class TestRegime:
    def test_aliases(self):
        assert normalize_regime("bullish") == TRENDING_UP
        assert normalize_regime("Bearish") == TRENDING_DOWN
        assert normalize_regime("neutral") == RANGING
        assert normalize_regime("sideways") == RANGING

    @pytest.mark.parametrize(
        ("p0a_regime", "expected_regime", "expected_setups"),
        [
            (
                "trend_up",
                TRENDING_UP,
                ("TREND_PULLBACK", "BREAKOUT_RETEST", "MOMENTUM_CONTINUATION"),
            ),
            (
                "trend_down",
                TRENDING_DOWN,
                ("RALLY_FAILURE", "BREAKDOWN_RETEST", "MOMENTUM_SHORT"),
            ),
            ("range", RANGING, ("RANGE_FADE_EXTREME", "RANGE_MEAN_REVERSION")),
            ("low_vol", NO_TRADE, ()),
            ("high_vol", NO_TRADE, ()),
            ("unclear", NO_TRADE, ()),
        ],
    )
    def test_canonical_p0a_regimes_use_taxonomy_setups(
        self, p0a_regime, expected_regime, expected_setups
    ):
        assessment = assess_regime(p0a_regime, atr_pct=2.0)

        assert assessment.regime == expected_regime
        assert assessment.allowed_setups == expected_setups

    def test_no_trade_regimes_are_not_unknown(self):
        for p0a_regime in ("low_vol", "high_vol", "unclear"):
            assessment = assess_regime(p0a_regime, atr_pct=2.0)

            assert assessment.regime == NO_TRADE
            assert assessment.regime != UNKNOWN
            assert assessment.allowed_setups == ()

    def test_unknown_regime_fails_closed(self):
        assessment = assess_regime("mystery", atr_pct=2.0)
        assert assessment.regime == UNKNOWN
        assert assessment.allowed_setups == ()

    def test_default_setups_follow_regime(self):
        assessment = assess_regime("trending_up", atr_pct=2.0)
        assert "TREND_PULLBACK" in assessment.allowed_setups
        assert assessment.volatility == "MEDIUM"

    def test_omitted_allowed_setups_use_regime_defaults(self):
        omitted = assess_regime("trend_up", atr_pct=2.0)
        explicit_null = assess_regime_from_mapping({"regime": "trend_up", "atr_pct": 2.0})

        assert omitted.allowed_setups == (
            "TREND_PULLBACK",
            "BREAKOUT_RETEST",
            "MOMENTUM_CONTINUATION",
        )
        assert explicit_null.allowed_setups == omitted.allowed_setups

    def test_explicit_empty_allowed_setups_stay_no_trade(self):
        assessment = assess_regime("trend_up", allowed_setups=[], atr_pct=2.0)

        assert assessment.regime == TRENDING_UP
        assert assessment.allowed_setups == ()

    def test_disabled_timeframe_empty_allowed_setups_stay_no_trade(self):
        assessment = assess_regime_from_mapping(
            {"regime": "range", "allowed_setups": [], "atr_pct": 2.0}
        )

        assert assessment.regime == RANGING
        assert assessment.allowed_setups == ()

    def test_explicit_setups_are_normalized(self):
        assessment = assess_regime(
            "trending down",
            allowed_setups=["breakdown-retest", "momentum continuation", "BREAKDOWN_RETEST"],
            atr_pct=4.0,
        )
        assert assessment.allowed_setups == ("BREAKDOWN_RETEST", "MOMENTUM_CONTINUATION")
        assert assessment.volatility == "HIGH"

    def test_mapping_adapter(self):
        assessment = assess_regime_from_mapping(
            {"regime": "RANGING", "allowed_setups": ["range_fade"], "atr_percentage": 1.0}
        )
        assert assessment.regime == RANGING
        assert assessment.allowed_setups == ("RANGE_FADE",)
        assert assessment.volatility == "LOW"


class TestVolatilityAndSize:
    @pytest.mark.parametrize(
        ("atr_pct", "expected"),
        [(4.0, "HIGH"), (3.0, "MEDIUM"), (2.0, "MEDIUM"), (1.0, "LOW")],
    )
    def test_classify_volatility(self, atr_pct, expected):
        assert classify_volatility(atr_pct) == expected

    @pytest.mark.parametrize(
        ("confidence", "expected"),
        [(80, 0.03), (70, 0.03), (69, 0.02), (50, 0.02), (49, 0.01), ("HIGH", 0.03), ("LOW", 0.01)],
    )
    def test_position_size(self, confidence, expected):
        assert compute_position_size(confidence) == pytest.approx(expected)


class TestAtrFallbackPrimitive:
    def test_long_stops(self):
        stop_loss, take_profit = build_atr_stops("LONG", 100.0, 1.0)
        assert stop_loss == pytest.approx(98.0)
        assert take_profit == pytest.approx(104.0)

    def test_short_stops(self):
        stop_loss, take_profit = build_atr_stops("SHORT", 100.0, 1.0)
        assert stop_loss == pytest.approx(102.0)
        assert take_profit == pytest.approx(96.0)

    @pytest.mark.parametrize("direction", ["SIDEWAYS", ""])
    def test_invalid_direction(self, direction):
        with pytest.raises(ValueError):
            build_atr_stops(direction, 100.0, 1.0)

    def test_invalid_atr(self):
        with pytest.raises(ValueError):
            build_atr_stops("LONG", 100.0, 0)


class TestPolicyCRiskPlan:
    def test_valid_long_llm_levels_pass_verbatim(self):
        plan = build_risk_plan(
            "LONG", 100.0, 1.0, confidence=80, atr_pct=1.0,
            stop_loss=97.0, take_profit=106.0,
        )
        assert plan.risk_approved
        assert plan.rejection_reason is None
        assert plan.target_source == TARGET_SOURCE_LLM
        assert plan.stop_loss == pytest.approx(97.0)
        assert plan.take_profit == pytest.approx(106.0)
        assert plan.size_pct == pytest.approx(0.03)
        assert plan.rr_ratio == pytest.approx(2.0)
        assert plan.volatility_level == "LOW"

    def test_valid_short_llm_levels_pass_verbatim(self):
        plan = build_risk_plan(
            "SHORT", 100.0, 1.0, confidence="MEDIUM",
            stop_loss=104.0, take_profit=92.0,
        )
        assert plan.risk_approved
        assert plan.stop_loss == pytest.approx(104.0)
        assert plan.take_profit == pytest.approx(92.0)

    def test_wrong_side_is_rejected_without_substitution(self):
        plan = build_risk_plan("LONG", 100.0, 1.0, confidence=80, stop_loss=102.0, take_profit=96.0)
        assert not plan.risk_approved
        assert "LONG stop_loss must be below entry_price" in plan.rejection_reason
        assert "LONG take_profit must be above entry_price" in plan.rejection_reason
        assert plan.stop_loss == pytest.approx(102.0)
        assert plan.take_profit == pytest.approx(96.0)

    def test_rr_below_minimum_is_rejected(self):
        plan = build_risk_plan(
            "LONG", 100.0, 1.0, confidence=80,
            stop_loss=99.0, take_profit=101.0, min_rr=1.5,
        )
        assert not plan.risk_approved
        assert "R/R" in plan.rejection_reason
        assert plan.stop_loss == pytest.approx(99.0)
        assert plan.take_profit == pytest.approx(101.0)

    def test_distance_bounds_are_rejected(self):
        plan = build_risk_plan("LONG", 100.0, 1.0, confidence=80, stop_loss=99.9, take_profit=100.2)
        assert not plan.risk_approved
        assert plan.rejection_reason.count("outside 0.5-10%") == 2
        assert plan.stop_loss == pytest.approx(99.9)
        assert plan.take_profit == pytest.approx(100.2)

    def test_missing_entry_is_rejected(self):
        plan = build_risk_plan("LONG", None, 1.0, confidence=80, stop_loss=97.0, take_profit=106.0)
        assert not plan.risk_approved
        assert plan.rejection_reason == "entry price unavailable"
        assert plan.stop_loss == pytest.approx(97.0)
        assert plan.take_profit == pytest.approx(106.0)

    def test_partial_llm_levels_reject_instead_of_falling_back(self):
        plan = build_risk_plan("LONG", 100.0, 1.0, confidence=80, stop_loss=97.0)
        assert not plan.risk_approved
        assert "must both be provided" in plan.rejection_reason
        assert plan.stop_loss == pytest.approx(97.0)
        assert plan.take_profit is None
        assert plan.target_source == TARGET_SOURCE_LLM

    def test_fallback_used_only_when_both_levels_are_absent(self):
        plan = build_risk_plan("LONG", 100.0, 1.0, confidence=80, atr_pct=1.0)
        assert plan.risk_approved
        assert plan.target_source == TARGET_SOURCE_FALLBACK
        assert plan.stop_loss == pytest.approx(98.0)
        assert plan.take_profit == pytest.approx(104.0)
        assert plan.rr_ratio == pytest.approx(2.0)

    def test_fallback_uses_structure_geometry_when_setup_is_provided(self):
        plan = build_risk_plan(
            "LONG", 96.0, 1.0, confidence=80,
            setup={"name": "range_mean_reversion", "tp_mode": "structure", "tp_atr": 4.5},
            range_low=95.0,
            range_high=105.0,
        )
        assert plan.risk_approved
        assert plan.target_source == TARGET_SOURCE_FALLBACK
        assert plan.take_profit == pytest.approx(100.0)
        assert plan.stop_loss == pytest.approx(94.0)

    def test_invalid_fallback_is_rejected_not_clamped(self):
        plan = build_risk_plan("LONG", 100.0, 0.01, confidence="LOW", atr_pct=0.1)
        assert not plan.risk_approved
        assert "fallback target invalid" in plan.rejection_reason
        assert plan.stop_loss == pytest.approx(99.98)
        assert plan.take_profit == pytest.approx(100.04)

    def test_invalid_direction_is_rejected(self):
        plan = build_risk_plan("SIDEWAYS", 100.0, 1.0, confidence=80, stop_loss=97.0, take_profit=106.0)
        assert not plan.risk_approved
        assert plan.rejection_reason == "direction must be LONG or SHORT"
