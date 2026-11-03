"""Tests for risk/volatility position sizing in the LLM-focus v2 layer."""

from __future__ import annotations

import pytest

from src.config.regime_taxonomy import (
    EXPOSURE_MODE_REJECT,
    EXPOSURE_MODE_SCALE_DOWN,
    RiskSettings,
    default_taxonomy_config,
    format_risk_summary,
    parse_taxonomy_config,
    resolve_risk_config,
)
from src.llm_focus.deterministic_risk import build_risk_plan
from src.llm_focus.risk_sizing import (
    CAP_PAIR_STAKE,
    CAP_PORTFOLIO_SCALE_DOWN,
    REASON_INVALID_STOP_DISTANCE,
    REASON_LEVERAGE_EXCEEDED,
    REASON_PORTFOLIO_EXPOSURE_EXCEEDED,
    REASON_SIZE_BELOW_MIN,
    REASON_SPOT_EXPOSURE_EXCEEDED,
    compute_risk_based_size,
    resolve_pair_max_stake,
    risk_sizing_enabled,
)


class TestRiskSettings:
    def test_build_in_defaults(self):
        settings = RiskSettings()

        assert settings.risk_per_trade_pct == pytest.approx(0.25)
        assert settings.min_size_pct == pytest.approx(0.1)
        assert settings.max_leverage == pytest.approx(1.0)
        assert settings.max_portfolio_exposure_pct is None
        assert settings.exposure_mode == EXPOSURE_MODE_SCALE_DOWN
        assert settings.pair_overrides == {}
        assert settings.regime_risk_multipliers == {}
        assert settings.portfolio_exposure_pct("spot") == pytest.approx(100.0)
        assert settings.portfolio_exposure_pct("futures") == pytest.approx(200.0)

    def test_taxonomy_risk_block_parses(self):
        taxonomy = parse_taxonomy_config(
            {
                "risk": {
                    "risk_per_trade_pct": 0.4,
                    "min_size_pct": 0.5,
                    "max_leverage": 3.0,
                    "max_portfolio_exposure_pct": 150.0,
                    "exposure_mode": "reject",
                    "pair_overrides": {"BEAMX/USDT": 0.15},
                    "regime_risk_multipliers": {"high_vol": 0.5, "low_vol": 1.0},
                }
            }
        )

        assert taxonomy.risk.risk_per_trade_pct == pytest.approx(0.4)
        assert taxonomy.risk.min_size_pct == pytest.approx(0.5)
        assert taxonomy.risk.max_leverage == pytest.approx(3.0)
        assert taxonomy.risk.max_portfolio_exposure_pct == pytest.approx(150.0)
        assert taxonomy.risk.exposure_mode == EXPOSURE_MODE_REJECT
        assert taxonomy.risk.pair_overrides["BEAMX/USDT"] == pytest.approx(0.15)
        assert taxonomy.risk.regime_risk_multipliers["high_vol"] == pytest.approx(0.5)
        assert taxonomy.as_dict()["risk"]["max_leverage"] == pytest.approx(3.0)

    def test_absent_risk_block_keeps_defaults(self):
        assert default_taxonomy_config().risk == RiskSettings()

    def test_per_pair_risk_override_resolves_with_symbol_aliases(self):
        settings = RiskSettings(pair_overrides={"BEAMX/USDT:USDT": 0.1})

        assert settings.risk_for_symbol("BEAMX/USDT") == pytest.approx(0.1)
        assert settings.risk_for_symbol("BEAMXUSDT") == pytest.approx(0.1)
        assert settings.risk_for_symbol("BTC/USDT") == pytest.approx(0.25)
        assert settings.risk_for_symbol(None) == pytest.approx(0.25)

    def test_regime_multiplier_is_neutral_by_default(self):
        settings = RiskSettings()

        assert settings.regime_multiplier("high_vol") == pytest.approx(1.0)
        assert settings.effective_risk_per_trade_pct(None, "high_vol") == pytest.approx(0.25)

    def test_regime_multiplier_scales_risk(self):
        settings = RiskSettings(
            risk_per_trade_pct=0.25,
            regime_risk_multipliers={"high_vol": 0.5},
        )

        assert settings.regime_multiplier("high_vol") == pytest.approx(0.5)
        assert settings.effective_risk_per_trade_pct(None, "high_vol") == pytest.approx(0.125)
        assert settings.effective_risk_per_trade_pct(None, "trend_up") == pytest.approx(0.25)

    def test_config_ini_overrides_win_over_taxonomy(self):
        taxonomy = parse_taxonomy_config(
            {
                "risk": {
                    "risk_per_trade_pct": 0.4,
                    "max_portfolio_exposure_pct": 50.0,
                    "exposure_mode": "reject",
                }
            }
        )
        overrides = {
            "risk_per_trade_pct": 0.3,
            "max_portfolio_exposure_pct": 25.0,
            "exposure_mode": "scale_down",
        }

        effective = resolve_risk_config(taxonomy, overrides)

        assert effective.risk_per_trade_pct == pytest.approx(0.3)
        assert effective.max_portfolio_exposure_pct == pytest.approx(25.0)
        assert effective.exposure_mode == EXPOSURE_MODE_SCALE_DOWN
        # Untouched taxonomy values survive the config.ini overlay.
        assert effective.min_size_pct == pytest.approx(0.1)
        assert effective.max_leverage == pytest.approx(1.0)

    def test_absent_config_ini_keeps_taxonomy_values(self):
        taxonomy = parse_taxonomy_config(
            {"risk": {"max_portfolio_exposure_pct": 50.0}}
        )

        effective = resolve_risk_config(taxonomy, {})
        assert effective.max_portfolio_exposure_pct == pytest.approx(50.0)

    @pytest.mark.parametrize(
        "risk",
        [
            {"risk_per_trade_pct": 0},
            {"max_portfolio_exposure_pct": 0},
            {"min_size_pct": -1},
            {"max_leverage": 0.5},
            {"exposure_mode": "clamp"},
            {"pair_overrides": {"BEAMX/USDT": 0}},
            {"regime_risk_multipliers": {"mystery": 0.5}},
        ],
    )
    def test_invalid_risk_values_are_rejected(self, risk):
        from src.config.regime_taxonomy import TaxonomyConfigError

        with pytest.raises(TaxonomyConfigError):
            parse_taxonomy_config({"risk": risk})

    def test_summary_lists_effective_values(self):
        summary = format_risk_summary(RiskSettings())

        assert "risk_per_trade=0.25%" in summary
        assert "pair_risk=none" in summary
        assert "regime_multipliers=none" in summary
        assert "min_size=0.1%" in summary
        assert "max_leverage=1x" in summary
        assert "portfolio_exposure=spot=100%/futures=200%" in summary
        assert "exposure_mode=scale_down" in summary


class TestRiskBasedSizing:
    @pytest.mark.parametrize(
        ("risk", "stop", "expected"),
        [
            (0.25, 2.0, 0.125),
            (0.25, 1.0, 0.25),
            (0.25, 4.0, 0.0625),
            (0.5, 2.0, 0.25),
            (0.5, 1.0, 0.5),
        ],
    )
    def test_recommended_risk_sizes(self, risk, stop, expected):
        decision = compute_risk_based_size(stop, RiskSettings(risk_per_trade_pct=risk))

        assert decision.approved
        assert decision.size_pct == pytest.approx(expected)
        assert decision.capped_by is None

    def test_default_risk_size_is_not_capped(self):
        decision = compute_risk_based_size(2.0, RiskSettings())

        assert decision.approved
        assert decision.size_pct == pytest.approx(0.125)
        assert decision.capped_by is None
        assert decision.risk_amount_pct == pytest.approx(0.25)

    def test_per_pair_risk_override_wins_over_global(self):
        settings = RiskSettings(
            risk_per_trade_pct=0.25,
            pair_overrides={"BEAMX/USDT": 0.1},
        )

        override = compute_risk_based_size(1.0, settings, symbol="BEAMX/USDT")
        default = compute_risk_based_size(1.0, settings, symbol="BTC/USDT")

        assert override.size_pct == pytest.approx(0.10)
        assert default.size_pct == pytest.approx(0.25)

    def test_regime_multiplier_applies_and_one_is_neutral(self):
        settings = RiskSettings(
            risk_per_trade_pct=0.25,
            regime_risk_multipliers={"high_vol": 0.5, "low_vol": 1.0},
        )

        high_vol = compute_risk_based_size(2.0, settings, regime="high_vol")
        low_vol = compute_risk_based_size(2.0, settings, regime="low_vol")
        absent = compute_risk_based_size(2.0, settings, regime="trend_up")

        assert high_vol.size_pct == pytest.approx(0.0625)
        assert low_vol.size_pct == pytest.approx(0.125)
        assert absent.size_pct == pytest.approx(0.125)
        assert high_vol.regime_multiplier == pytest.approx(0.5)

    def test_wide_stop_gives_smaller_size_than_tight_stop(self):
        wide = compute_risk_based_size(4.0, RiskSettings())
        tight = compute_risk_based_size(1.0, RiskSettings())

        assert wide.approved and tight.approved
        assert wide.size_pct == pytest.approx(0.0625)
        assert tight.size_pct == pytest.approx(0.25)
        assert wide.size_pct < tight.size_pct

    def test_portfolio_ceiling_scales_down_by_default(self):
        decision = compute_risk_based_size(
            2.0,
            RiskSettings(max_portfolio_exposure_pct=100.0),
            current_exposure_pct=90.0,
        )

        assert decision.approved
        assert decision.size_pct == pytest.approx(0.10)
        assert decision.capped_by == CAP_PORTFOLIO_SCALE_DOWN
        assert decision.resulting_exposure_pct == pytest.approx(100.0)

    def test_portfolio_ceiling_rejects_when_configured(self):
        decision = compute_risk_based_size(
            2.0,
            RiskSettings(
                max_portfolio_exposure_pct=100.0,
                exposure_mode=EXPOSURE_MODE_REJECT,
            ),
            current_exposure_pct=90.0,
        )

        assert not decision.approved
        assert decision.reason == REASON_PORTFOLIO_EXPOSURE_EXCEEDED
        assert decision.resulting_exposure_pct == pytest.approx(102.5)

    def test_exhausted_portfolio_scales_below_the_floor(self):
        decision = compute_risk_based_size(
            2.0,
            RiskSettings(max_portfolio_exposure_pct=100.0),
            current_exposure_pct=100.0,
        )

        assert not decision.approved
        assert decision.reason == REASON_SIZE_BELOW_MIN
        assert decision.size_pct == pytest.approx(0.0)

    @pytest.mark.parametrize("distance", [0.0, -1.0, float("nan"), float("inf")])
    def test_invalid_stop_distance_is_rejected_without_infinite_size(self, distance):
        decision = compute_risk_based_size(distance, RiskSettings())

        assert not decision.approved
        assert decision.reason == REASON_INVALID_STOP_DISTANCE
        assert decision.size_pct == 0.0

    def test_spot_rejects_above_100_percent_of_capital(self):
        decision = compute_risk_based_size(
            0.2,
            RiskSettings(),
            trading_mode="spot",
        )

        assert not decision.approved
        assert decision.reason == REASON_SPOT_EXPOSURE_EXCEEDED
        assert decision.size_pct == pytest.approx(1.25)
        assert decision.leverage == pytest.approx(1.0)

    def test_futures_allows_leverage_up_to_the_configured_maximum(self):
        decision = compute_risk_based_size(
            0.2,
            RiskSettings(max_leverage=2.0),
            trading_mode="futures",
        )

        assert decision.approved
        assert decision.size_pct == pytest.approx(1.25)
        assert decision.leverage == pytest.approx(1.25)

    def test_futures_rejects_leverage_above_the_maximum(self):
        decision = compute_risk_based_size(
            0.2,
            RiskSettings(max_leverage=1.0),
            trading_mode="futures",
        )

        assert not decision.approved
        assert decision.reason == REASON_LEVERAGE_EXCEEDED
        assert decision.leverage == pytest.approx(1.25)

    def test_size_below_minimum_is_skipped(self):
        decision = compute_risk_based_size(
            100.0,
            RiskSettings(min_size_pct=1.0),
        )

        assert not decision.approved
        assert decision.reason == REASON_SIZE_BELOW_MIN
        assert decision.size_pct == pytest.approx(0.0025)

    def test_optional_pair_max_stake_still_binds(self):
        decision = compute_risk_based_size(
            1.0,
            RiskSettings(),
            capital=10_000.0,
            pair_max_stake=500.0,
        )

        assert decision.approved
        assert decision.size_pct == pytest.approx(0.05)
        assert decision.capped_by == CAP_PAIR_STAKE
        assert decision.risk_amount_currency == pytest.approx(5.0)

    def test_pair_cap_is_read_from_the_canonical_taxonomy(self):
        taxonomy = parse_taxonomy_config(
            {"pair_overrides": {"BTC/USDT:USDT": {"max_stake": 300.0}}}
        )

        assert resolve_pair_max_stake(taxonomy, "BTC/USDT") == pytest.approx(300.0)
        assert resolve_pair_max_stake(taxonomy, "ETH/USDT") is None
        assert resolve_pair_max_stake(None, "BTC/USDT") is None

    def test_currency_risk_uses_capital_and_stop_distance(self):
        decision = compute_risk_based_size(
            2.0,
            RiskSettings(),
            capital=10_000.0,
        )

        assert decision.risk_amount_pct == pytest.approx(0.25)
        assert decision.risk_amount_currency == pytest.approx(25.0)


class TestRiskSizingGate:
    def test_v2_flag_is_off_by_default(self):
        assert not risk_sizing_enabled(None)
        assert not risk_sizing_enabled({})
        assert risk_sizing_enabled({"llm_focus_v2": True})

    def test_build_risk_plan_keeps_confidence_size_when_flag_is_off(self):
        plan = build_risk_plan(
            "LONG", 100.0, 1.0, confidence=80,
            stop_loss=97.0, take_profit=106.0,
            risk_sizing=False,
        )

        assert plan.risk_approved
        assert plan.size_pct == pytest.approx(0.03)
        assert plan.sizing is None

    def test_build_risk_plan_uses_risk_size_when_flag_is_on(self):
        plan = build_risk_plan(
            "LONG", 100.0, 1.0, confidence=80,
            stop_loss=97.0, take_profit=106.0,
            risk_sizing=True,
            risk_settings=RiskSettings(),
        )

        assert plan.risk_approved
        assert plan.size_pct == pytest.approx(0.25 / 3.0)
        assert plan.sizing is not None
        assert plan.sizing.approved

    def test_build_risk_plan_applies_symbol_and_regime(self):
        plan = build_risk_plan(
            "LONG", 100.0, 1.0, confidence=80,
            stop_loss=97.0, take_profit=106.0,
            risk_sizing=True,
            risk_settings=RiskSettings(
                risk_per_trade_pct=0.25,
                pair_overrides={"BEAMX/USDT": 0.1},
                regime_risk_multipliers={"high_vol": 0.5},
            ),
            symbol="BEAMX/USDT",
            regime="high_vol",
        )

        assert plan.risk_approved
        assert plan.size_pct == pytest.approx(0.05 / 3.0)
        assert plan.sizing.risk_per_trade_pct == pytest.approx(0.05)

    def test_build_risk_plan_portfolio_guard_rejects(self):
        plan = build_risk_plan(
            "LONG", 100.0, 1.0, confidence=80,
            stop_loss=97.0, take_profit=106.0,
            risk_sizing=True,
            risk_settings=RiskSettings(
                max_portfolio_exposure_pct=10.0,
                exposure_mode=EXPOSURE_MODE_REJECT,
            ),
            current_exposure_pct=5.0,
        )

        assert not plan.risk_approved
        assert plan.rejection_reason == REASON_PORTFOLIO_EXPOSURE_EXCEEDED
        assert plan.sizing is not None

    def test_build_risk_plan_floor_rejection_skips_the_trade(self):
        plan = build_risk_plan(
            "LONG", 100.0, 1.0, confidence=80,
            stop_loss=97.0, take_profit=106.0,
            risk_sizing=True,
            risk_settings=RiskSettings(min_size_pct=20.0),
        )

        assert not plan.risk_approved
        assert plan.rejection_reason == REASON_SIZE_BELOW_MIN
        assert plan.sizing is not None
        assert plan.sizing.reason == REASON_SIZE_BELOW_MIN
