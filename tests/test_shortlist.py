"""Tests for the pure recall-oriented shortlist."""

from types import SimpleNamespace

import pytest

from src.llm_focus.shortlist import (
    ADVISORY_JEV_BIAS_MISMATCH,
    ADVISORY_JEV_NOT_TRADEABLE,
    REASON_ATR_TOO_HIGH,
    REASON_ATR_TOO_LOW,
    REASON_ATR_UNAVAILABLE,
    REASON_NO_ALLOWED_SETUPS,
    REASON_OPEN_POSITION,
    REASON_REGIME_UNKNOWN,
    REASON_SETUP_NOT_ALLOWED,
    ShortlistCandidate,
    build_shortlist,
    candidate_from_mapping,
    evaluate_recall,
    maybe_build_shortlist,
    meets_recall_guardrail,
    merge_recall_reports,
)


def _candidate(**overrides):
    fields = {
        "symbol": "BTC/USDT",
        "regime": "TRENDING_UP",
        "allowed_setups": ("TREND_PULLBACK", "BREAKOUT_RETEST"),
        "setup": "TREND_PULLBACK",
        "atr_pct": 2.0,
        "has_open_position": False,
        "direction": "LONG",
        "jev_bias": "LONG",
        "jev_tradeable": True,
    }
    fields.update(overrides)
    return ShortlistCandidate(**fields)


class TestBuildShortlist:
    def test_accepts_valid_candidate(self):
        result = build_shortlist([_candidate()])
        assert result.symbols == ("BTC/USDT",)
        assert result.rejections == ()
        entry = result.candidates[0]
        assert entry.setup == "TREND_PULLBACK"
        assert entry.regime == "TRENDING_UP"
        assert entry.advisories == ()

    def test_rejects_unknown_regime(self):
        result = build_shortlist([_candidate(regime="UNKNOWN")])
        assert result.rejections[0].reasons == (REASON_REGIME_UNKNOWN,)

    def test_rejects_missing_allowed_setups(self):
        result = build_shortlist([_candidate(regime="UNKNOWN", allowed_setups=())])
        assert result.rejections[0].reasons == (
            REASON_REGIME_UNKNOWN,
            REASON_NO_ALLOWED_SETUPS,
        )

    def test_accepts_real_p0a_trend_up_candidate(self):
        result = build_shortlist(
            [
                _candidate(
                    regime="trend_up",
                    allowed_setups=(
                        "trend_pullback",
                        "breakout_retest",
                        "momentum_continuation",
                    ),
                )
            ]
        )
        assert result.symbols == ("BTC/USDT",)
        assert result.candidates[0].regime == "TRENDING_UP"

    def test_accepts_real_p0a_trend_down_setup(self):
        result = build_shortlist(
            [
                _candidate(
                    regime="trend_down",
                    allowed_setups=("rally_failure", "breakdown_retest", "momentum_short"),
                    setup="rally_failure",
                    direction="SHORT",
                    jev_bias="SHORT",
                )
            ]
        )
        assert result.symbols == ("BTC/USDT",)
        assert result.candidates[0].setup == "RALLY_FAILURE"

    def test_rejects_explicit_empty_allowed_setups_for_known_regime(self):
        result = build_shortlist(
            [_candidate(regime="trend_up", allowed_setups=(), setup="TREND_PULLBACK")]
        )
        assert result.symbols == ()
        assert result.rejections[0].reasons == (REASON_NO_ALLOWED_SETUPS,)

    def test_rejects_disabled_timeframe_empty_allowed_setups(self):
        result = build_shortlist(
            [_candidate(regime="range", allowed_setups=(), setup="RANGE_FADE_EXTREME")]
        )
        assert result.symbols == ()
        assert result.rejections[0].reasons == (REASON_NO_ALLOWED_SETUPS,)

    @pytest.mark.parametrize("regime", ["low_vol", "high_vol", "unclear"])
    def test_rejects_no_trade_regimes_as_no_allowed_setups(self, regime):
        result = build_shortlist([_candidate(regime=regime, allowed_setups=(), setup=None)])
        assert result.symbols == ()
        assert result.rejections[0].reasons == (REASON_NO_ALLOWED_SETUPS,)
        assert REASON_REGIME_UNKNOWN not in result.rejections[0].reasons

    def test_rejects_setup_outside_p0a_allowed_list(self):
        result = build_shortlist([_candidate(setup="RANGE_FADE")])
        assert result.rejections[0].reasons == (REASON_SETUP_NOT_ALLOWED,)

    def test_rejects_open_position(self):
        result = build_shortlist([_candidate(has_open_position=True)])
        assert REASON_OPEN_POSITION in result.rejections[0].reasons

    def test_rejects_missing_atr(self):
        result = build_shortlist([_candidate(atr_pct=None)])
        assert result.rejections[0].reasons == (REASON_ATR_UNAVAILABLE,)

    def test_rejects_atr_out_of_bounds(self):
        low = build_shortlist([_candidate(atr_pct=0.1)])
        high = build_shortlist([_candidate(atr_pct=9.0)])
        assert low.rejections[0].reasons == (REASON_ATR_TOO_LOW,)
        assert high.rejections[0].reasons == (REASON_ATR_TOO_HIGH,)

    def test_jev_is_advisory_and_does_not_drop_signal(self):
        result = build_shortlist(
            [
                _candidate(
                    direction="LONG",
                    jev_bias="SHORT",
                    jev_tradeable=False,
                )
            ]
        )
        assert result.symbols == ("BTC/USDT",)
        assert set(result.candidates[0].advisories) == {
            ADVISORY_JEV_NOT_TRADEABLE,
            ADVISORY_JEV_BIAS_MISMATCH,
        }

    def test_order_is_input_order(self):
        result = build_shortlist(
            [
                _candidate(symbol="ETH/USDT"),
                _candidate(symbol="BTC/USDT"),
                _candidate(symbol="SOL/USDT"),
            ]
        )
        assert result.symbols == ("ETH/USDT", "BTC/USDT", "SOL/USDT")

    def test_invalid_atr_bounds(self):
        with pytest.raises(ValueError):
            build_shortlist([], min_atr_pct=2.0, max_atr_pct=2.0)


class TestFlagIsolation:
    def test_disabled_flag_returns_none(self):
        assert maybe_build_shortlist(SimpleNamespace(SHORTLIST_V2=False), [_candidate()]) is None
        assert maybe_build_shortlist({}, [_candidate()]) is None

    def test_enabled_flag_runs_shortlist(self):
        result = maybe_build_shortlist(SimpleNamespace(SHORTLIST_V2=True), [_candidate()])
        assert result is not None
        assert result.symbols == ("BTC/USDT",)


class TestRecall:
    def test_recall_and_calls_saved(self):
        report = evaluate_recall(
            ["BTC/USDT", "ETH/USDT", "SOL/USDT"],
            ["BTC/USDT", "SOL/USDT"],
            full_llm_calls=10,
            shortlist_llm_calls=2,
        )
        assert report.total_signals == 3
        assert report.captured == 2
        assert report.missed == ("ETH/USDT",)
        assert report.recall == pytest.approx(2 / 3)
        assert report.llm_calls_saved == 8
        assert not meets_recall_guardrail(report, min_recall=0.95)
        assert meets_recall_guardrail(report, min_recall=0.5)

    def test_merge_reports(self):
        first = evaluate_recall(["BTC/USDT"], ["BTC/USDT"], full_llm_calls=10, shortlist_llm_calls=1)
        second = evaluate_recall(["ETH/USDT"], [], full_llm_calls=10, shortlist_llm_calls=0)
        merged = merge_recall_reports([first, second])
        assert merged.total_signals == 2
        assert merged.captured == 1
        assert merged.missed == ("ETH/USDT",)
        assert merged.llm_calls_saved == 19

    def test_no_reference_signals_counts_as_full_recall(self):
        report = evaluate_recall([], ["BTC/USDT"], full_llm_calls=10)
        assert report.recall == 1.0


class TestCandidateMapping:
    def test_nested_record(self):
        candidate = candidate_from_mapping(
            {
                "symbol": "BTC/USDT",
                "p0a": {
                    "regime": "TRENDING_UP",
                    "allowed_setups": "TREND_PULLBACK,BREAKOUT_RETEST",
                    "atr_pct": "2.5",
                },
                "setup": "TREND_PULLBACK",
                "open_position": False,
                "jev": {"bias": "LONG", "tradeable": True},
            }
        )
        assert candidate.symbol == "BTC/USDT"
        assert candidate.allowed_setups == ("TREND_PULLBACK", "BREAKOUT_RETEST")
        assert candidate.atr_pct == pytest.approx(2.5)
        assert candidate.jev_bias == "LONG"
