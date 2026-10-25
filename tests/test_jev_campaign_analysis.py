"""Unit tests for the JEV multi-symbol campaign aggregation.

No network: every test uses a synthetic report in the saved calibration shape.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import pytest

_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import jev_campaign_analysis as analysis  # noqa: E402


def _state(index: int, noul: float, forward: float) -> Dict[str, Any]:
    """Build one saved calibration state with an h1 label."""
    return {
        "index": index,
        "noul": noul,
        "forward_returns": {"1": forward},
        "worth_trading": {"1": abs(forward) * 100.0 >= 1.0},
    }


def _report(
    states: List[Dict[str, Any]],
    symbol: str = "TEST/USDT",
    market_type: str = "spot",
) -> Dict[str, Any]:
    """Wrap states in the minimal report shape the aggregator reads."""
    return {
        "dataset": {"symbol": symbol, "market_type": market_type,
                    "source": "synthetic.csv", "model": "test"},
        "scoring": {"calls": len(states), "scored": len(states), "errors": 0,
                    "rate_limit_errors": 0, "other_errors": 0},
        "aborted": False,
        "abort_reason": None,
        "states": states,
    }


def test_partition_and_top_bottom_select_use_rank_slices():
    """The 30 % cut keeps 30 % on each side and never overlaps."""
    records = [(float(index), index >= 7) for index in range(10)]

    rejected, kept = analysis._partition_by_score(records, 0.30)
    assert len(rejected) == 3
    assert len(kept) == 7
    assert max(score for score, _event in rejected) < min(score for score, _event in kept)

    top, bottom = analysis._top_bottom_select(0.30)(records)
    assert len(top) == 3 and len(bottom) == 3
    assert min(score for score, _event in top) > max(score for score, _event in bottom)


def test_wilson_interval_edges():
    """Wilson intervals handle the empty and all-events cases."""
    assert analysis.wilson_interval(0, 0) is None
    interval = analysis.wilson_interval(5, 10)
    assert interval is not None
    assert interval["low"] < 0.5 < interval["high"]
    certain = analysis.wilson_interval(10, 10)
    assert certain is not None and certain["high"] == 1.0


def test_symbol_summary_detects_a_clean_ranking_signal():
    """A perfect rank separation produces AUC 1 and a positive rejection."""
    states = [_state(index, 0.1 * (index + 1), 0.02 if index >= 6 else -0.005)
              for index in range(10)]
    summary = analysis._symbol_summary(_report(states))

    assert summary["prevalence"]["events"] == 4
    assert summary["tradeability_auc"]["auc"] == 1.0
    assert summary["tradeability_auc"]["proven_95"] is True
    assert summary["direction"]["auc"] == 1.0
    assert summary["direction"]["direction_proven_95"] is True
    assert summary["top_vs_bottom"]["top"]["rate"] == 1.0
    assert summary["top_vs_bottom"]["bottom"]["rate"] == 0.0
    assert summary["rank_rejection"][0]["better_than_noise"] is True
    assert analysis._verdict(summary) == "rank rejection separates event rates at all three cuts"


def test_symbol_summary_reports_noise_as_not_proven():
    """A constant score carries no ranking information (AUC exactly 0.5)."""
    states = [_state(index, 0.5, 0.02 if index % 2 == 0 else -0.005)
              for index in range(10)]
    summary = analysis._symbol_summary(_report(states))

    assert summary["tradeability_auc"]["auc"] == 0.5
    assert summary["tradeability_auc"]["proven_95"] is False
    assert summary["direction"]["direction_proven_95"] is False
    assert analysis._verdict(summary) == "no reliable rejection rule today"


@pytest.mark.parametrize("fraction", [0.30, 0.40, 0.50])
def test_bootstrap_difference_is_deterministic(fraction):
    """The same seed reproduces the same bootstrap interval."""
    records = [(float(index), index >= 6) for index in range(10)]
    first = analysis._bootstrap_difference(records, analysis._reject_select(fraction), resamples=200)
    second = analysis._bootstrap_difference(records, analysis._reject_select(fraction), resamples=200)
    assert first == second


def test_bootstrap_independent_delta_detects_large_event_rate_gap():
    """A 50 % vs 10 % event rate is reported as spot minus futures."""
    spot = [_state(index, 0.5, 0.02 if index < 25 else -0.005) for index in range(50)]
    futures = [_state(index, 0.5, 0.02 if index < 5 else -0.005) for index in range(50)]

    result = analysis._bootstrap_independent_delta(
        spot, futures, analysis._event_rate_stat, resamples=500
    )
    assert result is not None
    assert result["observed"] == pytest.approx(0.4)
    assert result["ci95"]["low"] > 0.0
    assert result["within_noise"] is False


def test_compare_markets_same_states_is_within_noise():
    """Identical spot and futures samples have zero delta and no significance."""
    states = [_state(index, 0.1 * (index + 1), 0.02 if index >= 6 else -0.005)
              for index in range(10)]
    report = _report(states)
    summary = analysis._symbol_summary(report)
    summary["verdict"] = analysis._verdict(summary)

    comparison = analysis._compare_markets(states, states, summary, summary)
    assert comparison["all_metric_differences_within_noise"] is True
    assert comparison["significant_metrics"] == []
    assert "same conclusion" in comparison["conclusion"]


def test_build_summary_reports_skipped_futures_symbol(tmp_path):
    """A spot-only symbol is listed as skipped, not silently compared."""
    states = [_state(index, 0.1 * (index + 1), 0.02 if index >= 6 else -0.005)
              for index in range(10)]
    spot_test = tmp_path / "calibration_test_200.json"
    spot_nexo = tmp_path / "calibration_nexo_200.json"
    futures_test = tmp_path / "calibration_test_200_futures.json"
    spot_test.write_text(json.dumps(_report(states, "TEST/USDT", "spot")), encoding="utf-8")
    spot_nexo.write_text(json.dumps(_report(states, "NEXO/USDT", "spot")), encoding="utf-8")
    futures_test.write_text(
        json.dumps(_report(states, "TEST/USDT", "futures")), encoding="utf-8"
    )

    summary = analysis.build_summary(
        [str(spot_test), str(spot_nexo)], [str(futures_test)]
    )
    assert len(summary["symbols"]) == 2
    assert len(summary["futures_symbols"]) == 1
    assert len(summary["comparisons"]) == 1
    assert summary["comparisons"][0]["symbol"] == "TEST/USDT"
    assert summary["futures_skipped"] == [
        {"symbol": "NEXO/USDT", "reason": "no Binance USDT-M perpetual market"}
    ]
    assert summary["multiple_testing"]["symbol_markets"] == 3
