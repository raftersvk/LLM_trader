"""Unit tests for the JEV multi-symbol campaign aggregation.

No network: every test uses a synthetic report in the saved calibration shape.
"""

from __future__ import annotations

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


def _report(states: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Wrap states in the minimal report shape the aggregator reads."""
    return {
        "dataset": {"symbol": "TEST/USDT", "source": "synthetic.csv", "model": "test"},
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
