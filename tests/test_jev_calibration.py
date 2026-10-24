"""Unit tests for the JEV tradeability calibration math.

No network and no JEV call: every test uses synthetic records and labels.
"""

from __future__ import annotations

import importlib.util
import sys
from argparse import Namespace
from pathlib import Path
from typing import Any, Dict

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "jev_calibration",
    Path(__file__).resolve().parents[1] / "scripts" / "jev_calibration.py",
)
assert _SPEC is not None and _SPEC.loader is not None
jev_calibration = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = jev_calibration
_SPEC.loader.exec_module(jev_calibration)

CalibrationRecord = jev_calibration.CalibrationRecord


def _record(
    noul: float | None,
    bias: str | None,
    base_action: str,
    forward: float,
    worth: bool | None = None,
) -> Any:
    """Build one calibration record for the primary 1-candle horizon."""
    correct = None
    if base_action == "BUY":
        correct = forward > 0
    elif base_action == "SELL":
        correct = forward < 0
    return CalibrationRecord(
        index=0,
        as_of="2026-01-01T00:00:00Z",
        noul=noul,
        bias=bias,
        forward_returns={1: forward},
        worth_trading={1: abs(forward) * 100.0 >= 1.0 if worth is None else worth},
        base_action=base_action,
        base_direction_correct={1: correct},
    )


def test_histogram_counts_bins_and_clamps_edges():
    """Values at 1.0 land in the last bucket and out-of-range values clamp."""
    histogram = jev_calibration.noul_histogram([0.0, 0.05, 0.1, 0.15, 0.55, 0.95, 1.0, 1.2])

    counts = {entry["bin"]: entry["count"] for entry in histogram}
    assert len(histogram) == 10
    assert counts["0.0-0.1"] == 2
    assert counts["0.1-0.2"] == 2
    assert counts["0.5-0.6"] == 1
    assert counts["0.9-1.0"] == 3
    assert sum(counts.values()) == 8


def test_noul_distribution_reports_summary_and_empty_case():
    """Distribution stats are exact and empty input is handled."""
    distribution = jev_calibration.noul_distribution([0.1, 0.2, 0.6, 0.3])
    assert distribution["count"] == 4
    assert distribution["min"] == 0.1
    assert distribution["max"] == 0.6
    assert distribution["median"] == pytest.approx(0.25)
    assert distribution["mean"] == pytest.approx(0.3)

    empty = jev_calibration.noul_distribution([])
    assert empty["count"] == 0
    assert empty["median"] is None
    assert sum(entry["count"] for entry in empty["histogram"]) == 0


def test_calibration_brier_known_values():
    """Brier is the mean squared probability error."""
    probabilities = [0.0, 1.0, 0.5, 0.5]
    labels = [False, True, True, False]
    assert jev_calibration.calibration_brier(probabilities, labels) == pytest.approx(0.125)
    assert jev_calibration.calibration_brier([], []) is None


def test_calibration_brier_rejects_length_mismatch():
    """Mismatched inputs are a programming error, not silent truncation."""
    with pytest.raises(ValueError):
        jev_calibration.calibration_brier([0.5], [True, False])


def test_reliability_bins_report_predicted_and_observed_rates():
    """Each reliability bin carries its mean prediction and observed rate."""
    bins = jev_calibration.calibration_reliability(
        [0.05, 0.15, 0.55, 0.62],
        [False, True, True, False],
        bins=5,
    )

    assert len(bins) == 5
    first = bins[0]
    assert first["count"] == 2
    assert first["mean_predicted"] == pytest.approx(0.1)
    assert first["observed_worth_rate"] == pytest.approx(0.5)
    assert bins[1]["count"] == 0
    assert "observed_worth_rate" not in bins[1]
    assert bins[2]["observed_worth_rate"] == pytest.approx(1.0)
    assert bins[3]["observed_worth_rate"] == pytest.approx(0.0)


def test_threshold_curve_mirrors_pre_gate_rules():
    """Coverage, trade count, precision, false holds and AUC at 0.5."""
    records = [
        _record(0.9, "long", "BUY", 0.02),    # taken, correct
        _record(0.3, "long", "BUY", 0.02),    # blocked: noul low
        _record(0.7, "none", "BUY", -0.02),   # blocked: bias none
        _record(0.6, "short", "BUY", -0.02),  # blocked: bias conflict
        _record(0.4, "long", "SELL", -0.02),  # blocked: noul low
        _record(0.8, "long", "HOLD", 0.005),  # not a base trade
    ]

    curve = jev_calibration.threshold_curve(records, [0.5], horizon=1)
    row = curve[0]

    assert row["threshold"] == 0.5
    assert row["coverage"] == pytest.approx(round(4 / 6, 4))  # noul >= 0.5
    assert row["trade_count"] == 1
    assert row["base_trade_precision"] == pytest.approx(1.0)
    assert row["hold_count"] == 4
    assert row["false_hold_count"] == 4
    assert row["false_hold_rate"] == pytest.approx(1.0)
    assert row["directional_auc"] == pytest.approx(1.0)


def test_threshold_curve_auc_is_none_without_both_classes():
    """A one-sided directional subset must not fabricate an AUC."""
    records = [
        _record(0.9, "long", "BUY", 0.02),
        _record(0.8, "long", "BUY", 0.03),
    ]
    row = jev_calibration.threshold_curve(records, [0.8], horizon=1)[0]
    assert row["coverage"] == pytest.approx(1.0)
    assert row["directional_auc"] is None


def test_threshold_curve_reports_none_precision_without_trades():
    """No surviving trade yields None precision, not a division by zero."""
    records = [_record(0.1, "long", "BUY", 0.02)]
    row = jev_calibration.threshold_curve(records, [0.5], horizon=1)[0]
    assert row["trade_count"] == 0
    assert row["base_trade_precision"] is None
    assert row["false_hold_rate"] == pytest.approx(1.0)


def test_threshold_curve_ignores_unscored_states():
    """Failed calls (noul None) must not enter coverage or precision."""
    records = [
        _record(0.9, "long", "BUY", 0.02),
        _record(None, None, "BUY", 0.02),
    ]
    row = jev_calibration.threshold_curve(records, [0.5], horizon=1)[0]
    assert row["coverage"] == pytest.approx(1.0)
    assert row["trade_count"] == 1
    assert row["hold_count"] == 0


def test_bias_direction_auc_uses_bias_as_score():
    """Long bias scores 1, short bias scores 0; ties count as half."""
    records = [
        _record(0.9, "long", "HOLD", 0.02),
        _record(0.9, "short", "HOLD", -0.02),
    ]
    assert jev_calibration.bias_direction_auc(records, 1) == pytest.approx(1.0)


def test_records_from_payload_round_trip():
    """A saved report's states rebuild identical records offline."""
    original = _record(0.42, "short", "SELL", -0.015)
    original.regime = "trend"
    original.confidence = 0.75

    rebuilt = jev_calibration.records_from_payload([original.to_payload()])[0]

    assert rebuilt.noul == pytest.approx(original.noul)
    assert rebuilt.bias == "short"
    assert rebuilt.regime == "trend"
    assert rebuilt.confidence == pytest.approx(0.75)
    assert rebuilt.forward_returns == {1: -0.015}
    assert rebuilt.worth_trading == {1: True}
    assert rebuilt.base_action == "SELL"
    assert rebuilt.base_direction_correct == {1: True}


def test_write_observations_emits_one_line_per_scored_state(tmp_path):
    """Observations use the required keys and skip failed JEV calls."""
    scored = _record(0.4, "long", "BUY", 0.02)
    scored.candle_open_time = "2026-10-15T09:00:00Z"
    scored.atr_pct = 0.01
    failed = CalibrationRecord(index=1, as_of="2026-10-15T10:00:00Z", noul=None)
    report = {
        "dataset": {"symbol": "BTC/USDT"},
        "states": [scored.to_payload(), failed.to_payload()],
    }
    path = tmp_path / "observations.jsonl"

    written = jev_calibration.write_observations(report, str(path))

    assert written == 1
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    payload = __import__("json").loads(lines[0])
    assert payload["pair"] == "BTC/USDT"
    assert payload["noul"] == 0.4
    assert payload["candle_open_time"] == "2026-10-15T09:00:00Z"
    assert payload["forward_1h"] == 0.02
    assert payload["cost_pct"] == 0.1


def test_payload_preserves_metric_inputs_at_full_precision():
    """noul, ATR and forward returns must not be rounded in the payload."""
    record = CalibrationRecord(
        index=0,
        as_of="2026-10-15T09:00:00Z",
        noul=0.45000000001,
        atr_pct=0.00123456789,
        forward_returns={1: 1e-9, 2: -1e-9, 3: 0.0, 4: 0.0000000002},
        worth_trading={1: False, 2: False, 3: False, 4: False},
        base_action="BUY",
        base_direction_correct={1: True, 2: False, 3: False, 4: True},
    )

    payload = record.to_payload()

    assert payload["noul"] == 0.45000000001
    assert payload["atr_pct"] == 0.00123456789
    assert payload["forward_returns"]["1"] == 1e-9
    assert payload["forward_returns"]["2"] == -1e-9
    assert payload["forward_returns"]["4"] == 0.0000000002


def test_aggregate_from_payload_reproduces_embedded_metrics():
    """A saved report rebuilt from its payload must match its own metrics."""
    records = [
        CalibrationRecord(
            index=index,
            as_of="2026-10-15T09:00:00Z",
            noul=noul,
            atr_pct=atr,
            bias=bias,
            regime=regime,
            forward_returns={1: forward, 2: forward * 2, 3: forward * 3, 4: forward * 4},
            worth_trading={horizon: abs(forward * horizon) * 100 >= 1.0
                           for horizon in (1, 2, 3, 4)},
            base_action=action,
            base_direction_correct={
                1: (action == "BUY" and forward > 0) or (action == "SELL" and forward < 0)
                if action in ("BUY", "SELL") else None,
                2: None, 3: None, 4: None,
            },
        )
        for index, (noul, atr, bias, regime, action, forward) in enumerate([
            (0.19999999999, 0.001, "long", "trend", "BUY", 1e-9),
            (0.45, 0.002, "long", "trend", "BUY", -1e-9),
            (0.45000000001, 0.003, "short", "range", "SELL", -1e-9),
            (0.9, 0.004, "none", "high_vol", "HOLD", 0.02),
        ])
    ]
    args = Namespace(
        candles="synthetic", symbol="BTC/USDT", timeframe="1h", candle_count=10,
        move_threshold_pct=1.0, delay_seconds=0.0, horizon=1,
        thresholds=(0.2, 0.45, 0.5),
    )
    metric_keys = ("noul", "brier", "reliability", "worth_rate", "base",
                   "threshold_curve", "by_horizon")

    first = jev_calibration.build_report(
        records, args, (1, 2, 3, 4), "2026-10-15T00:00:00Z", 0.0, False, None, "test"
    )
    rebuilt = jev_calibration.records_from_payload(first["states"])
    second = jev_calibration.build_report(
        rebuilt, args, (1, 2, 3, 4), "2026-10-15T00:00:00Z", 0.0, False, None, "test"
    )

    for key in metric_keys:
        assert second[key] == first[key], key
    # The exact 0.45 boundary state and the tiny positive return survive.
    assert first["states"][2]["noul"] == 0.45000000001
    assert first["states"][0]["forward_returns"]["1"] == 1e-9


def test_record_payload_is_json_friendly():
    """Serialized records keep labels and omit raw prompts."""
    payload = _record(0.42, "long", "BUY", 0.015).to_payload()
    assert payload["noul"] == 0.42
    assert payload["forward_returns"] == {"1": 0.015}
    assert payload["worth_trading"] == {"1": True}
    assert payload["base_direction_correct"] == {"1": True}
    assert isinstance(payload["forward_returns"], Dict)
