"""Unit tests for the JEV tradeability calibration math.

No network and no JEV call: every test uses synthetic records and labels.
"""

from __future__ import annotations

import importlib.util
import json
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


def test_threshold_curve_allows_missing_bias():
    """check_pre only blocks an explicit ``none`` bias, not a missing answer."""
    record = _record(0.9, None, "BUY", 0.02)
    row = jev_calibration.threshold_curve([record], [0.5], horizon=1)[0]

    assert row["trade_count"] == 1
    assert row["hold_count"] == 0
    assert row["base_trade_precision"] == pytest.approx(1.0)


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



def test_records_from_payload_preserves_explicit_zero_cost():
    """A stored 0.0 cost is not replaced by the 0.10 default."""
    payload = _full_record(0, 0.4).to_payload()
    payload["cost_pct"] = 0.0
    assert jev_calibration.records_from_payload([payload])[0].cost_pct == 0.0

    payload.pop("cost_pct")
    assert jev_calibration.records_from_payload([payload])[0].cost_pct == 0.10

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



def test_write_observations_merges_pairs_instead_of_truncating(tmp_path):
    """A BTC run followed by an ETH run keeps both pairs in the log."""
    path = tmp_path / "observations.jsonl"
    btc = _report([
        _full_record(0, 0.4, "2026-10-15T09:00:00Z"),
        _full_record(1, 0.5, "2026-10-15T10:00:00Z"),
    ], "BTC/USDT")
    eth = _report([
        _full_record(2, 0.3, "2026-10-15T09:00:00Z"),
        _full_record(3, 0.6, "2026-10-15T10:00:00Z"),
    ], "ETH/USDT")

    assert jev_calibration.write_observations(btc, str(path)) == 2
    assert jev_calibration.write_observations(eth, str(path)) == 4

    payloads = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert [payload["pair"] for payload in payloads] == ["BTC/USDT", "BTC/USDT", "ETH/USDT", "ETH/USDT"]
    assert len({(payload["pair"], payload["candle_open_time"]) for payload in payloads}) == 4


def test_write_observations_dedupes_on_pair_and_candle_time(tmp_path):
    """Re-scoring the same state replaces its line in place."""
    path = tmp_path / "observations.jsonl"
    first = _report([_full_record(0, 0.4, "2026-10-15T09:00:00Z")], "BTC/USDT")
    second = _report([_full_record(0, 0.9, "2026-10-15T09:00:00Z")], "BTC/USDT")

    jev_calibration.write_observations(first, str(path))
    assert jev_calibration.write_observations(second, str(path)) == 1

    payloads = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert len(payloads) == 1
    assert payloads[0]["noul"] == 0.9


def test_aborted_run_does_not_replace_an_existing_log(tmp_path):
    """A partial aborted run leaves a non-empty observation log untouched."""
    path = tmp_path / "observations.jsonl"
    good = _report([
        _full_record(0, 0.4, "2026-10-15T09:00:00Z"),
        _full_record(1, 0.5, "2026-10-15T10:00:00Z"),
    ], "BTC/USDT")
    jev_calibration.write_observations(good, str(path))
    before = path.read_text(encoding="utf-8")

    aborted = _report([_full_record(5, 0.8, "2026-10-16T00:00:00Z")], "BTC/USDT")
    aborted["aborted"] = True
    written = jev_calibration.write_observations(aborted, str(path))

    assert written == 2
    assert path.read_text(encoding="utf-8") == before


def test_aborted_run_writes_when_no_log_exists(tmp_path):
    """A first aborted run may create the log; later runs merge into it."""
    path = tmp_path / "observations.jsonl"
    aborted = _report([_full_record(0, 0.4, "2026-10-15T09:00:00Z")], "BTC/USDT")
    aborted["aborted"] = True

    assert jev_calibration.write_observations(aborted, str(path)) == 1
    assert path.exists()

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
        thresholds=(0.2, 0.45, 0.5), cost_pct=0.1,
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


def _full_record(
    index: int,
    noul: float,
    opened: str = "2026-10-15T09:00:00Z",
    cost_pct: float = 0.1,
) -> Any:
    """Record carrying every horizon so report validation succeeds."""
    forward = 0.01 * (index + 1)
    return CalibrationRecord(
        index=index,
        as_of=opened,
        noul=noul,
        candle_open_time=opened,
        bias="long",
        regime="trend",
        atr_pct=0.01,
        cost_pct=cost_pct,
        forward_returns={1: forward, 2: forward, 3: forward, 4: forward},
        worth_trading={horizon: False for horizon in (1, 2, 3, 4)},
        base_action="BUY",
        base_direction_correct={1: True, 2: True, 3: True, 4: True},
    )


def _report(records, symbol: str) -> Dict[str, Any]:
    """Build a minimal valid report for observation-log tests."""
    return jev_calibration.build_report(
        records, _source_args(symbol=symbol), (1, 2, 3, 4),
        "2026-10-15T00:00:00Z", 0.0, False, None, "test",
    )


def _source_args(**overrides: Any) -> Namespace:
    """Report-producing args with non-default provenance fields."""
    values = {
        "candles": "/tmp/does-not-exist-candles.csv",
        "symbol": "ETH/USDT",
        "timeframe": "4h",
        "candle_count": 600,
        "move_threshold_pct": 2.0,
        "delay_seconds": 0.25,
        "horizon": 2,
        "thresholds": (0.2, 0.5),
        "cost_pct": 0.1,
    }
    values.update(overrides)
    return Namespace(**values)



@pytest.mark.asyncio
async def test_aggregate_from_carries_dataset_cost_pct(tmp_path):
    """A flag-free rebuild keeps the report's stored cost, not the default."""
    records = [_full_record(0, 0.4, cost_pct=0.05), _full_record(1, 0.6, cost_pct=0.05)]
    report = jev_calibration.build_report(
        records, _source_args(cost_pct=0.05), (1, 2, 3, 4),
        "2026-10-15T00:00:00Z", 0.0, False, None, "test",
    )
    path = tmp_path / "calibration.json"
    path.write_text(json.dumps(report), encoding="utf-8")

    args = jev_calibration._parse_args(["--aggregate-from", str(path)])
    rebuilt = await jev_calibration.run_calibration(args)

    assert rebuilt["dataset"]["cost_pct"] == pytest.approx(0.05)
    assert all(state["cost_pct"] == pytest.approx(0.05) for state in rebuilt["states"])


@pytest.mark.asyncio
async def test_aggregate_from_carries_explicit_zero_cost(tmp_path):
    """Zero is a valid stored cost and must survive a flag-free rebuild."""
    records = [_full_record(0, 0.4, cost_pct=0.0)]
    report = jev_calibration.build_report(
        records, _source_args(cost_pct=0.0), (1, 2, 3, 4),
        "2026-10-15T00:00:00Z", 0.0, False, None, "test",
    )
    path = tmp_path / "calibration.json"
    path.write_text(json.dumps(report), encoding="utf-8")

    rebuilt = await jev_calibration.run_calibration(
        jev_calibration._parse_args(["--aggregate-from", str(path)])
    )

    assert rebuilt["dataset"]["cost_pct"] == 0.0
    assert rebuilt["states"][0]["cost_pct"] == 0.0


@pytest.mark.asyncio
async def test_aggregate_from_explicit_cost_overrides_dataset(tmp_path):
    """An explicit --cost-pct still overrides the stored cost."""
    records = [_full_record(0, 0.4, cost_pct=0.05)]
    report = jev_calibration.build_report(
        records, _source_args(cost_pct=0.05), (1, 2, 3, 4),
        "2026-10-15T00:00:00Z", 0.0, False, None, "test",
    )
    path = tmp_path / "calibration.json"
    path.write_text(json.dumps(report), encoding="utf-8")

    rebuilt = await jev_calibration.run_calibration(
        jev_calibration._parse_args(
            ["--aggregate-from", str(path), "--cost-pct", "0.2"]
        )
    )

    assert rebuilt["dataset"]["cost_pct"] == pytest.approx(0.2)
    assert rebuilt["states"][0]["cost_pct"] == pytest.approx(0.2)


def test_fresh_run_default_cost_is_applied_only_when_resolved():
    """--cost-pct stays None in the parser and defaults on a fresh run."""
    args = jev_calibration._parse_args(["--candles", "candles.csv"])
    assert args.cost_pct is None
    jev_calibration._apply_run_defaults(args)
    assert args.cost_pct == pytest.approx(0.10)

def test_validate_calibration_report_accepts_and_rejects():
    """Validation requires a dataset, a non-empty states list and state keys."""
    report = jev_calibration.build_report(
        [_full_record(0, 0.4), _full_record(1, 0.6)], _source_args(), (1, 2, 3, 4),
        "2026-10-15T00:00:00Z", 0.0, False, None, "test",
    )
    assert jev_calibration.validate_calibration_report(report) == []

    assert "missing dataset" in jev_calibration.validate_calibration_report({})
    empty = jev_calibration.validate_calibration_report({"dataset": {}, "states": []})
    assert "dataset missing symbol" in empty
    assert "states must be a non-empty list" in empty
    bad_state = jev_calibration.validate_calibration_report(
        {"dataset": {"symbol": "BTC/USDT", "timeframe": "1h",
                     "horizons": [1], "move_threshold_pct": 1.0},
         "states": [{"index": 0}]}
    )
    assert any("state 0 missing" in problem for problem in bad_state)


@pytest.mark.asyncio
async def test_aggregate_from_carries_dataset_provenance(tmp_path):
    """A flag-free rebuild keeps the source report's dataset fields."""
    report = jev_calibration.build_report(
        [_full_record(0, 0.4), _full_record(1, 0.6)], _source_args(), (2, 3),
        "2026-10-15T00:00:00Z", 0.0, False, None, "test",
    )
    path = tmp_path / "calibration_eth.json"
    path.write_text(json.dumps(report), encoding="utf-8")

    args = jev_calibration._parse_args(["--aggregate-from", str(path)])
    rebuilt = await jev_calibration.run_calibration(args)

    dataset = rebuilt["dataset"]
    assert dataset["source"] == "/tmp/does-not-exist-candles.csv"
    assert dataset["symbol"] == "ETH/USDT"
    assert dataset["timeframe"] == "4h"
    assert dataset["move_threshold_pct"] == 2.0
    assert dataset["delay_seconds"] == 0.25
    assert dataset["candles"] == 600
    assert dataset["horizons"] == [2, 3]
    assert set(rebuilt["worth_rate"]) == {"h2", "h3"}


@pytest.mark.asyncio
async def test_aggregate_from_explicit_flags_override_provenance(tmp_path):
    """Explicit CLI options win over the loaded report's dataset."""
    report = jev_calibration.build_report(
        [_full_record(0, 0.4), _full_record(1, 0.6)], _source_args(), (2, 3),
        "2026-10-15T00:00:00Z", 0.0, False, None, "test",
    )
    path = tmp_path / "calibration_eth.json"
    path.write_text(json.dumps(report), encoding="utf-8")

    args = jev_calibration._parse_args([
        "--aggregate-from", str(path),
        "--symbol", "SOL/USDT", "--timeframe", "15m",
        "--move-threshold-pct", "0.5", "--delay-seconds", "1.5",
        "--horizons", "1,4", "--horizon", "4",
    ])
    rebuilt = await jev_calibration.run_calibration(args)

    dataset = rebuilt["dataset"]
    assert dataset["symbol"] == "SOL/USDT"
    assert dataset["timeframe"] == "15m"
    assert dataset["move_threshold_pct"] == 0.5
    assert dataset["delay_seconds"] == 1.5
    assert dataset["horizons"] == [1, 4]
    assert set(rebuilt["worth_rate"]) == {"h1", "h4"}


@pytest.mark.asyncio
async def test_aggregate_from_rejects_invalid_payload(tmp_path):
    """A malformed saved report must stop before any rebuild."""
    path = tmp_path / "broken.json"
    path.write_text(json.dumps({"dataset": {}, "states": []}), encoding="utf-8")
    args = jev_calibration._parse_args(["--aggregate-from", str(path)])

    with pytest.raises(SystemExit, match="invalid calibration report"):
        await jev_calibration.run_calibration(args)


def test_record_payload_is_json_friendly():
    """Serialized records keep labels and omit raw prompts."""
    payload = _record(0.42, "long", "BUY", 0.015).to_payload()
    assert payload["noul"] == 0.42
    assert payload["forward_returns"] == {"1": 0.015}
    assert payload["worth_trading"] == {"1": True}
    assert payload["base_direction_correct"] == {"1": True}
    assert isinstance(payload["forward_returns"], Dict)


# ------------------------------------------------------- ranking and skill
def test_ranking_auc_known_value_and_ties():
    """AUC is rank-based; tied scores contribute half a win."""
    scores = [0.1, 0.4, 0.35, 0.8]
    labels = [False, False, True, True]
    assert jev_calibration.ranking_auc(scores, labels) == pytest.approx(0.75)
    assert jev_calibration.ranking_auc([0.5, 0.5], [True, False]) == pytest.approx(0.5)
    assert jev_calibration.ranking_auc([0.1, 0.2], [False, False]) is None
    assert jev_calibration.ranking_auc([], []) is None


def test_average_precision_known_value():
    """PR-AUC is average precision over ranked scores."""
    scores = [0.8, 0.4, 0.35, 0.1]
    labels = [True, False, True, False]
    assert jev_calibration.average_precision(scores, labels) == pytest.approx(5.0 / 6.0)
    assert jev_calibration.average_precision([0.2], [False]) is None


def test_brier_skill_score_reference_semantics():
    """Skill is 1 minus the ratio to the reference; a tie scores zero."""
    assert jev_calibration.brier_skill_score(0.25, 0.25) == pytest.approx(0.0)
    assert jev_calibration.brier_skill_score(0.10, 0.25) == pytest.approx(0.6)
    assert jev_calibration.brier_skill_score(0.10, None) is None
    assert jev_calibration.brier_skill_score(0.10, 0.0) is None


def test_logistic_baseline_is_finite_and_deterministic():
    """The one-feature logistic baseline produces a usable, repeatable Brier."""
    probabilities = [0.05, 0.1, 0.2, 0.4, 0.6, 0.7, 0.8, 0.9]
    labels = [False, False, False, False, True, False, True, True]

    first = jev_calibration.logistic_brier(probabilities, labels)
    second = jev_calibration.logistic_brier(probabilities, labels)

    assert first is not None and 0.0 <= first <= 1.0
    assert first == second
    model = jev_calibration.fit_logistic_regression(probabilities, labels)
    assert model is not None
    assert all(isinstance(value, float) for value in model)
    assert jev_calibration.fit_logistic_regression([0.2, 0.3], [True, True]) is None


def test_bootstrap_cis_are_deterministic():
    """Bootstrap CIs depend on the fixed seed only, not on call order."""
    scores = [0.1, 0.2, 0.3, 0.6, 0.7, 0.9]
    labels = [False, False, True, False, True, True]

    first_auc = jev_calibration.bootstrap_auc_ci(scores, labels, resamples=200)
    second_auc = jev_calibration.bootstrap_auc_ci(scores, labels, resamples=200)
    first_net = jev_calibration.bootstrap_net_ci([-0.2, 0.1, 0.3, -0.1], resamples=200)
    second_net = jev_calibration.bootstrap_net_ci([-0.2, 0.1, 0.3, -0.1], resamples=200)

    assert first_auc == second_auc
    assert first_net == second_net
    assert first_auc is not None and first_auc["low"] <= first_auc["high"]
    assert first_net is not None and first_net["low"] <= first_net["high"]
    assert jev_calibration.bootstrap_net_ci([0.1]) is None


# ---------------------------------------------------------------- economics
def _tail_records(forward: float, count: int = 10) -> list[Any]:
    """Records with a tradeable chronological tail and inert training states."""
    records = []
    for index in range(count):
        if index >= count - 3:
            record = _record(0.9, "long", "BUY", forward)
        else:
            record = _record(0.1, "long", "HOLD", 0.001)
        record.index = index
        record.candle_open_time = f"2026-10-15T{index:02d}:00:00Z"
        records.append(record)
    return records


def test_economics_cost_sensitivity_flags_sign_flip():
    """A result that flips sign inside the cost range is not an edge."""
    economics = jev_calibration.economics_report(_tail_records(0.0015), horizon=1)

    assert economics["evaluated_on"] == "oos"
    assert economics["oos_states"] == 3
    assert economics["oos_first"] == "2026-10-15T07:00:00Z"
    assert economics["cost_pct"]["0.05"]["net_pct"] > 0.0
    assert economics["cost_pct"]["0.20"]["net_pct"] < 0.0
    assert economics["sign_flip"] is True
    assert economics["verdict"] == "no demonstrated edge (cost-sensitive sign flip)"


def test_economics_uses_only_the_oos_tail():
    """Training states never contribute trades to the economics block."""
    records = _tail_records(0.0015)
    for record in records[:7]:
        record.noul = 0.99
        record.base_action = "BUY"
        record.base_direction_correct[1] = True
        record.forward_returns[1] = 0.05

    economics = jev_calibration.economics_report(records, horizon=1)

    assert economics["train_states"] == 7
    assert economics["oos_states"] == 3
    assert economics["cost_pct"]["0.10"]["trades"] == 3


def test_economics_keeps_symbols_separate():
    """Each call consumes only its own states; nothing is pooled across symbols."""
    btc = _tail_records(-0.01)
    eth = _tail_records(0.01)

    btc_economics = jev_calibration.economics_report(btc, horizon=1)
    eth_economics = jev_calibration.economics_report(eth, horizon=1)
    pooled = jev_calibration.economics_report(btc + eth, horizon=1)

    assert btc_economics["oos_states"] == 3
    assert eth_economics["oos_states"] == 3
    assert pooled["oos_states"] == 6
    assert btc_economics["cost_pct"]["0.10"]["net_pct"] < 0.0
    assert eth_economics["cost_pct"]["0.10"]["net_pct"] > 0.0


def test_report_has_ranking_metrics_and_low_prevalence_warning():
    """A rare label yields skill scores, AUC/PR-AUC and an explicit warning."""
    records = _tail_records(0.0015)
    report = jev_calibration.build_report(
        records, _source_args(), (1,), "2026-10-15T00:00:00Z", 0.0, False, None, "test"
    )
    metrics = report["by_horizon"]["h1"]

    assert metrics["prevalence"] == pytest.approx(0.0)
    assert metrics["brier_constant"] == pytest.approx(0.0)
    assert metrics["prevalence_warning"] is not None
    assert metrics["directional_auc"]["state_count"] == 3
    assert any("trivially strong" in warning for warning in report["warnings"])
    assert metrics["economics"]["evaluated_on"] == "oos"


def test_build_report_is_deterministic():
    """The same records must produce byte-identical JSON payloads."""
    records = _tail_records(0.0015)
    first = jev_calibration.build_report(
        records, _source_args(), (1, 2), "2026-10-15T00:00:00Z", 0.0, False, None, "test"
    )
    second = jev_calibration.build_report(
        records, _source_args(), (1, 2), "2026-10-15T00:00:00Z", 0.0, False, None, "test"
    )
    assert json.dumps(first) == json.dumps(second)
