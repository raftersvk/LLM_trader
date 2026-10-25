"""Unit tests for the read-only JEV post-gate veto impact report.

No network: the candle source is injected as a plain dict of candle series and
every test uses fixtures.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional

import pytest

_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import jev_validator_report as report  # noqa: E402

FIXTURES = Path(__file__).resolve().parent / "fixtures"
GOLDEN = FIXTURES / "jev_validator_report_expected.md"


def _candle(open_iso: str, close: float) -> report.Candle:
    """Build one flat OHLCV candle with a given close."""
    opened = datetime.fromisoformat(open_iso.replace("Z", "+00:00"))
    return report.Candle(
        open_at=opened, open=close, high=close, low=close, close=close, volume=1.0
    )


def _record(
    open_iso: str,
    action: str,
    risk_approved: bool = False,
    reasoning: str = "",
    jev: Optional[Dict[str, object]] = None,
    symbol: str = "BTC/USDT",
    decision_id: Optional[str] = None,
) -> report.SignalRecord:
    """Build one stored signal record."""
    opened = datetime.fromisoformat(open_iso.replace("Z", "+00:00"))
    return report.SignalRecord(
        decision_id=decision_id or f"{symbol}-{open_iso}-{action}",
        symbol=symbol,
        timeframe="1h",
        candle_open_time=opened,
        action=action,
        risk_approved=risk_approved,
        reasoning=reasoning,
        jev=jev,
    )


# ------------------------------------------------------------------ parsing
def test_parse_veto_noul_present_with_repeated_prefix():
    """The live double prefix parses and the noul comes from the text."""
    info = report.parse_veto_info(
        "some analysis [JEV veto: JEV veto: not tradeable (noul=0.273 < 0.5)]"
    )
    assert info.present is True
    assert info.kind == "tradeability"
    assert info.noul == pytest.approx(0.273)
    assert "not tradeable" in info.detail


def test_parse_veto_noul_absent_is_never_invented():
    """A tradeability veto without a numeric noul keeps noul None."""
    info = report.parse_veto_info(
        "text [JEV veto: JEV veto: not tradeable (noul=unknown < 0.5)]"
    )
    assert info.present is True
    assert info.kind == "tradeability"
    assert info.noul is None


def test_parse_veto_bias_and_absent_reasoning():
    """Bias reasons parse, and a missing reasoning is not a veto."""
    bias = report.parse_veto_info(
        "text [JEV veto: JEV veto: long bias conflicts with SELL]"
    )
    assert bias.kind == "bias"
    assert bias.noul is None
    assert report.parse_veto_info("").present is False
    assert report.parse_veto_info(None).present is False


def test_parse_veto_confidence_and_plain_reason():
    """Confidence and unknown reasons stay distinguishable."""
    confidence = report.parse_veto_info("[JEV veto: confidence 0.40 < 0.60]")
    assert confidence.kind == "confidence"
    other = report.parse_veto_info("[JEV veto: something unexpected]")
    assert other.kind == "other"


def test_classify_falls_back_to_payload_without_suffix():
    """A legacy vetoed row without the suffix is classified from jev."""
    tradeability = _record(
        "2026-01-01T00:00:00Z",
        "SELL",
        risk_approved=False,
        jev={"tradeability": False, "bias": "short", "regime": "range"},
    )
    assert report.classify_signal(tradeability) == ("vetoed", "tradeability", None)
    bias = _record(
        "2026-01-01T01:00:00Z",
        "BUY",
        risk_approved=False,
        jev={"tradeability": True, "bias": "short", "regime": "trend"},
    )
    assert report.classify_signal(bias) == ("vetoed", "bias", None)
    rejected = _record(
        "2026-01-01T02:00:00Z",
        "BUY",
        risk_approved=False,
        jev={"tradeability": True, "bias": "long", "regime": "trend"},
    )
    assert report.classify_signal(rejected) == ("other_rejected", None, None)


# ---------------------------------------------------------------------- P&L
def test_counterfactual_buy_and_sell_include_cost():
    """BUY profits when price rises; SELL profits when it falls; cost is paid."""
    candles = {
        datetime(2026, 1, 1, 0, tzinfo=timezone.utc): _candle(
            "2026-01-01T00:00:00Z", 100.0
        ),
        datetime(2026, 1, 1, 1, tzinfo=timezone.utc): _candle(
            "2026-01-01T01:00:00Z", 110.0
        ),
    }
    buy = _record("2026-01-01T00:00:00Z", "BUY")
    sell = _record("2026-01-01T00:00:00Z", "SELL")

    buy_result = report.counterfactual(buy, candles, 1, 0.10)
    sell_result = report.counterfactual(sell, candles, 1, 0.10)
    assert buy_result.status == "ok"
    assert buy_result.net_pct == pytest.approx(9.9)
    assert sell_result.net_pct == pytest.approx(-10.1)


def test_counterfactual_horizon_and_missing_candles():
    """A two-hour exit uses the +2 candle; gaps are reported, not zeroed."""
    candles = {
        datetime(2026, 1, 1, 0, tzinfo=timezone.utc): _candle(
            "2026-01-01T00:00:00Z", 100.0
        ),
        datetime(2026, 1, 1, 2, tzinfo=timezone.utc): _candle(
            "2026-01-01T02:00:00Z", 90.0
        ),
    }
    buy = _record("2026-01-01T00:00:00Z", "BUY")
    sell = _record("2026-01-01T00:00:00Z", "SELL")
    assert report.counterfactual(buy, candles, 2, 0.10).net_pct == pytest.approx(-10.1)
    assert report.counterfactual(sell, candles, 2, 0.10).net_pct == pytest.approx(9.9)
    assert report.counterfactual(buy, candles, 1, 0.10).status == "missing_exit"

    absent = _record("2026-01-01T05:00:00Z", "BUY")
    assert report.counterfactual(absent, candles, 1, 0.10).status == "missing_entry"


# --------------------------------------------------------------- aggregation
def test_summarize_avoided_net_and_false_veto_cost():
    """Avoided net sums all vetoes; false-veto cost sums only winners."""
    records = [
        _record(
            "2026-01-01T00:00:00Z",
            "SELL",
            reasoning="[JEV veto: not tradeable (noul=0.200 < 0.5)]",
        ),
        _record(
            "2026-01-01T01:00:00Z",
            "SELL",
            reasoning="[JEV veto: JEV veto: long bias conflicts with SELL]",
        ),
        _record("2026-01-01T02:00:00Z", "BUY", risk_approved=True),
    ]
    candles = {
        ("BTC/USDT", "1h"): [
            _candle("2026-01-01T00:00:00Z", 100.0),
            _candle("2026-01-01T01:00:00Z", 95.0),
            _candle("2026-01-01T02:00:00Z", 100.0),
            _candle("2026-01-01T03:00:00Z", 105.0),
        ]
    }
    evaluated = report.evaluate_signals(records, candles, (1,), 0.10)
    data = report.summarize(evaluated, (1,), 0.10, 1, report.LoadStats())

    stats = data.overall[1]
    assert stats.vetoed_total == 2
    assert stats.vetoed_evaluable == 2
    # First vetoed SELL: 100 -> 95 is +4.9 net (a false veto).
    # Second vetoed SELL: 100/95 - 1 is -5.263%, minus 0.1% => -5.363% net.
    assert stats.vetoed_net_pct == pytest.approx(4.9 - 5.363157894736842)
    assert stats.false_veto_count == 1
    assert stats.false_veto_cost_pct == pytest.approx(4.9)
    assert stats.loss_avoided_pct == pytest.approx(-5.363157894736842)
    assert stats.approved_evaluable == 1
    assert stats.approved_net_pct == pytest.approx(4.9)

    counts = data.symbols["BTC/USDT"].counts
    assert (counts.total, counts.approved, counts.vetoed) == (3, 1, 2)
    assert (counts.tradeability, counts.bias, counts.other_rejected) == (1, 1, 0)


def test_horizon_validated_requires_sample_and_small_false_cost():
    """The decision rule needs sample, negative avoided net and small false cost."""
    good = report.HorizonStats(
        horizon=1,
        vetoed_evaluable=report.MIN_VETO_SAMPLE,
        vetoed_net_pct=-5.0,
        false_veto_count=1,
        false_veto_cost_pct=0.5,
        loss_avoided_pct=-5.5,
    )
    assert report.horizon_validated(good, True) == (
        True,
        "net avoided is negative and false-veto cost is small",
    )
    assert report.horizon_validated(good, False)[0] is False

    too_few = report.HorizonStats(horizon=1, vetoed_evaluable=3, vetoed_net_pct=-5.0)
    assert report.horizon_validated(too_few, True)[0] is False

    costly = report.HorizonStats(
        horizon=1,
        vetoed_evaluable=report.MIN_VETO_SAMPLE,
        vetoed_net_pct=-1.0,
        false_veto_count=2,
        false_veto_cost_pct=4.0,
        loss_avoided_pct=-5.0,
    )
    assert report.horizon_validated(costly, True)[0] is False


# ------------------------------------------------------------------- loading
def test_load_signals_handles_malformed_absent_and_duplicates(tmp_path):
    """Bad lines are skipped; HOLD ignored; duplicate decisions collapse."""
    path = tmp_path / "BTC_USDT_1h.jsonl"
    lines = [
        json.dumps(
            {
                "decision_id": "hold-1",
                "symbol": "BTC/USDT",
                "timeframe": "1h",
                "candle_open_time": "2026-01-01T00:00:00Z",
                "action": "HOLD",
                "risk_approved": False,
                "reasoning": "",
            }
        ),
        json.dumps(
            {
                "decision_id": "veto-1",
                "symbol": "BTC/USDT",
                "timeframe": "1h",
                "candle_open_time": "2026-01-01T01:00:00Z",
                "action": "SELL",
                "risk_approved": False,
                "reasoning": "x [JEV veto: not tradeable (noul=0.2 < 0.5)]",
                "jev": {"tradeability": False, "bias": "none", "regime": "range"},
            }
        ),
        # Duplicate decision id: must be skipped.
        json.dumps(
            {
                "decision_id": "veto-1",
                "symbol": "BTC/USDT",
                "timeframe": "1h",
                "candle_open_time": "2026-01-01T01:00:00Z",
                "action": "SELL",
                "risk_approved": False,
                "reasoning": "duplicate",
            }
        ),
        "{not json",
        json.dumps({"symbol": "BTC/USDT", "action": "BUY"}),
        json.dumps(
            {
                "symbol": "BTC/USDT",
                "timeframe": "1h",
                "candle_open_time": "2026-01-01T02:00:00Z",
                "action": "BUY",
                "risk_approved": False,
            }
        ),
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    (tmp_path / "heartbeat_BTC_USDT_1h.json").write_text("{}\n", encoding="utf-8")

    records, stats = report.load_signals(str(tmp_path))

    assert stats.files_read == 1
    assert stats.holds_ignored == 1
    assert stats.duplicates == 1
    assert stats.malformed == 2
    assert len(records) == 2
    assert records[0].decision_id == "veto-1"
    assert report.classify_signal(records[0])[:2] == ("vetoed", "tradeability")
    # Absent jev and reasoning: rejected, but never treated as a JEV veto.
    assert report.classify_signal(records[1]) == ("other_rejected", None, None)


# -------------------------------------------------------------------- golden
def _golden_data() -> report.ReportData:
    """Small fixture covering an approved trade, a saved loss and a false veto."""
    records = [
        _record(
            "2026-01-01T00:00:00Z",
            "SELL",
            reasoning="analysis [JEV veto: JEV veto: not tradeable (noul=0.200 < 0.5)]",
            jev={"tradeability": False, "bias": "none", "regime": "range"},
        ),
        _record(
            "2026-01-01T01:00:00Z",
            "SELL",
            reasoning="analysis [JEV veto: JEV veto: long bias conflicts with SELL]",
            jev={"tradeability": True, "bias": "long", "regime": "trend"},
        ),
        _record("2026-01-01T02:00:00Z", "BUY", risk_approved=True),
    ]
    candles = {
        ("BTC/USDT", "1h"): [
            _candle("2026-01-01T00:00:00Z", 100.0),
            _candle("2026-01-01T01:00:00Z", 95.0),
            _candle("2026-01-01T02:00:00Z", 97.0),
            _candle("2026-01-01T03:00:00Z", 100.0),
            _candle("2026-01-01T04:00:00Z", 104.0),
        ]
    }
    evaluated = report.evaluate_signals(records, candles, (1, 2), 0.10)
    stats = report.LoadStats(
        files_read=1, lines_read=4, holds_ignored=1, malformed=0, duplicates=0
    )
    return report.summarize(evaluated, (1, 2), 0.10, 3, stats)


def test_golden_report_is_stable():
    """A fixed small input renders the exact same Markdown every time."""
    data = _golden_data()
    first = report.render_report(data, "<signals-dir>")
    second = report.render_report(_golden_data(), "<signals-dir>")
    assert first == second
    assert first == GOLDEN.read_text(encoding="utf-8")


def test_report_contains_the_two_judging_numbers():
    """The rendered report names avoided net and false-veto cost."""
    text = report.render_report(_golden_data(), "<signals-dir>")
    assert "Avoided net" in text
    assert "False-veto cost" in text
    assert "The sample is sufficient to judge the validator" in text
    assert "1h: **NOT VALIDATED**" in text


def test_cli_refuses_to_write_inside_signals_dir(tmp_path):
    """The report can never be written into the read-only signals directory."""
    signals_dir = tmp_path / "signals"
    signals_dir.mkdir()
    target = signals_dir / "report.md"
    with pytest.raises(SystemExit):
        report._ensure_outside(str(target), str(signals_dir), "the report")
    outside = tmp_path / "report.md"
    report._ensure_outside(str(outside), str(signals_dir), "the report")


def test_write_candles_csv_roundtrip(tmp_path):
    """The cache format round-trips for injected/fetched candle data."""
    path = tmp_path / "cache" / "BTC_USDT_1h.csv"
    candles = [_candle("2026-01-01T00:00:00Z", 100.0)]
    report.write_candles_csv(str(path), candles)
    loaded = report.read_candles_csv(str(path))
    assert len(loaded) == 1
    assert loaded[0].open_at == datetime(2026, 1, 1, tzinfo=timezone.utc)
    assert loaded[0].close == pytest.approx(100.0)


def test_load_signals_missing_directory_exits(tmp_path):
    """A missing directory is a clean CLI error, not a traceback."""
    with pytest.raises(SystemExit):
        report.load_signals(str(tmp_path / "nope"))


def test_load_signals_rejects_bad_timeframe_and_string_flag(tmp_path):
    """Unsupported timeframes are malformed; a string flag never approves."""
    path = tmp_path / "BTC_USDT_1h.jsonl"
    lines = [
        json.dumps(
            {
                "decision_id": "bad-tf",
                "symbol": "BTC/USDT",
                "timeframe": "2h",
                "candle_open_time": "2026-01-01T00:00:00Z",
                "action": "BUY",
                "risk_approved": True,
            }
        ),
        json.dumps(
            {
                "decision_id": "string-flag",
                "symbol": "BTC/USDT",
                "timeframe": "1h",
                "candle_open_time": "2026-01-01T01:00:00Z",
                "action": "BUY",
                "risk_approved": "false",
            }
        ),
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    records, stats = report.load_signals(str(tmp_path))

    assert stats.malformed == 1
    assert len(records) == 1
    assert records[0].risk_approved is False
    assert report.classify_signal(records[0]) == ("other_rejected", None, None)


def test_resolve_signals_dir_honors_env_and_explicit_override(monkeypatch, tmp_path):
    """The default follows LLM_TRADER_SIGNALS_DIR; an explicit flag wins."""
    monkeypatch.setenv("LLM_TRADER_SIGNALS_DIR", str(tmp_path))
    assert report.resolve_signals_dir(None) == str(tmp_path)
    assert report.resolve_signals_dir("/explicit/signals") == "/explicit/signals"
