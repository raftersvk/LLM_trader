"""Tests for the offline shortlist A/B script and recall accounting."""

import importlib.util
import json
from pathlib import Path

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "ab_shortlist.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("ab_shortlist_under_test", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _record(symbol, signal="HOLD", *, cycle="c1", setup="TREND_PULLBACK", allowed=None,
            atr_pct=2.0, open_position=False):
    return {
        "cycle": cycle,
        "symbol": symbol,
        "p0a": {
            "regime": "TRENDING_UP",
            "allowed_setups": allowed or ["TREND_PULLBACK", "BREAKOUT_RETEST"],
            "atr_pct": atr_pct,
        },
        "setup": setup,
        "open_position": open_position,
        "jev": {"bias": "LONG", "tradeable": True},
        "expected_signal": signal,
    }


class TestLoadRecords:
    def test_json_array(self, tmp_path):
        script = _load_script()
        path = tmp_path / "signals.json"
        path.write_text(json.dumps([_record("BTC/USDT"), _record("ETH/USDT")]), encoding="utf-8")
        records = script.load_records(path)
        assert [record["symbol"] for record in records] == ["BTC/USDT", "ETH/USDT"]

    def test_json_lines(self, tmp_path):
        script = _load_script()
        path = tmp_path / "signals.jsonl"
        lines = "\n".join(json.dumps(_record(symbol)) for symbol in ("BTC/USDT", "ETH/USDT"))
        path.write_text(lines, encoding="utf-8")
        records = script.load_records(path)
        assert len(records) == 2

    def test_object_with_records_key(self, tmp_path):
        script = _load_script()
        path = tmp_path / "signals.json"
        path.write_text(json.dumps({"records": [_record("BTC/USDT")]}), encoding="utf-8")
        assert len(script.load_records(path)) == 1


class TestRunAb:
    def test_recall_missed_reasons_and_calls_saved(self):
        script = _load_script()
        records = [
            _record("BTC/USDT", "BUY"),
            _record("ETH/USDT", "SELL", setup="RANGE_FADE"),
            _record("SOL/USDT", "HOLD"),
            _record("XRP/USDT", "BUY", cycle="c2", open_position=True),
            _record("ADA/USDT", "SELL", cycle="c2"),
        ]

        summary = script.run_ab(records, min_recall=0.95)

        assert summary["offline"] is True
        assert summary["cycles"] == 2
        assert summary["reference_signals"] == 4
        assert summary["captured"] == 2
        assert summary["recall"] == 0.5
        assert set(summary["missed"]) == {"ETH/USDT", "XRP/USDT"}
        assert summary["full_llm_calls"] == 5
        assert summary["shortlist_llm_calls"] == 3
        assert summary["llm_calls_saved"] == 2
        assert not summary["meets_recall_guardrail"]
        reasons = {item["symbol"]: item["reasons"] for item in summary["missed_with_reasons"]}
        assert reasons["ETH/USDT"] == ["setup_not_allowed"]
        assert reasons["XRP/USDT"] == ["open_position"]

    def test_full_pairs_override_models_ten_pair_path(self):
        script = _load_script()
        summary = script.run_ab([_record("BTC/USDT", "BUY")], full_pairs=10)
        assert summary["full_llm_calls"] == 10
        assert summary["shortlist_llm_calls"] == 1
        assert summary["llm_calls_saved"] == 9

    def test_guardrail_pass(self):
        script = _load_script()
        summary = script.run_ab(
            [_record("BTC/USDT", "BUY"), _record("ETH/USDT", "SELL")],
            min_recall=0.95,
        )
        assert summary["recall"] == 1.0
        assert summary["meets_recall_guardrail"]

    def test_cli_fails_below_guardrail(self, tmp_path, capsys):
        script = _load_script()
        path = tmp_path / "signals.json"
        path.write_text(
            json.dumps(
                [
                    _record("BTC/USDT", "BUY"),
                    _record("ETH/USDT", "SELL", setup="RANGE_FADE"),
                ]
            ),
            encoding="utf-8",
        )
        exit_code = script.main(["--dataset", str(path), "--fail-below-guardrail"])
        captured = capsys.readouterr()
        assert exit_code == 2
        assert "Recall guardrail" in captured.out
        assert "ETH/USDT" in captured.out
