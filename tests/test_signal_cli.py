"""Tests for the signal service `validate` subcommand and its script wrapper."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from src.signal_service.__main__ import (
    DEFAULT_SCHEMA_PATH,
    _build_parser,
    main,
)
from src.signal_service.contract import Signal, build_decision_id

CANDLE_OPEN = "2026-10-23T14:00:00Z"


def _valid_signal_line() -> str:
    signal = Signal(
        schema_version="1.0",
        decision_id=build_decision_id("BTC/USDT", "1h", CANDLE_OPEN),
        source="llm_trader",
        symbol="BTC/USDT",
        timeframe="1h",
        candle_open_time=CANDLE_OPEN,
        candle_close_time="2026-10-23T15:00:00Z",
        created_at="2026-10-23T15:00:02Z",
        expires_at="2026-10-23T16:00:00Z",
        action="HOLD",
        confidence=0.0,
        risk_approved=False,
        sl=None,
        tp=None,
        position_size_pct=0.05,
        reasoning="test",
        model="test-model",
        provider="test-provider",
    )
    signal.validate()
    return json.dumps(signal.to_dict())


def test_validate_defaults_to_the_shared_schema():
    args = _build_parser().parse_args(["validate"])
    assert args.schema == DEFAULT_SCHEMA_PATH


def test_validate_accepts_a_schema_override(tmp_path):
    args = _build_parser().parse_args(["validate", "--schema", str(tmp_path / "s.json")])
    assert args.schema == str(tmp_path / "s.json")


def test_validate_missing_schema_fails_cleanly(tmp_path, capsys):
    code = main([
        "validate",
        "--file", str(tmp_path / "signals.jsonl"),
        "--schema", str(tmp_path / "missing.json"),
    ])

    assert code == 2
    out = capsys.readouterr().out
    assert "schema file not found" in out
    assert "--schema" in out


def test_validate_unparsable_schema_fails_cleanly(tmp_path, capsys):
    schema = tmp_path / "broken.json"
    schema.write_text("{not json", encoding="utf-8")

    code = main(["validate", "--file", str(tmp_path / "signals.jsonl"), "--schema", str(schema)])

    assert code == 2
    assert "cannot load schema" in capsys.readouterr().out


def test_validate_uses_the_override_schema(tmp_path, capsys):
    schema = tmp_path / "schema.json"
    schema.write_text(json.dumps({"type": "object"}), encoding="utf-8")
    signals = tmp_path / "BTC_USDT_1h.jsonl"
    signals.write_text(_valid_signal_line() + "\n", encoding="utf-8")

    pytest.importorskip("jsonschema")
    code = main(["validate", "--file", str(signals), "--schema", str(schema)])

    assert code == 0
    assert "checked=1 failures=0" in capsys.readouterr().out


def test_script_delegates_and_reports_missing_schema(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [
            sys.executable, str(repo_root / "scripts" / "validate_signals.py"),
            "--file", str(tmp_path / "signals.jsonl"),
            "--schema", str(tmp_path / "missing.json"),
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 2
    assert "schema file not found" in result.stdout
