"""Tests for the signal service `validate` subcommand and its script wrapper."""

import json
import os
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


def test_defaults_are_repo_relative(monkeypatch):
    """Defaults derive from the repo, never from a personal home directory."""
    import importlib

    from src.signal_service import storage

    repo_root = storage.PROJECT_ROOT
    monkeypatch.delenv("LLM_TRADER_SIGNALS_DIR", raising=False)
    monkeypatch.delenv("SIGNAL_SCHEMA_PATH", raising=False)
    from src.signal_service import __main__ as cli

    importlib.reload(cli)

    assert cli.DEFAULT_SIGNALS_DIR == os.path.join(repo_root, "data", "signals")
    assert cli.DEFAULT_REPLAY_OUTPUT == os.path.join(repo_root, "data", "signals_backtest")
    assert cli.DEFAULT_SCHEMA_PATH == os.path.join(
        os.path.dirname(repo_root), "trading-stack", "contracts", "signal.schema.json",
    )


def test_signals_dir_default_can_be_overridden(monkeypatch, tmp_path):
    from src.signal_service import storage

    monkeypatch.setenv("LLM_TRADER_SIGNALS_DIR", str(tmp_path / "elsewhere"))
    assert storage.default_signals_dir() == str(tmp_path / "elsewhere")


def test_config_override_wires_the_loader_env(tmp_path, monkeypatch):
    from src.signal_service.__main__ import _apply_config_override

    monkeypatch.delenv("LLM_TRADER_CONFIG_INI", raising=False)
    config = tmp_path / "custom.ini"
    config.write_text("", encoding="utf-8")

    try:
        _apply_config_override(str(config))
        assert Path(os.environ["LLM_TRADER_CONFIG_INI"]) == config.resolve()
    finally:
        monkeypatch.delenv("LLM_TRADER_CONFIG_INI", raising=False)


def test_smoke_script_skips_missing_schema(tmp_path):
    """The offline smoke test must stay self-contained without the shared schema."""
    repo_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, str(repo_root / "scripts" / "smoke_signal_service.py")],
        cwd=repo_root,
        env={
            **{k: v for k, v in os.environ.items() if not k.startswith("LLM_TRADER_")},
            "SIGNAL_SCHEMA_PATH": str(tmp_path / "missing.json"),
        },
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "skipping JSON schema validation" in result.stderr


def test_has_closed_candle_matches_truncate_rule():
    """The skip guard must use the same close rule as truncate_context_to_as_of."""
    from datetime import datetime, timedelta, timezone

    from src.signal_service.__main__ import _has_closed_candle

    opened = datetime(2026, 10, 23, 12, tzinfo=timezone.utc)
    timestamps = [opened]

    assert _has_closed_candle(timestamps, "1h", opened + timedelta(hours=1)) is True
    assert _has_closed_candle(
        timestamps, "1h", opened + timedelta(minutes=59),
    ) is False
    assert _has_closed_candle([], "1h", opened + timedelta(hours=5)) is False


def test_replay_fails_clearly_when_window_precedes_csv(tmp_path):
    """A --start before the csv data must not die with a raw traceback."""
    import asyncio

    from src.signal_service.__main__ import _build_parser, _replay_command

    candles = tmp_path / "BTC_USDT_1h.csv"
    candles.write_text(
        "time,open,high,low,close,volume\n"
        "2026-10-23T12:00:00Z,110.0,120.0,100.0,115.0,333333.0\n",
        encoding="utf-8",
    )
    args = _build_parser().parse_args([
        "replay",
        "--pairs", "BTC/USDT",
        "--start", "2026-01-01",
        "--end", "2026-01-02",
        "--candles", str(candles),
        "--output", str(tmp_path / "signals"),
    ])

    with pytest.raises(ValueError) as excinfo:
        asyncio.run(_replay_command(args))

    message = str(excinfo.value)
    assert "no candle closed at or before" in message
    assert "check --start/--end/--as-of or the csv range" in message
