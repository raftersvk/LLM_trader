"""Unit tests for the aligned JEV candle fetcher (no network)."""

from __future__ import annotations

import importlib.util
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "fetch_jev_bench_candles",
    Path(__file__).resolve().parents[1] / "scripts" / "fetch_jev_bench_candles.py",
)
assert _SPEC is not None and _SPEC.loader is not None
fetch = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = fetch
_SPEC.loader.exec_module(fetch)


def test_market_symbol_maps_spot_to_perpetual_contract():
    """Futures symbols gain the USDT settle suffix; spot symbols are untouched."""
    assert fetch._market_symbol("SOL/USDT", "futures") == "SOL/USDT:USDT"
    assert fetch._market_symbol("SOL/USDT:USDT", "futures") == "SOL/USDT:USDT"
    assert fetch._market_symbol("SOL/USDT", "spot") == "SOL/USDT"
    assert fetch._base_name("SOL/USDT:USDT") == "SOL_USDT"


def test_timeframe_seconds_units():
    """Supported timeframe suffixes convert to seconds."""
    assert fetch.timeframe_seconds("1h") == 3600
    assert fetch.timeframe_seconds("15m") == 900
    assert fetch.timeframe_seconds("1d") == 86400


def test_timeframe_seconds_rejects_unknown_unit():
    """An unknown suffix fails loudly instead of guessing."""
    with pytest.raises(SystemExit):
        fetch.timeframe_seconds("1w")


def test_last_closed_open_excludes_forming_candle():
    """The current in-progress hour is never used as the end candle."""
    now = datetime(2026, 10, 25, 6, 40, tzinfo=timezone.utc)
    assert fetch._last_closed_open(now, 3600) == datetime(2026, 10, 25, 5, 0, tzinfo=timezone.utc)


def test_write_csv_matches_loader_format(tmp_path):
    """Rows use ISO-Z time and eight decimals, and the temp file is replaced."""
    path = tmp_path / "TEST_USDT_1h.csv"
    opened = int(datetime(2026, 10, 25, 5, 0, tzinfo=timezone.utc).timestamp() * 1000)
    fetch._write_csv(str(path), [[opened, 1.0, 2.0, 0.5, 1.5, 10.0]])

    lines = path.read_text(encoding="utf-8").splitlines()
    assert lines[0] == "time,open,high,low,close,volume"
    assert lines[1] == "2026-10-25T05:00:00Z,1.00000000,2.00000000,0.50000000,1.50000000,10.00000000"
    assert not (tmp_path / "TEST_USDT_1h.csv.tmp").exists()
