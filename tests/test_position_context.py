"""Tests for the enriched position-management context and metrics."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np

from src.signal_service.analyzer_adapter import PairAnalyzer
from src.signal_service.positions import PositionSnapshot, format_position_context

UTC = timezone.utc
START = datetime(2026, 10, 1, 0, 0, tzinfo=UTC)


class _Ctx:
    def __init__(self, candles, timestamps):
        self.ohlcv_candles = candles
        self.timestamps = timestamps
        self.current_price = float(candles[-1][4])
        self.timeframe = "4h"


class _Engine:
    def __init__(self, candles, timestamps):
        self.context = _Ctx(candles, timestamps)


def _candles(n=20):
    rows = []
    timestamps = []
    for i in range(n):
        open_at = START + timedelta(hours=4 * i)
        base = 100.0 + i
        rows.append([open_at.timestamp() * 1000.0, base, base + 2, base - 2, base + 1, 10.0])
        timestamps.append(open_at)
    return np.array(rows, dtype=np.float64), timestamps


def test_format_position_context_includes_management_metrics():
    position = PositionSnapshot(
        position_id="entry-1", symbol="BTC/USDT", side="long", entry_price=100.0,
        amount=1.0, stop_loss=95.0, take_profit=115.0,
        open_date="2026-10-01T00:00:00Z",
    )
    metrics = {
        "current_price": 108.0, "pnl_R": 1.6, "mfe_R": 2.1, "mae_R": -0.4,
        "hours_in_trade": 12.0, "atr": 3.2, "atr_pct": 2.96,
    }
    text = format_position_context(position, 100.0, 115.0, metrics)
    for fragment in (
        "current_price=108", "unrealised_pnl_R=+1.60", "mfe_R=+2.10 mae_R=-0.40",
        "hours_in_trade=12.0", "atr=3.2 (2.96% of price)",
        "answer UPDATE", "answer CLOSE",
    ):
        assert fragment in text


def test_format_position_context_degrades_without_metrics():
    position = PositionSnapshot(
        position_id="entry-1", symbol="BTC/USDT", side="short", entry_price=100.0,
        stop_loss=105.0,
    )
    text = format_position_context(position, 100.0, None, None)
    assert "side=short" in text
    assert "take_profit=unknown" in text
    assert "current_price" not in text


def test_position_metrics_computes_r_and_atr():
    candles, timestamps = _candles(20)
    analyzer = PairAnalyzer(engine=_Engine(candles, timestamps), symbol="BTC/USDT", timeframe="4h")
    position = PositionSnapshot(
        position_id="entry-1", symbol="BTC/USDT", side="long", entry_price=100.0,
        amount=1.0, stop_loss=95.0, take_profit=130.0,
        open_date=timestamps[0].isoformat().replace("+00:00", "Z"),
    )
    as_of = timestamps[-1] + timedelta(hours=4)
    metrics = analyzer.position_metrics(position, as_of=as_of)
    assert metrics["current_price"] == 120.0  # last close (100+19+1)
    assert metrics["pnl_R"] > 3.0
    assert metrics["mfe_R"] > metrics["pnl_R"]
    assert metrics["hours_in_trade"] == 80.0
    assert metrics["atr"] > 0
    assert 0 < metrics["atr_pct"] < 10


def test_position_metrics_without_open_date_degrades():
    """A missing open_date must not raise: hours are just omitted."""
    candles, timestamps = _candles(20)
    analyzer = PairAnalyzer(engine=_Engine(candles, timestamps), symbol="BTC/USDT", timeframe="4h")
    position = PositionSnapshot(
        position_id="entry-1", symbol="BTC/USDT", side="long", entry_price=100.0,
        amount=1.0, stop_loss=95.0,
    )
    metrics = analyzer.position_metrics(position, as_of=timestamps[-1] + timedelta(hours=4))
    assert metrics["current_price"] == 120.0
    assert "hours_in_trade" not in metrics
    assert metrics["pnl_R"] > 0
