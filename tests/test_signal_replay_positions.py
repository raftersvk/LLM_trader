"""Tests for the position-aware replay (reuse entries, manage open candles only)."""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

import numpy as np

from src.signal_service.contract import Signal
from src.signal_service.replay_positions import (
    _entry_index,
    _exit_on_candle,
    replay_positions_for_pair,
    SimPosition,
)
from src.signal_service.storage import UpdateStorage

UTC = timezone.utc
T0 = datetime(2026, 10, 1, 0, 0, tzinfo=UTC)
SYMBOL = "BTC/USDT"


def _candle(hours, open_, high, low, close):
    ts = T0 + timedelta(hours=hours)
    return [ts.timestamp() * 1000.0, open_, high, low, close, 1.0]


def _entry():
    return {
        "schema_version": "1.0",
        "decision_id": "entry-1",
        "source": "llm_trader",
        "symbol": SYMBOL,
        "timeframe": "4h",
        "candle_open_time": "2026-10-01T00:00:00Z",
        "candle_close_time": "2026-10-01T04:00:00Z",
        "created_at": "2026-10-01T04:00:02Z",
        "expires_at": "2026-10-01T08:00:00Z",
        "action": "BUY",
        "confidence": 0.7,
        "risk_approved": True,
        "sl": 95.0,
        "tp": 115.0,
        "position_size_pct": 0.05,
        "reasoning": "entry",
        "model": "m",
        "provider": "p",
        "prompt_version": "v1",
    }


def _update(action, revision, sl=None, tp=None, close_reason=None):
    return Signal(
        schema_version="1.1",
        decision_id=f"entry-1-u{revision}",
        source="llm_trader",
        symbol=SYMBOL,
        timeframe="4h",
        candle_open_time="2026-10-01T04:00:00Z",
        candle_close_time="2026-10-01T08:00:00Z",
        created_at="2026-10-01T08:00:00Z",
        expires_at="2026-10-01T12:00:00Z",
        action=action,
        confidence=0.0,
        risk_approved=True,
        sl=sl,
        tp=tp,
        position_size_pct=0.0,
        reasoning="manage",
        model="m",
        provider="p",
        position_id="entry-1",
        position_side="long",
        revision=revision,
        based_on_entry="entry-1",
        close_reason=close_reason,
        source_price=100.5,
    )


class _DummyInjector:
    def __init__(self):
        self.as_of = []

    def set_as_of(self, as_of):
        self.as_of.append(as_of)

    def restore(self):
        pass


class _FakeAnalyzer:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def position_metrics(self, snapshot, as_of=None):
        return {"current_price": snapshot.current_price}

    async def analyze(self, candle_open_time=None, position_context="", update_context=None):
        self.calls.append(
            {
                "candle": candle_open_time,
                "context": position_context,
                "revision": update_context.revision,
            }
        )
        return self.responses.pop(0)


def _run(tmp_path, analyzer, candles, timestamps, end_hours=16):
    storage = UpdateStorage(str(tmp_path / "BTC_USDT_4h.jsonl"))
    stats = asyncio.run(
        replay_positions_for_pair(
            analyzer=analyzer,
            symbol=SYMBOL,
            timeframe="4h",
            candles=np.array(candles, dtype=np.float64),
            timestamps=timestamps,
            entries=[_entry()],
            start=T0,
            end=T0 + timedelta(hours=end_hours),
            update_storage=storage,
            max_hold_hours=24.0,
            injector=_DummyInjector(),
        )
    )
    return stats, storage


def _ts(*hours):
    return [T0 + timedelta(hours=h) for h in hours]


def test_replay_analyses_only_candles_with_an_open_position(tmp_path):
    candles = [
        _candle(0, 100.0, 100.5, 99.5, 100.0),
        _candle(4, 100.0, 101.0, 99.0, 100.5),   # entry at this open; still open
        _candle(8, 100.5, 101.0, 100.2, 100.8), # still open; CLOSE decided
        _candle(12, 100.8, 101.0, 100.0, 100.2), # closed at this open
    ]
    analyzer = _FakeAnalyzer([
        _update("UPDATE", 1, sl=100.0, tp=115.0),
        _update("CLOSE", 2, close_reason="llm_close"),
    ])
    stats, storage = _run(tmp_path, analyzer, candles, _ts(0, 4, 8, 12))
    # Candle 00 has no position; 04 and 08 are managed; 12 is after the close.
    assert [c["candle"] for c in analyzer.calls] == [
        "2026-10-01T04:00:00Z", "2026-10-01T08:00:00Z",
    ]
    assert stats.analysis_calls == 2
    assert stats.updates == 2
    assert stats.closes == 1
    stored = [line for line in open(storage.path).read().splitlines() if line.strip()]
    assert len(stored) == 2
    # Replay must stamp the simulated candle close, not wall-clock now.
    import json
    first = json.loads(stored[0])
    assert first["created_at"] == "2026-10-01T08:00:00Z"


def test_replay_never_analyses_when_stop_is_hit_intrabar(tmp_path):
    candles = [
        _candle(0, 100.0, 100.5, 99.5, 100.0),
        _candle(4, 100.0, 100.5, 94.0, 96.0),  # low <= 95 -> SL
        _candle(8, 96.0, 97.0, 95.0, 96.0),
    ]
    analyzer = _FakeAnalyzer([])
    stats, _ = _run(tmp_path, analyzer, candles, _ts(0, 4, 8))
    assert analyzer.calls == []
    assert stats.analysis_calls == 0


def test_replay_never_analyses_when_tp_close_is_hit(tmp_path):
    candles = [
        _candle(0, 100.0, 100.5, 99.5, 100.0),
        _candle(4, 100.0, 116.0, 99.0, 116.0),  # close >= 115 -> TP
        _candle(8, 116.0, 117.0, 115.0, 116.0),
    ]
    analyzer = _FakeAnalyzer([])
    stats, _ = _run(tmp_path, analyzer, candles, _ts(0, 4, 8))
    assert analyzer.calls == []
    assert stats.analysis_calls == 0


def test_exit_on_candle_uses_stop_intrabar_and_tp_on_close():
    pos = SimPosition(
        enter_tag="e", side="long", entry=100.0, sl=95.0, tp=110.0,
        opened_at=T0, entry_sl=95.0, entry_tp=110.0,
    )
    assert _exit_on_candle(pos, _candle(4, 100.0, 111.0, 96.0, 105.0)) is None
    assert _exit_on_candle(pos, _candle(4, 100.0, 111.0, 96.0, 110.0)) == ("tp", 110.0)
    assert _exit_on_candle(pos, _candle(4, 100.0, 101.0, 94.0, 105.0)) == ("sl", 95.0)
    short = SimPosition(
        enter_tag="s", side="short", entry=100.0, sl=105.0, tp=90.0,
        opened_at=T0, entry_sl=105.0, entry_tp=90.0,
    )
    assert _exit_on_candle(short, _candle(4, 100.0, 106.0, 95.0, 100.0)) == ("sl", 105.0)
    assert _exit_on_candle(short, _candle(4, 100.0, 101.0, 89.0, 90.0)) == ("tp", 90.0)


def test_entry_index_ignores_unapproved_and_hold_rows():
    entries = [_entry(), {**_entry(), "decision_id": "h", "action": "HOLD"}]
    entries.append({**_entry(), "decision_id": "x", "risk_approved": False})
    index = _entry_index(entries)
    assert list(index) == [datetime(2026, 10, 1, 4, 0, tzinfo=UTC)]
