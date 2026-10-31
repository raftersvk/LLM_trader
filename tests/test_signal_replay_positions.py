"""Tests for the position-aware replay (reuse entries, manage open candles only)."""

from __future__ import annotations

import asyncio
import os
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
    def __init__(self, responses, delay=0.0):
        self.responses = list(responses)
        self.delay = delay
        self.calls = []

    def position_metrics(self, snapshot, as_of=None):
        return {"current_price": snapshot.current_price}

    async def analyze(self, candle_open_time=None, position_context="", update_context=None):
        self.calls.append(
            {
                "candle": candle_open_time,
                "context": position_context,
                "revision": update_context.revision,
                "entry_price": update_context.entry_price,
                "position_id": update_context.position_id,
                "based_on_entry": update_context.based_on_entry,
                "sl": update_context.stop_loss,
            }
        )
        if self.delay:
            await asyncio.sleep(self.delay)
        return self.responses.pop(0)


def _run(
    tmp_path,
    analyzer,
    candles,
    timestamps,
    end_hours=16,
    *,
    entries=None,
    max_hold_hours=24.0,
    timeout_seconds=300.0,
    market=None,
    start=None,
    storage=None,
):
    storage = storage or UpdateStorage(str(tmp_path / "BTC_USDT_4h.jsonl"))
    stats = asyncio.run(
        replay_positions_for_pair(
            analyzer=analyzer,
            symbol=SYMBOL,
            timeframe="4h",
            candles=np.array(candles, dtype=np.float64),
            timestamps=timestamps,
            entries=[_entry()] if entries is None else entries,
            start=start or T0,
            end=T0 + timedelta(hours=end_hours),
            update_storage=storage,
            max_hold_hours=max_hold_hours,
            timeout_seconds=timeout_seconds,
            market=market,
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


def test_replay_fills_at_the_next_candle_open(tmp_path):
    # Signal closes at 04:00; the fill must be the 04:00 open (102.0), not the
    # signal candle close.
    candles = [
        _candle(0, 100.0, 100.5, 99.5, 100.0),
        _candle(4, 102.0, 103.0, 101.0, 102.5),
    ]
    analyzer = _FakeAnalyzer([_update("HOLD", 1)])
    stats, _ = _run(tmp_path, analyzer, candles, _ts(0, 4), end_hours=8)
    assert stats.analysis_calls == 1
    assert analyzer.calls[0]["entry_price"] == 102.0
    assert analyzer.calls[0]["position_id"] == "replay:entry-1"


def test_replay_respects_max_hold_hours(tmp_path):
    candles = [
        _candle(0, 100.0, 100.5, 99.5, 100.0),
        _candle(4, 100.0, 101.0, 99.0, 100.5),
        _candle(8, 100.5, 101.0, 100.0, 100.8),
        _candle(12, 100.8, 101.0, 100.0, 100.2),
    ]
    analyzer = _FakeAnalyzer([_update("HOLD", 1)])
    stats, _ = _run(
        tmp_path, analyzer, candles, _ts(0, 4, 8, 12),
        max_hold_hours=4.0,
    )
    # Opened at 04:00, TTL at the 08:00 open (4h) closes it with no LLM call.
    assert [c["candle"] for c in analyzer.calls] == ["2026-10-01T04:00:00Z"]
    assert stats.analysis_calls == 1


def test_exit_on_candle_gap_fills_at_open():
    long_pos = SimPosition(
        enter_tag="e", side="long", entry=100.0, sl=95.0, tp=110.0,
        opened_at=T0, entry_sl=95.0, entry_tp=110.0,
    )
    # Gaps below the stop: the fill is the (worse) open, not the stop level.
    assert _exit_on_candle(long_pos, _candle(4, 90.0, 92.0, 88.0, 91.0)) == ("sl", 90.0)
    short_pos = SimPosition(
        enter_tag="s", side="short", entry=100.0, sl=105.0, tp=90.0,
        opened_at=T0, entry_sl=105.0, entry_tp=90.0,
    )
    assert _exit_on_candle(short_pos, _candle(4, 110.0, 112.0, 108.0, 111.0)) == ("sl", 110.0)


def test_replay_timeout_marks_failure_without_crashing(tmp_path):
    candles = [
        _candle(0, 100.0, 100.5, 99.5, 100.0),
        _candle(4, 100.0, 101.0, 99.0, 100.5),
    ]
    analyzer = _FakeAnalyzer([_update("HOLD", 1)], delay=0.05)
    stats, storage = _run(
        tmp_path, analyzer, candles, _ts(0, 4), end_hours=8, timeout_seconds=0.001,
    )
    assert stats.failed == 1
    assert stats.analysis_calls == 0
    assert not os.path.exists(storage.path)


def test_replay_market_scopes_update_ids(tmp_path):
    candles = [
        _candle(0, 100.0, 100.5, 99.5, 100.0),
        _candle(4, 100.0, 101.0, 99.0, 100.5),
    ]
    analyzer = _FakeAnalyzer([_update("UPDATE", 1, sl=100.0, tp=115.0)])
    stats, _ = _run(
        tmp_path, analyzer, candles, _ts(0, 4), end_hours=8, market="futures",
    )
    assert stats.updates == 1
    assert analyzer.calls[0]["position_id"] == "replay:futures:entry-1"
    # based_on_entry stays the real entry id so update/entry linkage still works.
    assert analyzer.calls[0]["based_on_entry"] == "entry-1"
    assert not analyzer.calls[0]["position_id"].startswith(("spot:", "futures:"))


def test_replay_rerun_reapplies_deduplicated_updates(tmp_path):
    candles = [
        _candle(0, 100.0, 100.5, 99.5, 100.0),
        _candle(4, 100.0, 101.0, 99.0, 100.5),
        _candle(8, 100.5, 101.0, 100.2, 100.8),
    ]
    storage = UpdateStorage(str(tmp_path / "BTC_USDT_4h.jsonl"))
    first = _FakeAnalyzer([
        _update("UPDATE", 1, sl=100.0, tp=115.0),
        _update("UPDATE", 2, sl=100.4, tp=115.0),
    ])
    stats1, _ = _run(
        tmp_path, first, candles, _ts(0, 4, 8), end_hours=12, storage=storage,
    )
    lines_after_first = len(open(storage.path).read().splitlines())
    assert stats1.updates == 2 and lines_after_first == 2

    # A rerun must reproduce the same trajectory: the deduplicated revision 1
    # still tightens the in-memory stop so revision 2 is requested.
    second = _FakeAnalyzer([
        _update("UPDATE", 1, sl=100.0, tp=115.0),
        _update("UPDATE", 2, sl=100.4, tp=115.0),
    ])
    stats2, _ = _run(
        tmp_path, second, candles, _ts(0, 4, 8), end_hours=12, storage=storage,
    )
    assert stats2.updates == 0
    assert [c["revision"] for c in second.calls] == [1, 2]
    assert lines_after_first == len(open(storage.path).read().splitlines())


def test_replay_off_grid_start_uses_csv_candles(tmp_path):
    candles = [
        _candle(0, 100.0, 100.5, 99.5, 100.0),
        _candle(4, 100.0, 101.0, 99.0, 100.5),
        _candle(8, 100.5, 101.0, 100.2, 100.8),
    ]
    analyzer = _FakeAnalyzer([_update("HOLD", 1)])
    stats, _ = _run(
        tmp_path, analyzer, candles, _ts(0, 4, 8),
        end_hours=8, start=T0 + timedelta(hours=1),
    )
    # The off-grid start must not silently drop the real 04:00 candle.
    assert stats.candles_seen == 1
    assert [c["candle"] for c in analyzer.calls] == ["2026-10-01T04:00:00Z"]


def test_entry_index_filters_other_symbols():
    other = {**_entry(), "decision_id": "e2", "symbol": "ETH/USDT"}
    index = _entry_index([_entry(), other], symbol=SYMBOL)
    assert list(index.values())[0]["decision_id"] == "entry-1"
    assert len(index) == 1


class _HoldEngine:
    """Minimal engine: always answers HOLD, no exchange access."""

    def __init__(self):
        from types import SimpleNamespace

        self.context = SimpleNamespace(ohlcv_candles=None, timestamps=[], current_price=None)
        self.model_manager = SimpleNamespace(
            supports_image_analysis=lambda *_a, **_k: True, last_finish_reason=None,
        )
        self.rag_engine = None
        self.data_collector = SimpleNamespace(collect_data=self._collect)

    async def _collect(self):
        return {"success": True, "errors": [], "market_context": "", "article_urls": {}}

    def initialize_for_symbol(self, **_kwargs):
        return None

    async def analyze_market(self, **_kwargs):
        return {"recommendation": "HOLD", "confidence": "LOW", "reasoning": "test"}


class _FakeRoot:
    def __init__(self, engine):
        self.engine = engine

    async def build_dependencies(self):
        class _ExchangeManager:
            async def find_symbol_exchange(self, _symbol):
                return (object(), "fake")

        return {
            "exchange_manager": _ExchangeManager(),
            "market_analyzer_factory": lambda _symbol: self.engine,
        }


def test_replay_positions_parser_exposes_options():
    from src.signal_service.__main__ import _build_parser

    args = _build_parser().parse_args([
        "replay-positions",
        "--start", "2026-01-01", "--end", "2026-01-02",
        "--candles", "c.csv", "--entries", "entries", "--output", "updates",
    ])
    assert args.timeframe == "4h"
    assert args.max_hold_hours == 24.0
    assert args.timeout_seconds == 300.0
    assert args.market is None


def test_replay_positions_command_reports_stats(tmp_path, monkeypatch, capsys):
    import json

    from src.signal_service import __main__ as cli

    engine = _HoldEngine()
    monkeypatch.setattr(cli, "_load_engine_factory", lambda: _FakeRoot(engine))

    candles = tmp_path / "BTC_USDT_4h.csv"
    candles.write_text(
        "time,open,high,low,close,volume\n"
        "2026-01-01T00:00:00Z,100.0,100.5,99.5,100.0,1.0\n"
        "2026-01-01T04:00:00Z,100.0,101.0,99.0,100.5,1.0\n"
        "2026-01-01T08:00:00Z,100.5,101.0,100.2,100.8,1.0\n",
        encoding="utf-8",
    )
    entries_dir = tmp_path / "entries"
    entries_dir.mkdir()
    entry = {
        **_entry(),
        "candle_open_time": "2026-01-01T00:00:00Z",
        "candle_close_time": "2026-01-01T04:00:00Z",
        "created_at": "2026-01-01T04:00:02Z",
        "expires_at": "2026-01-01T08:00:00Z",
    }
    (entries_dir / "BTC_USDT_4h.jsonl").write_text(json.dumps(entry) + "\n", encoding="utf-8")

    args = cli._build_parser().parse_args([
        "replay-positions",
        "--pairs", SYMBOL,
        "--start", "2026-01-01T00:00:00Z",
        "--end", "2026-01-01T08:00:00Z",
        "--candles", str(candles),
        "--entries", str(entries_dir),
        "--output", str(tmp_path / "updates"),
        "--no-news",
        "--market", "spot",
    ])
    asyncio.run(cli._replay_positions_command(args))

    payload = json.loads(capsys.readouterr().out)
    assert payload["symbol"] == SYMBOL
    assert payload["candles_seen"] == 2
    assert payload["candles_with_position"] == 1
    assert payload["analysis_calls"] == 1
    assert payload["updates"] == 0
    assert payload["positions"] == []


def test_replay_positions_command_requires_existing_entries(tmp_path, monkeypatch):
    import pytest

    from src.signal_service import __main__ as cli

    engine = _HoldEngine()
    monkeypatch.setattr(cli, "_load_engine_factory", lambda: _FakeRoot(engine))
    candles = tmp_path / "BTC_USDT_4h.csv"
    candles.write_text(
        "time,open,high,low,close,volume\n"
        "2026-01-01T00:00:00Z,100.0,100.5,99.5,100.0,1.0\n",
        encoding="utf-8",
    )
    args = cli._build_parser().parse_args([
        "replay-positions",
        "--start", "2026-01-01T00:00:00Z",
        "--end", "2026-01-01T08:00:00Z",
        "--candles", str(candles),
        "--entries", str(tmp_path / "missing"),
        "--output", str(tmp_path / "updates"),
    ])
    with pytest.raises(ValueError, match="entry signals not found"):
        asyncio.run(cli._replay_positions_command(args))
