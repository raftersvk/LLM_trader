"""Tests for the append-only JSONL signal storage."""

import json
import os
from datetime import datetime, timedelta, timezone


from src.signal_service.contract import Signal, build_decision_id, SIGNAL_SCHEMA_VERSION
from src.signal_service.storage import SignalStorage

CANDLE_OPEN = "2026-10-23T14:00:00Z"


def make_signal(symbol="BTC/USDT", candle_open=CANDLE_OPEN, action="BUY", **overrides):
    def frame(ts):
        return datetime.fromisoformat(ts.replace("Z", "+00:00"))
    from src.signal_service.contract import iso_z, timeframe_seconds
    period = timedelta(seconds=timeframe_seconds("1h"))
    base = dict(
        schema_version=SIGNAL_SCHEMA_VERSION,
        decision_id=build_decision_id(symbol, "1h", candle_open),
        source="llm_trader",
        symbol=symbol,
        timeframe="1h",
        candle_open_time=candle_open,
        candle_close_time=iso_z(frame(candle_open) + period),
        created_at=iso_z(frame(candle_open) + period + timedelta(seconds=2)),
        expires_at=iso_z(frame(candle_open) + 2 * period),
        action=action,
        confidence=0.9 if action == "BUY" else 0.0,
        risk_approved=action == "BUY",
        sl=67000.0 if action == "BUY" else None,
        tp=70000.0 if action == "BUY" else None,
        position_size_pct=0.05,
        reasoning="ok",
        model="deepseek-v4.1-flash",
        provider="opencode-go",
    )
    base.update(overrides)
    signal = Signal(**base)
    signal.validate()
    return signal


def test_append_and_load_roundtrip(tmp_path):
    storage = SignalStorage(str(tmp_path / "BTC_USDT_1h.jsonl"))
    signal = make_signal()
    assert storage.append(signal) is True
    loaded = storage.load_all()
    assert len(loaded) == 1
    assert loaded[0].to_dict() == signal.to_dict()


def test_dedup_by_symbol_timeframe_candle(tmp_path):
    storage = SignalStorage(str(tmp_path / "BTC_USDT_1h.jsonl"))
    signal = make_signal()
    assert storage.append(signal) is True
    assert storage.append(signal) is False
    assert len(storage.load_all()) == 1

    # Same candle, other symbol -> separate file, so no conflict.
    other = SignalStorage(str(tmp_path / "ETH_USDT_1h.jsonl"))
    assert other.append(make_signal(symbol="ETH/USDT")) is True


def test_dedup_recovered_after_restart(tmp_path):
    path = str(tmp_path / "BTC_USDT_1h.jsonl")
    first = SignalStorage(path)
    first.append(make_signal())

    # New storage instance = simulates restart: dedup keys are re-scanned.
    second = SignalStorage(path)
    duplicate = make_signal()
    assert second.is_duplicate(duplicate) is True
    assert second.append(duplicate) is False
    assert len(second.load_all()) == 1


def test_invalid_lines_never_corrupt_the_file(tmp_path):
    path = str(tmp_path / "BTC_USDT_1h.jsonl")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("{not valid json\n")
        handle.write(json.dumps({"symbol": "BTC/USDT"}))  # missing fields
        handle.write("\n")

    storage = SignalStorage(path)
    assert storage.load_all() == []  # invalid lines skipped
    assert storage.append(make_signal()) is True
    assert len(storage.load_all()) == 1
    with open(path, "r", encoding="utf-8") as handle:
        lines = handle.readlines()
    assert len(lines) == 3  # garbage preserved append-only, nothing rewritten


def test_ttl_handling(tmp_path):
    storage = SignalStorage(str(tmp_path / "BTC_USDT_1h.jsonl"))
    past = make_signal(candle_open="2020-01-01T00:00:00Z")
    now = make_signal(
        candle_open="2026-10-23T14:00:00Z",
        created_at="2026-10-23T15:00:02Z",
    )
    storage.append(past)
    storage.append(now)
    signals = storage.load_all()
    expired = [
        s for s in signals
        if s.is_expired(datetime(2026, 10, 23, 15, 0, tzinfo=timezone.utc))
    ]
    assert len(expired) == 1
    assert expired[0].candle_open_time == "2020-01-01T00:00:00Z"


def test_latest_for_candle(tmp_path):
    storage = SignalStorage(str(tmp_path / "BTC_USDT_1h.jsonl"))
    signal = make_signal()
    storage.append(signal)
    found = storage.latest_for_candle("BTC/USDT", "1h", CANDLE_OPEN)
    assert found is not None
    assert found.decision_id == signal.decision_id
    assert storage.latest_for_candle("ETH/USDT", "1h", CANDLE_OPEN) is None


def test_concurrent_writes_are_line_atomic(tmp_path):
    """Two storage handles on one file: both appends survive as separate lines."""
    path = str(tmp_path / "BTC_USDT_1h.jsonl")
    storage_a = SignalStorage(path)
    storage_b = SignalStorage(path)
    storage_a.append(make_signal(candle_open="2026-10-23T14:00:00Z"))
    storage_b.append(make_signal(candle_open="2026-10-23T15:00:00Z"))
    signals = SignalStorage(path).load_all()
    assert len(signals) == 2
    assert {s.candle_open_time for s in signals} == {
        "2026-10-23T14:00:00Z", "2026-10-23T15:00:00Z",
    }


def test_output_path_helpers(tmp_path):
    from src.signal_service.storage import signal_file_path
    path = signal_file_path(str(tmp_path), "BTC/USDT", "1h")
    assert os.path.basename(path) == "BTC_USDT_1h.jsonl"

def test_dedup_is_checked_inside_the_write_lock(tmp_path):
    """Two processes with primed caches must not append the same candle twice."""
    path = str(tmp_path / "BTC_USDT_1h.jsonl")
    first = SignalStorage(path)
    second = SignalStorage(path)
    signal = make_signal()

    # Both caches are loaded (empty) before either process writes.
    assert first.is_duplicate(signal) is False
    assert second.is_duplicate(signal) is False

    assert first.append(signal) is True
    # The second instance's stale cache must not let a duplicate through.
    assert second.append(signal) is False
    assert len(second.load_all()) == 1
