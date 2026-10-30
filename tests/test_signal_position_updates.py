"""Tests for contract v1.1 position updates, storage, sync and the service."""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone

import pytest

from src.signal_service.analyzer_adapter import PairAnalyzer
from src.signal_service.contract import Signal, SignalValidationError
from src.signal_service.positions import (
    PositionSnapshot,
    PositionsSnapshot,
    load_positions,
    utc_now_iso,
    write_positions_atomic,
)
from src.signal_service.position_sync import snapshot_from_trades
from src.signal_service.service import SignalService
from src.signal_service.storage import UpdateStorage

SYMBOL = "BTC/USDT"


def update_kwargs(**overrides):
    base = dict(
        schema_version="1.1",
        decision_id="entry-1-u1",
        source="llm_trader",
        symbol=SYMBOL,
        timeframe="1h",
        candle_open_time="2026-10-30T10:00:00Z",
        candle_close_time="2026-10-30T11:00:00Z",
        created_at="2026-10-30T11:00:02Z",
        expires_at="2026-10-30T12:00:00Z",
        action="UPDATE",
        confidence=0.0,
        risk_approved=True,
        sl=68000.0,
        tp=72000.0,
        position_size_pct=0.0,
        reasoning="tighten",
        model="m",
        provider="p",
        position_id="42",
        position_side="long",
        revision=1,
        based_on_entry="entry-1",
        source_price=69000.0,
    )
    base.update(overrides)
    return base


def entry_kwargs(**overrides):
    base = dict(
        schema_version="1.0",
        decision_id="entry-1",
        source="llm_trader",
        symbol=SYMBOL,
        timeframe="1h",
        candle_open_time="2026-10-30T10:00:00Z",
        candle_close_time="2026-10-30T11:00:00Z",
        created_at="2026-10-30T11:00:02Z",
        expires_at="2026-10-30T12:00:00Z",
        action="BUY",
        confidence=0.7,
        risk_approved=True,
        sl=66500.0,
        tp=72000.0,
        position_size_pct=0.05,
        reasoning="entry",
        model="m",
        provider="p",
    )
    base.update(overrides)
    return base


# ------------------------------------------------------------------ contract
def test_update_roundtrip_and_optional_fields():
    signal = Signal(**update_kwargs())
    signal.validate()
    data = signal.to_dict()
    assert data["schema_version"] == "1.1"
    assert data["position_id"] == "42"
    assert data["revision"] == 1
    restored = Signal.from_dict(data)
    assert restored.position_id == "42"
    assert restored.position_side == "long"


def test_update_requires_levels():
    with pytest.raises(SignalValidationError):
        Signal(**update_kwargs(sl=None, tp=None)).validate()


def test_update_requires_schema_11_and_identity():
    with pytest.raises(SignalValidationError):
        Signal(**update_kwargs(schema_version="1.0")).validate()
    with pytest.raises(SignalValidationError):
        Signal(**update_kwargs(position_id=None)).validate()
    with pytest.raises(SignalValidationError):
        Signal(**update_kwargs(revision=0)).validate()


def test_close_requires_reason_and_no_levels():
    close = Signal(**update_kwargs(action="CLOSE", sl=None, tp=None, close_reason="take_profit"))
    close.validate()
    with pytest.raises(SignalValidationError):
        Signal(**update_kwargs(action="CLOSE", sl=None, tp=None)).validate()
    with pytest.raises(SignalValidationError):
        Signal(**update_kwargs(action="CLOSE", close_reason="x")).validate()


# ------------------------------------------------------------------ storage
def test_update_storage_dedups_by_decision_id_and_tracks_revision(tmp_path):
    storage = UpdateStorage(str(tmp_path / "updates" / "BTC_USDT_1h.jsonl"))
    first = Signal(**update_kwargs(revision=1))
    assert storage.append(first) is True
    assert storage.append(Signal(**update_kwargs(revision=1))) is False
    second = Signal(**update_kwargs(decision_id="entry-1-u2", revision=2, sl=69000.0))
    assert storage.append(second) is True
    assert storage.last_revision("42") == 2


# ------------------------------------------------------------------ positions
def test_positions_atomic_roundtrip_and_symbol_mapping(tmp_path):
    snapshot = PositionsSnapshot(
        updated_at="2026-10-30T11:00:00Z",
        positions=[
            PositionSnapshot(
                position_id="42", symbol="BTC/USDT:USDT", side="short",
                entry_price=78000.0, stop_loss=79000.0, take_profit=76000.0,
            )
        ],
    )
    path = tmp_path / "positions.json"
    write_positions_atomic(str(path), snapshot)
    loaded = load_positions(str(path))
    assert len(loaded.for_symbol("BTC/USDT")) == 1
    assert loaded.for_symbol("ETH/USDT") == []


def test_missing_positions_file_is_empty(tmp_path):
    assert load_positions(str(tmp_path / "nope.json")).positions == []


def test_snapshot_from_trades_maps_freqtrade_fields():
    snapshot = snapshot_from_trades(
        [{"trade_id": 7, "pair": "BTC/USDT:USDT", "is_short": True,
          "open_rate": 78000.0, "amount": 0.1, "leverage": 2.0, "enter_tag": "e1"}]
    )
    position = snapshot.positions[0]
    assert position.position_id == "7"
    assert position.side == "short"
    assert position.enter_tag == "e1"


# ------------------------------------------------------------------ adapter
class _FakeEngine:
    class _MM:
        last_finish_reason = None
        def supports_image_analysis(self, *_a, **_k):
            return False

    class _Rag:
        async def retrieve_context(self, *_a, **_k):
            return ("", {})

        class _CB:
            def get_latest_article_urls(self):
                return {}
        context_builder = _CB()

    class _Ctx:
        current_price = 69000.0
        ohlcv_candles = None
        timeframe = "1h"

    def __init__(self, raw):
        self.last_llm_response = raw
        self.model_manager = self._MM()
        self.rag_engine = self._Rag()
        self.context = self._Ctx()
        self.calls = []

    async def analyze_market(self, **kwargs):
        self.calls.append(kwargs)
        return {"raw_response": self.last_llm_response, "recommendation": "UPDATE"}


def _analyzer(raw):
    return PairAnalyzer(engine=_FakeEngine(raw), symbol=SYMBOL, timeframe="1h")


from src.signal_service.analyzer_adapter import UpdateContext  # noqa: E402


def _ctx(revision=1):
    return UpdateContext(
        position_id="42", position_side="long", entry_price=67500.0,
        stop_loss=66500.0, take_profit=72000.0, revision=revision,
        based_on_entry="entry-1", current_price=69000.0,
    )


def test_adapter_emits_update_when_compliant():
    raw = json.dumps({
        "signal": "UPDATE", "confidence": "HIGH", "stop_loss": 68000.0,
        "take_profit": 72000.0, "reasoning": "tighten after progress",
    })
    signal = asyncio.run(
        _analyzer(raw).analyze(candle_open_time="2026-10-30T10:00:00Z", update_context=_ctx())
    )
    assert signal.action == "UPDATE"
    assert signal.schema_version == "1.1"
    assert signal.sl == 68000.0
    assert signal.revision == 1


def test_adapter_refuses_premature_tightening():
    raw = json.dumps({
        "signal": "UPDATE", "confidence": "HIGH", "stop_loss": 67200.0,
        "reasoning": "tighten too early",
    })
    early = UpdateContext(
        position_id="42", position_side="long", entry_price=67500.0,
        stop_loss=66500.0, take_profit=72000.0, revision=1,
        based_on_entry="entry-1", current_price=67800.0,
    )
    signal = asyncio.run(
        _analyzer(raw).analyze(candle_open_time="2026-10-30T10:00:00Z", update_context=early)
    )
    assert signal.action == "HOLD"
    assert "update refused" in signal.reasoning


# ------------------------------------------------------------------ service
class _FakeAnalyzer:
    symbol = SYMBOL
    timeframe = "1h"

    def __init__(self, signal):
        self.signal = signal
        self.calls = []

    async def analyze(self, **kwargs):
        self.calls.append(kwargs)
        return self.signal


def _service(tmp_path, analyzer):
    return SignalService(
        pairs=[SYMBOL],
        timeframe="1h",
        analyzer_factory=lambda _s: analyzer,
        signals_dir=str(tmp_path / "signals"),
        position_aware=True,
        positions_file=str(tmp_path / "positions.json"),
        updates_dir=str(tmp_path / "updates"),
        dedupe=True,
    )


def test_service_emits_update_and_skips_entry_when_position_open(tmp_path):
    update = Signal(**update_kwargs())
    analyzer = _FakeAnalyzer(update)
    service = _service(tmp_path, analyzer)
    snapshot = PositionsSnapshot(
        updated_at=utc_now_iso(),
        positions=[PositionSnapshot(
            position_id="42", symbol=SYMBOL, side="long", entry_price=67500.0,
            stop_loss=66500.0, take_profit=72000.0, current_price=69000.0,
            enter_tag="entry-1",
        )],
    )
    write_positions_atomic(service.positions_file, snapshot)
    result = asyncio.run(service.run_cycle(datetime(2026, 10, 30, 10, 0, tzinfo=timezone.utc)))
    assert result[SYMBOL] == "updates:1"
    assert not (tmp_path / "signals" / "BTC_USDT_1h.jsonl").exists()
    lines = (tmp_path / "updates" / "BTC_USDT_1h.jsonl").read_text().strip().splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["action"] == "UPDATE"
    assert analyzer.calls[0]["update_context"].revision == 1


def test_service_without_positions_is_stateless(tmp_path):
    entry = Signal(**entry_kwargs())
    analyzer = _FakeAnalyzer(entry)
    service = SignalService(
        pairs=[SYMBOL], timeframe="1h", analyzer_factory=lambda _s: analyzer,
        signals_dir=str(tmp_path / "signals"),
        position_aware=True, positions_file=str(tmp_path / "positions.json"),
        updates_dir=str(tmp_path / "updates"), dedupe=True,
    )
    result = asyncio.run(service.run_cycle(datetime(2026, 10, 30, 10, 0, tzinfo=timezone.utc)))
    assert result[SYMBOL] == "entry-1"
    assert (tmp_path / "signals" / "BTC_USDT_1h.jsonl").exists()
    assert not (tmp_path / "updates" / "BTC_USDT_1h.jsonl").exists()


# ------------------------------------------------------------------ more adapter
def test_update_mode_rejects_entry_action():
    raw = json.dumps({"signal": "BUY", "confidence": "HIGH", "stop_loss": 68000.0,
                      "take_profit": 72000.0, "reasoning": "entry schema"})
    signal = asyncio.run(
        _analyzer(raw).analyze(candle_open_time="2026-10-30T10:00:00Z", update_context=_ctx())
    )
    assert signal.action == "HOLD"
    assert "requires UPDATE or CLOSE" in signal.reasoning


def test_update_mode_emits_close():
    raw = json.dumps({"signal": "CLOSE", "confidence": "HIGH", "reasoning": "take profit"})
    signal = asyncio.run(
        _analyzer(raw).analyze(candle_open_time="2026-10-30T10:00:00Z", update_context=_ctx())
    )
    assert signal.action == "CLOSE"
    assert signal.close_reason == "llm_close"
    assert signal.sl is None and signal.tp is None


def test_update_mode_refuses_stop_beyond_price():
    raw = json.dumps({"signal": "UPDATE", "confidence": "HIGH", "stop_loss": 69500.0,
                      "reasoning": "above market"})
    signal = asyncio.run(
        _analyzer(raw).analyze(candle_open_time="2026-10-30T10:00:00Z", update_context=_ctx())
    )
    assert signal.action == "HOLD"
    assert "update refused" in signal.reasoning


def test_update_mode_bypasses_jev_pre_gate():
    class BombJev:
        def uses_pre(self):
            raise AssertionError("JEV pre-gate must not run for position updates")

        def uses_post(self):
            return False

    raw = json.dumps({"signal": "UPDATE", "confidence": "HIGH", "stop_loss": 68000.0,
                      "reasoning": "tighten"})
    analyzer = _analyzer(raw)
    analyzer.jev_gate = BombJev()
    signal = asyncio.run(
        analyzer.analyze(candle_open_time="2026-10-30T10:00:00Z", update_context=_ctx())
    )
    assert signal.action == "UPDATE"


# ------------------------------------------------------------------ more service
def test_position_aware_false_ignores_snapshot(tmp_path):
    entry = Signal(**entry_kwargs())
    analyzer = _FakeAnalyzer(entry)
    service = SignalService(
        pairs=[SYMBOL], timeframe="1h", analyzer_factory=lambda _s: analyzer,
        signals_dir=str(tmp_path / "signals"), position_aware=False,
        positions_file=str(tmp_path / "positions.json"),
        updates_dir=str(tmp_path / "updates"),
    )
    snapshot = PositionsSnapshot(
        updated_at=utc_now_iso(),
        positions=[PositionSnapshot(position_id="42", symbol=SYMBOL, side="long",
                                    entry_price=67500.0, current_price=69000.0)],
    )
    write_positions_atomic(service.positions_file, snapshot)
    result = asyncio.run(service.run_cycle(datetime(2026, 10, 30, 10, 0, tzinfo=timezone.utc)))
    assert result[SYMBOL] == "entry-1"
    assert not (tmp_path / "updates" / "BTC_USDT_1h.jsonl").exists()


def test_stale_snapshot_fails_open_to_entry(tmp_path):
    entry = Signal(**entry_kwargs())
    analyzer = _FakeAnalyzer(entry)
    service = _service(tmp_path, analyzer)
    service.max_snapshot_age_seconds = 60
    snapshot = PositionsSnapshot(
        updated_at="2020-01-01T00:00:00Z",
        positions=[PositionSnapshot(position_id="42", symbol=SYMBOL, side="long",
                                    entry_price=67500.0, stop_loss=66500.0,
                                    take_profit=72000.0, current_price=69000.0)],
    )
    write_positions_atomic(service.positions_file, snapshot)
    result = asyncio.run(service.run_cycle(datetime(2026, 10, 30, 10, 0, tzinfo=timezone.utc)))
    assert result[SYMBOL] == "entry-1"
    assert service.metrics["stale_snapshots"] == 1


def test_tp_unknown_still_emits_sl_only_update(tmp_path):
    update = Signal(**update_kwargs(sl=68000.0, tp=None))
    analyzer = _FakeAnalyzer(update)
    service = _service(tmp_path, analyzer)
    snapshot = PositionsSnapshot(
        updated_at=utc_now_iso(),
        positions=[PositionSnapshot(position_id="42", symbol=SYMBOL, side="long",
                                    entry_price=67500.0, stop_loss=66500.0,
                                    take_profit=None, current_price=69000.0,
                                    enter_tag="entry-1")],
    )
    write_positions_atomic(service.positions_file, snapshot)
    result = asyncio.run(service.run_cycle(datetime(2026, 10, 30, 10, 0, tzinfo=timezone.utc)))
    assert result[SYMBOL] == "updates:1"
    call_ctx = analyzer.calls[0]["update_context"]
    assert call_ctx.take_profit is None
    assert call_ctx.effective_entry_sl() == 66500.0


def test_revision_starts_at_one_without_existing_updates(tmp_path):
    update = Signal(**update_kwargs())
    analyzer = _FakeAnalyzer(update)
    service = _service(tmp_path, analyzer)
    snapshot = PositionsSnapshot(
        updated_at=utc_now_iso(),
        positions=[PositionSnapshot(position_id="42", symbol=SYMBOL, side="long",
                                    entry_price=67500.0, stop_loss=66500.0,
                                    take_profit=72000.0, current_price=69000.0)],
    )
    write_positions_atomic(service.positions_file, snapshot)
    asyncio.run(service.run_cycle(datetime(2026, 10, 30, 10, 0, tzinfo=timezone.utc)))
    assert analyzer.calls[0]["update_context"].revision == 1


# ------------------------------------------------------------------ position_sync
def test_position_sync_http_and_atomic_write(tmp_path):
    import httpx
    from src.signal_service.position_sync import FreqtradeReadOnlyClient, sync_once

    def handler(request):
        if request.url.path == "/api/v1/token/login":
            return httpx.Response(200, json={"access_token": "tok"})
        if request.url.path == "/api/v1/status":
            assert request.headers["authorization"] == "Bearer tok"
            return httpx.Response(200, json=[{
                "trade_id": 42, "pair": "BTC/USDT", "is_short": False,
                "open_rate": 67500.0, "amount": 0.1, "enter_tag": "e1",
            }])
        return httpx.Response(404)

    async def run():
        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as raw:
            client = FreqtradeReadOnlyClient("http://api", "u", "p", client=raw)
            return await sync_once(client, str(tmp_path / "positions.json"))

    assert asyncio.run(run()) == 1
    payload = json.loads((tmp_path / "positions.json").read_text())
    assert payload["positions"][0]["position_id"] == "42"
    assert not list(tmp_path.glob(".positions-*.tmp"))


def test_update_mode_emits_sl_only_and_never_invents_a_tp():
    # TP unknown on the position and none proposed by the model: the retained
    # v1.1 behaviour is a safe SL-only UPDATE, never a synthesised dynamic TP.
    raw = json.dumps({"signal": "UPDATE", "confidence": "HIGH", "stop_loss": 68000.0,
                      "reasoning": "tighten only"})
    no_tp_ctx = UpdateContext(
        position_id="42", position_side="long", entry_price=67500.0,
        stop_loss=66500.0, take_profit=None, revision=1,
        based_on_entry="entry-1", current_price=69000.0, entry_sl=66500.0,
    )
    signal = asyncio.run(
        _analyzer(raw).analyze(candle_open_time="2026-10-30T10:00:00Z", update_context=no_tp_ctx)
    )
    assert signal.action == "UPDATE"
    assert signal.sl == 68000.0
    assert signal.tp is None
