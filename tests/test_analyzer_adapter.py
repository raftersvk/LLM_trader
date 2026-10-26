"""Tests for the AnalysisEngine -> Signal adapter.

The adapter must analyze stateless, extract via PositionExtractor, hash the
raw response and compute risk_approved per the contract rules.
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta, timezone

import pytest

from src.signal_service.analyzer_adapter import (
    PairAnalyzer,
    ReplayDataInjector,
    analyze_and_store,
    analyze_with_timeout,
)
from src.signal_service.storage import SignalStorage, signal_file_path
from src.signal_service.contract import (
    SignalValidationError,
    build_decision_id,
    hash_raw_response,
    iso_z,
)

SYMBOL = "BTC/USDT"


class FakeModelManager:
    last_finish_reason = None

    def supports_image_analysis(self, *_args, **_kwargs) -> bool:
        return False


class FakeRagEngine:
    async def retrieve_context(self, *_args, **_kwargs):
        return ("", {})

    class _CB:
        def get_latest_article_urls(self):
            return {}

    context_builder = _CB()


class FakeContext:
    def __init__(self, current_price=None, candles=None):
        self.current_price = current_price
        self.ohlcv_candles = candles
        self.timeframe = "1h"


class FakeEngine:
    """Same surface as AnalysisEngine for the calls used by the adapter."""

    def __init__(self, raw_response=None, result=None, current_price=68000.0, config=None):
        self.last_llm_response = raw_response
        self.model_manager = FakeModelManager()
        self.rag_engine = FakeRagEngine()
        self.context = FakeContext(current_price=current_price)
        self.config = config
        self._result = result
        self.calls = []

    async def analyze_market(self, **kwargs):
        self.calls.append(kwargs)
        if self._result is None:
            raise RuntimeError("llm down")
        return self._result


RAW_BUY = (
    '{"signal": "BUY", "confidence": "HIGH", "stop_loss": 67000.0, '
    '"take_profit": 70000.0, "position_size": 5, "reasoning": "breakout"}'
)

RESULT_META = {
    "model": "deepseek-v4.1-flash",
    "provider": "litellm",
}


def assert_stateless(engine):
    call = engine.calls[-1]
    assert call["previous_response"] is None
    assert call["previous_indicators"] is None
    assert call["position_context"] == ""
    assert call["performance_context"] == ""
    assert call["brain_service"] is None
    assert call["dynamic_thresholds"] is None


@pytest.mark.asyncio
async def test_buy_signal_risk_approved():
    engine = FakeEngine(raw_response=RAW_BUY, result=dict(RESULT_META))
    analyzer = PairAnalyzer(engine=engine, symbol=SYMBOL, timeframe="1h")
    signal = await analyzer.analyze()
    assert signal.action == "BUY"
    assert signal.risk_approved is True
    assert signal.sl == 67000.0
    assert signal.tp == 70000.0
    assert signal.confidence == 0.90
    assert signal.model == "deepseek-v4.1-flash"
    assert signal.provider == "litellm"
    assert signal.raw_response_hash == hash_raw_response(RAW_BUY)
    assert signal.schema_version == "1.0"
    assert_stateless(engine)
    signal.validate()


@pytest.mark.asyncio
async def test_buy_coherence_and_rr_gate():
    # BUY with TP below price -> incoherent -> risk_approved=false, action kept
    raw = (
        '{"signal": "BUY", "confidence": "HIGH", "stop_loss": 67000.0, '
        '"take_profit": 67500.0, "position_size": 5, "reasoning": "meh"}'
    )
    engine = FakeEngine(raw_response=raw, result=dict(RESULT_META), current_price=68000.0)
    analyzer = PairAnalyzer(engine=engine, symbol=SYMBOL)
    signal = await analyzer.analyze()
    assert signal.action == "BUY"
    assert signal.risk_approved is False

    # RR = 100/1250 = 0.08 < 1.5 -> rejected
    raw_low_rr = (
        '{"signal": "BUY", "confidence": "HIGH", "stop_loss": 66750.0, '
        '"take_profit": 68100.0, "position_size": 5, "reasoning": "tight"}'
    )
    engine2 = FakeEngine(raw_response=raw_low_rr, result=dict(RESULT_META), current_price=68000.0)
    analyzer2 = PairAnalyzer(engine=engine2, symbol=SYMBOL)
    signal2 = await analyzer2.analyze()
    assert signal2.risk_approved is False


@pytest.mark.asyncio
async def test_confidence_below_threshold_vetoes_risk():
    raw = (
        '{"signal": "BUY", "confidence": "LOW", "stop_loss": 67000.0, '
        '"take_profit": 70000.0, "position_size": 5, "reasoning": "unsure"}'
    )
    engine = FakeEngine(raw_response=raw, result=dict(RESULT_META))
    analyzer = PairAnalyzer(engine=engine, symbol=SYMBOL, min_confidence=0.6)
    signal = await analyzer.analyze()
    assert signal.action == "BUY"
    assert signal.risk_approved is False
    assert signal.confidence == 0.35


@pytest.mark.asyncio
async def test_hold_signal():
    raw = '{"signal": "HOLD", "confidence": "MEDIUM", "reasoning": "no setup"}'
    engine = FakeEngine(raw_response=raw, result=dict(RESULT_META))
    analyzer = PairAnalyzer(engine=engine, symbol=SYMBOL)
    signal = await analyzer.analyze()
    assert signal.action == "HOLD"
    assert signal.risk_approved is False
    assert signal.reasoning.strip(), "reasoning must stay non-empty"
    signal.validate()


@pytest.mark.asyncio
async def test_valid_hold_is_never_risk_approved():
    """A valid HOLD must not look approved: it can never trigger an entry."""
    raw = '{"signal": "HOLD", "confidence": "HIGH", "reasoning": "no setup"}'
    engine = FakeEngine(raw_response=raw, result=dict(RESULT_META))
    analyzer = PairAnalyzer(engine=engine, symbol=SYMBOL)

    signal = await analyzer.analyze(candle_open_time="2026-10-23T14:00:00Z")

    assert signal.action == "HOLD"
    assert signal.risk_approved is False
    assert signal.reasoning.strip(), "reasoning must stay non-empty"
    assert signal.model == "deepseek-v4.1-flash"
    assert signal.provider == "litellm"
    signal.validate()


@pytest.mark.asyncio
async def test_hold_fallback_on_engine_error():
    engine = FakeEngine(raw_response=None, result=None)
    analyzer = PairAnalyzer(engine=engine, symbol=SYMBOL)
    signal = await analyzer.analyze()
    assert signal.action == "HOLD"
    assert signal.risk_approved is False
    assert signal.sl is None and signal.tp is None
    assert signal.raw_response_hash is None
    assert "analysis error" in signal.reasoning
    signal.validate()


@pytest.mark.asyncio
async def test_hold_fallback_on_timeout():
    class SlowEngine(FakeEngine):
        async def analyze_market(self, **kwargs):
            await asyncio.sleep(5)
            return dict(RESULT_META)

    import asyncio
    engine = SlowEngine(raw_response=RAW_BUY, result=dict(RESULT_META))
    analyzer = PairAnalyzer(engine=engine, symbol=SYMBOL)
    signal = await analyze_with_timeout(analyzer, timeout_seconds=0.1)
    assert signal.action == "HOLD"
    assert "timeout" in signal.reasoning


@pytest.mark.asyncio
async def test_no_raw_response_stored_in_signal():
    """The privacy guarantee: only reasoning survives, never the raw JSON."""
    engine = FakeEngine(raw_response=RAW_BUY, result=dict(RESULT_META))
    analyzer = PairAnalyzer(engine=engine, symbol=SYMBOL)
    signal = await analyzer.analyze()

    payload = signal.to_dict()
    dumped = json.dumps(payload)

    # No raw-response key and no raw JSON structure in the serialized signal.
    assert "raw_response" not in payload
    for token in ('"raw_response"', '"stop_loss"', '"take_profit"', '"position_size"', '"signal"', '"HIGH"'):
        assert token not in dumped, f"raw response token leaked into the signal: {token}"

    # The legitimate summarised reasoning text is preserved intact.
    assert payload["reasoning"] == "breakout"
    # Auditability comes from the hash, not from retaining the response.
    assert signal.raw_response_hash == hash_raw_response(RAW_BUY)


@pytest.mark.asyncio
async def test_jev_veto():
    engine = FakeEngine(raw_response=RAW_BUY, result=dict(RESULT_META))
    analyzer = PairAnalyzer(engine=engine, symbol=SYMBOL)

    def inject(jev):
        original = analyzer._extract_draft

        def patched(result, raw_response):
            draft = original(result, raw_response)
            draft.jev = jev
            return draft

        analyzer._extract_draft = patched  # type: ignore[assignment]

    # Coherent JEV: long bias + tradeable -> approved, and jev stored.
    inject({"model": "jev-1.13-free", "tradeability": True, "bias": "long"})
    approved = await analyzer.analyze()
    assert approved.risk_approved is True
    assert approved.jev["model"] == "jev-1.13-free"

    # JEV veto: tradeability=false -> risk_approved=false
    engine2 = FakeEngine(raw_response=RAW_BUY, result=dict(RESULT_META))
    analyzer2 = PairAnalyzer(engine=engine2, symbol=SYMBOL)
    inject_veto = analyzer2
    original2 = inject_veto._extract_draft

    def patched_veto(result, raw_response):
        draft = original2(result, raw_response)
        draft.jev = {"model": "jev-1.13-free", "tradeability": False, "bias": "none"}
        return draft

    inject_veto._extract_draft = patched_veto  # type: ignore[assignment]
    vetoed = await inject_veto.analyze()
    assert vetoed.risk_approved is False


def test_deterministic_candle_alignment():
    engine = FakeEngine(raw_response=RAW_BUY, result=dict(RESULT_META))
    analyzer = PairAnalyzer(engine=engine, symbol=SYMBOL, timeframe="1h")
    ref = datetime(2026, 10, 23, 14, 37, 0, tzinfo=timezone.utc)
    candle_open = analyzer.current_candle_bounds(ref)
    assert iso_z(candle_open) == "2026-10-23T14:00:00Z"


@pytest.mark.asyncio
async def test_explicit_candle_open_time():
    engine = FakeEngine(raw_response=RAW_BUY, result=dict(RESULT_META))
    analyzer = PairAnalyzer(engine=engine, symbol=SYMBOL, timeframe="1h")
    signal = await analyzer.analyze(candle_open_time="2026-10-23T14:00:00Z")
    assert signal.candle_open_time == "2026-10-23T14:00:00Z"
    assert signal.candle_close_time == "2026-10-23T15:00:00Z"
    assert signal.expires_at == "2026-10-23T16:00:00Z"
    assert signal.decision_id == build_decision_id(SYMBOL, "1h", "2026-10-23T14:00:00Z")


def test_replay_injector_no_look_ahead():
    """Replay slicing keeps only candles closed at or before as_of."""
    class FakeDataCollector:
        async def collect_data(self, *_args, **_kwargs):
            raise AssertionError("live collection must not run in replay")

    engine = FakeEngine(raw_response=RAW_BUY, result=dict(RESULT_META))
    engine.data_collector = FakeDataCollector()
    engine._enrich_market_context = lambda *a, **k: None
    analyzer = PairAnalyzer(engine=engine, symbol=SYMBOL, timeframe="1h")

    timestamps = [
        datetime(2026, 10, 23, h, tzinfo=timezone.utc) for h in range(10, 20)
    ]
    candles = [[float(h), 1.0, 2.0, 0.5, 1.5, 100.0] for h in range(10, 20)]
    injector = ReplayDataInjector(analyzer, candles, timestamps)
    injector.install()

    # as_of = close of the 14:00 candle: candles 10..14 visible, 15+ hidden.
    injector.set_as_of(datetime(2026, 10, 23, 15, 0, tzinfo=timezone.utc))
    kept_times = engine.context.timestamps
    assert max(kept_times) == datetime(2026, 10, 23, 14, tzinfo=timezone.utc)
    assert len(kept_times) == 5
    assert engine.context.current_price == 1.5

    # No candle at or before as_of -> clean error
    with pytest.raises(SignalValidationError):
        injector.set_as_of(datetime(2026, 10, 23, 5, 0, tzinfo=timezone.utc))

    injector.restore()
    assert callable(engine.data_collector.collect_data)

# --------------------------------------------------------------------------
# Regression tests: closed-candle propagation and engine initialization
# --------------------------------------------------------------------------

class FetchingDataCollector:
    """Fake collector that fills the context with a fixed OHLCV series."""

    def __init__(self, timestamps, candles):
        self.timestamps = timestamps
        self.candles = candles
        self.calls = 0

    async def collect_data(self, context, *_args, **_kwargs):
        self.calls += 1
        context.ohlcv_candles = list(self.candles)
        context.timestamps = list(self.timestamps)
        return {"success": True, "errors": [], "market_context": "", "article_urls": {}}


class ObservingEngine(FakeEngine):
    """Engine that collects data like the real one and records the visible window."""

    def __init__(self, collector, result=None):
        super().__init__(raw_response=RAW_BUY, result=result or dict(RESULT_META))
        self.data_collector = collector
        self.observed = {}

    async def analyze_market(self, **kwargs):
        self.calls.append(kwargs)
        await self.data_collector.collect_data(self.context)
        candles = self.context.ohlcv_candles or []
        self.observed = {
            "count": len(candles),
            "current_price": self.context.current_price,
            "last_open": self.context.timestamps[-1] if self.context.timestamps else None,
        }
        return self._result


def _hour(hour, day=23):
    return datetime(2026, 10, day, hour, tzinfo=timezone.utc)


def test_forming_candle_is_hidden_from_analysis():
    """(c) The candle still forming at analysis time must be truncated away."""
    timestamps = [_hour(13), _hour(14), _hour(15)]  # 15:00 is still forming
    candles = [
        [1.0, 1.0, 1.2, 0.9, 100.0, 10.0],
        [2.0, 2.0, 2.2, 1.9, 200.0, 20.0],
        [3.0, 3.0, 3.2, 2.9, 300.0, 30.0],
    ]
    collector = FetchingDataCollector(timestamps, candles)
    engine = ObservingEngine(collector)
    analyzer = PairAnalyzer(engine=engine, symbol=SYMBOL, timeframe="1h", use_news=False)
    original_collect = engine.data_collector.collect_data

    signal = asyncio.run(analyzer.analyze(candle_open_time="2026-10-23T14:00:00Z"))

    assert engine.observed["count"] == 2, "forming 15:00 candle must be hidden"
    assert engine.observed["last_open"] == _hour(14)
    assert engine.observed["current_price"] == 200.0, "price must come from the last closed candle"
    assert signal.candle_open_time == "2026-10-23T14:00:00Z"
    assert signal.candle_close_time == "2026-10-23T15:00:00Z"
    # The collector wrapper is installed for one analysis and then restored.
    assert engine.data_collector.collect_data == original_collect


def test_no_closed_candle_falls_back_to_hold():
    """(d) When only the forming candle exists, the adapter returns HOLD."""
    timestamps = [_hour(15)]  # nothing has closed yet for the 14:00 candle
    candles = [[3.0, 3.0, 3.2, 2.9, 300.0, 30.0]]
    engine = ObservingEngine(FetchingDataCollector(timestamps, candles))
    analyzer = PairAnalyzer(engine=engine, symbol=SYMBOL, timeframe="1h", use_news=False)

    signal = asyncio.run(analyzer.analyze(candle_open_time="2026-10-23T14:00:00Z"))

    assert signal.action == "HOLD"
    assert signal.risk_approved is False
    assert signal.sl is None and signal.tp is None
    assert "no closed candle" in signal.reasoning
    assert signal.candle_open_time == "2026-10-23T14:00:00Z"


class RecordingAnalyzer:
    """Minimal PairAnalyzer stand-in recording the candle it was asked about."""

    def __init__(self, symbol, timeframe="1h"):
        self.symbol = symbol
        self.timeframe = timeframe
        self.candles_seen = []

    async def analyze(self, candle_open_time=None):
        from src.signal_service.contract import Signal

        open_at = candle_open_time
        open_dt = datetime.fromisoformat(open_at.replace("Z", "+00:00"))
        signal = Signal(
            schema_version="1.0",
            decision_id=build_decision_id(self.symbol, self.timeframe, open_at),
            source="llm_trader",
            symbol=self.symbol,
            timeframe=self.timeframe,
            candle_open_time=open_at,
            candle_close_time=iso_z(open_dt + timedelta(hours=1)),
            created_at=iso_z(datetime.now(timezone.utc)),
            expires_at=iso_z(open_dt + timedelta(hours=2)),
            action="HOLD",
            confidence=0.0,
            risk_approved=False,
            sl=None,
            tp=None,
            position_size_pct=0.05,
            reasoning="test",
            model="test",
            provider="test",
        )
        signal.validate()
        self.candles_seen.append(candle_open_time)
        return signal


def test_run_cycle_passes_the_closed_candle(tmp_path):
    """(a) run_cycle must analyse the candle that just closed, not the forming one."""
    from src.signal_service.service import SignalService

    analyzers = {}

    def analyzer_factory(symbol):
        analyzers[symbol] = RecordingAnalyzer(symbol)
        return analyzers[symbol]

    service = SignalService(
        pairs=["BTC/USDT"],
        timeframe="1h",
        analyzer_factory=analyzer_factory,
        signals_dir=str(tmp_path),
    )
    results = asyncio.run(service.run_cycle(_hour(14)))

    assert analyzers["BTC/USDT"].candles_seen == ["2026-10-23T14:00:00Z"]
    assert results["BTC/USDT"] == "btc-usdt-1h-2026-10-23t14:00:00z"
    stored = service.storages["BTC/USDT"].load_all()
    assert len(stored) == 1
    assert stored[0].candle_open_time == "2026-10-23T14:00:00Z"
    assert stored[0].candle_close_time == "2026-10-23T15:00:00Z"


class RecordingEngine(FakeEngine):
    """Engine recording initialize_for_symbol calls."""

    def __init__(self, log):
        super().__init__(raw_response=RAW_BUY, result=dict(RESULT_META))
        self.initialized = []
        self._log = log

    def initialize_for_symbol(self, symbol, exchange, timeframe=None):
        self.initialized.append({"symbol": symbol, "exchange": exchange, "timeframe": timeframe})
        self._log.append(symbol)


class FakeExchangeManager:
    def __init__(self, supported):
        self.supported = supported
        self.calls = []

    async def find_symbol_exchange(self, symbol):
        self.calls.append(symbol)
        if symbol in self.supported:
            return self.supported[symbol], "binance"
        return None, None


def test_engines_are_initialized_once_per_pair():
    """(b) Every pair gets exactly one initialized engine with its own exchange."""
    from src.signal_service.__main__ import _build_initialized_engines

    btc_exchange = object()
    eth_exchange = object()
    manager = FakeExchangeManager({"BTC/USDT": btc_exchange, "ETH/USDT": eth_exchange})
    created = []
    log = []

    def engine_factory(symbol):
        engine = RecordingEngine(log)
        created.append(engine)
        return engine

    engines = asyncio.run(_build_initialized_engines(
        ["BTC/USDT", "ETH/USDT"], engine_factory, manager, "1h",
    ))

    assert set(engines) == {"BTC/USDT", "ETH/USDT"}
    assert manager.calls == ["BTC/USDT", "ETH/USDT"]
    assert len(created) == 2
    assert created[0].initialized == [
        {"symbol": "BTC/USDT", "exchange": btc_exchange, "timeframe": "1h"}
    ]
    assert created[1].initialized == [
        {"symbol": "ETH/USDT", "exchange": eth_exchange, "timeframe": "1h"}
    ]
    assert log == ["BTC/USDT", "ETH/USDT"]


def test_unknown_pair_raises_clear_error():
    """A pair unknown to every exchange must fail loudly, not run uninitialized."""
    from src.signal_service.__main__ import _build_initialized_engines

    manager = FakeExchangeManager({})

    def engine_factory(_symbol):  # pragma: no cover - must not be called
        raise AssertionError("engine_factory must not run for an unknown pair")

    with pytest.raises(RuntimeError, match="No exchange supports"):
        asyncio.run(_build_initialized_engines(["XYZ/USDT"], engine_factory, manager, "1h"))


def test_debug_cycle_replays_last_closed_candle(tmp_path):
    """Debug mode analyses the last closed 1h candle every cycle, not once an hour."""
    from src.signal_service.service import SignalService

    analyzers = {}

    def analyzer_factory(symbol):
        analyzers[symbol] = RecordingAnalyzer(symbol)
        return analyzers[symbol]

    service = SignalService(
        pairs=["BTC/USDT"],
        timeframe="1h",
        analyzer_factory=analyzer_factory,
        signals_dir=str(tmp_path),
        debug_cycle_seconds=0.01,
        max_cycles=3,
        dedupe=False,
    )
    asyncio.run(service.run())

    seen = analyzers["BTC/USDT"].candles_seen
    assert len(seen) == 3, "three debug cycles must run without waiting an hour"
    assert len(set(seen)) == 1, "every cycle replays the same closed candle"

    candle_open = datetime.fromisoformat(seen[0].replace("Z", "+00:00"))
    now = datetime.now(timezone.utc)
    assert candle_open + timedelta(hours=1) <= now, "the replayed candle must be closed"
    assert now - (candle_open + timedelta(hours=1)) < timedelta(hours=1)

    stored = service.storages["BTC/USDT"].load_all()
    assert len(stored) == 3, "with dedupe disabled each debug cycle is written"


def test_debug_cycle_dedupes_by_default(tmp_path):
    """Without --no-dedup, repeated debug analyses of one candle are not duplicated."""
    from src.signal_service.service import SignalService

    analyzers = {}

    def analyzer_factory(symbol):
        analyzers[symbol] = RecordingAnalyzer(symbol)
        return analyzers[symbol]

    service = SignalService(
        pairs=["BTC/USDT"],
        timeframe="1h",
        analyzer_factory=analyzer_factory,
        signals_dir=str(tmp_path),
        debug_cycle_seconds=0.01,
        max_cycles=3,
    )
    asyncio.run(service.run())

    assert len(analyzers["BTC/USDT"].candles_seen) == 3
    assert len(service.storages["BTC/USDT"].load_all()) == 1


class FakeEngineConfig:
    """Minimal Config surface used for model/provider fallbacks."""

    LITELLM_MODEL = "deepseek-v4.1-flash"
    PROVIDER = "litellm"


@pytest.mark.asyncio
async def test_empty_truncated_response_is_not_a_clean_hold():
    """finish_reason=length + empty content must be an explicit, unapproved HOLD."""
    engine = FakeEngine(raw_response="", result=dict(RESULT_META), config=FakeEngineConfig())
    engine.model_manager.last_finish_reason = "length"
    analyzer = PairAnalyzer(engine=engine, symbol=SYMBOL, timeframe="1h")

    signal = await analyzer.analyze(candle_open_time="2026-10-23T14:00:00Z")

    assert signal.action == "HOLD"
    assert signal.risk_approved is False
    assert signal.sl is None and signal.tp is None
    assert signal.reasoning.strip(), "reasoning must explain the failure"
    assert "invalid LLM response" in signal.reasoning
    assert "finish_reason=length" in signal.reasoning
    assert signal.model == "deepseek-v4.1-flash"
    assert signal.provider == "litellm"
    assert signal.raw_response_hash is None


@pytest.mark.asyncio
@pytest.mark.parametrize("reasoning", ["", "   ", "\n\t "])
async def test_empty_reasoning_response_is_an_invalid_hold(reasoning):
    """An empty reasoning must never become a max-confidence, clean HOLD."""
    raw = "```json\n" + json.dumps(
        {"action": "HOLD", "confidence": 100, "reasoning": reasoning}
    ) + "\n```"
    engine = _attach_unified_parser(
        FakeEngine(raw_response=raw, result=dict(RESULT_META), config=FakeEngineConfig())
    )
    analyzer = PairAnalyzer(engine=engine, symbol=SYMBOL, timeframe="1h")

    signal = await analyzer.analyze(candle_open_time="2026-10-24T17:00:00Z")

    assert signal.action == "HOLD"
    assert signal.confidence == 0.0
    assert signal.risk_approved is False
    assert signal.sl is None and signal.tp is None
    assert "empty reasoning" in signal.reasoning
    assert signal.model == "deepseek-v4.1-flash"
    assert signal.provider == "litellm"
    assert signal.raw_response_hash == hash_raw_response(raw)
    signal.validate()


@pytest.mark.asyncio
async def test_non_empty_reasoning_keeps_action_and_confidence():
    """A short but non-empty reasoning is legitimate: pass through untouched."""
    raw = "```json\n" + json.dumps(
        {"action": "HOLD", "confidence": 100, "reasoning": "ok"}
    ) + "\n```"
    engine = _attach_unified_parser(FakeEngine(raw_response=raw, result=dict(RESULT_META)))
    analyzer = PairAnalyzer(engine=engine, symbol=SYMBOL, timeframe="1h")

    signal = await analyzer.analyze(candle_open_time="2026-10-24T20:00:00Z")

    assert signal.action == "HOLD"
    assert signal.confidence == 1.0
    assert signal.reasoning == "ok"
    assert signal.raw_response_hash == hash_raw_response(raw)


@pytest.mark.asyncio
async def test_non_empty_reasoning_keeps_buy_action():
    """A BUY with a non-empty reasoning is never downgraded by the guard."""
    engine = _attach_unified_parser(FakeEngine(raw_response=RAW_BUY, result=dict(RESULT_META)))
    analyzer = PairAnalyzer(engine=engine, symbol=SYMBOL, timeframe="1h")

    signal = await analyzer.analyze()

    assert signal.action == "BUY"
    assert signal.confidence == 0.90
    assert signal.reasoning == "breakout"


@pytest.mark.asyncio
async def test_invalid_response_reports_real_model_not_unspecified():
    """A parsed response that reports 'unspecified' must use the configured model."""
    engine = FakeEngine(
        raw_response=RAW_BUY,
        result={"model": "unspecified", "provider": "unspecified", "raw_response": RAW_BUY},
        config=FakeEngineConfig(),
    )
    analyzer = PairAnalyzer(engine=engine, symbol=SYMBOL, timeframe="1h")

    signal = await analyzer.analyze()

    assert signal.model == "deepseek-v4.1-flash"
    assert signal.provider == "litellm"
    assert signal.raw_response_hash == hash_raw_response(RAW_BUY)


@pytest.mark.asyncio
async def test_error_result_is_unapproved_hold_with_reason():
    """An error result is an explicit, unapproved HOLD, never approved silently."""
    engine = FakeEngine(
        raw_response="",
        result={"error": "Invalid response format", "raw_response": ""},
        config=FakeEngineConfig(),
    )
    analyzer = PairAnalyzer(engine=engine, symbol=SYMBOL)

    signal = await analyzer.analyze(candle_open_time="2026-10-23T14:00:00Z")

    assert signal.action == "HOLD"
    assert signal.risk_approved is False
    assert "invalid LLM response: Invalid response format" in signal.reasoning
    assert signal.model == "deepseek-v4.1-flash"
    assert signal.provider == "litellm"


def test_truncate_context_accepts_numpy_candles():
    """NumPy OHLCV arrays must never be evaluated by truthiness (ambiguous)."""
    np = pytest.importorskip("numpy")
    from types import SimpleNamespace
    from src.signal_service.analyzer_adapter import truncate_context_to_as_of

    candles = np.array([
        [1.0, 1.0, 2.0, 0.5, 100.0, 10.0],
        [2.0, 2.0, 3.0, 1.5, 200.0, 20.0],
        [3.0, 3.0, 4.0, 2.5, 300.0, 30.0],
    ])
    timestamps = [_hour(13), _hour(14), _hour(15)]

    context = SimpleNamespace(ohlcv_candles=None, timestamps=None, current_price=None)
    kept = truncate_context_to_as_of(context, candles, timestamps, "1h", _hour(15))

    assert kept is True
    assert len(context.ohlcv_candles) == 2
    assert context.current_price == 200.0

    empty_context = SimpleNamespace(ohlcv_candles=None, timestamps=None, current_price=None)
    assert truncate_context_to_as_of(
        empty_context, np.empty((0, 6)), [], "1h", _hour(15),
    ) is False
    assert empty_context.ohlcv_candles is None


def test_csv_loader_emits_ccxt_six_column_arrays(tmp_path):
    """A replayed CSV must produce the same 6-column layout as the live collector.

    Regression: the loader used to drop the timestamp column and emit
    [open, high, low, close, volume], so `truncate_context_to_as_of` read the
    VOLUME as `current_price` and the chart generator (which needs
    ohlcv[:, 1:6]) could not index the rows.
    """
    np = pytest.importorskip("numpy")
    from src.signal_service.__main__ import _load_candles_csv

    path = tmp_path / "BTC_USDT_1h.csv"
    path.write_text(
        "time,open,high,low,close,volume\n"
        "2026-10-23T10:00:00Z,100.0,110.0,90.0,105.0,111111.0\n"
        "2026-10-23T11:00:00Z,105.0,115.0,95.0,110.0,222222.0\n"
        "2026-10-23T12:00:00Z,110.0,120.0,100.0,115.0,333333.0\n"
        "2026-10-23T13:00:00Z,115.0,125.0,105.0,120.0,444444.0\n",
        encoding="utf-8",
    )

    candles, timestamps = _load_candles_csv(str(path))

    assert isinstance(candles, np.ndarray)
    assert candles.shape == (4, 6)
    assert candles.dtype == np.float64
    # Column 0 is the CCXT timestamp in milliseconds.
    assert candles[0, 0] == datetime(2026, 10, 23, 10, tzinfo=timezone.utc).timestamp() * 1000
    assert candles[0, 1:].tolist() == [100.0, 110.0, 90.0, 105.0, 111111.0]
    assert [ts.hour for ts in timestamps] == [10, 11, 12, 13]


@pytest.mark.asyncio
async def test_csv_replay_round_trip_uses_close_not_volume(tmp_path):
    """CSV -> loader -> ReplayDataInjector -> truncate must expose close prices."""
    np = pytest.importorskip("numpy")
    from src.analyzer.pattern_engine.chart_generator import ChartGenerator
    from src.signal_service.__main__ import _load_candles_csv

    path = tmp_path / "BTC_USDT_1h.csv"
    path.write_text(
        "time,open,high,low,close,volume\n"
        "2026-10-23T10:00:00Z,100.0,110.0,90.0,105.0,111111.0\n"
        "2026-10-23T11:00:00Z,105.0,115.0,95.0,110.0,222222.0\n"
        "2026-10-23T12:00:00Z,110.0,120.0,100.0,115.0,333333.0\n"
        "2026-10-23T13:00:00Z,115.0,125.0,105.0,120.0,444444.0\n",
        encoding="utf-8",
    )
    candles, timestamps = _load_candles_csv(str(path))

    engine = FakeEngine(raw_response=RAW_BUY, result=dict(RESULT_META))
    engine.data_collector = FetchingDataCollector(timestamps, candles)
    analyzer = PairAnalyzer(engine=engine, symbol=SYMBOL, timeframe="1h")
    injector = ReplayDataInjector(analyzer, candles, timestamps)
    injector.install()
    try:
        # as_of = close of the 12:00 candle: only candles 10..12 are visible.
        injector.set_as_of(datetime(2026, 10, 23, 13, tzinfo=timezone.utc))
        context = engine.context
        assert len(context.timestamps) == 3
        # 115.0 is the close of the last visible candle; 333333.0 is its volume.
        assert context.current_price == 115.0
        assert np.asarray(context.ohlcv_candles)[:, 4].tolist() == [105.0, 110.0, 115.0]
        assert np.asarray(context.ohlcv_candles)[:, 5].tolist() == [111111.0, 222222.0, 333333.0]

        # The chart generator slices ohlcv[:, 1:6]; that requires the CCXT layout.
        chart_generator = ChartGenerator()
        figure = chart_generator._create_simple_candlestick_chart(  # pylint: disable=protected-access
            context.ohlcv_candles, SYMBOL, "1h", 600, 900, timestamps=context.timestamps,
        )
        assert len(figure.data) >= 2
        assert figure.data[0].close.tolist() == [105.0, 110.0, 115.0]
    finally:
        injector.restore()


RAW_BUY_ZERO_SL = (
    '{"signal": "BUY", "confidence": "HIGH", "stop_loss": 0, '
    '"take_profit": 70000.0, "position_size": 5, "reasoning": "bad stop"}'
)


@pytest.mark.asyncio
async def test_contract_invalid_output_falls_back_to_unapproved_hold():
    """A response that fails Signal.validate() must still yield a HOLD signal."""
    engine = FakeEngine(
        raw_response=RAW_BUY_ZERO_SL,
        result={**RESULT_META, "raw_response": RAW_BUY_ZERO_SL},
        config=FakeEngineConfig(),
    )
    analyzer = PairAnalyzer(engine=engine, symbol=SYMBOL, timeframe="1h")

    signal = await analyzer.analyze(candle_open_time="2026-10-23T14:00:00Z")

    assert signal.action == "HOLD"
    assert signal.risk_approved is False
    assert "invalid LLM output" in signal.reasoning
    assert "sl must be a positive number or null" in signal.reasoning
    # Real model/provider, never 'unspecified'.
    assert signal.model == "deepseek-v4.1-flash"
    assert signal.provider == "litellm"
    assert signal.raw_response_hash == hash_raw_response(RAW_BUY_ZERO_SL)


@pytest.mark.asyncio
async def test_contract_invalid_output_is_stored_with_heartbeat(tmp_path):
    """The fallback HOLD must be persisted and refresh the heartbeat."""
    engine = FakeEngine(
        raw_response=RAW_BUY_ZERO_SL,
        result={**RESULT_META, "raw_response": RAW_BUY_ZERO_SL},
        config=FakeEngineConfig(),
    )
    analyzer = PairAnalyzer(engine=engine, symbol=SYMBOL, timeframe="1h")
    signals_dir = tmp_path / "signals"
    storage = SignalStorage(signal_file_path(str(signals_dir), SYMBOL, "1h"))

    signal = await analyze_and_store(
        analyzer, storage, timeout_seconds=5.0, candle_open_time="2026-10-23T14:00:00Z",
    )

    assert signal.action == "HOLD"
    rows = storage.path and open(storage.path, encoding="utf-8").read().splitlines()
    assert len(rows) == 1
    stored = json.loads(rows[0])
    assert stored["action"] == "HOLD"
    assert stored["risk_approved"] is False
    heartbeat = json.loads((signals_dir / "heartbeat_BTC_USDT_1h.json").read_text(encoding="utf-8"))
    assert heartbeat["symbol"] == SYMBOL
    assert heartbeat["candle_open_time"] == "2026-10-23T14:00:00Z"


class FailingStorage:
    """Storage stub whose append always fails (e.g. disk or permission error)."""

    def __init__(self, path):
        self.path = path

    def append(self, _signal, dedupe=True):
        raise OSError("disk full")


@pytest.mark.asyncio
async def test_heartbeat_refreshed_when_append_fails(tmp_path):
    """A failed append must not leave heartbeat.json stale for the pair."""
    engine = FakeEngine(
        raw_response=RAW_BUY, result=dict(RESULT_META), config=FakeEngineConfig(),
    )
    analyzer = PairAnalyzer(engine=engine, symbol=SYMBOL, timeframe="1h")
    signals_dir = tmp_path / "signals"
    signals_dir.mkdir()
    storage = FailingStorage(str(signals_dir / "BTC_USDT_1h.jsonl"))

    with pytest.raises(OSError):
        await analyze_and_store(
            analyzer, storage, 5.0, candle_open_time="2026-10-23T14:00:00Z",
        )

    heartbeat = json.loads((signals_dir / "heartbeat_BTC_USDT_1h.json").read_text(encoding="utf-8"))
    assert heartbeat["symbol"] == SYMBOL
    assert heartbeat["candle_open_time"] == "2026-10-23T14:00:00Z"


class RaisingAnalyzer:
    """Analyzer whose analysis blows up outside analyze()'s own fallback."""

    symbol = SYMBOL
    timeframe = "1h"

    async def analyze(self, candle_open_time=None):
        raise RuntimeError("boom")


def test_run_cycle_refreshes_heartbeat_on_failure(tmp_path):
    """A pair whose cycle errors still gets a fresh heartbeat (liveness)."""
    from src.signal_service.service import SignalService

    service = SignalService(
        pairs=[SYMBOL],
        timeframe="1h",
        analyzer_factory=lambda _symbol: RaisingAnalyzer(),
        signals_dir=str(tmp_path),
    )

    results = asyncio.run(service.run_cycle(_hour(14)))

    assert results[SYMBOL] == "error"
    heartbeat = json.loads((tmp_path / "heartbeat_BTC_USDT_1h.json").read_text(encoding="utf-8"))
    assert heartbeat["candle_open_time"] == "2026-10-23T14:00:00Z"
    assert SYMBOL in service.heartbeats


def _attach_unified_parser(engine):
    """Give the fake engine the AnalysisResultProcessor surface the bot has."""
    from types import SimpleNamespace
    from src.parsing.unified_parser import UnifiedParser

    engine.result_processor = SimpleNamespace(
        unified_parser=UnifiedParser(logger=logging.getLogger("tests")),
    )
    return engine


FENCED_NUMERIC_CONFIDENCE = (
    "```json\n"
    '{"signal": "BUY", "confidence": 45, "stop_loss": 67000.0, '
    '"take_profit": 70000.0, "position_size": 5, "reasoning": "weak setup"}\n'
    "```"
)


@pytest.mark.asyncio
async def test_numeric_confidence_uses_json_path_and_is_vetted():
    """Numeric confidence must survive extraction (45 -> 0.45) and be vetted.

    The JSON path keeps the raw 0-100 score rather than collapsing it to a
    label, so a low-confidence signal still fails the 0.6 min_confidence gate.
    """
    engine = _attach_unified_parser(FakeEngine(
        raw_response=FENCED_NUMERIC_CONFIDENCE,
        result={**RESULT_META, "raw_response": FENCED_NUMERIC_CONFIDENCE},
        config=FakeEngineConfig(),
    ))
    analyzer = PairAnalyzer(engine=engine, symbol=SYMBOL, timeframe="1h")

    signal = await analyzer.analyze(candle_open_time="2026-10-23T14:00:00Z")

    assert signal.action == "BUY"
    assert signal.confidence == 0.45
    assert signal.risk_approved is False


FENCED_CONFIDENCE_57 = (
    "```json\n"
    '{"signal": "BUY", "confidence": 57, "stop_loss": 67000.0, '
    '"take_profit": 70000.0, "position_size": 5, "reasoning": "decent setup"}\n'
    "```"
)


@pytest.mark.asyncio
async def test_numeric_confidence_57_is_preserved_as_055():
    """57 must round to 0.55, not collapse to the MEDIUM label's 0.65."""
    engine = _attach_unified_parser(FakeEngine(
        raw_response=FENCED_CONFIDENCE_57,
        result={**RESULT_META, "raw_response": FENCED_CONFIDENCE_57},
        config=FakeEngineConfig(),
    ))
    analyzer = PairAnalyzer(engine=engine, symbol=SYMBOL, timeframe="1h")

    signal = await analyzer.analyze(candle_open_time="2026-10-23T14:00:00Z")

    assert signal.confidence == 0.55


FENCED_NAN_CONFIDENCE = (
    "```json\n"
    '{"signal": "BUY", "confidence": NaN, "stop_loss": 67000.0, '
    '"take_profit": 70000.0, "position_size": 5, "reasoning": "bad score"}\n'
    "```"
)


@pytest.mark.asyncio
async def test_nan_confidence_never_becomes_max_confidence():
    """A NaN score must not be clamped to 1.0 and approved.

    Python's json.loads accepts NaN; the extractor drops the non-finite score
    so the historical LOW label fallback (0.35) applies instead.
    """
    engine = _attach_unified_parser(FakeEngine(
        raw_response=FENCED_NAN_CONFIDENCE,
        result={**RESULT_META, "raw_response": FENCED_NAN_CONFIDENCE},
        config=FakeEngineConfig(),
    ))
    analyzer = PairAnalyzer(engine=engine, symbol=SYMBOL, timeframe="1h")

    signal = await analyzer.analyze(candle_open_time="2026-10-23T14:00:00Z")

    assert signal.confidence == 0.35
    assert signal.risk_approved is False


@pytest.mark.asyncio
async def test_label_only_confidence_keeps_the_legacy_mapping():
    """Without a numeric score, the label fallback is unchanged."""
    raw = (
        '{"signal": "HOLD", "confidence": "MEDIUM", "reasoning": "no setup"}'
    )
    engine = _attach_unified_parser(FakeEngine(
        raw_response=raw,
        result={**RESULT_META, "raw_response": raw},
        config=FakeEngineConfig(),
    ))
    analyzer = PairAnalyzer(engine=engine, symbol=SYMBOL, timeframe="1h")

    signal = await analyzer.analyze(candle_open_time="2026-10-23T14:00:00Z")

    assert signal.confidence == 0.65


@pytest.mark.asyncio
async def test_missing_confidence_defaults_to_medium_as_before():
    """A payload without confidence keeps the historical MEDIUM (0.65) default."""
    raw = '{"signal": "HOLD", "reasoning": "no setup"}'
    engine = _attach_unified_parser(FakeEngine(
        raw_response=raw,
        result={**RESULT_META, "raw_response": raw},
        config=FakeEngineConfig(),
    ))
    analyzer = PairAnalyzer(engine=engine, symbol=SYMBOL, timeframe="1h")

    signal = await analyzer.analyze(candle_open_time="2026-10-23T14:00:00Z")

    assert signal.confidence == 0.65


FENCED_ACTION_KEY = (
    "```json\n"
    '{"action": "SELL", "confidence": "HIGH", "stop_loss": 70000.0, '
    '"take_profit": 64000.0, "position_size": 5, "reasoning": "breakdown"}\n'
    "```"
)


@pytest.mark.asyncio
async def test_action_key_is_honoured_like_the_live_bot():
    """A response keyed on `action` must not silently degrade to HOLD."""
    engine = _attach_unified_parser(FakeEngine(
        raw_response=FENCED_ACTION_KEY,
        result={**RESULT_META, "raw_response": FENCED_ACTION_KEY},
        config=FakeEngineConfig(),
    ))
    analyzer = PairAnalyzer(engine=engine, symbol=SYMBOL, timeframe="1h")

    signal = await analyzer.analyze(candle_open_time="2026-10-23T14:00:00Z")

    assert signal.action == "SELL"
    assert signal.risk_approved is True
    assert signal.sl == 70000.0
    assert signal.tp == 64000.0


def test_each_pair_has_its_own_heartbeat_file(tmp_path):
    """Writing one pair's heartbeat must never overwrite another pair's file."""
    from src.signal_service.storage import write_heartbeat

    signals_dir = tmp_path / "signals"
    write_heartbeat(str(signals_dir), "BTC/USDT", "1h", "2026-10-23T14:00:00Z")
    write_heartbeat(str(signals_dir), "ETH/USDT", "1h", "2026-10-23T13:00:00Z")

    btc = json.loads((signals_dir / "heartbeat_BTC_USDT_1h.json").read_text(encoding="utf-8"))
    eth = json.loads((signals_dir / "heartbeat_ETH_USDT_1h.json").read_text(encoding="utf-8"))
    assert btc["symbol"] == "BTC/USDT"
    assert btc["candle_open_time"] == "2026-10-23T14:00:00Z"
    assert eth["symbol"] == "ETH/USDT"
    assert eth["candle_open_time"] == "2026-10-23T13:00:00Z"


def test_run_cycle_keeps_per_pair_heartbeats_and_summary(tmp_path):
    """A healthy pair must not mask a stuck pair in the persisted liveness files."""
    from src.signal_service.service import SignalService

    def analyzer_factory(symbol):
        if symbol == SYMBOL:
            return RaisingAnalyzer()
        return RecordingAnalyzer(symbol)

    service = SignalService(
        pairs=[SYMBOL, "ETH/USDT"],
        timeframe="1h",
        analyzer_factory=analyzer_factory,
        signals_dir=str(tmp_path),
    )

    results = asyncio.run(service.run_cycle(_hour(14)))

    assert results[SYMBOL] == "error"
    btc = json.loads((tmp_path / "heartbeat_BTC_USDT_1h.json").read_text(encoding="utf-8"))
    eth = json.loads((tmp_path / "heartbeat_ETH_USDT_1h.json").read_text(encoding="utf-8"))
    assert btc["candle_open_time"] == "2026-10-23T14:00:00Z"
    assert eth["candle_open_time"] == "2026-10-23T14:00:00Z"
    summary = json.loads((tmp_path / "heartbeat_summary.json").read_text(encoding="utf-8"))
    assert set(summary["pairs"]) == {SYMBOL, "ETH/USDT"}


def test_csv_loader_rejects_short_rows(tmp_path):
    """A row missing volume must fail cleanly, not crash np.reshape later."""
    from src.signal_service.__main__ import _load_candles_csv

    path = tmp_path / "short.csv"
    path.write_text(
        "time,open,high,low,close,volume\n"
        "2026-10-23T10:00:00Z,100.0,110.0,90.0,105.0\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="expected 6 columns"):
        _load_candles_csv(str(path))


def test_csv_loader_rejects_newest_first_files(tmp_path):
    """A descending csv would misalign candles/timestamps and look ahead."""
    from src.signal_service.__main__ import _load_candles_csv

    path = tmp_path / "descending.csv"
    path.write_text(
        "time,open,high,low,close,volume\n"
        "2026-10-23T12:00:00Z,110.0,120.0,100.0,115.0,333333.0\n"
        "2026-10-23T11:00:00Z,105.0,115.0,95.0,110.0,222222.0\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="strictly ascending"):
        _load_candles_csv(str(path))


def test_csv_loader_rejects_duplicate_timestamps(tmp_path):
    """Duplicate open times make position-based slicing ambiguous."""
    from src.signal_service.__main__ import _load_candles_csv

    path = tmp_path / "duplicate.csv"
    path.write_text(
        "time,open,high,low,close,volume\n"
        "2026-10-23T11:00:00Z,105.0,115.0,95.0,110.0,222222.0\n"
        "2026-10-23T11:00:00Z,106.0,116.0,96.0,111.0,222223.0\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="strictly ascending"):
        _load_candles_csv(str(path))


class MarkerRagEngine:
    """RAG stub whose retrieval is observable (the neutralized one returns empty)."""

    class _CB:
        def get_latest_article_urls(self):
            return {}

    context_builder = _CB()

    async def retrieve_context(self, *_args, **_kwargs):
        return ("LIVE NEWS CONTEXT", {})


def _replay_analyzer(engine, *, use_news=True, use_chart=True):
    engine.data_collector = FetchingDataCollector([], [])
    analyzer = PairAnalyzer(
        engine=engine, symbol=SYMBOL, timeframe="1h", use_news=use_news, use_chart=use_chart,
    )
    injector = ReplayDataInjector(
        analyzer, [[0.0, 100.0, 110.0, 90.0, 105.0, 1000.0]], [_hour(14)],
    )
    injector.install()
    return analyzer, injector


@pytest.mark.asyncio
async def test_replay_install_binds_chart_ablation_flag():
    """--use-chart must control replay exactly as it controls run."""
    engine = FakeEngine(raw_response=RAW_BUY, result=dict(RESULT_META))
    analyzer, injector = _replay_analyzer(engine, use_chart=True)
    try:
        assert engine.model_manager.supports_image_analysis("any") is True
    finally:
        injector.restore()

    engine = FakeEngine(raw_response=RAW_BUY, result=dict(RESULT_META))
    analyzer, injector = _replay_analyzer(engine, use_chart=False)
    try:
        assert engine.model_manager.supports_image_analysis("any") is False
    finally:
        injector.restore()


@pytest.mark.asyncio
async def test_replay_install_binds_news_ablation_flag():
    """--no-news must control replay exactly as it controls run."""
    engine = FakeEngine(raw_response=RAW_BUY, result=dict(RESULT_META))
    engine.rag_engine = MarkerRagEngine()
    analyzer, injector = _replay_analyzer(engine, use_news=True)
    try:
        assert await engine.rag_engine.retrieve_context() == ("LIVE NEWS CONTEXT", {})
    finally:
        injector.restore()

    engine = FakeEngine(raw_response=RAW_BUY, result=dict(RESULT_META))
    engine.rag_engine = MarkerRagEngine()
    analyzer, injector = _replay_analyzer(engine, use_news=False)
    try:
        assert await engine.rag_engine.retrieve_context() == ("", {})
    finally:
        injector.restore()


class SharedModelManager:
    """Process-wide model manager stub shared by several per-pair engines."""

    last_finish_reason = None

    def supports_image_analysis(self, *_args, **_kwargs) -> bool:
        return True


@pytest.mark.asyncio
async def test_flags_reapplied_for_shared_model_manager():
    """Each pair re-binds its chart flag before its own analysis."""
    shared_manager = SharedModelManager()
    engine_a = FakeEngine(raw_response=RAW_BUY, result=dict(RESULT_META))
    engine_b = FakeEngine(raw_response=RAW_BUY, result=dict(RESULT_META))
    engine_a.model_manager = shared_manager
    engine_b.model_manager = shared_manager
    analyzer_a = PairAnalyzer(engine=engine_a, symbol=SYMBOL, timeframe="1h", use_chart=False)
    analyzer_b = PairAnalyzer(engine=engine_b, symbol=SYMBOL, timeframe="1h", use_chart=True)

    await analyzer_a.analyze()
    assert shared_manager.supports_image_analysis("BTC/USDT") is False
    await analyzer_b.analyze()
    assert shared_manager.supports_image_analysis("BTC/USDT") is True
    # A's second cycle must not inherit B's flag.
    await analyzer_a.analyze()
    assert shared_manager.supports_image_analysis("BTC/USDT") is False


@pytest.mark.asyncio
async def test_news_ablation_restored_for_shared_rag_engine():
    """A later news-enabled pair un-does another pair's neutralization."""
    shared_rag = MarkerRagEngine()
    engine_a = FakeEngine(raw_response=RAW_BUY, result=dict(RESULT_META))
    engine_b = FakeEngine(raw_response=RAW_BUY, result=dict(RESULT_META))
    engine_a.rag_engine = shared_rag
    engine_b.rag_engine = shared_rag
    analyzer_a = PairAnalyzer(engine=engine_a, symbol=SYMBOL, timeframe="1h", use_news=True)
    analyzer_b = PairAnalyzer(engine=engine_b, symbol=SYMBOL, timeframe="1h", use_news=False)

    await analyzer_a.analyze()
    assert await shared_rag.retrieve_context() == ("LIVE NEWS CONTEXT", {})
    await analyzer_b.analyze()
    assert await shared_rag.retrieve_context() == ("", {})
    # A's second cycle must get live news back, not B's neutralized stub.
    await analyzer_a.analyze()
    assert await shared_rag.retrieve_context() == ("LIVE NEWS CONTEXT", {})


PROSE_WITHOUT_JSON = (
    "1) MARKET STRUCTURE: price is below the SMAs. "
    "2) MOMENTUM: weak. ... 12) DECISION: HOLD"
)
REPAIRED_HOLD = '{"action": "HOLD", "confidence": 50, "reasoning": "repaired decision"}'


class RepairModelManager(FakeModelManager):
    """Fake model manager recording the single JSON-repair request."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    async def send_prompt(self, **kwargs):
        self.calls.append(kwargs)
        return self.replies.pop(0)


def _json_parse_error_result(raw: str) -> dict:
    return {
        **RESULT_META,
        "raw_response": raw,
        "parse_error": "Failed to parse response",
        "response_validation": {
            "schema": "1.0",
            "status": "invalid",
            "valid": False,
            "errors": [{"field": "response", "message": "Failed to parse response JSON", "type": "json_parse_error"}],
        },
    }


@pytest.mark.asyncio
async def test_json_parse_error_is_repaired_once_and_the_decision_is_used():
    """Malformed first answer -> one short repair -> parsed valid decision."""
    engine = _attach_unified_parser(FakeEngine(
        raw_response=PROSE_WITHOUT_JSON,
        result=_json_parse_error_result(PROSE_WITHOUT_JSON),
        config=FakeEngineConfig(),
    ))
    repair = RepairModelManager([REPAIRED_HOLD])
    engine.model_manager = repair
    analyzer = PairAnalyzer(
        engine=engine, symbol=SYMBOL, timeframe="1h", json_response_format=True,
    )

    signal = await analyzer.analyze(candle_open_time="2026-10-25T09:00:00Z")

    assert len(repair.calls) == 1
    assert "Return ONLY the JSON object matching the response schema, no prose." in repair.calls[0]["prompt"]
    assert "response_format" not in repair.calls[0]
    assert engine.calls[-1]["response_format"] == {"type": "json_object"}
    assert engine.calls[-1]["json_object_output"] is True
    assert signal.action == "HOLD"
    assert signal.reasoning == "repaired decision"
    assert signal.risk_approved is False
    signal.validate()


@pytest.mark.asyncio
async def test_json_parse_error_twice_falls_back_to_exact_hold_reason():
    """A second malformed answer must stop immediately with the documented reason."""
    engine = _attach_unified_parser(FakeEngine(
        raw_response=PROSE_WITHOUT_JSON,
        result=_json_parse_error_result(PROSE_WITHOUT_JSON),
        config=FakeEngineConfig(),
    ))
    repair = RepairModelManager([PROSE_WITHOUT_JSON + " still prose"])
    engine.model_manager = repair
    analyzer = PairAnalyzer(engine=engine, symbol=SYMBOL, timeframe="1h")

    signal = await analyzer.analyze(candle_open_time="2026-10-25T09:00:00Z")

    assert len(repair.calls) == 1, "repair must run exactly once, never loop"
    assert signal.action == "HOLD"
    assert signal.risk_approved is False
    assert signal.reasoning == "invalid LLM output: JSON parse error after repair"
    assert signal.raw_response_hash == hash_raw_response(PROSE_WITHOUT_JSON)
    signal.validate()


RESPONSE_FORMAT_ECHO = '{"type": "json_object"}'


@pytest.mark.asyncio
async def test_response_format_echo_is_repaired_once_without_response_format():
    """Live echo -> one repair WITHOUT response_format -> the real decision is used."""
    engine = _attach_unified_parser(FakeEngine(
        raw_response=RESPONSE_FORMAT_ECHO,
        result=dict(RESULT_META),
        config=FakeEngineConfig(),
    ))
    repair = RepairModelManager([REPAIRED_HOLD])
    engine.model_manager = repair
    analyzer = PairAnalyzer(
        engine=engine, symbol=SYMBOL, timeframe="1h", json_response_format=True,
    )

    signal = await analyzer.analyze(candle_open_time="2026-10-25T09:00:00Z")

    assert len(repair.calls) == 1
    assert "Return ONLY the JSON object matching the response schema, no prose." in repair.calls[0]["prompt"]
    assert "response_format" not in repair.calls[0], "repair must not repeat the forced format"
    assert engine.calls[-1]["response_format"] == {"type": "json_object"}
    assert signal.action == "HOLD"
    assert signal.reasoning == "repaired decision"
    assert signal.risk_approved is False
    signal.validate()


@pytest.mark.asyncio
async def test_response_format_echo_twice_falls_back_to_hold_without_looping():
    """A provider that echoes again after one repair stops at HOLD."""
    engine = _attach_unified_parser(FakeEngine(
        raw_response=RESPONSE_FORMAT_ECHO,
        result=dict(RESULT_META),
        config=FakeEngineConfig(),
    ))
    repair = RepairModelManager([RESPONSE_FORMAT_ECHO])
    engine.model_manager = repair
    analyzer = PairAnalyzer(
        engine=engine, symbol=SYMBOL, timeframe="1h", json_response_format=True,
    )

    signal = await analyzer.analyze(candle_open_time="2026-10-25T09:00:00Z")

    assert len(repair.calls) == 1, "repair must run exactly once, never loop"
    assert signal.action == "HOLD"
    assert signal.risk_approved is False
    assert signal.reasoning == "invalid LLM output: JSON parse error after repair"
    signal.validate()


@pytest.mark.asyncio
async def test_json_format_is_off_by_default():
    """Default false keeps the prose prompt and sends no response_format."""
    engine = _attach_unified_parser(FakeEngine(
        raw_response=RAW_BUY,
        result=dict(RESULT_META),
        config=FakeEngineConfig(),
    ))
    analyzer = PairAnalyzer(engine=engine, symbol=SYMBOL, timeframe="1h")

    await analyzer.analyze(candle_open_time="2026-10-25T09:00:00Z")

    assert engine.calls[-1]["response_format"] is None
    assert engine.calls[-1]["json_object_output"] is False


@pytest.mark.asyncio
async def test_json_format_can_be_disabled_for_signal_analysis():
    """The [signal_service] opt-in can be turned off explicitly."""
    engine = _attach_unified_parser(FakeEngine(
        raw_response=RAW_BUY,
        result=dict(RESULT_META),
        config=FakeEngineConfig(),
    ))
    analyzer = PairAnalyzer(
        engine=engine, symbol=SYMBOL, timeframe="1h", json_response_format=False,
    )

    await analyzer.analyze(candle_open_time="2026-10-25T09:00:00Z")

    assert engine.calls[-1]["response_format"] is None


@pytest.mark.asyncio
async def test_truncated_parse_error_is_rejected_before_repair():
    """finish_reason=length must keep its guard and never be repaired into a trade."""
    raw = PROSE_WITHOUT_JSON
    engine = _attach_unified_parser(FakeEngine(
        raw_response=raw,
        result=_json_parse_error_result(raw),
        config=FakeEngineConfig(),
    ))
    repair = RepairModelManager(['{"action": "BUY", "confidence": "HIGH", "stop_loss": 1, "take_profit": 2, "reasoning": "x"}'])
    engine.model_manager = repair
    engine.model_manager.last_finish_reason = "length"
    analyzer = PairAnalyzer(engine=engine, symbol=SYMBOL, timeframe="1h")

    signal = await analyzer.analyze(candle_open_time="2026-10-25T09:00:00Z")

    assert repair.calls == [], "a truncated answer must not be repaired"
    assert signal.action == "HOLD"
    assert signal.risk_approved is False
    assert "finish_reason=length" in signal.reasoning
    assert "truncated content" in signal.reasoning


@pytest.mark.asyncio
async def test_empty_parse_error_keeps_the_empty_content_diagnostic():
    """An empty completion must be reported as empty, not as a failed repair."""
    engine = _attach_unified_parser(FakeEngine(
        raw_response="",
        result=_json_parse_error_result(""),
        config=FakeEngineConfig(),
    ))
    repair = RepairModelManager([REPAIRED_HOLD])
    engine.model_manager = repair
    analyzer = PairAnalyzer(engine=engine, symbol=SYMBOL, timeframe="1h")

    signal = await analyzer.analyze(candle_open_time="2026-10-25T09:00:00Z")

    assert repair.calls == [], "an empty completion has nothing to repair"
    assert signal.action == "HOLD"
    assert signal.risk_approved is False
    assert "empty content" in signal.reasoning
    assert "JSON parse error after repair" not in signal.reasoning


@pytest.mark.asyncio
async def test_content_filtered_parse_error_is_rejected_before_repair():
    """finish_reason=content_filter keeps its guard and is never repaired."""
    raw = PROSE_WITHOUT_JSON
    engine = _attach_unified_parser(FakeEngine(
        raw_response=raw,
        result=_json_parse_error_result(raw),
        config=FakeEngineConfig(),
    ))
    repair = RepairModelManager([REPAIRED_HOLD])
    engine.model_manager = repair
    engine.model_manager.last_finish_reason = "content_filter"
    analyzer = PairAnalyzer(engine=engine, symbol=SYMBOL, timeframe="1h")

    signal = await analyzer.analyze(candle_open_time="2026-10-25T09:00:00Z")

    assert repair.calls == []
    assert signal.action == "HOLD"
    assert signal.risk_approved is False
    assert "content filtered" in signal.reasoning


ERROR_PAYLOAD = (
    "```json\n"
    '{"analysis": {"summary": "Analysis unavailable: Empty response from API. Please try again later."}}\n'
    "```"
)


@pytest.mark.asyncio
async def test_repair_error_payload_is_not_mistaken_for_a_decision():
    """format_error_response() is JSON too, but must not count as a repair."""
    engine = _attach_unified_parser(FakeEngine(
        raw_response=PROSE_WITHOUT_JSON,
        result=_json_parse_error_result(PROSE_WITHOUT_JSON),
        config=FakeEngineConfig(),
    ))
    repair = RepairModelManager([ERROR_PAYLOAD])
    engine.model_manager = repair
    analyzer = PairAnalyzer(engine=engine, symbol=SYMBOL, timeframe="1h")

    signal = await analyzer.analyze(candle_open_time="2026-10-25T09:00:00Z")

    assert len(repair.calls) == 1
    assert signal.action == "HOLD"
    assert signal.risk_approved is False
    assert signal.reasoning == "invalid LLM output: JSON parse error after repair"
    signal.validate()


@pytest.mark.asyncio
async def test_repair_honours_the_json_format_opt_out():
    """When structured output is disabled, the repair must not send response_format."""
    engine = _attach_unified_parser(FakeEngine(
        raw_response=PROSE_WITHOUT_JSON,
        result=_json_parse_error_result(PROSE_WITHOUT_JSON),
        config=FakeEngineConfig(),
    ))
    repair = RepairModelManager([REPAIRED_HOLD])
    engine.model_manager = repair
    analyzer = PairAnalyzer(
        engine=engine, symbol=SYMBOL, timeframe="1h", json_response_format=False,
    )

    signal = await analyzer.analyze(candle_open_time="2026-10-25T09:00:00Z")

    assert len(repair.calls) == 1
    assert "response_format" not in repair.calls[0]
    assert engine.calls[-1]["response_format"] is None
    assert engine.calls[-1]["json_object_output"] is False
    assert signal.reasoning == "repaired decision"
