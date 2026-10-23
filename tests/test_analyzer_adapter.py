"""Tests for the AnalysisEngine -> Signal adapter.

The adapter must analyze stateless, extract via PositionExtractor, hash the
raw response and compute risk_approved per the contract rules.
"""

import asyncio
from datetime import datetime, timedelta, timezone

import pytest

from src.signal_service.analyzer_adapter import PairAnalyzer, ReplayDataInjector, analyze_with_timeout
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
    engine = FakeEngine(raw_response=RAW_BUY, result=dict(RESULT_META))
    analyzer = PairAnalyzer(engine=engine, symbol=SYMBOL)
    signal = await analyzer.analyze()
    dumped = repr(signal.to_dict())
    assert "breakout" not in dumped or "reasoning" in dumped
    # only the reasoning text (a summarised form) is allowed, never the full
    # raw response; verify the raw text is limited to its reasoning field.
    assert signal.to_dict().get("raw_response") is None


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
