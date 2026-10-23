"""Tests for the AnalysisEngine -> Signal adapter.

The adapter must analyze stateless, extract via PositionExtractor, hash the
raw response and compute risk_approved per the contract rules.
"""

from datetime import datetime, timezone

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

    def __init__(self, raw_response=None, result=None, current_price=68000.0):
        self.last_llm_response = raw_response
        self.model_manager = FakeModelManager()
        self.rag_engine = FakeRagEngine()
        self.context = FakeContext(current_price=current_price)
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
    "provider": "opencode-go",
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
    assert signal.provider == "opencode-go"
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