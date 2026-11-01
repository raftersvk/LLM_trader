"""Tests for position ENTRY CONTEXT and live position metrics.

The blocks are prompt-only: the signal contract is unchanged. Metrics are
R-multiples of the original entry SL, never of a tightened current stop.
"""

from __future__ import annotations

import asyncio
import json
import re
from datetime import datetime, timezone

import pytest

from src.signal_service.analyzer_adapter import PairAnalyzer, UpdateContext
from src.signal_service.contract import Signal
from src.signal_service.position_metrics import (
    EntryContext,
    build_position_metrics,
    format_entry_context_block,
    format_position_metrics_block,
    format_regime_comparison_block,
)
from src.signal_service.positions import (
    PositionSnapshot,
    PositionsSnapshot,
    utc_now_iso,
    write_positions_atomic,
)
from src.signal_service.service import SignalService
from src.signal_service.storage import SignalStorage

SYMBOL = "BTC/USDT"
AS_OF = datetime(2026, 10, 30, 11, 0, tzinfo=timezone.utc)

CANDLES = [
    [1698652800, 67500.0, 68500.0, 67000.0, 68000.0, 1.0],  # 09:00Z
    [1698656400, 68000.0, 69500.0, 67200.0, 69000.0, 1.0],  # 10:00Z
]
TIMESTAMPS = [
    datetime(2026, 10, 30, 9, 0, tzinfo=timezone.utc),
    datetime(2026, 10, 30, 10, 0, tzinfo=timezone.utc),
]
# A still-forming candle must never contribute its high/low.
FUTURE_CANDLE = [1698660000, 69000.0, 99999.0, 1.0, 69000.0, 1.0]
FUTURE_TIMESTAMP = datetime(2026, 10, 30, 11, 0, tzinfo=timezone.utc)


def entry_kwargs(**overrides):
    base = dict(
        schema_version="1.0",
        decision_id="entry-1",
        source="llm_trader",
        symbol=SYMBOL,
        timeframe="1h",
        candle_open_time="2026-10-30T08:00:00Z",
        candle_close_time="2026-10-30T09:00:00Z",
        created_at="2026-10-30T09:00:02Z",
        expires_at="2026-10-30T10:00:00Z",
        action="BUY",
        confidence=0.72,
        risk_approved=True,
        sl=66500.0,
        tp=72000.0,
        position_size_pct=0.05,
        reasoning="pullback to SMA20 rejected",
        model="m",
        provider="p",
        setup="trend_pullback",
        regime="trend_up",
        htf_alignment="aligned",
    )
    base.update(overrides)
    return base


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


# ------------------------------------------------------------------ metrics
def test_long_metrics_use_entry_sl_and_closed_candles_only():
    metrics = build_position_metrics(
        entry_price=67500.0,
        entry_sl=66500.0,  # risk 1000, not the tightened 68000 stop
        side="long",
        current_price=69000.0,
        candles=CANDLES + [FUTURE_CANDLE],
        timestamps=TIMESTAMPS + [FUTURE_TIMESTAMP],
        opened_at="2026-10-30T08:30:00Z",
        as_of=AS_OF,
        atr=300.0,
    )
    assert metrics.risk_basis == 66500.0
    assert metrics.pnl_r == 1.5
    # The 99999 high of the still-forming candle is excluded.
    assert metrics.mfe_r == 2.0
    assert metrics.mae_r == 0.5
    assert metrics.mfe_pct == pytest.approx(2.962962962962963)
    assert metrics.mae_pct == pytest.approx(-0.7407407407407407)
    assert metrics.atr == 300.0
    assert metrics.atr_pct is not None and 0.43 < metrics.atr_pct < 0.44
    assert metrics.hours_in_trade == 2.5


def test_short_metrics_are_symmetric():
    metrics = build_position_metrics(
        entry_price=78000.0,
        entry_sl=79000.0,  # risk 1000
        side="short",
        current_price=76000.0,
        candles=[[1698656400, 78000.0, 78500.0, 75500.0, 76000.0, 1.0]],
        timestamps=[datetime(2026, 10, 30, 10, 0, tzinfo=timezone.utc)],
        opened_at="2026-10-30T09:30:00Z",
        as_of=AS_OF,
    )
    assert metrics.pnl_r == 2.0
    assert metrics.mfe_r == 2.5
    assert metrics.mae_r == 0.5
    assert metrics.mfe_pct > 0
    assert metrics.mae_pct < 0


def test_missing_entry_sl_has_no_r_basis():
    metrics = build_position_metrics(
        entry_price=67500.0,
        entry_sl=None,
        side="long",
        current_price=69000.0,
        as_of=AS_OF,
    )
    assert metrics.risk_basis is None
    assert metrics.pnl_r is None
    assert metrics.mfe_r is None
    assert metrics.mae_r is None


# ------------------------------------------------------------------ formatting
def test_missing_entry_signal_renders_explicit_fallback():
    block = format_entry_context_block(None)
    assert "ENTRY CONTEXT" in block
    assert "unavailable" in block


def test_blocks_render_all_required_fields():
    entry = EntryContext(
        setup="trend_pullback",
        regime="trend_up",
        htf_alignment="aligned",
        confidence=0.72,
        reasoning="pullback to SMA20 rejected",
        invalidation="Close below the pullback low.",
        sl=66500.0,
        tp=72000.0,
        created_at="2026-10-30T09:00:02Z",
    )
    metrics = build_position_metrics(
        entry_price=67500.0,
        entry_sl=66500.0,
        side="long",
        current_price=69000.0,
        candles=CANDLES,
        timestamps=TIMESTAMPS,
        opened_at="2026-10-30T08:30:00Z",
        as_of=AS_OF,
    )
    block = format_entry_context_block(entry)
    for token in (
        "setup=trend_pullback", "entry_regime=trend_up", "entry_htf_alignment=aligned",
        "entry_confidence=0.720", "invalidation=Close below the pullback low.",
        "entry_sl=66500", "entry_tp=72000", "entry_created_at=2026-10-30T09:00:02Z",
        "entry_reasoning=pullback to SMA20 rejected",
    ):
        assert token in block
    metric_block = format_position_metrics_block(metrics)
    for token in (
        "current_price=69000", "pnl_R=1.50", "mfe_R=2.00", "mae_R=0.50",
        "mfe_pct=2.96", "mae_pct=-0.74", "hours_in_trade=2.5",
        "R basis=entry_sl=66500", "not the current stop",
    ):
        assert token in metric_block
    regime = format_regime_comparison_block(
        "trend_up", {"regime": "trend_up", "allowed_setups": ["trend_pullback"]}
    )
    assert "entry_regime=trend_up" in regime
    assert "regime_match=yes" in regime
    # The REGIME SNAPSHOT block is the only source of the current regime.
    assert "current_regime=" not in regime
    assert "allowed_setups=" not in regime


def test_regime_comparison_detects_mismatch_and_missing_snapshot():
    mismatch = format_regime_comparison_block(
        "trend_up", {"regime": "range", "allowed_setups": ["range_mean_reversion"]}
    )
    assert "regime_match=no" in mismatch
    fallback = format_regime_comparison_block("trend_up", None)
    assert "regime_match=unknown" in fallback
    assert "current_regime" not in fallback


# ------------------------------------------------------------------ adapter
class _FakeEngine:
    class _MM:
        last_finish_reason = None

        def supports_image_analysis(self, *_a, **_k):
            return False

    class _Ctx:
        current_price = 69000.0
        timeframe = "1h"

        def __init__(self):
            self.ohlcv_candles = CANDLES + [FUTURE_CANDLE]
            self.timestamps = TIMESTAMPS + [FUTURE_TIMESTAMP]
            self.technical_data = {"atr": 300.0, "atr_percent": 0.43}

    def __init__(self, raw, with_context=True):
        self.last_llm_response = raw
        self.model_manager = self._MM()
        self.context = self._Ctx() if with_context else None
        self.calls = []

    async def analyze_market(self, **kwargs):
        self.calls.append(kwargs)
        return {"raw_response": self.last_llm_response, "recommendation": "UPDATE"}


def _ctx(**overrides):
    base = dict(
        position_id="42", position_side="long", entry_price=67500.0,
        stop_loss=67000.0, take_profit=72000.0, revision=1,
        based_on_entry="entry-1", current_price=69000.0, entry_sl=66500.0,
        entry_context=EntryContext(
            setup="trend_pullback", regime="trend_up", htf_alignment="aligned",
            confidence=0.72, reasoning="pullback rejected",
            invalidation="Close below the pullback low.",
            sl=66500.0, tp=72000.0, created_at="2026-10-30T09:00:02Z",
        ),
        position_opened_at="2026-10-30T08:30:00Z",
    )
    base.update(overrides)
    return UpdateContext(**base)


def test_adapter_prompt_contains_entry_context_and_metrics():
    raw = json.dumps({"signal": "UPDATE", "confidence": "HIGH", "stop_loss": 68000.0,
                      "reasoning": "tighten"})
    engine = _FakeEngine(raw)
    signal = asyncio.run(
        PairAnalyzer(engine=engine, symbol=SYMBOL, timeframe="1h").analyze(
            candle_open_time="2026-10-30T10:00:00Z", update_context=_ctx()
        )
    )
    assert signal.action == "UPDATE"
    prompt = engine.calls[0]["position_context"]
    assert "ENTRY CONTEXT" in prompt
    assert "POSITION METRICS" in prompt
    assert "POSITION REGIME CHECK" in prompt
    assert "setup=trend_pullback" in prompt
    assert "invalidation=Close below the pullback low." in prompt
    # R is based on entry_sl=66500, not the current 68000 stop.
    assert "R basis=entry_sl=66500" in prompt
    assert "pnl_R=1.50" in prompt
    # The forming candle's 99999 high is excluded from MFE.
    assert "mfe_R=2.00" in prompt


def test_adapter_prompt_degrades_without_engine_context():
    raw = json.dumps({"signal": "UPDATE", "confidence": "HIGH", "stop_loss": 68000.0,
                      "reasoning": "tighten"})
    engine = _FakeEngine(raw, with_context=False)
    signal = asyncio.run(
        PairAnalyzer(engine=engine, symbol=SYMBOL, timeframe="1h").analyze(
            candle_open_time="2026-10-30T10:00:00Z", update_context=_ctx()
        )
    )
    assert signal.action == "UPDATE"
    prompt = engine.calls[0]["position_context"]
    assert "ENTRY CONTEXT" in prompt
    assert "POSITION METRICS" in prompt
    assert "atr=unknown" in prompt


def test_adapter_prompt_falls_back_when_entry_signal_is_absent():
    raw = json.dumps({"signal": "UPDATE", "confidence": "HIGH", "stop_loss": 68000.0,
                      "reasoning": "tighten"})
    engine = _FakeEngine(raw)
    signal = asyncio.run(
        PairAnalyzer(engine=engine, symbol=SYMBOL, timeframe="1h").analyze(
            candle_open_time="2026-10-30T10:00:00Z",
            update_context=_ctx(entry_context=None, entry_sl=None),
        )
    )
    assert signal.action == "UPDATE"
    prompt = engine.calls[0]["position_context"]
    assert "unavailable: no entry signal" in prompt
    assert "R basis=entry_sl=unknown" in prompt


def test_stateless_entry_prompt_is_unchanged():
    engine = _FakeEngine(json.dumps({"signal": "HOLD", "confidence": "LOW",
                                     "reasoning": "wait"}))
    asyncio.run(
        PairAnalyzer(engine=engine, symbol=SYMBOL, timeframe="1h").analyze(
            candle_open_time="2026-10-30T10:00:00Z"
        )
    )
    assert engine.calls[0]["position_context"] == ""


# ------------------------------------------------------- stale-context regression
class _Collector:
    """Simulates a collector that only knows the fresh (current) candles."""

    def __init__(self, context):
        self.context = context
        self.called = 0

    async def collect_data(self, context):
        self.called += 1
        context.ohlcv_candles = list(CANDLES + [FUTURE_CANDLE])
        context.timestamps = list(TIMESTAMPS + [FUTURE_TIMESTAMP])
        context.current_price = 69000.0


class _PromptBuilder:
    """Minimal prompt builder exposing the provider installed by the adapter."""

    regime_snapshot_provider = None


class _RefreshEngine(_FakeEngine):
    """Engine whose context starts one candle old and is refreshed on collect."""

    def __init__(self, raw):
        super().__init__(raw)
        self.context = self._Ctx()
        # Start the cycle one candle behind: only the 09:00 candle exists.
        self.context.ohlcv_candles = [CANDLES[0]]
        self.context.timestamps = [TIMESTAMPS[0]]
        self.context.technical_data = {}
        self.data_collector = _Collector(self.context)
        self.prompt_builder = _PromptBuilder()
        self.prompt = ""

    async def _calculate_technical_indicators(self):
        self.context.technical_data = {
            "atr": 300.0,
            "atr_percent": 0.43,
            "adx": 35.0,
            "plus_di": 30.0,
            "minus_di": 10.0,
            "choppiness": 35.0,
            "sma_20": 68000.0,
        }

    async def _process_long_term_data(self):
        return None

    async def analyze_market(self, **kwargs):
        self.calls.append(kwargs)
        provider = self.prompt_builder.regime_snapshot_provider
        snapshot_block = provider(self.context) if provider else ""
        self.prompt = kwargs.get("position_context", "")
        if snapshot_block:
            self.prompt = f"{self.prompt}\n\n{snapshot_block}"
        return {"raw_response": self.last_llm_response, "recommendation": "UPDATE"}


def test_metrics_use_refreshed_closed_candle_not_previous_cycle():
    raw = json.dumps({"signal": "UPDATE", "confidence": "HIGH", "stop_loss": 68000.0,
                      "reasoning": "tighten"})
    engine = _RefreshEngine(raw)
    signal = asyncio.run(
        PairAnalyzer(engine=engine, symbol=SYMBOL, timeframe="1h").analyze(
            candle_open_time="2026-10-30T10:00:00Z", update_context=_ctx()
        )
    )
    assert signal.action == "UPDATE"
    assert engine.data_collector.called == 1
    prompt = engine.calls[0]["position_context"]
    # The 10:00 candle (high 69500) must contribute; the stale 09:00-only
    # context would have capped MFE at 68500 (mfe_R=1.00).
    assert "mfe_R=2.00" in prompt
    assert "mae_R=0.50" in prompt
    assert "atr=300" in prompt
    assert "atr_pct=0.43" in prompt


def test_current_regime_is_rendered_only_by_the_snapshot_block():
    raw = json.dumps({"signal": "UPDATE", "confidence": "HIGH", "stop_loss": 68000.0,
                      "reasoning": "tighten"})
    engine = _RefreshEngine(raw)
    signal = asyncio.run(
        PairAnalyzer(engine=engine, symbol=SYMBOL, timeframe="1h").analyze(
            candle_open_time="2026-10-30T10:00:00Z", update_context=_ctx()
        )
    )
    assert signal.action == "UPDATE"
    prompt = engine.prompt
    assert prompt.count("## REGIME SNAPSHOT") == 1
    assert prompt.count("allowed_setups=") == 1
    assert "current_regime" not in prompt
    assert "regime_match=yes" in prompt
    # ``entry_regime=`` is not a word-boundary ``regime=``: the only current
    # regime label is the one in the deterministic snapshot block.
    assert len(re.findall(r"\bregime=", prompt)) == 1


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
        pairs=[SYMBOL], timeframe="1h", analyzer_factory=lambda _s: analyzer,
        signals_dir=str(tmp_path / "signals"), position_aware=True,
        positions_file=str(tmp_path / "positions.json"),
        updates_dir=str(tmp_path / "updates"), dedupe=True,
    )


def _write_entry(tmp_path, **overrides):
    storage = SignalStorage(str(tmp_path / "signals" / "BTC_USDT_1h.jsonl"))
    storage.append(Signal(**entry_kwargs(**overrides)))
    return storage


def test_service_joins_entry_signal_by_enter_tag(tmp_path):
    _write_entry(tmp_path)
    analyzer = _FakeAnalyzer(Signal(**update_kwargs()))
    service = _service(tmp_path, analyzer)
    snapshot = PositionsSnapshot(updated_at=utc_now_iso(), positions=[
        PositionSnapshot(
            position_id="42", symbol=SYMBOL, side="long", entry_price=67500.0,
            stop_loss=68000.0, take_profit=72000.0, current_price=69000.0,
            enter_tag="entry-1", open_date="2026-10-30T08:30:00Z",
        )
    ])
    write_positions_atomic(service.positions_file, snapshot)
    asyncio.run(service.run_cycle(datetime(2026, 10, 30, 10, 0, tzinfo=timezone.utc)))

    ctx = analyzer.calls[0]["update_context"]
    assert ctx.entry_sl == 66500.0
    assert ctx.entry_context.setup == "trend_pullback"
    assert ctx.entry_context.regime == "trend_up"
    assert ctx.entry_context.htf_alignment == "aligned"
    assert ctx.entry_context.confidence == 0.72
    assert ctx.entry_context.created_at == "2026-10-30T09:00:02Z"
    assert ctx.position_opened_at == "2026-10-30T08:30:00Z"


def test_service_position_without_enter_tag_has_no_entry_context(tmp_path):
    analyzer = _FakeAnalyzer(Signal(**update_kwargs()))
    service = _service(tmp_path, analyzer)
    snapshot = PositionsSnapshot(updated_at=utc_now_iso(), positions=[
        PositionSnapshot(
            position_id="42", symbol=SYMBOL, side="long", entry_price=67500.0,
            stop_loss=66500.0, take_profit=72000.0, current_price=69000.0,
        )
    ])
    write_positions_atomic(service.positions_file, snapshot)
    asyncio.run(service.run_cycle(datetime(2026, 10, 30, 10, 0, tzinfo=timezone.utc)))

    ctx = analyzer.calls[0]["update_context"]
    assert ctx.entry_context is None
    assert ctx.entry_sl is None
    assert ctx.stop_loss == 66500.0


def test_service_enter_tag_without_stored_signal_falls_back(tmp_path):
    analyzer = _FakeAnalyzer(Signal(**update_kwargs()))
    service = _service(tmp_path, analyzer)
    snapshot = PositionsSnapshot(updated_at=utc_now_iso(), positions=[
        PositionSnapshot(
            position_id="42", symbol=SYMBOL, side="long", entry_price=67500.0,
            stop_loss=66500.0, take_profit=72000.0, current_price=69000.0,
            enter_tag="missing-entry",
        )
    ])
    write_positions_atomic(service.positions_file, snapshot)
    asyncio.run(service.run_cycle(datetime(2026, 10, 30, 10, 0, tzinfo=timezone.utc)))

    ctx = analyzer.calls[0]["update_context"]
    assert ctx.entry_context is None
    assert ctx.entry_sl is None


def test_service_joins_spot_and_futures_positions_independently(tmp_path):
    _write_entry(tmp_path, decision_id="spot-entry", setup="trend_pullback")
    storage = SignalStorage(str(tmp_path / "signals" / "BTC_USDT_1h.jsonl"))
    storage.append(Signal(**entry_kwargs(
        decision_id="fut-entry", action="SELL", sl=79000.0, tp=74000.0,
        setup="rally_failure", regime="trend_down",
    )), dedupe=False)
    analyzer = _FakeAnalyzer(Signal(**update_kwargs()))
    service = _service(tmp_path, analyzer)
    snapshot = PositionsSnapshot(updated_at=utc_now_iso(), positions=[
        PositionSnapshot(
            position_id="spot:42", symbol=SYMBOL, side="long", entry_price=67500.0,
            stop_loss=68000.0, take_profit=72000.0, current_price=69000.0,
            enter_tag="spot-entry", market="spot",
        ),
        PositionSnapshot(
            position_id="futures:42", symbol="BTC/USDT:USDT", side="short",
            entry_price=78000.0, stop_loss=78500.0, take_profit=74000.0,
            current_price=76000.0, enter_tag="fut-entry", market="futures",
        ),
    ])
    write_positions_atomic(service.positions_file, snapshot)
    asyncio.run(service.run_cycle(datetime(2026, 10, 30, 10, 0, tzinfo=timezone.utc)))

    assert len(analyzer.calls) == 2
    by_side = {call["update_context"].position_side: call["update_context"]
               for call in analyzer.calls}
    assert by_side["long"].entry_context.setup == "trend_pullback"
    assert by_side["long"].entry_sl == 66500.0
    assert by_side["short"].entry_context.setup == "rally_failure"
    assert by_side["short"].entry_context.regime == "trend_down"
    assert by_side["short"].entry_sl == 79000.0


def test_entry_signal_without_optional_taxonomy_still_joins(tmp_path):
    _write_entry(tmp_path, setup=None, regime=None, htf_alignment=None)
    analyzer = _FakeAnalyzer(Signal(**update_kwargs()))
    service = _service(tmp_path, analyzer)
    snapshot = PositionsSnapshot(updated_at=utc_now_iso(), positions=[
        PositionSnapshot(
            position_id="42", symbol=SYMBOL, side="long", entry_price=67500.0,
            stop_loss=68000.0, take_profit=72000.0, current_price=69000.0,
            enter_tag="entry-1",
        )
    ])
    write_positions_atomic(service.positions_file, snapshot)
    asyncio.run(service.run_cycle(datetime(2026, 10, 30, 10, 0, tzinfo=timezone.utc)))

    ctx = analyzer.calls[0]["update_context"]
    assert ctx.entry_context is not None
    assert ctx.entry_context.setup is None
    assert ctx.entry_context.invalidation is None
    assert ctx.entry_sl == 66500.0
