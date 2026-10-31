"""Tests for the deterministic P0 regime snapshot and its JEV/prompt wiring.

No network, no model and no exchange access: every input is a plain dict or a
synthetic close series.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from src.analyzer.analysis_context import AnalysisContext
from src.analyzer.prompts.prompt_builder import PromptBuilder
from src.signal_service.jev_client import JevResponse
from src.signal_service.analyzer_adapter import PairAnalyzer
from src.signal_service.jev_gate import JevGate, build_state
from src.signal_service.regime_snapshot import (
    build_regime_snapshot,
    format_regime_snapshot_block,
)

SYMBOL = "ARB/USDT"
BASE = datetime(2026, 1, 1, tzinfo=timezone.utc)


def _trend_up_inputs() -> Dict[str, Any]:
    return {
        "technical_data": {
            "adx": 32.0,
            "plus_di": 31.0,
            "minus_di": 11.0,
            "choppiness": 30.0,
            "atr": 2.0,
            "atr_percent": 2.0,
            "sma_20": 100.0,
            "supertrend_direction": 1,
        },
        "long_term_data": {
            "daily_adx": 30.0,
            "daily_plus_di": 28.0,
            "daily_minus_di": 12.0,
            "sma_values": {50: 90.0, 200: 80.0},
        },
        "weekly_macro": {
            "weekly_macro_trend": {
                "trend_direction": "Bullish",
                "price_above_200w_sma": True,
            }
        },
        "closes_1h": [99.0, 100.0, 101.0, 103.0, 105.0],
    }


def _trend_down_inputs() -> Dict[str, Any]:
    return {
        "technical_data": {
            "adx": 32.0,
            "plus_di": 11.0,
            "minus_di": 31.0,
            "choppiness": 30.0,
            "atr": 2.0,
            "atr_percent": 2.0,
            "sma_20": 100.0,
            "supertrend_direction": -1,
        },
        "long_term_data": {
            "daily_adx": 30.0,
            "daily_plus_di": 12.0,
            "daily_minus_di": 28.0,
            "sma_values": {50: 90.0, 200: 100.0},
        },
        "weekly_macro": {
            "weekly_macro_trend": {"trend_direction": "Bearish"}
        },
        "closes_1h": [105.0, 103.0, 101.0, 99.0, 95.0],
    }


def _range_inputs() -> Dict[str, Any]:
    return {
        "technical_data": {
            "adx": 14.0,
            "plus_di": 21.0,
            "minus_di": 20.0,
            "choppiness": 70.0,
            "atr": 1.5,
            "atr_percent": 1.5,
            "sma_20": 100.0,
            "supertrend_direction": 0,
        },
        "closes_1h": [100.0, 100.5, 99.5, 100.2, 100.0],
    }


# ------------------------------------------------------------------ snapshots
def test_trend_up_snapshot():
    snapshot = build_regime_snapshot(**_trend_up_inputs(), timeframe="1h")

    assert snapshot["regime"] == "trend_up"
    assert snapshot["confidence"] >= 0.8
    assert snapshot["htf_alignment"] == "aligned"
    assert snapshot["extension_atr"] == pytest.approx(2.5)
    assert snapshot["allowed_setups"] == [
        "trend_pullback",
        "breakout_retest",
        "momentum_continuation",
    ]


def test_trend_down_snapshot():
    snapshot = build_regime_snapshot(**_trend_down_inputs(), timeframe="1h")

    assert snapshot["regime"] == "trend_down"
    assert snapshot["confidence"] >= 0.8
    assert snapshot["htf_alignment"] == "aligned"
    assert snapshot["extension_atr"] == pytest.approx(-2.5)
    assert snapshot["allowed_setups"] == [
        "rally_failure",
        "breakdown_retest",
        "momentum_short",
    ]


def test_range_snapshot():
    snapshot = build_regime_snapshot(**_range_inputs(), timeframe="1h")

    assert snapshot["regime"] == "range"
    assert snapshot["htf_alignment"] == "neutral"
    assert snapshot["allowed_setups"] == [
        "range_fade_extreme",
        "range_mean_reversion",
    ]


def test_htf_counter_trend_is_not_aligned():
    inputs = _trend_up_inputs()
    inputs["weekly_macro"] = {"weekly_macro_trend": {"trend_direction": "Bearish"}}
    inputs["long_term_data"] = {
        "daily_adx": 30.0,
        "daily_plus_di": 12.0,
        "daily_minus_di": 28.0,
    }

    snapshot = build_regime_snapshot(**inputs, timeframe="1h")

    assert snapshot["regime"] == "trend_up"
    assert snapshot["htf_alignment"] == "counter"


def test_relative_strength_vs_btc_7d_and_30d():
    pair_daily = [100.0 + index for index in range(40)]
    btc_daily = [100.0 + index * 0.5 for index in range(40)]

    snapshot = build_regime_snapshot(
        technical_data={"adx": 14.0, "choppiness": 70.0},
        closes_1d=pair_daily,
        btc_closes_1d=btc_daily,
    )

    expected_7d = ((pair_daily[-1] / pair_daily[-8]) - (btc_daily[-1] / btc_daily[-8])) * 100
    expected_30d = ((pair_daily[-1] / pair_daily[-31]) - (btc_daily[-1] / btc_daily[-31])) * 100
    assert snapshot["rs_btc_7d"] == pytest.approx(expected_7d, abs=0.02)
    assert snapshot["rs_btc_30d"] == pytest.approx(expected_30d, abs=0.02)


def test_missing_inputs_degrade_to_unclear():
    snapshot = build_regime_snapshot()
    assert snapshot["regime"] == "unclear"
    assert snapshot["no_trade"] is True
    assert snapshot["allowed_setups"] == []


def test_high_vol_snapshot_is_no_trade():
    snapshot = build_regime_snapshot(
        technical_data={"adx": 15.0, "choppiness": 45.0, "atr_percent": 5.0},
        closes_1h=[100.0, 101.0, 99.0, 100.0, 102.0],
    )

    assert snapshot["regime"] == "high_vol"
    assert snapshot["no_trade"] is True
    assert snapshot["allowed_setups"] == []


def test_low_vol_snapshot_is_no_trade():
    snapshot = build_regime_snapshot(
        technical_data={
            "adx": 18.0,
            "choppiness": 60.0,
            "atr_percent": 0.5,
            "sma_20": 100.0,
        },
        closes_1h=[100.0, 100.1, 100.0, 100.1, 100.0],
        atr_pct_median_90d=2.0,
    )

    assert snapshot["regime"] == "low_vol"
    assert snapshot["no_trade"] is True
    assert snapshot["allowed_setups"] == []


# -------------------------------------------------------------- prompt block
def _prompt_builder() -> PromptBuilder:
    builder = PromptBuilder(
        timeframe="1h",
        format_utils=MagicMock(),
        template_manager=MagicMock(),
        overview_formatter=MagicMock(),
        long_term_formatter=MagicMock(),
        technical_formatter=MagicMock(),
        market_formatter=MagicMock(),
    )
    for method in (
        "build_trading_context",
        "build_sentiment_section",
        "build_coin_details_section",
        "build_market_data_section",
        "build_market_period_metrics_section",
    ):
        setattr(builder, method, MagicMock(return_value=""))
    builder.technical_analysis_formatter.format_technical_analysis.return_value = ""
    return builder


def test_regime_snapshot_block_is_compact_and_lists_setups():
    snapshot = build_regime_snapshot(**_trend_down_inputs(), timeframe="1h")
    block = format_regime_snapshot_block(snapshot)

    assert block.startswith("## REGIME SNAPSHOT (deterministic)")
    assert "regime=trend_down" in block
    assert "htf=aligned" in block
    assert "tradeable=yes" in block
    assert "allowed_setups=rally_failure,breakdown_retest,momentum_short" in block
    assert len(block.splitlines()) == 4


def test_prompt_builder_injects_regime_block_only_when_configured():
    snapshot = build_regime_snapshot(**_trend_up_inputs(), timeframe="1h")
    builder = _prompt_builder()

    without = builder.build_prompt(AnalysisContext(symbol=SYMBOL))
    assert "## REGIME SNAPSHOT" not in without

    builder.regime_snapshot_provider = lambda _context: format_regime_snapshot_block(snapshot)
    with_block = builder.build_prompt(AnalysisContext(symbol=SYMBOL))

    assert "## REGIME SNAPSHOT (deterministic)" in with_block
    assert "regime=trend_up" in with_block


# ------------------------------------------------------------------- JEV state
def test_build_state_carries_regime_fields_only_when_supplied():
    snapshot = build_regime_snapshot(**_trend_down_inputs(), timeframe="1h")
    candles = [[100.0, 102.0, 99.0, 100.0 + index, 10.0] for index in range(6)]
    timestamps = [BASE + timedelta(hours=index) for index in range(6)]

    plain = build_state(
        SYMBOL, "1h", candles, timestamps, as_of=BASE + timedelta(hours=6)
    )
    enriched = build_state(
        SYMBOL,
        "1h",
        candles,
        timestamps,
        as_of=BASE + timedelta(hours=6),
        regime_snapshot=snapshot,
    )

    assert "adx" not in plain
    assert enriched["adx"] == pytest.approx(32.0)
    assert enriched["choppiness"] == pytest.approx(30.0)
    assert enriched["htf_alignment"] == "aligned"
    assert enriched["extension_atr"] == pytest.approx(-2.5)
    assert enriched["deterministic_regime"] == "trend_down"
    assert enriched["rs_btc_7d"] is None
    assert enriched["rs_btc_30d"] is None


def _response(tradeable: float = 0.8, bias: str = "long") -> JevResponse:
    return JevResponse(
        model="jev-1.13",
        answers={
            "tradeability": {"type": "noul", "noul": tradeable},
            "bias": {"type": "choice", "choice": bias, "confidence": 0.8},
            "regime": {"type": "choice", "choice": "trend", "confidence": 0.7},
            "confidence": {
                "type": "score",
                "score": 3.0,
                "legend": {"0": "none", "1": "weak", "2": "fair", "3": "strong", "4": "very strong"},
            },
        },
        latency_ms=10,
    )


class _FakeClient:
    def __init__(self, response: Optional[JevResponse] = None) -> None:
        self.response = response or _response()
        self.calls: List[Dict[str, Any]] = []

    async def ask(self, state: Dict[str, Any], questions: Dict[str, Any]) -> JevResponse:
        self.calls.append(state)
        return self.response


@pytest.mark.asyncio
@pytest.mark.parametrize("regime", ["trend_down", "range", "unclear", "high_vol", "low_vol"])
async def test_short_is_accepted_on_short_bias_in_every_regime(regime):
    """Bias-only policy: regime fields never gate a short that says short."""
    client = _FakeClient(_response(tradeable=0.01, bias="short"))
    gate = JevGate(client=client, mode="post")
    state = {
        "symbol": SYMBOL,
        "deterministic_regime": regime,
        "setup": "rally_failure",
        "no_trade": regime in ("unclear", "high_vol", "low_vol"),
    }

    result = await gate.check_post(state, "SELL")

    assert result.allowed is True
    assert result.signal is not None and result.signal.bias == "short"


@pytest.mark.asyncio
@pytest.mark.parametrize("regime", ["trend_down", "range"])
@pytest.mark.parametrize("bias", ["long", "none"])
async def test_short_without_short_bias_is_vetoed_even_in_clear_regimes(regime, bias):
    gate = JevGate(client=_FakeClient(_response(tradeable=0.9, bias=bias)), mode="post")
    state = {"symbol": SYMBOL, "deterministic_regime": regime, "setup": "rally_failure"}

    result = await gate.check_post(state, "SELL")

    assert result.allowed is False
    assert "SELL requires short bias" in result.reason


@pytest.mark.asyncio
@pytest.mark.parametrize("bias", ["short", "long", "none"])
@pytest.mark.parametrize("regime", ["trend_down", "unclear"])
async def test_long_is_never_filtered_by_the_jev_bias(bias, regime):
    gate = JevGate(client=_FakeClient(_response(tradeable=0.01, bias=bias)), mode="post")
    state = {"symbol": SYMBOL, "deterministic_regime": regime, "setup": "momentum_continuation"}

    result = await gate.check_post(state, "BUY")

    assert result.allowed is True


@pytest.mark.asyncio
async def test_noul_is_logged_not_enforced(caplog):
    gate = JevGate(client=_FakeClient(_response(tradeable=0.0, bias="short")), mode="post")

    with caplog.at_level("INFO"):
        result = await gate.check_post({"symbol": SYMBOL, "deterministic_regime": "trend_down"}, "SELL")

    assert result.allowed is True
    assert "noul=0.000" in caplog.text
    assert "advisory only" in caplog.text

def test_adapter_installs_prompt_provider_using_the_prompt_context():
    """The prompt block must use the engine's fresh context, not a stale copy."""
    builder = SimpleNamespace(regime_snapshot_provider=None)
    engine = SimpleNamespace(prompt_builder=builder)
    analyzer = PairAnalyzer(engine=engine, symbol=SYMBOL, timeframe="1h")
    context = SimpleNamespace(
        technical_data=_trend_up_inputs()["technical_data"],
        long_term_data=_trend_up_inputs()["long_term_data"],
        weekly_macro_indicators=_trend_up_inputs()["weekly_macro"],
        ohlcv_candles=[[0, 99.0, 106.0, 98.0, 105.0, 10.0]],
        weekly_ohlcv=None,
    )

    analyzer._install_regime_snapshot_provider()
    block = builder.regime_snapshot_provider(context)

    assert block.startswith("## REGIME SNAPSHOT (deterministic)")
    assert "regime=trend_up" in block

def test_adapter_snapshot_drops_the_forming_candle():
    """Only candles closed at ``as_of`` may enter the regime snapshot."""
    builder = SimpleNamespace(regime_snapshot_provider=None)
    engine = SimpleNamespace(prompt_builder=builder)
    analyzer = PairAnalyzer(engine=engine, symbol=SYMBOL, timeframe="1h")
    context = SimpleNamespace(
        technical_data=_trend_up_inputs()["technical_data"],
        long_term_data=_trend_up_inputs()["long_term_data"],
        weekly_macro_indicators=_trend_up_inputs()["weekly_macro"],
        ohlcv_candles=[
            [0, 98.0, 101.0, 97.0, 100.0, 10.0],
            [3_600_000, 100.0, 106.0, 99.0, 105.0, 10.0],
            [7_200_000, 105.0, 131.0, 104.0, 130.0, 10.0],
        ],
        timestamps=[BASE, BASE + timedelta(hours=1), BASE + timedelta(hours=2)],
        weekly_ohlcv=None,
    )

    snapshot = analyzer._regime_snapshot_from_context(context, BASE + timedelta(hours=2))

    assert snapshot is not None
    assert snapshot["last_close"] == pytest.approx(105.0)
    assert snapshot["extension_atr"] == pytest.approx(2.5)

class _RefreshingEngine:
    """Engine double that mimics collect -> truncate -> recompute ordering."""

    def __init__(self) -> None:
        self._raw_candles = [
            [0, 98.0, 101.0, 97.0, 100.0, 10.0],
            [3_600_000, 100.0, 106.0, 99.0, 105.0, 10.0],
            [7_200_000, 105.0, 131.0, 104.0, 130.0, 10.0],
        ]
        self._raw_timestamps = [
            BASE,
            BASE + timedelta(hours=1),
            BASE + timedelta(hours=2),
        ]
        self.context = SimpleNamespace(
            ohlcv_candles=list(self._raw_candles),
            timestamps=list(self._raw_timestamps),
            current_price=130.0,
            technical_data={},
            technical_history={},
            long_term_data={"data": []},
            weekly_macro_indicators=None,
            weekly_ohlcv=None,
        )
        self.data_collector = self
        self.collect_args: List[Any] = []
        self.calculate_calls = 0
        self.process_calls = 0
        self.last_llm_response = json.dumps(
            {
                "signal": "SELL",
                "confidence": "HIGH",
                "stop_loss": 112.0,
                "take_profit": 90.0,
                "position_size": 0.05,
                "reasoning": "trend_down rally failure",
            }
        )

    async def collect_data(self, context: Any = None) -> Dict[str, Any]:
        """Refetch raw data and drop the computed long-term fields, as the real collector does."""
        target = context or self.context
        self.collect_args.append(context)
        target.ohlcv_candles = list(self._raw_candles)
        target.timestamps = list(self._raw_timestamps)
        target.current_price = 130.0
        target.long_term_data = {"data": [[0, 90, 91, 89, 90, 1], [172_800_000, 90, 91, 89, 90, 1]]}
        target.weekly_ohlcv = [[0, 80, 81, 79, 80, 1]]
        return {"success": True}

    async def _calculate_technical_indicators(self) -> None:
        self.calculate_calls += 1
        self.context.technical_data = {
            "adx": 30.0,
            "plus_di": 10.0,
            "minus_di": 30.0,
            "choppiness": 40.0,
            "atr": 2.0,
            "atr_percent": 2.0,
            "sma_20": 110.0,
            "supertrend_direction": -1,
        }
        self.context.technical_history = {"atr_percent": [1.8, 2.0, 2.2]}

    async def _process_long_term_data(self) -> None:
        self.process_calls += 1
        self.context.long_term_data.update(
            {
                "daily_adx": 30.0,
                "daily_plus_di": 10.0,
                "daily_minus_di": 28.0,
                "sma_values": {50: 90.0, 200: 100.0},
                "macro_trend": {"trend_direction": "Bearish"},
            }
        )

    async def analyze_market(self, **_kwargs: Any) -> Dict[str, Any]:
        await self.data_collector.collect_data(self.context)
        await self._calculate_technical_indicators()
        await self._process_long_term_data()
        return {
            "recommendation": "SELL",
            "confidence": "HIGH",
            "stop_loss": 112.0,
            "take_profit": 90.0,
            "position_size": 0.05,
            "reasoning": "trend_down rally failure",
            "model": "deepseek-v4.1-flash",
            "provider": "opencode-go",
        }


@pytest.mark.asyncio
async def test_refresh_recomputes_indicators_and_restores_daily_fields():
    engine = _RefreshingEngine()
    analyzer = PairAnalyzer(engine=engine, symbol=SYMBOL, timeframe="1h")
    as_of = BASE + timedelta(hours=2)

    await analyzer._refresh_context(as_of)

    assert engine.collect_args == [engine.context]
    assert engine.calculate_calls == 1
    assert engine.process_calls == 1
    assert len(engine.context.ohlcv_candles) == 2
    assert engine.context.technical_data["adx"] == pytest.approx(30.0)
    assert engine.context.long_term_data["daily_adx"] == pytest.approx(30.0)

    state = analyzer._jev_state(as_of)

    assert state is not None
    assert state["last_close"] == pytest.approx(105.0)
    assert state["adx"] == pytest.approx(30.0)
    assert state["deterministic_regime"] == "trend_down"


@pytest.mark.asyncio
async def test_refresh_replaces_stale_indicators_with_fresh_values():
    """A new cycle must never feed the previous cycle's ADX/ATR/daily trend."""
    engine = _RefreshingEngine()
    engine.context.technical_data = {
        "adx": 99.0,
        "plus_di": 50.0,
        "minus_di": 1.0,
        "atr_percent": 99.0,
        "sma_20": 1.0,
    }
    engine.context.technical_history = {"atr_percent": [99.0, 99.0]}
    engine.context.long_term_data = {
        "data": [[0, 90.0, 91.0, 89.0, 90.0, 1.0]],
        "daily_adx": 99.0,
        "macro_trend": {"trend_direction": "Bullish"},
    }
    analyzer = PairAnalyzer(engine=engine, symbol=SYMBOL, timeframe="1h")
    as_of = BASE + timedelta(hours=2)

    await analyzer._refresh_context(as_of)
    state = analyzer._jev_state(as_of)
    snapshot = analyzer._regime_snapshot_from_context(engine.context, as_of)

    assert state is not None
    assert state["adx"] == pytest.approx(30.0)
    assert state["deterministic_regime"] == "trend_down"
    assert snapshot is not None and snapshot["daily_trend"] == "bearish"


@pytest.mark.asyncio
async def test_missing_recompute_degrades_to_unclear_instead_of_stale():
    """Without a recompute hook the derived fields are cleared, not reused."""

    class _NoRecomputeEngine:
        def __init__(self) -> None:
            self.context = SimpleNamespace(
                ohlcv_candles=[
                    [0, 98.0, 101.0, 97.0, 100.0, 10.0],
                    [3_600_000, 100.0, 106.0, 99.0, 105.0, 10.0],
                    [7_200_000, 105.0, 131.0, 104.0, 130.0, 10.0],
                ],
                timestamps=[BASE, BASE + timedelta(hours=1), BASE + timedelta(hours=2)],
                current_price=130.0,
                technical_data={"adx": 99.0, "atr_percent": 99.0, "sma_20": 1.0},
                technical_history={"atr_percent": [99.0]},
                long_term_data={
                    "data": [[0, 90.0, 91.0, 89.0, 90.0, 1.0]],
                    "daily_adx": 99.0,
                    "macro_trend": {"trend_direction": "Bullish"},
                },
                weekly_macro_indicators={
                    "weekly_macro_trend": {"trend_direction": "Bullish"}
                },
                weekly_ohlcv=[[0, 80.0, 81.0, 79.0, 80.0, 1.0]],
            )
            self.data_collector = self

        async def collect_data(self, context: Any = None) -> Dict[str, Any]:
            target = context or self.context
            target.ohlcv_candles = list(self.context.ohlcv_candles)
            target.timestamps = list(self.context.timestamps)
            target.long_term_data = {"data": [[0, 90.0, 91.0, 89.0, 90.0, 1.0]]}
            target.weekly_ohlcv = [[0, 80.0, 81.0, 79.0, 80.0, 1.0]]
            return {"success": True}

    engine = _NoRecomputeEngine()
    analyzer = PairAnalyzer(engine=engine, symbol=SYMBOL, timeframe="1h")
    as_of = BASE + timedelta(hours=2)

    await analyzer._refresh_context(as_of)
    state = analyzer._jev_state(as_of)

    assert state is not None
    assert state["last_close"] == pytest.approx(105.0)
    assert state["adx"] is None
    assert state["deterministic_regime"] == "unclear"
    assert state["no_trade"] is True


@pytest.mark.asyncio
async def test_post_gate_state_matches_the_analysed_candle():
    """Pre and post gates must see the same closed candle and indicators."""
    engine = _RefreshingEngine()
    client = _FakeClient(_response(tradeable=0.8, bias="short"))
    analyzer = PairAnalyzer(
        engine=engine,
        symbol=SYMBOL,
        timeframe="1h",
        jev_gate=JevGate(client=client, mode="pre_post"),
    )

    signal = await analyzer.analyze(candle_open_time="2026-01-01T01:00:00Z")

    assert signal.action == "SELL"
    assert len(client.calls) == 2
    pre_state, post_state = client.calls
    assert pre_state["last_close"] == post_state["last_close"] == pytest.approx(105.0)
    assert pre_state["adx"] == post_state["adx"] == pytest.approx(30.0)
    assert pre_state["candles_used"] == post_state["candles_used"] == 2
    assert pre_state["deterministic_regime"] == post_state["deterministic_regime"] == "trend_down"


def test_adapter_treats_raw_long_term_rows_as_missing():
    """A raw daily payload must not break the snapshot (daily trend stays None)."""
    engine = SimpleNamespace(prompt_builder=SimpleNamespace(regime_snapshot_provider=None))
    analyzer = PairAnalyzer(engine=engine, symbol=SYMBOL, timeframe="1h")
    context = SimpleNamespace(
        technical_data=_trend_up_inputs()["technical_data"],
        long_term_data=[[0, 90.0, 91.0, 89.0, 90.0, 1.0]],
        weekly_macro_indicators=None,
        ohlcv_candles=[[0, 99.0, 106.0, 98.0, 105.0, 10.0]],
        weekly_ohlcv=None,
    )

    snapshot = analyzer._regime_snapshot_from_context(context)

    assert snapshot is not None
    assert snapshot["regime"] == "trend_up"
    assert snapshot["daily_trend"] is None
    assert snapshot["htf_alignment"] == "neutral"
