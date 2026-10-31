"""Tests for the deterministic P0 regime snapshot and its JEV/prompt wiring.

No network, no model and no exchange access: every input is a plain dict or a
synthetic close series.
"""

from __future__ import annotations

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
async def test_range_regime_lowers_the_pre_gate_threshold():
    client = _FakeClient(_response(tradeable=0.45, bias="long"))
    gate = JevGate(client=client, mode="pre")

    directional = await gate.check_pre({"symbol": SYMBOL, "deterministic_regime": "trend_down"})
    in_range = await gate.check_pre({"symbol": SYMBOL, "deterministic_regime": "range"})

    assert directional.allowed is False
    assert in_range.allowed is True
    assert in_range.signal is not None and in_range.signal.tradeability is True


@pytest.mark.asyncio
@pytest.mark.parametrize("symbol", ["ARB/USDT", "ETH/USDT"])
@pytest.mark.parametrize(
    ("action", "bias"),
    [("SELL", "long"), ("BUY", "short")],
)
async def test_clear_regime_does_not_veto_lagging_bias_conflict(symbol, action, bias):
    gate = JevGate(client=_FakeClient(_response(tradeable=0.9, bias=bias)), mode="post")
    state = {"symbol": symbol, "deterministic_regime": "trend_down", "setup": "rally_failure"}

    result = await gate.check_post(state, action)

    assert result.allowed is True
    assert "conflicts" not in result.reason


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("setup", "action", "bias"),
    [("momentum_short", "SELL", "long"), ("momentum_continuation", "BUY", "short")],
)
async def test_momentum_setup_keeps_bias_conflict_veto(setup, action, bias):
    gate = JevGate(client=_FakeClient(_response(tradeable=0.9, bias=bias)), mode="post")
    state = {"symbol": SYMBOL, "deterministic_regime": "trend_down", "setup": setup}

    result = await gate.check_post(state, action)

    assert result.allowed is False
    assert "conflicts" in result.reason


@pytest.mark.asyncio
async def test_unclear_regime_keeps_bias_conflict_veto():
    gate = JevGate(client=_FakeClient(_response(tradeable=0.9, bias="long")), mode="post")

    result = await gate.check_post({"symbol": SYMBOL, "deterministic_regime": "unclear"}, "SELL")

    assert result.allowed is False
    assert "conflicts" in result.reason

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
