"""Tests for the JEV gate, parser, state builder and adapter wiring.

No network is used: every JEV call goes through a fake client.
"""

from __future__ import annotations

import logging
import json
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import pytest

from src.signal_service.analyzer_adapter import PairAnalyzer
from src.signal_service.jev_client import JevError, JevRateLimitError, JevResponse
from src.signal_service.jev_gate import (
    ALLOWED_BIAS,
    JevGate,
    JevMode,
    build_questions,
    build_state,
    closed_candles,
    parse_jev_signal,
    parse_mode,
)

SYMBOL = "BTC/USDT"
BASE = datetime(2026, 1, 1, tzinfo=timezone.utc)


def _response(
    tradeable: float = 0.8,
    bias: str = "long",
    regime: str = "trend",
    score: Optional[float] = 3.0,
    model: str = "jev-1.13",
) -> JevResponse:
    answers: Dict[str, Dict[str, Any]] = {
        "tradeability": {"type": "noul", "noul": tradeable},
        "bias": {"type": "choice", "choice": bias, "confidence": 0.8},
        "regime": {"type": "choice", "choice": regime, "confidence": 0.7},
    }
    if score is not None:
        answers["confidence"] = {
            "type": "score",
            "score": score,
            "legend": {"0": "none", "1": "weak", "2": "fair", "3": "strong", "4": "very strong"},
        }
    return JevResponse(model=model, answers=answers, latency_ms=120)


class FakeJevClient:
    """Minimal JevClient surface used by the gate."""

    def __init__(self, response: Optional[JevResponse] = None, error: Optional[BaseException] = None) -> None:
        self.response = response or _response()
        self.error = error
        self.calls: List[Dict[str, Any]] = []

    async def ask(self, state: Dict[str, Any], questions: Dict[str, Dict[str, Any]]) -> JevResponse:
        self.calls.append(state)
        if self.error is not None:
            raise self.error
        return self.response


def _candles(count: int = 6) -> List[List[float]]:
    rows = []
    for index in range(count):
        close = 100.0 + index
        rows.append([close - 1, close + 1, close - 2, close, 10.0 + index])
    return rows


def _timestamps(count: int = 6) -> List[datetime]:
    return [BASE + timedelta(hours=index) for index in range(count)]


# --------------------------------------------------------------------- parser
def test_parse_noul_choice_score():
    signal = parse_jev_signal(_response(tradeable=0.82, bias="long", regime="high_vol", score=3.0))

    assert signal.model == "jev-1.13"
    assert signal.tradeability is True
    assert signal.tradeability_probability == pytest.approx(0.82)
    assert signal.bias == "long"
    assert signal.regime == "high_vol"
    assert signal.confidence == pytest.approx(0.75)
    assert signal.score_raw == pytest.approx(3.0)


def test_parse_decision_3_score_normalized_by_legend_length():
    """Decision 3.0 score is the expected level among 2-10 legend levels."""
    three = JevResponse(
        model="vllm-sr/d3-flash",
        answers={
            "confidence": {
                "type": "score",
                "score": 1.69,
                "probabilities": {"0": 0.05, "1": 0.21, "2": 0.74},
                "confidence": 0.55,
                "legend": {"0": "low", "1": "medium", "2": "high"},
            }
        },
        latency_ms=90,
    )
    signal = parse_jev_signal(three)
    assert signal.confidence == pytest.approx(0.845)
    assert signal.score_raw == pytest.approx(1.69)

    ten = JevResponse(
        model="vllm-sr/d3-flash",
        answers={
            "confidence": {
                "type": "score",
                "score": 9.0,
                "legend": {str(level): f"level {level}" for level in range(10)},
            }
        },
        latency_ms=90,
    )
    # Nine is the highest level: the normalized confidence saturates at 1.0.
    assert parse_jev_signal(ten).confidence == pytest.approx(1.0)


def test_parse_score_with_negative_legend_indices_uses_offset():
    """A legend with negative indices still maps its top level to 1.0."""
    response = JevResponse(
        model="vllm-sr/d3-flash",
        answers={
            "confidence": {
                "type": "score",
                "score": 1.0,
                "legend": {"-1": "bearish", "0": "neutral", "1": "bullish"},
            }
        },
        latency_ms=90,
    )
    signal = parse_jev_signal(response)
    assert signal.confidence == pytest.approx(1.0)
    assert signal.score_raw == pytest.approx(1.0)

    mid = JevResponse(
        model="vllm-sr/d3-flash",
        answers={
            "confidence": {
                "type": "score",
                "score": 0.0,
                "legend": {"-1": "bearish", "0": "neutral", "1": "bullish"},
            }
        },
        latency_ms=90,
    )
    assert parse_jev_signal(mid).confidence == pytest.approx(0.5)


def test_parse_tradeability_threshold_and_missing_fields():
    below = parse_jev_signal(_response(tradeable=0.2))
    assert below.tradeability is False

    empty = JevResponse(model="jev-1.13", answers={}, latency_ms=10)
    signal = parse_jev_signal(empty)
    assert signal.tradeability is None
    assert signal.bias is None
    assert signal.regime is None
    assert signal.confidence is None


def test_parse_rejects_unknown_choice_values(caplog):
    response = _response(bias="sideways", regime="unknown")
    with caplog.at_level(logging.WARNING, logger="src.signal_service.jev_gate"):
        signal = parse_jev_signal(response)

    assert signal.bias is None
    assert signal.regime is None
    assert "unexpected bias choice" in caplog.text
    assert "unexpected regime choice" in caplog.text


def test_jev_payload_matches_contract_keys():
    signal = parse_jev_signal(_response())
    payload = signal.to_payload()
    assert set(payload) <= {"model", "tradeability", "bias", "regime"}
    assert payload == {
        "model": "jev-1.13",
        "tradeability": True,
        "bias": "long",
        "regime": "trend",
    }


def test_build_questions_uses_typed_questions():
    questions = build_questions()
    assert set(questions) == {"tradeability", "bias", "regime", "confidence"}
    assert questions["tradeability"]["type"] == "noul"
    assert set(questions["tradeability"]["criteria"]) == {"true", "false"}
    assert questions["bias"]["type"] == "choice"
    assert set(questions["bias"]["criteria"]) == ALLOWED_BIAS
    assert questions["regime"]["type"] == "choice"
    assert questions["confidence"]["type"] == "score"
    assert 2 <= len(questions["confidence"]["criteria"]) <= 10


def test_parse_mode():
    assert parse_mode("off") is JevMode.OFF
    assert parse_mode("PRE_POST") is JevMode.PRE_POST
    with pytest.raises(ValueError):
        parse_mode("sometimes")


# ---------------------------------------------------------------- state builder
def test_state_has_no_look_ahead():
    candles = _candles()
    timestamps = _timestamps()
    # Reference time is the close of the 3rd candle (00:00..02:00).
    as_of = BASE + timedelta(hours=3)

    state = build_state(SYMBOL, "1h", candles, timestamps, as_of=as_of)

    assert state["candles_used"] == 3
    assert state["last_close"] == 102.0
    assert state["recent_closes"] == [100.0, 101.0, 102.0]
    assert state["last_candle_close_time"] == "2026-01-01T03:00:00Z"
    assert state["as_of"] == "2026-01-01T03:00:00Z"
    # No news and no position: v1 state is price/volume only.
    assert "news" not in state
    assert "position" not in state
    assert "position_context" not in state


def test_state_excludes_forming_candle():
    candles = _candles()
    timestamps = _timestamps()
    as_of = BASE + timedelta(hours=2, minutes=59)

    kept = closed_candles(candles, timestamps, "1h", as_of)

    assert len(kept) == 2
    assert all(open_at + timedelta(hours=1) <= as_of for _row, open_at in kept)


def test_state_requires_enough_closed_candles():
    with pytest.raises(ValueError):
        build_state(SYMBOL, "1h", _candles(2), _timestamps(2), as_of=BASE + timedelta(minutes=30))


def test_state_is_json_serializable():
    import json

    state = build_state(SYMBOL, "1h", _candles(), _timestamps(), as_of=BASE + timedelta(hours=6))
    assert json.loads(json.dumps(state))["symbol"] == SYMBOL


# ----------------------------------------------------------------------- gate
@pytest.mark.asyncio
async def test_off_mode_makes_no_call():
    client = FakeJevClient()
    gate = JevGate(client=client, mode="off")
    state = build_state(SYMBOL, "1h", _candles(), _timestamps(), as_of=BASE + timedelta(hours=6))

    pre = await gate.check_pre(state)
    post = await gate.check_post(state, "BUY")

    assert client.calls == []
    assert pre.allowed is True
    assert post.allowed is True
    assert pre.consulted is False


@pytest.mark.asyncio
async def test_pre_mode_blocks_untradeable_market():
    gate = JevGate(client=FakeJevClient(_response(tradeable=0.2)), mode="pre")
    state = {"symbol": SYMBOL}

    result = await gate.check_pre(state)

    assert result.allowed is False
    assert result.consulted is True
    assert "not tradeable" in result.reason
    assert result.signal is not None and result.signal.tradeability is False


@pytest.mark.asyncio
async def test_pre_mode_blocks_none_bias():
    gate = JevGate(client=FakeJevClient(_response(bias="none")), mode="pre")

    result = await gate.check_pre({"symbol": SYMBOL})

    assert result.allowed is False
    assert "bias none" in result.reason


@pytest.mark.asyncio
async def test_pre_mode_allows_tradeable_directional_market():
    gate = JevGate(client=FakeJevClient(_response(tradeable=0.9, bias="short")), mode="pre")

    result = await gate.check_pre({"symbol": SYMBOL})

    assert result.allowed is True
    assert result.signal is not None and result.signal.bias == "short"


@pytest.mark.asyncio
async def test_post_mode_never_blocks_hold_and_does_not_call_jev():
    client = FakeJevClient(_response(tradeable=0.0, bias="none"))
    gate = JevGate(client=client, mode="post")

    result = await gate.check_post({"symbol": SYMBOL}, "HOLD")

    assert result.allowed is True
    assert client.calls == []
    assert "HOLD exempt" in result.reason


@pytest.mark.asyncio
async def test_post_mode_vetoes_conflicting_bias():
    buy_gate = JevGate(client=FakeJevClient(_response(bias="short")), mode="post")
    sell_gate = JevGate(client=FakeJevClient(_response(bias="long")), mode="post")

    buy = await buy_gate.check_post({"symbol": SYMBOL}, "BUY")
    sell = await sell_gate.check_post({"symbol": SYMBOL}, "SELL")

    assert buy.allowed is False and "short bias conflicts with BUY" in buy.reason
    assert sell.allowed is False and "long bias conflicts with SELL" in sell.reason


@pytest.mark.asyncio
async def test_post_mode_vetoes_untradeable_market():
    gate = JevGate(client=FakeJevClient(_response(tradeable=0.1, bias="long")), mode="post")

    result = await gate.check_post({"symbol": SYMBOL}, "BUY")

    assert result.allowed is False
    assert "JEV veto: not tradeable" in result.reason


@pytest.mark.asyncio
async def test_post_mode_approves_aligned_decision():
    gate = JevGate(client=FakeJevClient(_response(tradeable=0.9, bias="long")), mode="post")

    result = await gate.check_post({"symbol": SYMBOL}, "BUY")

    assert result.allowed is True
    assert "approved" in result.reason


@pytest.mark.asyncio
@pytest.mark.parametrize("error", [
    JevError("boom"),
    JevRateLimitError("free usage limit reached", status=429),
])
async def test_gate_fails_open_on_jev_errors(error):
    gate = JevGate(client=FakeJevClient(error=error), mode="post")

    result = await gate.check_post({"symbol": SYMBOL}, "BUY")

    assert result.allowed is True
    assert result.error is not None
    assert "fail-open" in result.reason


class ExplodingJevClient:
    """Client whose transport raises an arbitrary (non-JevError) exception."""

    def __init__(self, error: BaseException) -> None:
        self.error = error

    async def ask(self, state: Dict[str, Any], questions: Dict[str, Dict[str, Any]]) -> JevResponse:
        raise self.error


class BadPayloadJevClient:
    """Client returning something that is not a JevResponse: parsing fails."""

    async def ask(self, state: Dict[str, Any], questions: Dict[str, Dict[str, Any]]) -> Any:
        return {"model": "jev-1.13", "answers": None}


@pytest.mark.asyncio
@pytest.mark.parametrize("error", [
    RuntimeError("transport bug"),
    TypeError("state is not JSON serializable"),
    ValueError("unexpected payload"),
])
async def test_gate_fails_open_on_non_jev_errors(error):
    """The fail-open contract covers every client failure, not only JevError."""
    gate = JevGate(client=ExplodingJevClient(error), mode="pre_post")

    pre = await gate.check_pre({"symbol": SYMBOL})
    post = await gate.check_post({"symbol": SYMBOL}, "BUY")

    for result in (pre, post):
        assert result.allowed is True
        assert result.consulted is True
        assert "fail-open" in result.reason
        assert result.error is not None
        assert type(error).__name__ in result.error
    assert type(error).__name__ in gate.last_error


@pytest.mark.asyncio
async def test_gate_fails_open_on_parser_error():
    """A response that cannot be parsed must fail open, not abort the analysis."""
    gate = JevGate(client=BadPayloadJevClient(), mode="post")

    result = await gate.check_post({"symbol": SYMBOL}, "BUY")

    assert result.allowed is True
    assert result.consulted is True
    assert "fail-open" in result.reason
    assert result.error is not None
    assert gate.last_error is not None


@pytest.mark.asyncio
async def test_gate_without_client_fails_open():
    gate = JevGate(client=None, mode="pre")
    result = await gate.check_pre({"symbol": SYMBOL})
    assert result.allowed is True
    assert result.consulted is False


# --------------------------------------------------------- adapter integration
class FakeContext:
    def __init__(self, candles: List[List[float]], timestamps: List[datetime], current_price: float) -> None:
        self.ohlcv_candles = candles
        self.timestamps = timestamps
        self.current_price = current_price


class FakeEngine:
    """Minimal AnalysisEngine surface used by PairAnalyzer."""

    def __init__(self, result: Dict[str, Any], candles: List[List[float]], timestamps: List[datetime]) -> None:
        self._result = result
        # The real engine always exposes the raw LLM text; emulate it so the
        # adapter takes the normal extraction path.
        self.last_llm_response = json.dumps({
            "signal": result.get("recommendation", "HOLD"),
            "confidence": result.get("confidence", "LOW"),
            "stop_loss": result.get("stop_loss"),
            "take_profit": result.get("take_profit"),
            "position_size": result.get("position_size"),
            "reasoning": result.get("reasoning", ""),
        })
        self.context = FakeContext(candles, timestamps, candles[-1][3])
        self.calls = 0

    async def analyze_market(self, **_kwargs: Any) -> Dict[str, Any]:
        self.calls += 1
        return dict(self._result)


def _engine_result(action: str = "BUY") -> Dict[str, Any]:
    return {
        "recommendation": action,
        "confidence": "HIGH",
        "stop_loss": 102.0,
        "take_profit": 114.0,
        "position_size": 0.05,
        "reasoning": "test reasoning",
        "model": "deepseek-v4.1-flash",
        "provider": "opencode-go",
    }


def _analyzer(gate: Optional[JevGate], action: str = "BUY") -> PairAnalyzer:
    candles = _candles()
    engine = FakeEngine(_engine_result(action), candles, _timestamps())
    return PairAnalyzer(engine=engine, symbol=SYMBOL, timeframe="1h", jev_gate=gate)


@pytest.mark.asyncio
async def test_adapter_off_mode_unchanged_and_no_call():
    client = FakeJevClient()
    analyzer = _analyzer(JevGate(client=client, mode="off"))

    signal = await analyzer.analyze(candle_open_time="2026-01-01T05:00:00Z")

    assert client.calls == []
    assert analyzer.engine.calls == 1
    assert signal.action == "BUY"
    assert signal.risk_approved is True
    assert signal.jev is None


@pytest.mark.asyncio
async def test_adapter_pre_blocks_llm_call():
    client = FakeJevClient(_response(tradeable=0.2))
    analyzer = _analyzer(JevGate(client=client, mode="pre"))

    signal = await analyzer.analyze(candle_open_time="2026-01-01T05:00:00Z")

    assert analyzer.engine.calls == 0
    assert signal.action == "HOLD"
    assert signal.source == "jev"
    assert signal.risk_approved is False
    assert signal.model == "jev-1.13"
    assert "JEV pre-gate: not tradeable" in signal.reasoning
    assert signal.jev == {
        "model": "jev-1.13",
        "tradeability": False,
        "bias": "long",
        "regime": "trend",
    }


@pytest.mark.asyncio
async def test_adapter_post_veto_sets_risk_false_with_reason():
    client = FakeJevClient(_response(tradeable=0.9, bias="short"))
    analyzer = _analyzer(JevGate(client=client, mode="post"))

    signal = await analyzer.analyze(candle_open_time="2026-01-01T05:00:00Z")

    assert analyzer.engine.calls == 1
    assert signal.action == "BUY"
    assert signal.risk_approved is False
    assert "JEV veto: short bias conflicts with BUY" in signal.reasoning
    assert signal.jev["bias"] == "short"


@pytest.mark.asyncio
async def test_adapter_pre_post_calls_jev_twice():
    client = FakeJevClient(_response(tradeable=0.9, bias="long"))
    analyzer = _analyzer(JevGate(client=client, mode="pre_post"))

    signal = await analyzer.analyze(candle_open_time="2026-01-01T05:00:00Z")

    assert len(client.calls) == 2
    assert analyzer.engine.calls == 1
    assert signal.risk_approved is True
    assert signal.jev["model"] == "jev-1.13"


@pytest.mark.asyncio
async def test_adapter_pre_post_skips_llm_when_pre_blocks():
    client = FakeJevClient(_response(tradeable=0.0, bias="none"))
    analyzer = _analyzer(JevGate(client=client, mode="pre_post"))

    signal = await analyzer.analyze(candle_open_time="2026-01-01T05:00:00Z")

    assert len(client.calls) == 1
    assert analyzer.engine.calls == 0
    assert signal.action == "HOLD"


@pytest.mark.asyncio
async def test_adapter_hold_is_never_blocked_in_post_mode():
    client = FakeJevClient(_response(tradeable=0.0, bias="none"))
    analyzer = _analyzer(JevGate(client=client, mode="post"), action="HOLD")

    signal = await analyzer.analyze(candle_open_time="2026-01-01T05:00:00Z")

    assert signal.action == "HOLD"
    assert "JEV veto" not in signal.reasoning
    assert client.calls == []


class _RefreshingCollector:
    """Emulates the live data collector: refreshes the engine context."""

    def __init__(self, engine: "LiveOrderingEngine") -> None:
        self.engine = engine

    async def collect_data(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        self.engine.context.ohlcv_candles = list(self.engine._fresh_candles)
        self.engine.context.timestamps = list(self.engine._fresh_timestamps)
        return {"success": True}


class LiveOrderingEngine(FakeEngine):
    """Engine whose context is stale until an analysis refreshes it.

    Mirrors real live ordering: the scheduler's context still holds the
    candles truncated during the previous cycle, and collection inside
    analyze_market brings in the candle the current cycle is about.
    """

    def __init__(self, result: Dict[str, Any], fresh_candles: List[List[float]], fresh_timestamps: List[datetime]) -> None:
        stale_candles = fresh_candles[:-1]
        stale_timestamps = fresh_timestamps[:-1]
        super().__init__(result, stale_candles, stale_timestamps)
        self._fresh_candles = fresh_candles
        self._fresh_timestamps = fresh_timestamps

    async def analyze_market(self, **_kwargs: Any) -> Dict[str, Any]:
        if getattr(self, "data_collector", None) is not None:
            await self.data_collector.collect_data()
        return await super().analyze_market(**_kwargs)


def _live_engine(action: str = "BUY", with_collector: bool = True) -> LiveOrderingEngine:
    engine = LiveOrderingEngine(_engine_result(action), _candles(), _timestamps())
    if with_collector:
        engine.data_collector = _RefreshingCollector(engine)
    return engine


@pytest.mark.asyncio
async def test_adapter_post_gate_uses_fresh_context_after_analysis():
    """Regression: the post gate must not decide on a one-candle-stale series.

    In live ordering the context is refreshed inside analyze_market, so the
    state rebuilt after the analysis includes the candle this cycle analysed.
    """
    client = FakeJevClient(_response(tradeable=0.9, bias="short"))
    engine = _live_engine()
    analyzer = PairAnalyzer(engine=engine, symbol=SYMBOL, timeframe="1h", jev_gate=JevGate(client=client, mode="post"))

    signal = await analyzer.analyze(candle_open_time="2026-01-01T05:00:00Z")

    assert signal.jev is not None
    # Stale context had 5 candles; the fresh one has the analysed 6th candle.
    assert client.calls[0]["candles_used"] == 6
    assert client.calls[0]["last_close"] == 105.0
    assert client.calls[0]["last_candle_close_time"] == "2026-01-01T06:00:00Z"
    assert analyzer.engine.calls == 1


@pytest.mark.asyncio
async def test_adapter_pre_gate_refreshes_context_before_gating():
    """Regression: the pre-gate must see the candle that just closed.

    With a collector exposed, analyze() collects and truncates to as_of before
    consulting JEV, so the pre-gate state includes the analysed candle.
    """
    client = FakeJevClient(_response(tradeable=0.2))
    engine = _live_engine()
    analyzer = PairAnalyzer(engine=engine, symbol=SYMBOL, timeframe="1h", jev_gate=JevGate(client=client, mode="pre"))

    signal = await analyzer.analyze(candle_open_time="2026-01-01T05:00:00Z")

    assert client.calls[0]["candles_used"] == 6
    assert client.calls[0]["last_close"] == 105.0
    assert signal.action == "HOLD"
    assert signal.source == "jev"
    assert analyzer.engine.calls == 0  # pre blocked: LLM never called


@pytest.mark.asyncio
async def test_adapter_pre_gate_no_collector_skips_state():
    """Without a collector the context cannot be refreshed: the gate fails open."""
    client = FakeJevClient(_response(tradeable=0.2))
    engine = _live_engine(with_collector=False)
    analyzer = PairAnalyzer(engine=engine, symbol=SYMBOL, timeframe="1h", jev_gate=JevGate(client=client, mode="pre"))

    signal = await analyzer.analyze(candle_open_time="2026-01-01T05:00:00Z")

    # Without a collector the context cannot be refreshed first: the pre-gate
    # decides on the stale (5-candle) context anyway, documenting the residual
    # one-candle lag for engine doubles without a data_collector. The
    # production engine always exposes one.
    assert client.calls[0]["candles_used"] == 5
    assert analyzer.engine.calls == 0  # pre blocked: LLM never called
    assert signal.action == "HOLD"
    assert signal.source == "jev"


@pytest.mark.asyncio
async def test_adapter_survives_non_jev_gate_exception():
    """A broken JEV transport must not abort analyze(): a signal is still produced."""
    analyzer = _analyzer(JevGate(client=ExplodingJevClient(RuntimeError("transport bug")), mode="pre_post"))

    signal = await analyzer.analyze(candle_open_time="2026-01-01T05:00:00Z")

    assert analyzer.engine.calls == 1
    assert signal.action == "BUY"
    assert signal.risk_approved is True
    assert signal.jev is None


def test_jev_state_handles_numpy_candles():
    """The real engine exposes OHLCV as a NumPy array; truthiness must not raise."""
    np = pytest.importorskip("numpy")
    from src.signal_service.jev_gate import _mean

    timestamps = _timestamps()
    engine = FakeEngine(_engine_result("HOLD"), list(_candles()), timestamps)
    # Same shape/type as the live AnalysisContext.
    engine.context.ohlcv_candles = np.array(_candles(), dtype=float)
    engine.context.timestamps = timestamps
    analyzer = PairAnalyzer(engine=engine, symbol=SYMBOL, timeframe="1h")

    state = analyzer._jev_state(timestamps[-1] + timedelta(hours=1))

    assert state is not None
    assert state["candles_used"] == len(timestamps)

    # Empty ndarray: no ValueError, no state.
    engine.context.ohlcv_candles = np.empty((0, 5))
    assert analyzer._jev_state(timestamps[-1] + timedelta(hours=1)) is None

    # NumPy-safe mean helper.
    assert _mean(np.array([1.0, 3.0])) == 2.0
    assert _mean(np.array([])) is None


def test_jev_state_maps_non_finite_values_to_none():
    """OHLCV gaps (NaN from the exchange layer) must not poison the JSON payload."""
    np = pytest.importorskip("numpy")
    timestamps = _timestamps()
    as_of = timestamps[-1] + timedelta(hours=1)

    # A fully missing last candle: every metric derived from it is non-finite.
    candles = _candles()
    candles[-1] = [float("nan")] * 5
    state = build_state(SYMBOL, "1h", np.array(candles, dtype=float), timestamps, as_of)
    assert state["last_close"] is None
    assert state["recent_closes"][-1] is None
    assert state["atr_pct"] is None
    assert state["volume_ratio"] is None
    # Strict JSON: a bare NaN literal would raise here.
    assert json.loads(json.dumps(state, allow_nan=False))["last_close"] is None

    # A gap in an earlier volume only: the last close stays valid, the volume
    # ratio (20-candle window) becomes None instead of NaN.
    candles = _candles()
    candles[3][4] = float("nan")
    state = build_state(SYMBOL, "1h", np.array(candles, dtype=float), timestamps, as_of)
    assert state["last_close"] == 105.0
    assert state["volume_ratio"] is None
    json.dumps(state, allow_nan=False)


def test_round_maps_non_finite_to_none():
    from src.signal_service.jev_gate import _round

    assert _round(float("nan")) is None
    assert _round(float("inf")) is None
    assert _round(float("-inf")) is None
    assert _round(None) is None
    assert _round("not-a-number") is None
    assert _round(1.23456789, digits=2) == 1.23


def test_build_state_zero_close_denominator():
    """A corrupt candle with close 0.0 must yield None, not ZeroDivisionError."""
    candles = _candles()
    candles[4][3] = 0.0  # the reference close for the 1-period return
    timestamps = _timestamps()

    state = build_state(SYMBOL, "1h", candles, timestamps, as_of=timestamps[-1] + timedelta(hours=1))

    assert state["returns"]["1"] is None
    assert state["returns"]["4"] is not None
    json.dumps(state, allow_nan=False)


@pytest.mark.asyncio
async def test_adapter_survives_zero_close_candle():
    """A zero-close candle must not abort analyze(): the signal is still stored."""
    candles = _candles()
    candles[4][3] = 0.0
    engine = FakeEngine(_engine_result("BUY"), candles, _timestamps())
    client = FakeJevClient(_response(tradeable=0.9, bias="long"))
    analyzer = PairAnalyzer(
        engine=engine, symbol=SYMBOL, timeframe="1h",
        jev_gate=JevGate(client=client, mode="post"),
    )

    signal = await analyzer.analyze(candle_open_time="2026-01-01T05:00:00Z")

    assert analyzer.engine.calls == 1
    assert signal.action == "BUY"
    assert client.calls[0]["returns"]["1"] is None


def test_jev_state_fails_open_on_unexpected_state_error(monkeypatch):
    """Any build_state exception is swallowed by _jev_state (fail-open)."""
    engine = FakeEngine(_engine_result("HOLD"), _candles(), _timestamps())
    analyzer = PairAnalyzer(engine=engine, symbol=SYMBOL, timeframe="1h")

    def _boom(*_args: Any, **_kwargs: Any) -> Dict[str, Any]:
        raise RuntimeError("unexpected state failure")

    monkeypatch.setattr("src.signal_service.analyzer_adapter.build_state", _boom)

    assert analyzer._jev_state(_timestamps()[-1] + timedelta(hours=1)) is None
