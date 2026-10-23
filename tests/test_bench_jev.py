"""Tests for the JEV benchmark scaffold pacing.

No network: the gate runs without a client, so every JEV call fails open, and
the pacing helper is replaced by a recording fake.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "bench_jev", Path(__file__).resolve().parents[1] / "scripts" / "bench_jev.py"
)
assert _SPEC is not None and _SPEC.loader is not None
bench_jev = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = bench_jev
_SPEC.loader.exec_module(bench_jev)


class _AlwaysBuyBase:
    """Base pipeline stub: every state yields a BUY decision."""

    async def decide(self, _candles: List[Any], _index: int, _state: Dict[str, Any]) -> Any:
        return bench_jev.BaseDecision("BUY", 1.0)


def _samples(count: int) -> List[Tuple[int, Dict[str, Any], Dict[int, float]]]:
    return [(index, {"symbol": "BTC/USDT"}, {}) for index in range(count)]


@pytest.mark.asyncio
async def test_run_mode_paces_every_jev_call(monkeypatch):
    """Real runs pause before each gate call, not once per mode."""
    sleeps: List[float] = []

    async def _record(delay_seconds: float) -> None:
        sleeps.append(delay_seconds)

    monkeypatch.setattr(bench_jev, "_pace", _record)
    gate = bench_jev.JevGate(client=None, mode="pre_post")

    await bench_jev.run_mode(
        "jev_pre_post", gate, _AlwaysBuyBase(), [], _samples(3), folds=1,
        delay_seconds=0.25,
    )

    # pre + post per state: 3 states -> 6 calls, each paced.
    assert sleeps == [0.25] * 6


@pytest.mark.asyncio
async def test_run_mode_llm_alone_never_paces(monkeypatch):
    """llm_alone makes no JEV call, so it must not sleep at all."""
    sleeps: List[float] = []

    async def _record(delay_seconds: float) -> None:
        sleeps.append(delay_seconds)

    monkeypatch.setattr(bench_jev, "_pace", _record)
    gate = bench_jev.JevGate(client=None, mode="off")

    await bench_jev.run_mode(
        "llm_alone", gate, _AlwaysBuyBase(), [], _samples(4), folds=1,
        delay_seconds=0.25,
    )

    assert sleeps == []


@pytest.mark.asyncio
async def test_pace_is_noop_without_delay(monkeypatch):
    """A zero delay must not sleep, so offline runs stay fast."""
    real_sleep_calls: List[float] = []

    async def _record_sleep(seconds: float) -> None:
        real_sleep_calls.append(seconds)

    monkeypatch.setattr(bench_jev.asyncio, "sleep", _record_sleep)
    await bench_jev._pace(0.0)
    await bench_jev._pace(-1.0)

    assert real_sleep_calls == []


class _FakeExchangeManager:
    """Exchange manager stub resolving one supported pair."""

    def __init__(self, exchange: Any = "fake-exchange") -> None:
        self._exchange = exchange
        self.asked: List[str] = []

    async def find_symbol_exchange(self, symbol: str):
        self.asked.append(symbol)
        return (self._exchange, "fake")


class _RecordingEngine:
    """Engine stub recording initialize_for_symbol and exposing a context."""

    def __init__(self) -> None:
        self.context = None
        self.init_calls: List[Dict[str, Any]] = []

    def initialize_for_symbol(self, symbol=None, exchange=None, timeframe=None):
        self.init_calls.append(
            {"symbol": symbol, "exchange": exchange, "timeframe": timeframe}
        )
        self.context = type("Ctx", (), {"ohlcv_candles": None, "timestamps": []})()


@pytest.mark.asyncio
async def test_llm_backend_initializes_engine_before_use(monkeypatch):
    """The engine must be initialized, otherwise the injector crashes on context=None."""
    import types

    engine = _RecordingEngine()
    exchange_manager = _FakeExchangeManager()

    class _Root:
        async def build_dependencies(self):
            return {
                "exchange_manager": exchange_manager,
                "market_analyzer_factory": lambda _symbol: engine,
            }

    monkeypatch.setitem(
        sys.modules, "start", types.SimpleNamespace(CompositionRoot=_Root),
    )
    backend = bench_jev.LLMBackend("BTC/USDT", "1h")

    await backend.initialize()

    assert backend.available is True, backend.reason
    assert exchange_manager.asked == ["BTC/USDT"]
    assert engine.init_calls == [
        {"symbol": "BTC/USDT", "exchange": "fake-exchange", "timeframe": "1h"}
    ]
    assert backend.exchange_id == "fake"


@pytest.mark.asyncio
async def test_llm_backend_reports_unsupported_pair(monkeypatch):
    """An unsupported pair must mark the backend unavailable, not crash."""
    import types

    engine = _RecordingEngine()

    class _Root:
        async def build_dependencies(self):
            return {
                "exchange_manager": _FakeExchangeManager(exchange=None),
                "market_analyzer_factory": lambda _symbol: engine,
            }

    monkeypatch.setitem(
        sys.modules, "start", types.SimpleNamespace(CompositionRoot=_Root),
    )
    backend = bench_jev.LLMBackend("NOPE/USDT", "1h")

    await backend.initialize()

    assert backend.available is False
    assert "No exchange supports" in (backend.reason or "")
    assert engine.init_calls == []


@pytest.mark.asyncio
async def test_unavailable_llm_backend_does_not_crash_the_run():
    """--base llm without config/keys must still yield a HOLD report, not TypeError."""
    backend = bench_jev.LLMBackend("BTC/USDT", "1h")  # never initialized: unavailable
    gate = bench_jev.JevGate(client=None, mode="off")

    run = await bench_jev.run_mode(
        "llm_alone", gate, backend, [], _samples(2), folds=1,
    )

    assert [prediction.action for prediction in run.predictions] == ["HOLD", "HOLD"]
    assert all("unavailable" in (prediction.error or "") for prediction in run.predictions)
