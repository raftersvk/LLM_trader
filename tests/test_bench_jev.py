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

from src.signal_service.jev_client import JevError

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


class _ErroringClient:
    """JEV client stub: every ask raises the typed gate error."""

    def __init__(self) -> None:
        self.calls = 0

    async def ask(self, _state: Dict[str, Any], _questions: Dict[str, Any]) -> Any:
        self.calls += 1
        raise JevError("simulated JEV outage")


@pytest.mark.asyncio
async def test_run_mode_aborts_on_jev_error_when_enabled():
    """A real run must stop on the first JEV error, not report fail-open data."""
    client = _ErroringClient()
    gate = bench_jev.JevGate(client=client, mode="pre")

    with pytest.raises(SystemExit, match="JEV error during real benchmark"):
        await bench_jev.run_mode(
            "jev_pre", gate, _AlwaysBuyBase(), [], _samples(2), folds=1,
            abort_on_jev_error=True,
        )

    assert client.calls == 1


@pytest.mark.asyncio
async def test_run_mode_continues_on_jev_error_by_default():
    """Without the flag, the historical continue-on-error behavior is kept."""
    client = _ErroringClient()
    gate = bench_jev.JevGate(client=client, mode="pre")

    run = await bench_jev.run_mode(
        "jev_pre", gate, _AlwaysBuyBase(), [], _samples(2), folds=1,
    )

    assert client.calls == 2
    assert run.jev_errors == 2
    assert len(run.predictions) == 2


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


def test_gate_for_builds_a_gate_without_the_removed_noul_threshold():
    """Benchmark startup must use the current JevGate signature."""
    gate = bench_jev._gate_for("post", None, 0.6)

    assert isinstance(gate, bench_jev.JevGate)
    assert gate.mode.value == "post"
    assert gate.min_confidence == 0.6


def test_cli_no_longer_exposes_the_removed_tradeability_threshold():
    """The CLI flag was retired with the noul threshold."""
    with pytest.raises(SystemExit):
        bench_jev._build_parser().parse_args(["--tradeability-threshold", "0.5"])


def test_default_output_is_repo_relative(monkeypatch):
    """The report default must not point into a personal home directory."""
    import os

    monkeypatch.delenv("JEV_BENCH_OUTPUT", raising=False)

    assert bench_jev.default_output_path() == os.path.join(
        bench_jev.PROJECT_ROOT, "data", "jev_bench", "bench_jev.json",
    )


def test_default_output_respects_env_override(monkeypatch, tmp_path):
    target = str(tmp_path / "custom_report.json")
    monkeypatch.setenv("JEV_BENCH_OUTPUT", target)

    assert bench_jev.default_output_path() == target


def test_build_samples_calibration_context_is_optional():
    """v1 samples stay unchanged; v2 can add the past-only reference block."""
    candles = bench_jev.synthetic_candles(60, 5)
    plain = bench_jev.build_samples(candles, "BTC/USDT", "1h", horizons=(1,))
    contextual = bench_jev.build_samples(
        candles, "BTC/USDT", "1h", horizons=(1,), calibration_context=True
    )

    assert "calibration_context" not in plain[-1][1]
    block = contextual[-1][1]["calibration_context"]
    assert block["lookback_hours"] == 200
    assert 0.0 <= block["base_rate_move_1pct"] <= 1.0
    assert block["realized_vol_24h_pct"] is not None
    # The optional block is the only difference between the two states.
    assert {
        key: value for key, value in contextual[-1][1].items()
        if key != "calibration_context"
    } == plain[-1][1]
