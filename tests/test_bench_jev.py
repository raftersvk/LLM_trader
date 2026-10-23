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

    def decide(self, _state: Dict[str, Any]) -> Any:
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
        is_llm=False, delay_seconds=0.25,
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
        is_llm=False, delay_seconds=0.25,
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
