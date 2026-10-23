"""Tests for DashboardServer start-up behaviour."""
import asyncio
import threading
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from src.dashboard.server import DashboardServer


def _build_server() -> DashboardServer:
    config = SimpleNamespace(
        DATA_DIR=".",
        DEMO_QUOTE_CAPITAL=10000.0,
        DASHBOARD_ENABLE_CORS=False,
        DASHBOARD_CORS_ORIGINS=[],
    )
    return DashboardServer(
        brain_service=MagicMock(),
        vector_memory=MagicMock(),
        analysis_engine=MagicMock(),
        config=config,
        logger=MagicMock(),
        unified_parser=MagicMock(),
        persistence=MagicMock(),
        exchange_manager=MagicMock(),
    )


@pytest.mark.asyncio
async def test_start_does_not_block_event_loop_while_waiting(monkeypatch):
    """A slow server must not starve the trading loops sharing the loop."""
    loop_alive = threading.Event()
    heartbeats = 0

    class _FakeServer:
        started = False

        def run(self):
            loop_alive.wait(timeout=2.0)
            self.started = True

    fake_server = _FakeServer()
    monkeypatch.setattr(
        "src.dashboard.server.uvicorn.Server", lambda config: fake_server
    )
    server = _build_server()

    async def heartbeat():
        nonlocal heartbeats
        while not fake_server.started:
            heartbeats += 1
            loop_alive.set()
            await asyncio.sleep(0.01)

    beat_task = asyncio.create_task(heartbeat())
    try:
        await server.start()
    finally:
        beat_task.cancel()
        await asyncio.gather(beat_task, return_exceptions=True)

    assert fake_server.started is True
    assert heartbeats > 0
