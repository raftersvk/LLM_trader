"""Tests for DashboardServer start/stop lifecycle."""
import asyncio
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


class _FakeServer:
    """Minimal uvicorn.Server stand-in."""

    def __init__(self) -> None:
        self.started = False
        self.should_exit = False
        self.serve_calls = 0

    async def serve(self) -> None:
        self.serve_calls += 1
        self.started = True
        while not self.should_exit:
            await asyncio.sleep(0.01)


@pytest.mark.asyncio
async def test_start_does_not_block_event_loop_while_waiting(monkeypatch):
    """Server start-up must not starve the trading loops sharing the loop."""
    fake_server = _FakeServer()
    monkeypatch.setattr(
        "src.dashboard.server.uvicorn.Server", lambda config: fake_server
    )
    server = _build_server()
    heartbeats = 0

    async def heartbeat():
        nonlocal heartbeats
        while not fake_server.started:
            heartbeats += 1
            await asyncio.sleep(0)

    beat_task = asyncio.create_task(heartbeat())
    try:
        task = await server.start()
        assert task is server.server_task
        assert fake_server.started is True
        assert heartbeats > 0
    finally:
        beat_task.cancel()
        await asyncio.gather(beat_task, return_exceptions=True)
        await server.stop()


@pytest.mark.asyncio
async def test_start_is_idempotent_and_stop_awaits_server(monkeypatch):
    """A second start reuses the task; stop releases it before a restart."""
    fake_server = _FakeServer()
    monkeypatch.setattr(
        "src.dashboard.server.uvicorn.Server", lambda config: fake_server
    )
    server = _build_server()

    first_task = await server.start()
    second_task = await server.start()

    assert second_task is first_task
    assert fake_server.serve_calls == 1

    await server.stop()

    assert fake_server.should_exit is True
    assert first_task.done()
    assert server.server_task is None
    assert server._server is None


@pytest.mark.asyncio
async def test_start_logs_failure_when_server_cannot_run(monkeypatch):
    """A failing server (e.g. port conflict) is logged, not left orphaned."""
    class _FailingServer:
        started = False
        should_exit = False

        async def serve(self) -> None:
            raise RuntimeError("bind failed")

    failing_server = _FailingServer()
    monkeypatch.setattr(
        "src.dashboard.server.uvicorn.Server", lambda config: failing_server
    )
    server = _build_server()

    task = await server.start()

    assert task.done()
    assert any(
        "bind failed" in str(call) for call in server.logger.error.call_args_list
    )
    await server.stop()
