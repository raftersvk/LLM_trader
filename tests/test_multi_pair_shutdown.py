"""Multi-pair shutdown must release the supervisor and stop every pair loop."""
import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from start import CompositionRoot


class _FakeShutdownManager:
    def __init__(self, already_shutting_down: bool = False):
        self._shutting_down = already_shutting_down
        self.callbacks = []

    async def shutdown_gracefully(self) -> None:
        self._shutting_down = True
        for callback in self.callbacks:
            await callback()

    @property
    def is_shutting_down(self) -> bool:
        return self._shutting_down

    def register_shutdown_callback(self, callback):
        self.callbacks.append(callback)

    async def trigger(self) -> None:
        for callback in self.callbacks:
            await callback()


def _root(shutdown_manager) -> CompositionRoot:
    root = object.__new__(CompositionRoot)
    root.shutdown_manager = shutdown_manager
    root.logger = MagicMock()
    return root


@pytest.mark.asyncio
async def test_shutdown_callback_releases_multi_pair_supervisor():
    """Graceful shutdown stops the pairs and lets the supervisor return."""
    manager = MagicMock()
    manager.stop_all_pairs = AsyncMock()
    shutdown = _FakeShutdownManager()
    root = _root(shutdown)

    task = asyncio.create_task(root._supervise_multi_pair(manager))
    await asyncio.sleep(0)
    assert not task.done()

    await shutdown.trigger()

    await asyncio.wait_for(task, timeout=1.0)
    manager.stop_all_pairs.assert_awaited_once()


@pytest.mark.asyncio
async def test_supervisor_returns_when_shutdown_already_started():
    """A shutdown requested before the wait does not hang the loop."""
    manager = MagicMock()
    manager.stop_all_pairs = AsyncMock()
    root = _root(_FakeShutdownManager(already_shutting_down=True))

    await asyncio.wait_for(root._supervise_multi_pair(manager), timeout=1.0)

    manager.stop_all_pairs.assert_awaited_once()


@pytest.mark.asyncio
async def test_supervisor_stops_pairs_when_cancelled():
    """Cancellation still stops the pair loops before propagating."""
    manager = MagicMock()
    manager.stop_all_pairs = AsyncMock()
    root = _root(_FakeShutdownManager())

    task = asyncio.create_task(root._supervise_multi_pair(manager))
    await asyncio.sleep(0)
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task
    manager.stop_all_pairs.assert_awaited_once()


@pytest.mark.asyncio
async def test_multi_pair_registers_quit_and_starts_keyboard_listener():
    """Multi-pair mode must expose 'q' and start the listener the bot would own."""
    keyboard = MagicMock()
    keyboard.start_listening = AsyncMock()
    keyboard.stop_listening = AsyncMock()
    shutdown = _FakeShutdownManager()
    root = _root(shutdown)
    toggle = AsyncMock()

    task = root._register_multi_pair_keyboard(keyboard, toggle)
    await asyncio.sleep(0)

    commands = {call.args[0]: call.args[1] for call in keyboard.register_command.call_args_list}
    assert set(commands) == {"d", "q"}
    assert commands["d"] is toggle
    assert keyboard.stop_listening in shutdown.callbacks
    keyboard.start_listening.assert_awaited_once()

    await commands["q"]()
    await asyncio.wait_for(task, timeout=1.0)
