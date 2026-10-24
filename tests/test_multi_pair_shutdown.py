"""Multi-pair shutdown must release the supervisor and stop every pair loop."""
import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from start import CompositionRoot


class _FakeShutdownManager:
    def __init__(self, already_shutting_down: bool = False):
        self._shutting_down = already_shutting_down
        self.callbacks = []
        self._shutdown_task = None

    @property
    def shutdown_task(self):
        return self._shutdown_task

    async def wait_for_shutdown(self) -> None:
        if self._shutdown_task is not None:
            try:
                await self._shutdown_task
            except asyncio.CancelledError:
                pass

    def start_shutdown(self):
        if self._shutdown_task is None or self._shutdown_task.done():
            self._shutdown_task = asyncio.create_task(self.shutdown_gracefully())
        return self._shutdown_task

    async def shutdown_gracefully(self) -> None:
        self._shutdown_task = asyncio.current_task()
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


@pytest.mark.asyncio
async def test_quit_runs_shutdown_off_the_listener_task():
    """The q callback must not await shutdown inside the cancellable listener."""
    started = asyncio.Event()
    release = asyncio.Event()

    class _SlowShutdown(_FakeShutdownManager):
        async def shutdown_gracefully(self) -> None:
            self._shutting_down = True
            started.set()
            await release.wait()

    keyboard = MagicMock()
    keyboard.start_listening = AsyncMock()
    keyboard.stop_listening = AsyncMock()
    shutdown = _SlowShutdown()
    root = _root(shutdown)

    root._register_multi_pair_keyboard(keyboard, AsyncMock())
    commands = {call.args[0]: call.args[1] for call in keyboard.register_command.call_args_list}

    await asyncio.wait_for(commands["q"](), timeout=1.0)
    await asyncio.wait_for(started.wait(), timeout=1.0)

    # The callback returned while shutdown is still running elsewhere.
    assert shutdown.is_shutting_down is True
    release.set()
    await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_supervisor_returns_when_callbacks_abort_after_shutdown_started():
    """A bot callback cancelling the chain must not strand the supervisor."""
    manager = MagicMock()
    manager.stop_all_pairs = AsyncMock()
    shutdown = _FakeShutdownManager()

    async def _bot_shutdown_cancels_listener() -> None:
        raise asyncio.CancelledError()

    # Registered before the supervisor, like CryptoTradingBot.shutdown.
    shutdown.callbacks.append(_bot_shutdown_cancels_listener)
    root = _root(shutdown)

    supervisor = asyncio.create_task(root._supervise_multi_pair(manager))
    await asyncio.sleep(0)

    runner = asyncio.create_task(shutdown.shutdown_gracefully())
    with pytest.raises(asyncio.CancelledError):
        await runner

    await asyncio.wait_for(supervisor, timeout=1.0)
    manager.stop_all_pairs.assert_awaited_once()


@pytest.mark.asyncio
async def test_supervisor_waits_for_the_shutdown_task_to_finish_cleanup():
    """Returning on the pair-stop event alone would truncate later cleanup."""
    manager = MagicMock()
    manager.stop_all_pairs = AsyncMock()
    cleanup_started = asyncio.Event()
    cleanup_release = asyncio.Event()
    cleanup_finished = asyncio.Event()

    class _SlowCleanupManager(_FakeShutdownManager):
        async def shutdown_gracefully(self) -> None:
            self._shutdown_task = asyncio.current_task()
            self._shutting_down = True
            for callback in self.callbacks:
                await callback()
            cleanup_started.set()
            await cleanup_release.wait()
            cleanup_finished.set()

    shutdown = _SlowCleanupManager()
    root = _root(shutdown)

    supervisor = asyncio.create_task(root._supervise_multi_pair(manager))
    await asyncio.sleep(0)
    shutdown.start_shutdown()
    await asyncio.wait_for(cleanup_started.wait(), timeout=1.0)

    # The pair-stop callback has already released the event, but cleanup runs on.
    await asyncio.sleep(0.1)
    assert not supervisor.done()
    assert cleanup_finished.is_set() is False

    cleanup_release.set()
    await asyncio.wait_for(supervisor, timeout=1.0)
    assert cleanup_finished.is_set() is True
    manager.stop_all_pairs.assert_awaited_once()
