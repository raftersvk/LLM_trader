"""Tests for MultiPairManager start-lifecycle guarding."""
import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.managers.multi_pair_manager import MultiPairManager


class _FakeBot:
    """Bot whose run() blocks until the test releases it."""

    def __init__(self, started: asyncio.Event, release: asyncio.Event) -> None:
        self.config = SimpleNamespace(TIMEFRAME="1h")
        self._started = started
        self._release = release

    async def run(self, symbol: str, timeframe: str) -> None:
        self._started.set()
        await self._release.wait()


def _manager(factory, pairs=None) -> MultiPairManager:
    return MultiPairManager(
        logger=MagicMock(),
        config=SimpleNamespace(TIMEFRAME="1h", CRYPTO_PAIRS=pairs or ["BTC/USDT"]),
        dependencies_factory=factory,
    )


@pytest.mark.asyncio
async def test_concurrent_start_pair_creates_only_one_loop():
    """A second start during bot creation must not spawn a duplicate loop."""
    started = asyncio.Event()
    release = asyncio.Event()
    calls = []
    factory = MagicMock()

    async def create_bot(symbol):
        calls.append(symbol)
        await asyncio.sleep(0)
        return _FakeBot(started, release)

    factory.create_bot_for_pair = create_bot
    manager = _manager(factory)

    results = await asyncio.gather(
        manager.start_pair("BTC/USDT"),
        manager.start_pair("BTC/USDT"),
    )

    assert sorted(results) == [False, True]
    assert calls == ["BTC/USDT"]
    context = manager.pair_contexts["BTC/USDT"]
    assert context.bot_task is not None

    await asyncio.wait_for(started.wait(), timeout=1)
    release.set()
    await asyncio.wait_for(context.bot_task, timeout=1)


@pytest.mark.asyncio
async def test_start_pair_resets_state_when_bot_creation_fails():
    """A creation failure must not leave the pair permanently unstartable."""
    started = asyncio.Event()
    release = asyncio.Event()
    factory = MagicMock()
    factory.create_bot_for_pair = AsyncMock(side_effect=RuntimeError("boom"))
    manager = _manager(factory)

    assert await manager.start_pair("BTC/USDT") is False

    context = manager.pair_contexts["BTC/USDT"]
    assert context.is_running is False
    assert context.bot_task is None
    assert "boom" in (context.last_error or "")

    factory.create_bot_for_pair = AsyncMock(return_value=_FakeBot(started, release))
    assert await manager.start_pair("BTC/USDT") is True

    await asyncio.wait_for(started.wait(), timeout=1)
    release.set()
    await asyncio.wait_for(context.bot_task, timeout=1)


@pytest.mark.asyncio
async def test_start_all_pairs_continues_after_one_pair_fails():
    """One failing pair must not abort the remaining pairs."""
    started = asyncio.Event()
    release = asyncio.Event()
    factory = MagicMock()

    async def create_bot(symbol):
        if symbol == "BTC/USDT":
            raise RuntimeError("boom")
        return _FakeBot(started, release)

    factory.create_bot_for_pair = create_bot
    manager = _manager(factory, pairs=["BTC/USDT", "ETH/USDT"])

    await manager.start_all_pairs()

    assert manager.pair_contexts["BTC/USDT"].is_running is False
    eth_context = manager.pair_contexts["ETH/USDT"]
    assert eth_context.is_running is True

    await asyncio.wait_for(started.wait(), timeout=1)
    release.set()
    await asyncio.wait_for(eth_context.bot_task, timeout=1)
