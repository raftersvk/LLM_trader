"""Integration test for the multi-pair bot factory.

Building one pair bot must not raise (regression: ``pair_deps`` lacked the
``trading_strategy`` key, so every pair failed to start).
"""
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.factories.bot_factory import BotDependencyFactory


def _config() -> MagicMock:
    config = MagicMock()
    config.TIMEFRAME = "4h"
    config.DEMO_QUOTE_CAPITAL = 10000.0
    config.QUOTE_CURRENCY = "USDC"
    config.EXECUTOR_API_ENABLED = False
    config.EXECUTOR_MAX_POSITION_USDC = 0
    config.REFERENCE_EXIT_CHECK_SECONDS = 10
    config.get_config.return_value = 120.0
    return config


def _base_dependencies() -> dict:
    return {
        "shutdown_manager": MagicMock(),
        "exchange_manager": MagicMock(),
        "coingecko_api": MagicMock(),
        "market_api": MagicMock(),
        "alternative_me_api": MagicMock(),
        "http_session": MagicMock(),
        "model_manager": MagicMock(),
        "rag_engine": MagicMock(),
        "discord_notifier": MagicMock(),
        "keyboard_handler": MagicMock(),
        "exit_monitor": MagicMock(),
        "market_analyzer": MagicMock(),
        "ev_formatter": MagicMock(),
        "parser": MagicMock(),
        "sentiment_analyst": MagicMock(),
        "executor_handler": MagicMock(),
        "dashboard_state": MagicMock(),
        "guard_pipeline": MagicMock(),
        "post_mortem_service": MagicMock(),
        "tightening_policy": MagicMock(),
        "post_mortem_repo": MagicMock(),
        "prompt_budget": MagicMock(),
    }


@pytest.mark.asyncio
async def test_create_bot_for_pair_wires_the_pair_strategy(tmp_path, monkeypatch):
    """A pair bot is created with its own strategy and persistence."""
    monkeypatch.chdir(tmp_path)
    logger = MagicMock()
    logger.getChild.return_value = logger

    with patch("src.factories.bot_factory.chromadb.PersistentClient"), patch(
        "src.factories.bot_factory.SentenceTransformer"
    ), patch("src.app.CryptoTradingBot.initialize", new=AsyncMock()):
        factory = BotDependencyFactory(_base_dependencies(), _config(), logger)
        bot = await factory.create_bot_for_pair("ETH/USDT")

    assert bot is not None
    assert factory._bot_instances["ETH/USDT"] is bot
    assert bot.trading_strategy is not None
    assert bot.trading_strategy is not factory.base_dependencies.get("market_analyzer")
    assert bot.persistence is not None
    assert bot.brain_service is not None
    assert bot.brain_service.context_provider.post_mortem_repo is (
        factory.base_dependencies["post_mortem_repo"]
    )
    assert bot.brain_service.context_provider.prompt_budget is (
        factory.base_dependencies["prompt_budget"]
    )
    assert bot.memory_service.initial_capital == _config().DEMO_QUOTE_CAPITAL

    # Each pair owns an isolated persistence directory.
    assert "pair_ETH_USDT" in str(bot.persistence.data_dir)

    # Regression: the 9th positional argument is conditions_extractor, not the
    # position factory. A mis-wired strategy blows up on the first analysis.
    strategy = bot.trading_strategy
    assert strategy._conditions is not strategy.position_factory
    assert strategy._conditions.__class__.__name__ == "MarketConditionsExtractor"
    assert strategy.guard_pipeline is factory.base_dependencies["guard_pipeline"]
    assert strategy.post_mortem_service is factory.base_dependencies["post_mortem_service"]
    assert strategy.dashboard_state is factory.base_dependencies["dashboard_state"]
    conditions = strategy._conditions.extract_market_conditions({"price": 100.0})
    assert conditions is not None


@pytest.mark.asyncio
async def test_each_pair_gets_its_own_analyzer_engine(tmp_path, monkeypatch):
    """Pairs must not receive the shared engine/collaborator set."""
    monkeypatch.chdir(tmp_path)
    logger = MagicMock()
    logger.getChild.return_value = logger
    engines = []

    def _engine_factory():
        engine = MagicMock(name=f"engine-{len(engines)}")
        engines.append(engine)
        return engine

    base = _base_dependencies()
    base["analyzer_engine_factory"] = _engine_factory

    with patch("src.factories.bot_factory.chromadb.PersistentClient"), patch(
        "src.factories.bot_factory.SentenceTransformer"
    ), patch("src.app.CryptoTradingBot.initialize", new=AsyncMock()):
        factory = BotDependencyFactory(base, _config(), logger)
        first = await factory.create_bot_for_pair("BTC/USDC")
        second = await factory.create_bot_for_pair("ETH/USDC")

    assert len(engines) == 2
    assert first.market_analyzer is engines[0]
    assert second.market_analyzer is engines[1]
    assert first.market_analyzer is not second.market_analyzer
