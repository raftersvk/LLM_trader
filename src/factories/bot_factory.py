"""Factory for creating bot instances per trading pair."""

import asyncio
import os
from pathlib import Path
from typing import Dict, Any
from src.app import CryptoTradingBot
from src.config.loader import Config
from src.logger.logger import Logger


class BotDependencyFactory:
    """Creates isolated bot dependencies for each trading pair."""

    def __init__(
        self,
        base_dependencies: Dict[str, Any],
        config: Config,
        logger: Logger
    ):
        self.base_dependencies = base_dependencies
        self.config = config
        self.logger = logger
        self._bot_instances: Dict[str, CryptoTradingBot] = {}

    async def create_bot_for_pair(self, symbol: str) -> CryptoTradingBot:
        """Create an isolated bot instance for a trading pair.

        Each pair gets:
        - Its own AnalysisEngine (initialized with its symbol)
        - Its own persistence directory
        - Its own vector memory (separate ChromaDB)
        """
        if symbol in self._bot_instances:
            # Reuse existing instance if already created
            return self._bot_instances[symbol]

        # Clone shared dependencies (non-symbol-specific)
        deps = self._clone_dependencies(symbol)

        # Create the bot
        bot = CryptoTradingBot(
            logger=self.logger,
            config=self.config,
            shutdown_manager=self.base_dependencies.get('shutdown_manager'),
            **deps
        )

        await bot.initialize()
        self._bot_instances[symbol] = bot
        return bot

    def _clone_dependencies(self, symbol: str) -> Dict[str, Any]:
        """Clone/create dependencies specific to a pair."""
        # Shared dependencies (read-only or thread-safe)
        shared = [
            'exchange_manager',
            'coingecko_api',
            'news_client',
            'market_api',
            'categories_api',
            'alternative_me_api',
            'cryptocompare_session',
            'model_manager',
            'discord_notifier',
            'keyboard_handler',
            'rag_engine',
        ]

        deps = {}
        for key in shared:
            if key in self.base_dependencies:
                deps[key] = self.base_dependencies[key]

        # Create pair-specific persistence
        safe_symbol = symbol.replace("/", "_").replace("-", "_")
        pair_data_dir = f"data/trading/pair_{safe_symbol}"
        persistence_path = Path(pair_data_dir)
        persistence_path.mkdir(parents=True, exist_ok=True)
        persistence = PersistenceManager(
            logger=self.logger,
            data_dir=str(persistence_path)
        )
        deps['persistence'] = persistence

        # Dependencies that must be pair-specific
        # will be initialized dynamically during bot.run()
        # via the initialize_for_symbol() call

        return deps

    async def cleanup_bot(self, symbol: str):
        """Clean up resources for a specific bot."""
        if symbol in self._bot_instances:
            bot = self._bot_instances.pop(symbol)
            await bot.shutdown()


# Import here to avoid circular imports
from src.managers.persistence_manager import PersistenceManager