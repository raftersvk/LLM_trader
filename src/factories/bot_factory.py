"""Factory for creating bot instances per trading pair."""

import chromadb
from pathlib import Path
from typing import Dict, Any
from sentence_transformers import SentenceTransformer
from src.app import CryptoTradingBot
from src.config.loader import Config
from src.logger.logger import Logger
from src.managers.persistence_manager import PersistenceManager


def _get_best_device():
    import torch
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


class BotDependencyFactory:
    """Creates isolated bot dependencies for each trading pair."""

    def __init__(
        self,
        base_dependencies: Dict[str, Any],
        config: Config,
        logger: Logger,
        engine_registry=None,
        vector_memory_registry=None,
    ):
        self.base_dependencies = base_dependencies
        self.config = config
        self.logger = logger
        self._bot_instances: Dict[str, CryptoTradingBot] = {}
        self.engine_registry = engine_registry
        self.vector_memory_registry = vector_memory_registry

        from src.trading.portfolio_capital import PortfolioCapitalService
        self.portfolio_capital_service = PortfolioCapitalService(
            logger, initial_capital=config.DEMO_QUOTE_CAPITAL
        )

    async def create_bot_for_pair(self, symbol: str) -> CryptoTradingBot:
        """Create an isolated bot instance for a trading pair.

        Each pair gets:
        - Its own AnalysisEngine (initialized with its symbol)
        - Its own persistence directory
        - Its own vector memory (separate ChromaDB)
        """
        self.logger.info("BotFactory: creating bot for %s", symbol)
        if symbol in self._bot_instances:
            self.logger.info("BotFactory: reusing existing bot for %s", symbol)
            return self._bot_instances[symbol]

        self.logger.debug("BotFactory: cloning shared dependencies for %s", symbol)
        deps = self._clone_dependencies(symbol)
        self.logger.debug("BotFactory: creating pair-specific services for %s", symbol)
        pair_services = self._create_pair_services(symbol)

        deps['persistence'] = pair_services['persistence']
        deps['brain_service'] = pair_services['brain_service']
        deps['statistics_service'] = pair_services['statistics_service']
        deps['memory_service'] = pair_services['memory_service']
        deps['trading_strategy'] = pair_services['strategy']

        self.logger.debug("BotFactory: instantiating CryptoTradingBot for %s", symbol)
        bot_logger = self.logger.getChild(symbol.replace("/", "_"))
        bot_logger.set_symbol(symbol)
        bot = CryptoTradingBot(
            logger=bot_logger,
            config=self.config,
            shutdown_manager=self.base_dependencies.get('shutdown_manager'),
            exchange_manager=deps.get('exchange_manager'),
            market_analyzer=deps.get('market_analyzer'),
            trading_strategy=deps['trading_strategy'],
            discord_notifier=deps.get('discord_notifier'),
            keyboard_handler=deps.get('keyboard_handler'),
            rag_engine=deps.get('rag_engine'),
            coingecko_api=deps.get('coingecko_api'),
            news_client=deps.get('news_client'),
            market_api=deps.get('market_api'),
            categories_api=deps.get('categories_api'),
            alternative_me_api=deps.get('alternative_me_api'),
            cryptocompare_session=deps.get('cryptocompare_session'),
            persistence=deps['persistence'],
            model_manager=deps.get('model_manager'),
            brain_service=deps['brain_service'],
            statistics_service=deps['statistics_service'],
            memory_service=deps['memory_service'],
            discord_task=None,
        )

        self.logger.debug("BotFactory: calling bot.initialize() for %s", symbol)
        await bot.initialize()
        self._bot_instances[symbol] = bot

        if self.engine_registry and hasattr(bot, 'market_analyzer') and bot.market_analyzer:
            self.engine_registry.register(symbol, bot.market_analyzer)
        if self.vector_memory_registry and 'vector_memory' in pair_services:
            self.vector_memory_registry.register(symbol, pair_services['vector_memory'])

        self.logger.info("BotFactory: bot for %s ready", symbol)
        return bot

    def _clone_dependencies(self, symbol: str) -> Dict[str, Any]:
        """Clone shared (non-symbol-specific) dependencies."""
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
            'parser',
        ]

        deps = {}
        for key in shared:
            if key in self.base_dependencies:
                deps[key] = self.base_dependencies[key]

        market_analyzer_factory = self.base_dependencies.get('market_analyzer_factory')
        if market_analyzer_factory:
            deps['market_analyzer'] = market_analyzer_factory(symbol)

        return deps

    def _create_pair_services(self, symbol: str) -> Dict[str, Any]:
        """Create services that must be pair-specific."""
        safe_symbol = symbol.replace("/", "_").replace("-", "_")

        pair_data_dir = f"data/trading/pair_{safe_symbol}"
        persistence_path = Path(pair_data_dir)
        persistence_path.mkdir(parents=True, exist_ok=True)
        self.logger.debug("BotFactory: persistence for %s at %s", symbol, persistence_path)
        persistence = PersistenceManager(
            logger=self.logger,
            data_dir=str(persistence_path)
        )

        brain_path = f"data/trading/brain_{safe_symbol}_{self.config.TIMEFRAME}"
        Path(brain_path).mkdir(parents=True, exist_ok=True)
        chroma_client = chromadb.PersistentClient(path=brain_path)
        embed_device = _get_best_device()
        embedding_model = SentenceTransformer("BAAI/bge-small-en-v1.5", device=embed_device)
        from src.trading.vector_memory import VectorMemoryService
        vector_memory = VectorMemoryService(self.logger, chroma_client, embedding_model=embedding_model)

        from src.trading.brain import TradingBrainService
        brain_service = TradingBrainService(self.logger, persistence, vector_memory)

        from src.trading.memory import TradingMemoryService
        memory_service = TradingMemoryService(self.logger, persistence, max_memory=10, vector_memory=vector_memory)

        from src.trading.statistics import TradingStatisticsService
        statistics_service = TradingStatisticsService(self.logger, persistence)

        from src.managers.risk_manager import RiskManager
        risk_manager = RiskManager(self.logger, self.config)

        from src.trading.position_extractor import PositionExtractor
        from src.factories.position_factory import PositionFactory
        from src.trading.trading_strategy import TradingStrategy
        parser = self.base_dependencies.get('parser')
        strategy = TradingStrategy(
            self.logger, persistence, brain_service, statistics_service, memory_service,
            risk_manager, self.config, PositionExtractor(self.logger, parser), PositionFactory(self.logger),
            portfolio_capital_service=self.portfolio_capital_service
        )

        return {
            'persistence': persistence,
            'brain_service': brain_service,
            'memory_service': memory_service,
            'statistics_service': statistics_service,
            'strategy': strategy,
            'risk_manager': risk_manager,
            'vector_memory': vector_memory,
        }

    async def cleanup_bot(self, symbol: str):
        """Clean up resources for a specific bot."""
        if symbol in self._bot_instances:
            bot = self._bot_instances.pop(symbol)
            await bot.shutdown()