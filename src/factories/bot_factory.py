"""Factory for creating bot instances per trading pair."""

import chromadb
from pathlib import Path
from typing import Dict, Any
from sentence_transformers import SentenceTransformer
from src.app import BotServices, CryptoTradingBot
from src.config.loader import Config
from src.logger.logger import Logger
from src.managers.persistence_manager import PersistenceManager
from src.trading import PositionStatusMonitor


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
        shared_deps = self._clone_dependencies(symbol)
        self.logger.debug("BotFactory: creating pair-specific services for %s", symbol)
        pair_services = self._create_pair_services(symbol)

        self.logger.debug("BotFactory: instantiating CryptoTradingBot for %s", symbol)
        bot_logger = self.logger.getChild(symbol.replace("/", "_"))
        bot_logger.set_symbol(symbol)

        pair_deps = dict(self.base_dependencies)
        pair_deps.update(shared_deps)
        pair_deps.update(pair_services)
        # _create_pair_services exposes the strategy as "strategy"; BotServices and
        # the position monitor read it as "trading_strategy".
        pair_deps["trading_strategy"] = pair_services["strategy"]
        pair_deps["market_analyzer"] = await self._create_engine_for_pair(symbol)

        def _create_position_monitor(bot: CryptoTradingBot) -> PositionStatusMonitor:
            try:
                reconcile_interval_seconds = float(
                    self.config.get_config("executor_api", "reconcile_interval_seconds", 120.0)
                )
            except (TypeError, ValueError):
                reconcile_interval_seconds = 120.0
            return PositionStatusMonitor(
                logger=bot_logger,
                config=self.config,
                persistence=pair_deps["persistence"],
                trading_strategy=pair_deps["trading_strategy"],
                exit_monitor=pair_deps["exit_monitor"],
                notifier=pair_deps.get("discord_notifier"),
                active_tasks=bot.active_tasks,
                is_running=lambda: bot.running,
                fetch_current_ticker=bot.fetch_current_ticker,
                interruptible_sleep=bot.interruptible_sleep,
                get_symbol=lambda: bot.current_symbol,
                manages_exits=lambda: not self.config.EXECUTOR_API_ENABLED,
                reconcile_position=pair_deps["trading_strategy"].reconcile_local_position,
                reconcile_interval_seconds=reconcile_interval_seconds,
                reference_exits_enabled=pair_deps["trading_strategy"].executor_query_available,
                reference_exit_check_seconds=self.config.REFERENCE_EXIT_CHECK_SECONDS,
                forward_local_close=bot.forward_local_exit_request,
            )

        bot = CryptoTradingBot(
            BotServices(
                logger=bot_logger,
                config=self.config,
                shutdown_manager=pair_deps.get("shutdown_manager"),
                position_monitor_factory=_create_position_monitor,
                exchange_manager=pair_deps.get("exchange_manager"),
                market_analyzer=pair_deps["market_analyzer"],
                trading_strategy=pair_deps["trading_strategy"],
                discord_notifier=pair_deps.get("discord_notifier"),
                keyboard_handler=pair_deps.get("keyboard_handler"),
                rag_engine=pair_deps.get("rag_engine"),
                coingecko_api=pair_deps.get("coingecko_api"),
                market_api=pair_deps.get("market_api"),
                alternative_me_api=pair_deps.get("alternative_me_api"),
                http_session=pair_deps.get("http_session"),
                persistence=pair_deps["persistence"],
                model_manager=pair_deps.get("model_manager"),
                brain_service=pair_deps["brain_service"],
                statistics_service=pair_deps["statistics_service"],
                memory_service=pair_deps["memory_service"],
                exit_monitor=pair_deps["exit_monitor"],
                sentiment_analyst=pair_deps.get("sentiment_analyst"),
                executor_handler=pair_deps.get("executor_handler"),
                ev_formatter=pair_deps.get("ev_formatter"),
                dashboard_state=pair_deps.get("dashboard_state"),
                discord_task=None,
            )
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

    async def _create_engine_for_pair(self, symbol: str):
        """Build a symbol-scoped AnalysisEngine sharing the provisioned collaborators."""
        base = self.base_dependencies.get("market_analyzer")
        if base is None:
            return None
        from src.analyzer.analysis_engine import AnalysisEngine

        engine = AnalysisEngine(
            self.logger.getChild(symbol.replace("/", "_")),
            base.rag_engine,
            base.model_manager,
            base.market_api,
            self.config,
            base.technical_calculator,
            base.pattern_analyzer,
            base.prompt_builder,
            base.data_collector,
            base.metrics_calculator,
            base.result_processor,
            base.chart_generator,
        )
        self.logger.debug("BotFactory: created AnalysisEngine for %s", symbol)
        return engine

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