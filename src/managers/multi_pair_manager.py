"""Manages concurrent trading analysis across multiple pairs."""

import asyncio
from typing import Dict, List, Optional
from dataclasses import dataclass
from src.logger.logger import Logger
from src.config.loader import Config


@dataclass
class PairTradingContext:
    """Trading context for a specific pair."""
    symbol: str
    timeframe: str
    bot_task: Optional[asyncio.Task] = None
    is_running: bool = False
    last_error: Optional[str] = None
    analysis_count: int = 0


class MultiPairManager:
    """Orchestrates multiple trading bots for different pairs."""
    
    def __init__(
        self,
        logger: Logger,
        config: Config,
        dependencies_factory  # Factory to create dependencies per pair
    ):
        self.logger = logger
        self.config = config
        self.dependencies_factory = dependencies_factory
        self.pair_contexts: Dict[str, PairTradingContext] = {}
        self._lock = asyncio.Lock()
        
    def get_pairs(self) -> List[str]:
        """Return the list of configured pairs."""
        return self.config.CRYPTO_PAIRS
    
    async def initialize_pair(self, symbol: str) -> PairTradingContext:
        """Initialize the context for a specific pair."""
        timeframe = self.config.TIMEFRAME
        context = PairTradingContext(
            symbol=symbol,
            timeframe=timeframe,
            is_running=False
        )
        self.pair_contexts[symbol] = context
        self.logger.info("Initialized pair context: %s (%s)", symbol, timeframe)
        return context
    
    async def start_pair(self, symbol: str) -> bool:
        """Start the trading loop for a pair."""
        self.logger.info("start_pair %s: checking context", symbol)
        async with self._lock:
            self.logger.info("start_pair %s: lock acquired", symbol)
            if symbol not in self.pair_contexts:
                await self.initialize_pair(symbol)

            context = self.pair_contexts[symbol]
            if context.is_running:
                self.logger.warning("Pair %s already running", symbol)
                return False
            context.is_running = True

        self.logger.info("start_pair %s: creating bot (outside lock)", symbol)
        self.logger.info("start_pair %s: calling create_bot_for_pair...", symbol)
        try:
            bot = await self.dependencies_factory.create_bot_for_pair(symbol)
        except Exception as e:
            self.logger.error("Failed to create bot for %s: %s", symbol, e, exc_info=True)
            async with self._lock:
                if symbol in self.pair_contexts:
                    self.pair_contexts[symbol].is_running = False
                    self.pair_contexts[symbol].last_error = str(e)
            return False
        self.logger.info("start_pair %s: create_bot_for_pair returned", symbol)

        async with self._lock:
            context = self.pair_contexts[symbol]
            # Re-verify under the lock: never replace a live trading loop.
            if context.bot_task and not context.bot_task.done():
                self.logger.warning("Pair %s already running", symbol)
                return False
            context.bot_task = asyncio.create_task(
                self._run_pair_loop(bot, symbol),
                name=f"TradingBot-{symbol.replace('/', '_')}"
            )
            self.logger.info("Pair %s task started (task_id=%s)", symbol, context.bot_task.get_name())
            return True
    
    async def _run_pair_loop(self, bot, symbol: str):
        """Execution loop for a specific pair."""
        self.logger.info("Pair %s loop starting", symbol)
        try:
            await bot.run(symbol, bot.config.TIMEFRAME)
        except asyncio.CancelledError:
            self.logger.info("Trading loop cancelled for %s", symbol)
            raise
        except Exception as e:
            self.logger.error("Error in trading loop for %s: %s", symbol, e, exc_info=True)
            async with self._lock:
                if symbol in self.pair_contexts:
                    self.pair_contexts[symbol].last_error = str(e)
                    self.pair_contexts[symbol].is_running = False
        finally:
            async with self._lock:
                if symbol in self.pair_contexts:
                    self.pair_contexts[symbol].is_running = False
            self.logger.info("Pair %s loop ended", symbol)
    
    async def stop_pair(self, symbol: str) -> bool:
        """Stop the loop for a pair."""
        async with self._lock:
            if symbol not in self.pair_contexts:
                return False
            
            context = self.pair_contexts[symbol]
            if context.bot_task and not context.bot_task.done():
                context.bot_task.cancel()
                try:
                    await asyncio.wait_for(context.bot_task, timeout=5.0)
                except asyncio.TimeoutError:
                    self.logger.warning("Timeout waiting for %s to stop", symbol)
            
            context.is_running = False
            self.logger.info("Stopped trading loop for %s", symbol)
            return True
    
    async def start_all_pairs(self):
        """Start all configured pairs."""
        pairs = self.get_pairs()
        self.logger.info("start_all_pairs: got %d pairs %s", len(pairs), pairs)
        self.logger.info("Starting %d pairs: %s", len(pairs), ", ".join(pairs))

        for symbol in pairs:
            self.logger.info("start_all_pairs: starting pair %s", symbol)
            await self.start_pair(symbol)
            self.logger.info("start_all_pairs: pair %s started", symbol)

        self.logger.info("All %d pairs started", len(pairs))
    
    async def stop_all_pairs(self):
        """Stop all running pairs."""
        tasks = []
        for symbol in self.pair_contexts:
            tasks.append(self.stop_pair(symbol))
        
        await asyncio.gather(*tasks, return_exceptions=True)
        self.logger.info("All pairs stopped")
    
    def get_status(self) -> Dict:
        """Return the status of all pairs."""
        status = {}
        for symbol, context in self.pair_contexts.items():
            status[symbol] = {
                "is_running": context.is_running,
                "timeframe": context.timeframe,
                "analysis_count": context.analysis_count,
                "last_error": context.last_error,
                "task_active": context.bot_task is not None and not context.bot_task.done() if context.bot_task else False
            }
        return status