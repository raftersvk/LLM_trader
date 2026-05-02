"""Manages concurrent trading analysis across multiple pairs."""

import asyncio
from typing import Dict, List, Optional
from dataclasses import dataclass, field
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
        async with self._lock:
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
        async with self._lock:
            if symbol not in self.pair_contexts:
                await self.initialize_pair(symbol)
            
            context = self.pair_contexts[symbol]
            if context.is_running and context.bot_task and not context.bot_task.done():
                self.logger.warning("Pair %s already running", symbol)
                return False
            
            # Create a new independent bot for this pair
            bot = await self.dependencies_factory.create_bot_for_pair(symbol)
            
            # Start the async task
            context.bot_task = asyncio.create_task(
                self._run_pair_loop(bot, symbol),
                name=f"TradingBot-{symbol.replace('/', '_')}"
            )
            context.is_running = True
            self.logger.info("Started trading loop for %s", symbol)
            return True
    
    async def _run_pair_loop(self, bot, symbol: str):
        """Execution loop for a specific pair."""
        try:
            await bot.run(symbol, bot.config.TIMEFRAME)
        except asyncio.CancelledError:
            self.logger.info("Trading loop cancelled for %s", symbol)
            raise
        except Exception as e:
            self.logger.error("Error in trading loop for %s: %s", symbol, e)
            async with self._lock:
                if symbol in self.pair_contexts:
                    self.pair_contexts[symbol].last_error = str(e)
                    self.pair_contexts[symbol].is_running = False
        finally:
            async with self._lock:
                if symbol in self.pair_contexts:
                    self.pair_contexts[symbol].is_running = False
    
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
        self.logger.info("Starting %d pairs: %s", len(pairs), ", ".join(pairs))
        
        tasks = []
        for symbol in pairs:
            tasks.append(self.start_pair(symbol))
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        successful = sum(1 for r in results if r is True)
        self.logger.info("Started %d/%d pairs successfully", successful, len(pairs))
        return results
    
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