"""Router for performance tracking and statistics."""
import asyncio
import json
from pathlib import Path
from typing import Dict, Any, Optional

from fastapi import APIRouter

from src.trading.statistics_calculator import StatisticsCalculator, TradingStatistics


class PerformanceRouter:
    """Handles performance history and statistics endpoints."""
    def __init__(self, config, logger, dashboard_state):
        self.router = APIRouter(prefix="/api/performance", tags=["performance"])
        self.config = config
        self.logger = logger
        self.dashboard_state = dashboard_state

        self.router.add_api_route("/history", self.get_performance_history, methods=["GET"])
        self.router.add_api_route("/stats", self.get_statistics, methods=["GET"])
        # New endpoint for consolidated multi-pair statistics
        self.router.add_api_route("/stats/all", self.get_consolidated_statistics, methods=["GET"])

    def _get_persistence_for_symbol(self, symbol: Optional[str]) -> Any:
        """Get a PersistenceManager instance for a specific symbol or the primary one."""
        from src.managers.persistence_manager import PersistenceManager
        
        if symbol:
            safe_symbol = symbol.replace("/", "_").replace("-", "_")
            data_dir = Path("data/trading/pair_{}".format(safe_symbol))
        else:
            # Default to the primary pair for backward compatibility
            primary_symbol = self.config.CRYPTO_PAIRS[0] if self.config.CRYPTO_PAIRS else self.config.CRYPTO_PAIR
            safe_symbol = primary_symbol.replace("/", "_").replace("-", "_")
            data_dir = Path("data/trading/pair_{}".format(safe_symbol))
        
        return PersistenceManager(logger=self.logger, data_dir=str(data_dir))

    def _load_json_file(self, file_path: Path) -> Dict[str, Any]:
        """Load JSON from a file synchronously."""
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _default_statistics(self) -> Dict[str, Any]:
        """Return a default statistics payload seeded from configured capital."""
        initial_capital = getattr(self.config, "DEMO_QUOTE_CAPITAL", 10000.0)
        return TradingStatistics(
            initial_capital=initial_capital,
            current_capital=initial_capital,
        ).to_dict()

    def _process_trade_history(self, trades: list, stats: Dict[str, Any]) -> list:
        """Process trade history into an equity curve synchronously."""
        equity_curve = []
        initial_capital = stats.get("initial_capital", 10000.0)
        running_capital = initial_capital
        # pylint: disable=protected-access
        closed_trades = StatisticsCalculator._extract_closed_trades(trades)
        closed_trade_idx = 0
        open_position = None
        for trade in trades:
            ts = trade.get("timestamp")
            action = trade.get("action", "").upper()
            if action in ("BUY", "SELL"):
                open_position = trade
                equity_curve.append({
                    "time": ts,
                    "value": round(running_capital, 2),
                    "action": action,
                    "price": trade.get("price")
                })
            elif action in ("CLOSE", "CLOSE_LONG", "CLOSE_SHORT") and open_position:
                if closed_trade_idx < len(closed_trades):
                    running_capital += closed_trades[closed_trade_idx].pnl_quote
                    closed_trade_idx += 1
                equity_curve.append({
                    "time": ts,
                    "value": round(running_capital, 2),
                    "action": action,
                    "price": trade.get("price")
                })
                open_position = None
        return equity_curve

    async def get_performance_history(self, symbol: Optional[str] = None) -> Dict[str, Any]:
        """Get historical performance data for the chart, optionally for a specific symbol."""
        cached = self.dashboard_state.get_cached("performance_history", ttl_seconds=60.0)
        if cached:
            return cached
        try:
            data_dir = self.config.DATA_DIR
        except AttributeError:
            data_dir = "data"
        
        persistence = self._get_persistence_for_symbol(symbol)
        trade_history_file = persistence.data_dir / "trade_history.json"
        stats_file = persistence.data_dir / "statistics.json"
        equity_curve = []
        stats = self._default_statistics()
        if stats_file.exists():
            try:
                stats = await asyncio.to_thread(self._load_json_file, stats_file)
            except Exception:
                self.logger.error("Failed to load stats file for symbol %s", symbol, exc_info=True)
        if trade_history_file.exists():
            try:
                trades = await asyncio.to_thread(self._load_json_file, trade_history_file)
                equity_curve = await asyncio.to_thread(self._process_trade_history, trades, stats)
            except Exception:
                self.logger.error("Failed to process trade history for symbol %s", symbol, exc_info=True)
                return {"error": "Failed to load trade history"}
        result = {
            "history": equity_curve,
            "stats": stats
        }
        self.dashboard_state.set_cached("performance_history", result)
        return result

    async def get_statistics(self, symbol: Optional[str] = None) -> Dict[str, Any]:
        """Get trading statistics summary for a specific symbol or the primary one."""
        cached = self.dashboard_state.get_cached("statistics", ttl_seconds=60.0)
        if cached:
            return cached
        persistence = self._get_persistence_for_symbol(symbol)
        stats_file = persistence.data_dir / "statistics.json"
        if stats_file.exists():
            try:
                result = await asyncio.to_thread(self._load_json_file, stats_file)
                self.dashboard_state.set_cached("statistics", result)
                return result
            except Exception:
                self.logger.error("Failed to load statistics for symbol %s", symbol, exc_info=True)
                return {"error": "Failed to load stats"}
        result = self._default_statistics()
        self.dashboard_state.set_cached("statistics", result)
        return result

    async def get_consolidated_statistics(self) -> Dict[str, Any]:
        """Get consolidated statistics across all configured pairs."""
        try:
            pairs = self.config.CRYPTO_PAIRS
            if not pairs:
                return {"error": "No pairs configured"}
            
            consolidated_stats = []
            total_pnl = 0.0
            total_trades = 0
            winning_trades = 0
            
            for symbol in pairs:
                persistence = self._get_persistence_for_symbol(symbol)
                stats_file = persistence.data_dir / "statistics.json"
                if stats_file.exists():
                    try:
                        stats = await asyncio.to_thread(self._load_json_file, stats_file)
                        # Accumulate for consolidated view
                        total_pnl += stats.get("total_pnl_quote", 0.0)
                        total_trades += stats.get("total_trades", 0)
                        winning_trades += stats.get("winning_trades", 0)
                        consolidated_stats.append({
                            "symbol": symbol,
                            "stats": stats
                        })
                    except Exception:
                        self.logger.error("Failed to load stats for symbol %s", symbol, exc_info=True)
                        consolidated_stats.append({
                            "symbol": symbol,
                            "error": "Failed to load statistics"
                        })
                else:
                    consolidated_stats.append({
                        "symbol": symbol,
                        "stats": self._default_statistics()
                    })
            
            win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0.0
            
            return {
                "pairs": consolidated_stats,
                "consolidated": {
                    "total_pnl_quote": total_pnl,
                    "total_trades": total_trades,
                    "winning_trades": winning_trades,
                    "win_rate": win_rate
                }
            }
        except Exception:
            self.logger.error("Failed to generate consolidated statistics", exc_info=True)
            return {"error": "Failed to generate consolidated statistics"}
