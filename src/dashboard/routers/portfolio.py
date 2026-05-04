"""Router for portfolio-level aggregated endpoints."""
import asyncio
import json
from pathlib import Path
from typing import Dict, Any, Optional

from fastapi import APIRouter


class PortfolioRouter:
    """Handles portfolio-level endpoints aggregating across all pairs."""
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    def __init__(self, config, logger, dashboard_state, engine_registry, vector_memory_registry):
        self.router = APIRouter(prefix="/api/portfolio", tags=["portfolio"])
        self.config = config
        self.logger = logger
        self.dashboard_state = dashboard_state
        self.engine_registry = engine_registry
        self.vector_memory_registry = vector_memory_registry

        self.router.add_api_route("/summary", self.get_portfolio_summary, methods=["GET"])

    def _load_json_file(self, file_path: Path) -> Optional[Dict[str, Any]]:
        """Load JSON from a file synchronously."""
        if not file_path.exists():
            return None
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _get_pair_data_dir(self, symbol: str) -> Path:
        """Get the data directory for a specific pair."""
        safe_symbol = symbol.replace("/", "_").replace("-", "_")
        return Path(f"data/trading/pair_{safe_symbol}")

    async def get_portfolio_summary(self) -> Dict[str, Any]:
        """Get aggregated portfolio KPIs across all configured pairs."""
        cached = self.dashboard_state.get_cached("portfolio_summary", ttl_seconds=30.0)
        if cached:
            return cached

        pairs = self.config.CRYPTO_PAIRS
        total_pnl = 0.0
        total_trades = 0
        winning_trades = 0
        active_positions = 0
        total_capital = 0.0
        initial_capital = 0.0
        pair_summaries = []

        for symbol in pairs:
            pair_dir = self._get_pair_data_dir(symbol)
            pair_stats = {
                "symbol": symbol,
                "current_price": None,
                "trend": "--",
                "action": "--",
                "confidence": "--",
                "position": None,
                "stats": None,
            }

            pair_state = self.dashboard_state.pairs.get(symbol)
            if pair_state:
                pair_stats["current_price"] = pair_state.current_price

            prev_response_file = pair_dir / "previous_response.json"
            prev_data = await asyncio.to_thread(self._load_json_file, prev_response_file)
            if prev_data:
                from src.dashboard.routers.brain import _extract_market_status
                status = _extract_market_status(prev_data, None)
                pair_stats["trend"] = status.get("trend", "--")
                pair_stats["action"] = status.get("action", "--")
                pair_stats["confidence"] = status.get("confidence", "--")

            stats_file = pair_dir / "statistics.json"
            stats = await asyncio.to_thread(self._load_json_file, stats_file)
            if stats:
                pair_stats["stats"] = stats
                total_pnl += stats.get("total_pnl_quote", 0.0)
                total_trades += stats.get("total_trades", 0)
                winning_trades += stats.get("winning_trades", 0)
                total_capital += stats.get("current_capital", 0.0)
                initial_capital += stats.get("initial_capital", 0.0)

            position_file = pair_dir / "positions.json"
            pos_data = await asyncio.to_thread(self._load_json_file, position_file)
            if pos_data:
                active_positions += 1
                pair_stats["position"] = pos_data

            pair_summaries.append(pair_stats)

        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0.0
        total_pnl_pct = ((total_capital - initial_capital) / initial_capital * 100) if initial_capital > 0 else 0.0

        result = {
            "total_pnl_quote": total_pnl,
            "total_pnl_pct": total_pnl_pct,
            "total_capital": total_capital,
            "initial_capital": initial_capital,
            "total_trades": total_trades,
            "winning_trades": winning_trades,
            "win_rate": win_rate,
            "active_positions": active_positions,
            "total_pairs": len(pairs),
            "pairs": pair_summaries,
        }
        self.dashboard_state.set_cached("portfolio_summary", result)
        return result
