"""Router for system monitoring and fleet status."""
from typing import Dict, Any

from fastapi import APIRouter


class SystemRouter:
    """Handles system monitoring endpoints for bot fleet status."""
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    def __init__(self, config, logger, dashboard_state, multi_pair_manager=None):
        self.router = APIRouter(prefix="/api/system", tags=["system"])
        self.config = config
        self.logger = logger
        self.dashboard_state = dashboard_state
        self.multi_pair_manager = multi_pair_manager

        self.router.add_api_route("/status", self.get_system_status, methods=["GET"])

    async def get_system_status(self) -> Dict[str, Any]:
        """Get system status for all pairs and the bot fleet."""
        cached = self.dashboard_state.get_cached("system_status", ttl_seconds=10.0)
        if cached:
            return cached

        pairs = self.config.CRYPTO_PAIRS
        pair_statuses = {}

        if self.multi_pair_manager:
            pair_statuses = self.multi_pair_manager.get_status()
        else:
            for symbol in pairs:
                pair_state = self.dashboard_state.pairs.get(symbol)
                pair_statuses[symbol] = {
                    "is_running": False,
                    "timeframe": self.config.TIMEFRAME,
                    "analysis_count": pair_state.analysis_count if pair_state else 0,
                    "last_error": None,
                    "task_active": False,
                }

        active_count = sum(1 for s in pair_statuses.values() if s.get("is_running"))

        countdowns = {}
        for symbol in pairs:
            countdowns[symbol] = self.dashboard_state.get_countdown_data(symbol)

        result = {
            "total_pairs": len(pairs),
            "active_pairs": active_count,
            "pairs": pair_statuses,
            "countdowns": countdowns,
            "api_costs": self.dashboard_state.api_costs,
        }
        self.dashboard_state.set_cached("system_status", result)
        return result
