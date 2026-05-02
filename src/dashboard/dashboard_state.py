"""Shared dashboard state for real-time updates.

This module holds state that is updated by the trading bot and read by the dashboard.
It enables WebSocket broadcasts and API endpoints to share live data.
"""
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional, Dict, Any
import asyncio
import time
from src.dashboard.routers.ws_router import broadcast


@dataclass
class PairState:
    """State for a specific trading pair."""
    symbol: str
    next_check_utc: Optional[datetime] = None
    last_analysis_time: Optional[datetime] = None
    current_position: Optional[Dict[str, Any]] = None
    current_price: Optional[float] = None
    analysis_count: int = 0


@dataclass
class DashboardState:
    """Shared state between bot and dashboard - now supports multiple pairs."""
    # Global state
    bot_status: str = "running"
    api_costs: Dict[str, float] = field(default_factory=lambda: {"openrouter": 0.0, "google": 0.0})
    last_request_cost: Optional[float] = None

    # Per-pair state
    pairs: Dict[str, PairState] = field(default_factory=dict)

    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False)
    _cache: dict[str, Any] = field(default_factory=dict)
    _cache_timestamps: dict[str, float] = field(default_factory=dict)

    async def update_price(self, symbol: str, price: float) -> None:
        """Update current price for a specific pair (no broadcast to avoid spam)."""
        async with self._lock:
            if symbol not in self.pairs:
                self.pairs[symbol] = PairState(symbol=symbol)
            self.pairs[symbol].current_price = price

    async def update_next_check(self, symbol: str, next_time: datetime) -> None:
        """Update next check time for a pair and broadcast to clients."""
        async with self._lock:
            if symbol not in self.pairs:
                self.pairs[symbol] = PairState(symbol=symbol)
            self.pairs[symbol].next_check_utc = next_time
        await self._broadcast({"type": "countdown", "symbol": symbol, "next_check_utc": next_time.isoformat()})

    async def update_position(self, symbol: str, position_data: Optional[Dict[str, Any]]) -> None:
        """Update current position for a pair and broadcast to clients."""
        async with self._lock:
            if symbol not in self.pairs:
                self.pairs[symbol] = PairState(symbol=symbol)
            self.pairs[symbol].current_position = position_data
        await self._broadcast({"type": "position", "symbol": symbol, "data": position_data})

    async def update_analysis_complete(self, symbol: str) -> None:
        """Signal that analysis has completed for a specific pair."""
        async with self._lock:
            if symbol not in self.pairs:
                self.pairs[symbol] = PairState(symbol=symbol)
            self.pairs[symbol].last_analysis_time = datetime.now(timezone.utc)
            self.pairs[symbol].analysis_count += 1
        await self._broadcast({"type": "analysis_complete", "symbol": symbol})

    async def update_api_costs(self, provider: str, cost: float) -> None:
        """Update API costs for a provider and broadcast to clients."""
        async with self._lock:
            if provider in self.api_costs:
                self.api_costs[provider] += cost
            self.last_request_cost = cost
        await self._broadcast({"type": "cost_update", "provider": provider, "cost": cost, "total": self.api_costs})

    async def reset_api_costs(self) -> None:
        """Reset all API costs to zero."""
        async with self._lock:
            self.api_costs = {"openrouter": 0.0, "google": 0.0}
            self.last_request_cost = None
        await self._broadcast({"type": "cost_reset", "total": self.api_costs})

    async def _broadcast(self, data: Dict[str, Any]) -> None:
        """Broadcast data to all connected WebSocket clients."""
        await broadcast(data)

    def get_pair_state(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get state for a specific pair."""
        if symbol not in self.pairs:
            return None

        state = self.pairs[symbol]
        return {
            "symbol": symbol,
            "next_check_utc": state.next_check_utc.isoformat() if state.next_check_utc else None,
            "last_analysis_time": state.last_analysis_time.isoformat() if state.last_analysis_time else None,
            "current_price": state.current_price,
            "analysis_count": state.analysis_count,
            "current_position": state.current_position
        }

    def get_all_pairs_state(self) -> Dict[str, Any]:
        """Get state for all pairs."""
        return {
            symbol: self.get_pair_state(symbol)
            for symbol in self.pairs
        }

    def get_countdown_data(self, symbol: Optional[str] = None) -> Dict[str, Any]:
        """Get current countdown state for REST API, optionally for a specific symbol."""
        target_symbol = symbol or (next(iter(self.pairs)) if self.pairs else None)
        if not target_symbol or target_symbol not in self.pairs:
            return {"next_check_utc": None, "seconds_remaining": None}
        
        state = self.pairs[target_symbol]
        if not state.next_check_utc:
            return {"next_check_utc": None, "seconds_remaining": None}
        now = datetime.now(timezone.utc)
        remaining = (state.next_check_utc.replace(tzinfo=timezone.utc) 
                     if state.next_check_utc.tzinfo is None 
                     else state.next_check_utc - now).total_seconds()
        return {
            "next_check_utc": state.next_check_utc.isoformat(),
            "seconds_remaining": max(0, int(remaining))
        }

    def get_cached(self, key: str, ttl_seconds: float = 30.0) -> Optional[Any]:
        """Retrieve a cached value if it is within TTL."""
        cached_time = self._cache_timestamps.get(key, 0)
        if time.time() - cached_time > ttl_seconds:
            return None
        return self._cache.get(key)

    def set_cached(self, key: str, value: Any) -> None:
        """Store a value in cache with current timestamp, enforcing max size."""
        if len(self._cache) >= 100 and key not in self._cache:
            if self._cache_timestamps:
                oldest_key = min(self._cache_timestamps, key=self._cache_timestamps.get)
                self.invalidate_cache(oldest_key)
        self._cache[key] = value
        self._cache_timestamps[key] = time.time()

    def invalidate_cache(self, key: str) -> None:
        """Remove cached value."""
        self._cache.pop(key, None)
        self._cache_timestamps.pop(key, None)


dashboard_state = DashboardState()