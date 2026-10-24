"""Shared dashboard state for real-time updates.

This module holds state that is updated by the trading bot and read by the dashboard.
It enables WebSocket broadcasts and API endpoints to share live data.

The ConnectionManager is injected by the composition root (start.py) — never
constructed here. The multi-pair fields (``pairs`` and ``PairState``) are the
fork's multi-pair dashboard extension; the brain-rebuild lifecycle is upstream's.
"""
import asyncio
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from src.dashboard.routers.ws_router import ConnectionManager


@dataclass
class PairState:
    """State for a specific trading pair (multi-pair fork extension)."""
    symbol: str
    next_check_utc: Optional[datetime] = None
    last_analysis_time: Optional[datetime] = None
    current_position: Optional[Dict[str, Any]] = None
    current_price: Optional[float] = None
    analysis_count: int = 0


@dataclass
class DashboardState:
    """Shared state between bot and dashboard."""
    # pylint: disable=too-many-instance-attributes
    # Global (single-pair) state kept for upstream routers.
    next_check_utc: datetime | None = None
    current_price: float | None = None
    brain_rebuild_status: str = "idle"
    brain_rebuild_started_at: datetime | None = None
    brain_rebuild_completed_at: datetime | None = None
    brain_rebuild_message: str = ""
    brain_rebuild_sequence: int = 0

    # Fork multi-pair state.
    bot_status: str = "running"
    api_costs: Dict[str, float] = field(
        default_factory=lambda: {"openrouter": 0.0, "google": 0.0}
    )
    last_request_cost: Optional[float] = None
    pairs: Dict[str, PairState] = field(default_factory=dict)

    _cache: dict[str, Any] = field(default_factory=dict)
    cache_timestamps: dict[str, float] = field(default_factory=dict)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False)
    connection_manager: ConnectionManager | None = field(default=None, repr=False)

    # ------------------------------------------------------------------
    # Price / countdown updates (accept both single-pair and per-pair calls)
    # ------------------------------------------------------------------
    async def update_price(self, symbol_or_price, price: float | None = None) -> None:
        """Update current price; fork callers pass (symbol, price)."""
        if isinstance(symbol_or_price, str) and price is not None:
            async with self._lock:
                if symbol_or_price not in self.pairs:
                    self.pairs[symbol_or_price] = PairState(symbol=symbol_or_price)
                self.pairs[symbol_or_price].current_price = price
            return
        async with self._lock:
            self.current_price = float(symbol_or_price)

    async def update_next_check(self, symbol_or_time, next_time: datetime | None = None) -> None:
        """Update next check time; fork callers pass (symbol, next_time)."""
        if isinstance(symbol_or_time, str) and next_time is not None:
            async with self._lock:
                if symbol_or_time not in self.pairs:
                    self.pairs[symbol_or_time] = PairState(symbol=symbol_or_time)
                self.pairs[symbol_or_time].next_check_utc = next_time
            await self.broadcast(
                {"type": "countdown", "symbol": symbol_or_time, "next_check_utc": next_time.isoformat()}
            )
            return
        async with self._lock:
            self.next_check_utc = symbol_or_time
        await self.broadcast({"type": "countdown", "next_check_utc": symbol_or_time.isoformat()})

    async def broadcast(self, data: dict[str, Any]) -> None:
        """Broadcast data to all connected WebSocket clients.

        No-op when no ConnectionManager was injected (tests, headless runs).
        """
        if self.connection_manager:
            await self.connection_manager.broadcast(data)

    async def _broadcast(self, data: Dict[str, Any]) -> None:
        """Backwards-compatible alias used by the fork's multi-pair methods."""
        await self.broadcast(data)

    def get_countdown_data(self, symbol: Optional[str] = None) -> Dict[str, Any]:
        """Get current countdown state for REST API, optionally for one symbol."""
        if symbol or self.pairs:
            target_symbol = symbol or (next(iter(self.pairs)) if self.pairs else None)
            if not target_symbol or target_symbol not in self.pairs:
                return {"next_check_utc": None, "seconds_remaining": None}
            state = self.pairs[target_symbol]
            if not state.next_check_utc:
                return {"next_check_utc": None, "seconds_remaining": None}
            now = datetime.now(timezone.utc)
            next_check = (
                state.next_check_utc.replace(tzinfo=timezone.utc)
                if state.next_check_utc.tzinfo is None
                else state.next_check_utc
            )
            remaining = (next_check - now).total_seconds()
            return {
                "next_check_utc": state.next_check_utc.isoformat(),
                "seconds_remaining": max(0, int(remaining)),
            }
        if not self.next_check_utc:
            return {"next_check_utc": None, "seconds_remaining": None}
        now = datetime.now(timezone.utc)
        next_check = (
            self.next_check_utc.replace(tzinfo=timezone.utc)
            if self.next_check_utc.tzinfo is None
            else self.next_check_utc
        )
        remaining = (next_check - now).total_seconds()
        return {
            "next_check_utc": self.next_check_utc.isoformat(),
            "seconds_remaining": max(0, int(remaining)),
        }

    # ------------------------------------------------------------------
    # Fork multi-pair helpers
    # ------------------------------------------------------------------
    async def update_position(self, symbol: str, position_data: Optional[Dict[str, Any]]) -> None:
        """Update current position for a pair and broadcast to clients."""
        async with self._lock:
            if symbol not in self.pairs:
                self.pairs[symbol] = PairState(symbol=symbol)
            self.pairs[symbol].current_position = position_data
        await self.broadcast({"type": "position", "symbol": symbol, "data": position_data})

    async def update_analysis_complete(self, symbol: str) -> None:
        """Signal that analysis has completed for a specific pair."""
        async with self._lock:
            if symbol not in self.pairs:
                self.pairs[symbol] = PairState(symbol=symbol)
            self.pairs[symbol].last_analysis_time = datetime.now(timezone.utc)
            self.pairs[symbol].analysis_count += 1
        await self.broadcast({"type": "analysis_complete", "symbol": symbol})

    async def update_api_costs(self, provider: str, cost: float) -> None:
        """Update API costs for a provider and broadcast to clients."""
        async with self._lock:
            if provider in self.api_costs:
                self.api_costs[provider] += cost
            self.last_request_cost = cost
        await self.broadcast(
            {"type": "cost_update", "provider": provider, "cost": cost, "total": self.api_costs}
        )

    async def reset_api_costs(self) -> None:
        """Reset all API costs to zero."""
        async with self._lock:
            self.api_costs = {"openrouter": 0.0, "google": 0.0}
            self.last_request_cost = None
        await self.broadcast({"type": "cost_reset", "total": self.api_costs})

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
            "current_position": state.current_position,
        }

    def get_all_pairs_state(self) -> Dict[str, Any]:
        """Get state for all pairs."""
        return {symbol: self.get_pair_state(symbol) for symbol in self.pairs}

    # ------------------------------------------------------------------
    # Cache helpers (shared by both sides)
    # ------------------------------------------------------------------
    def get_cached(self, key: str, ttl_seconds: float = 30.0) -> Any | None:
        """Retrieve a cached value if it is within TTL."""
        cached_time = self.cache_timestamps.get(key, 0)
        if time.time() - cached_time > ttl_seconds:
            return None
        return self._cache.get(key)

    def set_cached(self, key: str, value: Any) -> None:
        """Store a value in cache with current timestamp, enforcing max size."""
        if len(self._cache) >= 100 and key not in self._cache:  # noqa: SIM102
            if self.cache_timestamps:
                oldest_key = min(
                    self.cache_timestamps, key=lambda cache_key: self.cache_timestamps[cache_key]
                )
                self.invalidate_cache(oldest_key)
        self._cache[key] = value
        self.cache_timestamps[key] = time.time()

    def invalidate_cache(self, key: str) -> None:
        """Remove cached value."""
        self._cache.pop(key, None)
        self.cache_timestamps.pop(key, None)

    def invalidate_cache_prefix(self, prefix: str) -> None:
        """Remove cached values whose keys start with prefix."""
        for key in list(self._cache):
            if key.startswith(prefix):
                self.invalidate_cache(key)

    def invalidate_brain_caches(self) -> None:
        """Remove dashboard caches affected by trade close and brain learning."""
        for key in (
            "brain_status", "position", "rules", "performance_history",
            "statistics", "decision_summary",
        ):
            self.invalidate_cache(key)
        for prefix in ("vectors_", "decision_summary"):
            self.invalidate_cache_prefix(prefix)

    # ------------------------------------------------------------------
    # Brain rebuild lifecycle (upstream)
    # ------------------------------------------------------------------
    def get_brain_lifecycle(self) -> dict[str, Any]:
        """Return current brain rebuild lifecycle state for API responses."""
        return {
            "status": self.brain_rebuild_status,
            "started_at": self.brain_rebuild_started_at.isoformat() if self.brain_rebuild_started_at else None,
            "completed_at": self.brain_rebuild_completed_at.isoformat() if self.brain_rebuild_completed_at else None,
            "message": self.brain_rebuild_message,
            "sequence": self.brain_rebuild_sequence,
        }

    async def mark_brain_rebuild_started(self, message: str = "Brain learning from closed trade") -> None:
        """Mark brain rebuild as started and notify dashboard clients."""
        async with self._lock:
            self.brain_rebuild_sequence += 1
            self.brain_rebuild_status = "updating"
            self.brain_rebuild_started_at = datetime.now(timezone.utc)
            self.brain_rebuild_completed_at = None
            self.brain_rebuild_message = message
            lifecycle = self.get_brain_lifecycle()
        self.invalidate_brain_caches()
        await self.broadcast({"type": "brain_rebuild_started", "data": lifecycle})

    async def mark_brain_rebuild_completed(self, message: str = "Brain state rebuilt") -> None:
        """Mark brain rebuild as complete, clear stale caches, and notify clients."""
        async with self._lock:
            self.brain_rebuild_status = "rebuilt"
            self.brain_rebuild_completed_at = datetime.now(timezone.utc)
            self.brain_rebuild_message = message
            lifecycle = self.get_brain_lifecycle()
        self.invalidate_brain_caches()
        await self.broadcast({"type": "brain_rebuild_completed", "data": lifecycle})

    async def mark_brain_rebuild_failed(self, message: str) -> None:
        """Mark brain rebuild as failed and notify dashboard clients."""
        async with self._lock:
            self.brain_rebuild_status = "error"
            self.brain_rebuild_completed_at = datetime.now(timezone.utc)
            self.brain_rebuild_message = message
            lifecycle = self.get_brain_lifecycle()
        self.invalidate_brain_caches()
        await self.broadcast({"type": "brain_rebuild_failed", "data": lifecycle})

    async def broadcast_brain_state_updated(self, message: str = "Brain state refreshed") -> None:
        """Notify clients that brain-bound panels should refresh."""
        await self.broadcast({
            "type": "brain_state_updated",
            "data": {"message": message, "lifecycle": self.get_brain_lifecycle()},
        })


dashboard_state = DashboardState()
