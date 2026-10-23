"""Shared portfolio capital service for multi-pair trading.

Manages a unified capital pool across all trading pairs, ensuring
that each pair correctly calculates available funds based on all
open positions across the portfolio.
"""

import asyncio
from typing import Dict, Optional
from dataclasses import dataclass, field
from datetime import datetime

from src.logger.logger import Logger
from src.utils.data_utils import SerializableMixin


@dataclass(slots=True)
class PositionAllocation(SerializableMixin):
    """Represents an allocated position in the portfolio."""
    symbol: str
    quote_amount: float
    entry_price: float
    quantity: float
    direction: str
    opened_at: datetime = field(default_factory=datetime.now)


@dataclass(slots=True)
class PortfolioCapitalState(SerializableMixin):
    """Global portfolio capital state."""
    initial_capital: float = 10000.0
    current_capital: float = 10000.0
    total_realized_pnl: float = 0.0
    positions: Dict[str, PositionAllocation] = field(default_factory=dict)


class PortfolioCapitalService:
    """Service for managing shared portfolio capital across pairs.

    Responsibilities:
    - Track total portfolio capital
    - Track allocated funds (open positions) per symbol
    - Calculate available capital per pair
    - Reserve/release capital when positions open/close
    """

    def __init__(self, logger: Logger, initial_capital: float = 10000.0):
        """Initialize portfolio capital service.

        Args:
            logger: Logger instance
            initial_capital: Total initial capital for the portfolio
        """
        self.logger = logger
        self._initial_capital = initial_capital
        self._state = PortfolioCapitalState(initial_capital=initial_capital, current_capital=initial_capital)
        self._lock = asyncio.Lock()

    @property
    def initial_capital(self) -> float:
        return self._initial_capital

    @property
    def current_capital(self) -> float:
        return self._state.current_capital

    @property
    def total_allocated(self) -> float:
        """Get total allocated capital across all open positions."""
        return sum(pos.quote_amount for pos in self._state.positions.values())

    @property
    def total_unallocated(self) -> float:
        """Get unallocated capital (available for new positions)."""
        return self._state.current_capital - self.total_allocated

    def get_allocated_for_symbol(self, symbol: str) -> float:
        """Get allocated capital for a specific symbol."""
        if symbol in self._state.positions:
            return self._state.positions[symbol].quote_amount
        return 0.0

    def get_available_capital(self, symbol: str, pair_capital: float) -> float:
        """Calculate available capital for a specific pair.

        This accounts for:
        - Total portfolio capital (initial + realized P&L)
        - Already allocated positions across all pairs

        Args:
            symbol: Trading pair symbol
            pair_capital: Current capital tracked by this pair's statistics

        Returns:
            Available capital for this pair to use
        """
        total_allocated = self.total_allocated

        available = self._state.current_capital - total_allocated
        return max(0.0, available)

    def reserve_position(self, symbol: str, quote_amount: float, entry_price: float,
                        quantity: float, direction: str) -> bool:
        """Reserve capital for a new position.

        Args:
            symbol: Trading pair symbol
            quote_amount: Amount in quote currency
            entry_price: Entry price
            quantity: Base asset quantity
            direction: LONG or SHORT

        Returns:
            True if reservation succeeded, False if capital is unavailable or
            the symbol already has an allocation
        """
        if symbol in self._state.positions:
            self.logger.warning(
                "Cannot reserve $%s for %s: position already allocated",
                f"{quote_amount:,.2f}", symbol
            )
            return False

        available = self.total_unallocated
        if available < quote_amount:
            self.logger.warning(
                "Cannot reserve $%s for %s: only $%s available",
                f"{quote_amount:,.2f}", symbol, f"{available:,.2f}"
            )
            return False

        allocation = PositionAllocation(
            symbol=symbol,
            quote_amount=quote_amount,
            entry_price=entry_price,
            quantity=quantity,
            direction=direction
        )
        self._state.positions[symbol] = allocation
        self.logger.info(
            "Reserved $%s for %s %s position. Total allocated: $%s",
            f"{quote_amount:,.2f}", symbol, direction, f"{self.total_allocated:,.2f}"
        )
        return True

    def release_position(self, symbol: str, realized_pnl: float = 0.0) -> Optional[float]:
        """Release capital when a position closes.

        Args:
            symbol: Trading pair symbol
            realized_pnl: Realized P&L from the closed position

        Returns:
            The quote_amount that was released, or None if position not found
        """
        if symbol not in self._state.positions:
            self.logger.warning("Attempted to release non-existent position for %s", symbol)
            return None

        released = self._state.positions[symbol].quote_amount
        del self._state.positions[symbol]

        self._state.total_realized_pnl += realized_pnl
        self._state.current_capital = self._initial_capital + self._state.total_realized_pnl

        self.logger.info(
            "Released $%s from %s. Realized P&L: $%s, New capital: $%s",
            f"{released:,.2f}", symbol, f"{realized_pnl:,.2f}", f"{self._state.current_capital:,.2f}"
        )
        return released

    def update_realized_pnl(self, pnl: float) -> None:
        """Update current capital with realized P&L.

        Args:
            pnl: Realized P&L amount (can be negative)
        """
        self._state.total_realized_pnl += pnl
        self._state.current_capital = self._initial_capital + self._state.total_realized_pnl

    def get_state(self) -> PortfolioCapitalState:
        """Get current portfolio state."""
        return self._state

    def has_open_position(self, symbol: str) -> bool:
        """Check if there's an open position for a symbol."""
        return symbol in self._state.positions

    def get_allocation_summary(self) -> Dict[str, float]:
        """Get summary of all allocations."""
        return {symbol: pos.quote_amount for symbol, pos in self._state.positions.items()}
