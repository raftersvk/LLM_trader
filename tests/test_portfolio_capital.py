"""Tests for shared portfolio capital allocation bookkeeping."""
from unittest.mock import MagicMock

from src.trading.portfolio_capital import PortfolioCapitalService


def _service(initial_capital: float = 10000.0) -> PortfolioCapitalService:
    return PortfolioCapitalService(logger=MagicMock(), initial_capital=initial_capital)


def test_reserve_position_rejects_insufficient_capital():
    """A reservation larger than unallocated capital must fail."""
    service = _service()

    reserved = service.reserve_position("BTC/USDT", 11000.0, 100.0, 110.0, "LONG")

    assert reserved is False
    assert service.total_allocated == 0.0


def test_reserve_position_rejects_duplicate_symbol_without_overwriting():
    """Re-reserving the same symbol must not silently replace the allocation."""
    service = _service()
    assert service.reserve_position("BTC/USDT", 4000.0, 100.0, 40.0, "LONG") is True

    reserved = service.reserve_position("BTC/USDT", 3000.0, 120.0, 25.0, "LONG")

    assert reserved is False
    assert service.total_allocated == 4000.0
    assert service._state.positions["BTC/USDT"].quote_amount == 4000.0
