"""Tests for quote-currency P&L conversion used by portfolio capital release."""
from datetime import datetime, timezone

import pytest

from src.trading.data_models import Position
from src.trading.trading_strategy import TradingStrategy


def _position(direction: str, entry_price: float, size: float, entry_fee: float = 0.0) -> Position:
    return Position(
        entry_price=entry_price,
        stop_loss=entry_price * 0.9,
        take_profit=entry_price * 1.1,
        size=size,
        entry_time=datetime.now(timezone.utc),
        confidence="HIGH",
        direction=direction,
        symbol="BTC/USDT",
        entry_fee=entry_fee,
    )


def test_long_profit_is_converted_from_percent_to_quote():
    """A +5% LONG on a 10000 allocation must release ~500, not ~5."""
    position = _position("LONG", entry_price=100.0, size=100.0)

    net_pnl = TradingStrategy._calculate_net_pnl_quote(position, current_price=105.0, closing_fee=0.0)

    assert net_pnl == pytest.approx(500.0)


def test_short_profit_and_loss_are_direction_aware():
    """SHORT P&L must invert the price delta."""
    position = _position("SHORT", entry_price=100.0, size=100.0)

    assert TradingStrategy._calculate_net_pnl_quote(position, 95.0, 0.0) == pytest.approx(500.0)
    assert TradingStrategy._calculate_net_pnl_quote(position, 105.0, 0.0) == pytest.approx(-500.0)


def test_fees_are_deducted_from_quote_pnl():
    """Both entry and closing fees reduce the realised quote amount."""
    position = _position("LONG", entry_price=100.0, size=100.0, entry_fee=7.5)

    net_pnl = TradingStrategy._calculate_net_pnl_quote(position, 105.0, closing_fee=7.875)

    assert net_pnl == pytest.approx(500.0 - 7.5 - 7.875)
