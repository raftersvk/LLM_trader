"""Tests for position opening when the shared capital reservation fails."""
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.trading.data_models import RiskAssessment
from src.trading.trading_strategy import TradingStrategy


def _risk_assessment() -> RiskAssessment:
    return RiskAssessment(
        direction="LONG",
        entry_price=100.0,
        stop_loss=95.0,
        take_profit=110.0,
        quantity=40.0,
        size_pct=0.4,
        quote_amount=4000.0,
        entry_fee=3.0,
        sl_distance_pct=0.05,
        tp_distance_pct=0.10,
        rr_ratio=2.0,
        volatility_level="MEDIUM",
    )


def _strategy(reserve_result: bool):
    persistence = MagicMock()
    persistence.load_position.return_value = None
    persistence.async_save_position = AsyncMock()
    persistence.async_save_trade_decision = AsyncMock()

    risk_manager = MagicMock()
    risk_manager.calculate_entry_parameters.return_value = _risk_assessment()

    portfolio = MagicMock()
    portfolio.current_capital = 10000.0
    portfolio.get_available_capital.return_value = 10000.0
    portfolio.reserve_position.return_value = reserve_result

    position_factory = MagicMock()
    position_factory.create_position.side_effect = AssertionError("position must not be created")

    strategy = TradingStrategy(
        logger=MagicMock(),
        persistence=persistence,
        brain_service=MagicMock(),
        statistics_service=MagicMock(),
        memory_service=MagicMock(),
        risk_manager=risk_manager,
        config=MagicMock(),
        position_factory=position_factory,
        portfolio_capital_service=portfolio,
    )
    return strategy, persistence, position_factory


@pytest.mark.asyncio
async def test_open_position_aborts_when_reservation_fails():
    """A lost reservation race must not create or persist a position."""
    strategy, persistence, position_factory = _strategy(reserve_result=False)

    decision = await strategy._open_new_position(
        signal="BUY",
        confidence="HIGH",
        stop_loss=95.0,
        take_profit=110.0,
        position_size=0.4,
        current_price=100.0,
        symbol="BTC/USDT",
        reasoning="test",
    )

    assert decision is None
    assert strategy.current_position is None
    position_factory.create_position.assert_not_called()
    persistence.async_save_position.assert_not_awaited()
    persistence.async_save_trade_decision.assert_not_awaited()
