"""Per-pair dashboard state reaches the portfolio and system routers."""
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from src.dashboard.dashboard_state import DashboardState
from src.dashboard.routers.portfolio import PortfolioRouter
from src.dashboard.routers.system import SystemRouter


def _config(tmp_path) -> SimpleNamespace:
    return SimpleNamespace(
        CRYPTO_PAIRS=["BTC/USDC", "ETH/USDC"],
        TIMEFRAME="1h",
        DATA_DIR=str(tmp_path / "data"),
        QUOTE_CURRENCY="USDC",
    )


@pytest.mark.asyncio
async def test_multi_pair_state_feeds_portfolio_and_system(tmp_path, monkeypatch):
    """Per-symbol updates populate pairs, not just the global slots."""
    monkeypatch.chdir(tmp_path)
    state = DashboardState()
    next_check = datetime.now(timezone.utc) + timedelta(minutes=7)

    await state.update_price("BTC/USDC", 100.0)
    await state.update_price("ETH/USDC", 200.0)
    await state.update_next_check("BTC/USDC", next_check)
    await state.update_next_check("ETH/USDC", next_check)
    await state.update_position("BTC/USDC", {"symbol": "BTC/USDC"})
    await state.update_analysis_complete("BTC/USDC")

    assert state.pairs["BTC/USDC"].current_price == 100.0
    assert state.pairs["ETH/USDC"].current_price == 200.0
    assert state.pairs["BTC/USDC"].current_position == {"symbol": "BTC/USDC"}
    assert state.pairs["BTC/USDC"].analysis_count == 1
    assert state.get_countdown_data("ETH/USDC")["seconds_remaining"] is not None

    config = _config(tmp_path)
    portfolio = PortfolioRouter(config, MagicMock(), state, None, None)
    summary = await portfolio.get_portfolio_summary()
    prices = {row["symbol"]: row["current_price"] for row in summary["pairs"]}
    assert prices == {"BTC/USDC": 100.0, "ETH/USDC": 200.0}

    system = SystemRouter(config, MagicMock(), state, None)
    status = await system.get_system_status()
    assert status["pairs"]["BTC/USDC"]["analysis_count"] == 1
    assert status["countdowns"]["BTC/USDC"]["seconds_remaining"] is not None
    assert status["countdowns"]["ETH/USDC"]["seconds_remaining"] is not None


@pytest.mark.asyncio
async def test_single_pair_global_price_still_reaches_portfolio(tmp_path, monkeypatch):
    """A global-only update keeps the single-pair portfolio price working."""
    monkeypatch.chdir(tmp_path)
    state = DashboardState()
    await state.update_price(150.0)

    config = SimpleNamespace(
        CRYPTO_PAIRS=["BTC/USDC"],
        TIMEFRAME="1h",
        DATA_DIR=str(tmp_path / "data"),
        QUOTE_CURRENCY="USDC",
    )
    portfolio = PortfolioRouter(config, MagicMock(), state, None, None)

    summary = await portfolio.get_portfolio_summary()

    assert summary["pairs"][0]["current_price"] == 150.0


@pytest.mark.asyncio
async def test_bot_publishes_updates_scoped_to_the_current_symbol():
    """The bot must send the pair symbol, not only the global slots."""
    from src.app import CryptoTradingBot

    bot = object.__new__(CryptoTradingBot)
    bot.current_symbol = "BTC/USDC"
    bot.dashboard_state = DashboardState()
    bot.logger = MagicMock()
    bot.persistence = MagicMock()
    bot.persistence.load_position.return_value = None

    await bot._publish_price(123.0)
    next_check = datetime.now(timezone.utc) + timedelta(minutes=3)
    await bot._publish_next_check(next_check)
    await bot._sync_pair_dashboard_state()

    assert bot.dashboard_state.current_price == 123.0
    assert bot.dashboard_state.pairs["BTC/USDC"].current_price == 123.0
    countdown = bot.dashboard_state.get_countdown_data("BTC/USDC")
    assert countdown["seconds_remaining"] is not None
    assert bot.dashboard_state.pairs["BTC/USDC"].analysis_count == 1
    assert bot.dashboard_state.pairs["BTC/USDC"].current_position is None
