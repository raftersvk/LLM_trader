"""Tests for the Kraken reference fee model."""

import pytest

from src.llm_focus.fees import (
    KRAKEN_FUTURES_TIER_1,
    KRAKEN_SPOT_TIER_1,
    KRAKEN_SPOT_TIER_2,
    KRAKEN_SPOT_TIER_3,
    KRAKEN_STABLECOIN_FX,
    SOURCE_DATE,
    SOURCE_URL,
    fee_sensitivity,
    get_fee_rate,
    round_trip_cost_pct,
)


class TestFeeConstants:
    def test_source_metadata(self):
        assert SOURCE_URL == "https://www.kraken.com/fr/features/fee-schedule"
        assert SOURCE_DATE == "2026-11-02"

    def test_spot_tiers(self):
        assert (KRAKEN_SPOT_TIER_1.maker_pct, KRAKEN_SPOT_TIER_1.taker_pct) == (0.40, 0.80)
        assert (KRAKEN_SPOT_TIER_2.maker_pct, KRAKEN_SPOT_TIER_2.taker_pct) == (0.30, 0.60)
        assert (KRAKEN_SPOT_TIER_3.maker_pct, KRAKEN_SPOT_TIER_3.taker_pct) == (0.22, 0.38)

    def test_stablecoin_and_futures(self):
        assert (KRAKEN_STABLECOIN_FX.maker_pct, KRAKEN_STABLECOIN_FX.taker_pct) == (0.20, 0.20)
        assert (KRAKEN_FUTURES_TIER_1.maker_pct, KRAKEN_FUTURES_TIER_1.taker_pct) == (0.02, 0.05)

    def test_get_fee_rate(self):
        assert get_fee_rate("kraken", "spot", 2) is KRAKEN_SPOT_TIER_2
        assert get_fee_rate("kraken", "spot", 1, stablecoin_fx=True) is KRAKEN_STABLECOIN_FX
        assert get_fee_rate("kraken", "futures", 1) is KRAKEN_FUTURES_TIER_1

    def test_unknown_schedule(self):
        with pytest.raises(ValueError):
            get_fee_rate("binance", "spot", 1)


class TestFeeSensitivity:
    def test_spot_tier1_round_trip_is_160bp(self):
        assert round_trip_cost_pct() == pytest.approx(1.60)

    def test_taker_is_default(self):
        sensitivity = fee_sensitivity(10_000)
        assert sensitivity.liquidity == "taker"
        assert sensitivity.entry_fee_quote == pytest.approx(80.0)
        assert sensitivity.exit_fee_quote == pytest.approx(80.0)
        assert sensitivity.total_fee_quote == pytest.approx(160.0)
        assert sensitivity.round_trip_pct == pytest.approx(1.60)

    def test_maker_sensitivity(self):
        sensitivity = fee_sensitivity(10_000, liquidity="maker")
        assert sensitivity.total_fee_quote == pytest.approx(80.0)
        assert sensitivity.round_trip_pct == pytest.approx(0.80)

    def test_futures_sensitivity(self):
        assert round_trip_cost_pct("kraken", "futures", 1) == pytest.approx(0.10)

    def test_stablecoin_sensitivity(self):
        assert round_trip_cost_pct("kraken", "spot", 1, stablecoin_fx=True) == pytest.approx(0.40)

    def test_invalid_liquidity_and_notional(self):
        with pytest.raises(ValueError):
            fee_sensitivity(100, liquidity="mid")
        with pytest.raises(ValueError):
            fee_sensitivity(0)
