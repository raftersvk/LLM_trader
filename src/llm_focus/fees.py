"""Reference venue fees for v2 backtests and sensitivity checks.

Source: https://www.kraken.com/fr/features/fee-schedule
Relevé le 2026-11-02.

Market entries/exits use the taker rate by default. The spot Tier 1 round trip
is 1.60% (0.80% entry + 0.80% exit), which is enough to disqualify most
intra-hour strategies; earlier backtests using 0.05/0.10/0.20% materially
understated spot costs.
"""

from __future__ import annotations

from dataclasses import dataclass

SOURCE_URL = "https://www.kraken.com/fr/features/fee-schedule"
SOURCE_DATE = "2026-11-02"

DEFAULT_LIQUIDITY = "taker"
VALID_LIQUIDITY = ("maker", "taker")


@dataclass(frozen=True)
class FeeRate:
    """Maker/taker fee rates in percent."""

    maker_pct: float
    taker_pct: float
    label: str
    source: str = SOURCE_URL
    as_of: str = SOURCE_DATE

    def rate_pct(self, liquidity: str = DEFAULT_LIQUIDITY) -> float:
        """Return the requested liquidity rate in percent."""
        normalized = str(liquidity).strip().lower()
        if normalized not in VALID_LIQUIDITY:
            raise ValueError("liquidity must be 'maker' or 'taker'")
        return self.maker_pct if normalized == "maker" else self.taker_pct


KRAKEN_SPOT_TIER_1 = FeeRate(0.40, 0.80, "Kraken spot crypto Tier 1")
KRAKEN_SPOT_TIER_2 = FeeRate(0.30, 0.60, "Kraken spot crypto Tier 2")
KRAKEN_SPOT_TIER_3 = FeeRate(0.22, 0.38, "Kraken spot crypto Tier 3")
KRAKEN_STABLECOIN_FX = FeeRate(0.20, 0.20, "Kraken stablecoin/FX pairs")
KRAKEN_FUTURES_TIER_1 = FeeRate(0.02, 0.05, "Kraken futures Tier 1")

FEE_SCHEDULES: dict[tuple[str, str, int], FeeRate] = {
    ("kraken", "spot", 1): KRAKEN_SPOT_TIER_1,
    ("kraken", "spot", 2): KRAKEN_SPOT_TIER_2,
    ("kraken", "spot", 3): KRAKEN_SPOT_TIER_3,
    ("kraken", "futures", 1): KRAKEN_FUTURES_TIER_1,
}


@dataclass(frozen=True)
class FeeSensitivity:
    """Fee cost for one round trip at a given notional."""

    venue: str
    market: str
    tier: int
    liquidity: str
    notional_quote: float
    entry_fee_quote: float
    exit_fee_quote: float
    total_fee_quote: float
    round_trip_pct: float
    rate: FeeRate


def get_fee_rate(
    venue: str = "kraken",
    market: str = "spot",
    tier: int = 1,
    stablecoin_fx: bool = False,
) -> FeeRate:
    """Return the reference fee rate for a venue/market/tier."""
    normalized_venue = str(venue).strip().lower()
    normalized_market = str(market).strip().lower()
    if normalized_venue == "kraken" and normalized_market == "spot" and stablecoin_fx:
        return KRAKEN_STABLECOIN_FX
    key = (normalized_venue, normalized_market, int(tier))
    try:
        return FEE_SCHEDULES[key]
    except KeyError as exc:
        raise ValueError(f"no fee schedule for venue={venue} market={market} tier={tier}") from exc


def round_trip_cost_pct(
    venue: str = "kraken",
    market: str = "spot",
    tier: int = 1,
    stablecoin_fx: bool = False,
    liquidity: str = DEFAULT_LIQUIDITY,
) -> float:
    """Return the round-trip cost in percent (entry + exit)."""
    rate = get_fee_rate(venue, market, tier, stablecoin_fx=stablecoin_fx)
    return 2 * rate.rate_pct(liquidity)


def fee_sensitivity(
    notional_quote: float,
    venue: str = "kraken",
    market: str = "spot",
    tier: int = 1,
    stablecoin_fx: bool = False,
    liquidity: str = DEFAULT_LIQUIDITY,
) -> FeeSensitivity:
    """Return entry/exit/total fees for a notional, exposing venue/tier/liquidity."""
    if notional_quote <= 0:
        raise ValueError("notional_quote must be positive")
    rate = get_fee_rate(venue, market, tier, stablecoin_fx=stablecoin_fx)
    per_side = notional_quote * rate.rate_pct(liquidity) / 100
    total = per_side * 2
    return FeeSensitivity(
        venue=str(venue).strip().lower(),
        market=str(market).strip().lower(),
        tier=int(tier),
        liquidity=str(liquidity).strip().lower(),
        notional_quote=notional_quote,
        entry_fee_quote=per_side,
        exit_fee_quote=per_side,
        total_fee_quote=total,
        round_trip_pct=2 * rate.rate_pct(liquidity),
        rate=rate,
    )
