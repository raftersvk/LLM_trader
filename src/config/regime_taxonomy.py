"""External regime/taxonomy/threshold configuration.

The regime snapshot, the 1h setup taxonomy rendered into the prompt and the
JEV gate all read their policy from one document instead of hard-coded
constants. YAML is the canonical format; a JSON document is accepted as a
fallback (both for a ``.json`` path and when PyYAML is unavailable, because
JSON is a subset of YAML).

The loader is fail-safe by design:

* a missing canonical file returns the built-in defaults, which reproduce the
  behaviour that shipped with the code;
* an existing file is parsed strictly, so a typo or a wrong type raises
  :class:`TaxonomyConfigError` instead of silently changing trading policy.

No network, no model and no exchange access.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, ClassVar, Mapping

from src.llm_focus.fees import KRAKEN_FUTURES_TIER_1, get_fee_rate

DEFAULT_TAXONOMY_CONFIG_PATH = (
    "/home/djelinux/TradingBots/trading-stack/freqtrade/user_data/regime_taxonomy.yaml"
)
TAXONOMY_CONFIG_ENV = "TAXONOMY_CONFIG_PATH"
LEGACY_TAXONOMY_CONFIG_ENV = "REGIME_TAXONOMY_CONFIG"

REGIMES: tuple[str, ...] = (
    "trend_up",
    "trend_down",
    "range",
    "low_vol",
    "high_vol",
    "unclear",
)
NO_TRADE_REGIMES: tuple[str, ...] = ("low_vol", "high_vol", "unclear")
ALLOWED_SIDES: tuple[str, ...] = ("long", "short", "both")
TP_MODE_ATR_MULTIPLE = "atr_multiple"
TP_MODE_STRUCTURE = "structure"
VALID_TP_MODES: tuple[str, ...] = (TP_MODE_ATR_MULTIPLE, TP_MODE_STRUCTURE)
STRUCTURE_MID_RANGE = "mid_range"
STRUCTURE_VWAP = "vwap"
STRUCTURE_MEASURED_MOVE = "measured_move"
VALID_TP_SUBMODES: tuple[str, ...] = (
    STRUCTURE_MID_RANGE,
    STRUCTURE_VWAP,
    STRUCTURE_MEASURED_MOVE,
)
COST_VENUES: tuple[str, ...] = ("spot", "futures")
COST_ORDER_TYPES: tuple[str, ...] = ("taker", "maker")
COST_FIELDS: frozenset[str] = frozenset({
    "venue",
    "tier",
    "entry_order_type",
    "maker_pct",
    "taker_pct",
    "min_edge_multiple",
    "min_sl_atr_multiple",
    "max_sl_atr_multiple",
})
#: Flat per-pair execution fields shared with the Freqtrade strategy (W3).
PAIR_OVERRIDE_FIELDS: frozenset[str] = frozenset({
    "sl_atr",
    "tp_atr",
    "max_hold_bars",
    "max_hold_hours",
    "max_stake",
    "leverage",
    "trail_atr",
    "be_at_r",
    "trail_after_r",
})

DEFAULT_ENABLED_TIMEFRAMES: tuple[str, ...] = ("1h",)
DEFAULT_JEV_LOG_PATH = "runtime/logs/jev_decisions.jsonl"


class TaxonomyConfigError(ValueError):
    """Raised when a taxonomy document does not match the schema."""


@dataclass(frozen=True)
class RegimeThresholds:
    """Numeric cut-offs used to classify the 1h regime.

    The defaults reproduce the P0-B rules: ADX >= 25 for a trend, ADX < 20 for
    a range, choppiness > 61.8 for a range, ATR% > max(3%, 2.5x the 90-bar
    median) for high volatility and ATR% < 0.55x the median with choppiness >
    55 for low volatility.
    """

    adx_trend: float = 25.0
    adx_weak: float = 20.0
    range_choppiness: float = 61.8
    high_vol_atr_pct_floor: float = 3.0
    high_vol_atr_multiple: float = 2.5
    low_vol_atr_multiple: float = 0.55
    low_vol_choppiness: float = 55.0

    _FIELDS: ClassVar[tuple[str, ...]] = (
        "adx_trend",
        "adx_weak",
        "range_choppiness",
        "high_vol_atr_pct_floor",
        "high_vol_atr_multiple",
        "low_vol_atr_multiple",
        "low_vol_choppiness",
    )

    @classmethod
    def from_mapping(
        cls,
        data: Mapping[str, Any] | None,
        base: "RegimeThresholds | None" = None,
        label: str = "regime_thresholds",
    ) -> "RegimeThresholds":
        """Validate a threshold mapping, optionally on top of a base."""
        base = base or cls()
        if data is None:
            return base
        if not isinstance(data, Mapping):
            raise TaxonomyConfigError(f"{label} must be a mapping")
        unknown = set(data) - set(cls._FIELDS)
        if unknown:
            raise TaxonomyConfigError(
                f"{label} has unknown keys: {', '.join(sorted(map(str, unknown)))}"
            )
        values: dict[str, Any] = {}
        for name in cls._FIELDS:
            if name in data:
                values[name] = _number(data[name], f"{label}.{name}", minimum=0.0, allow_zero=False)
            else:
                values[name] = getattr(base, name)
        thresholds = cls(**values)
        if thresholds.adx_trend <= thresholds.adx_weak:
            raise TaxonomyConfigError(f"{label}.adx_trend must be greater than adx_weak")
        for name in ("range_choppiness", "low_vol_choppiness"):
            value = getattr(thresholds, name)
            if value > 100.0:
                raise TaxonomyConfigError(f"{label}.{name} must be <= 100")
        return thresholds

    def as_dict(self) -> dict[str, float]:
        """Return the validated thresholds as a plain mapping."""
        return {name: float(getattr(self, name)) for name in self._FIELDS}


@dataclass(frozen=True)
class SetupDefinition:
    """One setup family: side, narrative wording and ATR bracket policy."""

    name: str
    side: str
    description: str
    entry: str
    invalidation: str
    regimes: tuple[str, ...]
    max_hold_bars: int
    sl_atr: float | None = None
    tp_atr: float | None = None
    trail_atr: float | None = None
    be_at_r: float | None = None
    tp_mode: str = TP_MODE_ATR_MULTIPLE
    tp_submode: str | None = None

    _FIELDS: ClassVar[tuple[str, ...]] = (
        "side",
        "description",
        "entry",
        "invalidation",
        "regimes",
        "max_hold_bars",
        "sl_atr",
        "tp_atr",
        "trail_atr",
        "be_at_r",
        "tp_mode",
        "tp_submode",
    )
    _REQUIRED_FIELDS: ClassVar[tuple[str, ...]] = (
        "side",
        "description",
        "entry",
        "invalidation",
        "regimes",
        "max_hold_bars",
    )

    @classmethod
    def from_mapping(
        cls,
        name: str,
        data: Mapping[str, Any] | None,
        base: "SetupDefinition | None" = None,
    ) -> "SetupDefinition":
        """Validate one setup definition, optionally overriding a base."""
        if not isinstance(name, str) or not name.strip():
            raise TaxonomyConfigError("setup names must be non-empty strings")
        if not isinstance(data, Mapping):
            raise TaxonomyConfigError(f"setups.{name} must be a mapping")
        unknown = set(data) - set(cls._FIELDS)
        if unknown:
            raise TaxonomyConfigError(
                f"setups.{name} has unknown keys: {', '.join(sorted(map(str, unknown)))}"
            )
        if base is None and set(cls._REQUIRED_FIELDS) - set(data):
            missing = ", ".join(sorted(set(cls._REQUIRED_FIELDS) - set(data)))
            raise TaxonomyConfigError(f"setups.{name} is missing keys: {missing}")

        side = _string(
            data.get("side", base.side if base else None), f"setups.{name}.side"
        ).lower()
        if side not in ALLOWED_SIDES:
            raise TaxonomyConfigError(
                f"setups.{name}.side must be one of {'|'.join(ALLOWED_SIDES)}"
            )
        regimes_raw = data.get("regimes", base.regimes if base else ())
        regimes = _regime_tuple(regimes_raw, f"setups.{name}.regimes")
        return cls(
            name=name,
            side=side,
            description=_string(
                data.get("description", base.description if base else None),
                f"setups.{name}.description",
            ),
            entry=_string(
                data.get("entry", base.entry if base else None), f"setups.{name}.entry"
            ),
            invalidation=_string(
                data.get("invalidation", base.invalidation if base else None),
                f"setups.{name}.invalidation",
            ),
            regimes=regimes,
            max_hold_bars=_positive_int(
                data.get("max_hold_bars", base.max_hold_bars if base else None),
                f"setups.{name}.max_hold_bars",
            ),
            sl_atr=_optional_number(
                data.get("sl_atr", base.sl_atr if base else None),
                f"setups.{name}.sl_atr",
                allow_zero=False,
            ),
            tp_atr=_optional_number(
                data.get("tp_atr", base.tp_atr if base else None),
                f"setups.{name}.tp_atr",
                allow_zero=False,
            ),
            trail_atr=_optional_number(
                data.get("trail_atr", base.trail_atr if base else None),
                f"setups.{name}.trail_atr",
                allow_zero=True,
            ),
            be_at_r=_optional_number(
                data.get("be_at_r", base.be_at_r if base else None),
                f"setups.{name}.be_at_r",
                allow_zero=True,
            ),
            tp_mode=_tp_mode(
                data.get("tp_mode", base.tp_mode if base else TP_MODE_ATR_MULTIPLE),
                f"setups.{name}.tp_mode",
            ),
            tp_submode=_optional_tp_submode(
                data.get("tp_submode", base.tp_submode if base else None),
                f"setups.{name}.tp_submode",
            ),
        )

    def as_dict(self) -> dict[str, Any]:
        """Return the setup as a JSON/YAML-compatible mapping.

        Optional fields that are absent from the source stay absent, so a
        setup without ``sl_atr``/``tp_atr`` keeps the signal's absolute levels
        and an omitted ``trail_atr``/``be_at_r`` does not shadow legacy config.
        """
        data: dict[str, Any] = {
            "side": self.side,
            "description": self.description,
            "entry": self.entry,
            "invalidation": self.invalidation,
            "regimes": list(self.regimes),
            "max_hold_bars": self.max_hold_bars,
        }
        for name in ("sl_atr", "tp_atr", "trail_atr", "be_at_r"):
            value = getattr(self, name)
            if value is not None:
                data[name] = value
        data["tp_mode"] = self.tp_mode
        if self.tp_submode is not None:
            data["tp_submode"] = self.tp_submode
        return data


@dataclass(frozen=True)
class JevSettings:
    """JEV gate policy that belongs to the taxonomy document."""

    bias_short_only: bool = True
    log_decisions: bool = True
    log_path: str = DEFAULT_JEV_LOG_PATH

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any] | None, base: "JevSettings | None" = None) -> "JevSettings":
        """Validate the ``jev`` section."""
        base = base or cls()
        if data is None:
            return base
        if not isinstance(data, Mapping):
            raise TaxonomyConfigError("jev must be a mapping")
        unknown = set(data) - {"bias_short_only", "log_decisions", "log_path"}
        if unknown:
            raise TaxonomyConfigError(
                f"jev has unknown keys: {', '.join(sorted(map(str, unknown)))}"
            )
        return cls(
            bias_short_only=_boolean(
                data.get("bias_short_only", base.bias_short_only), "jev.bias_short_only"
            ),
            log_decisions=_boolean(
                data.get("log_decisions", base.log_decisions), "jev.log_decisions"
            ),
            log_path=_string(data.get("log_path", base.log_path), "jev.log_path"),
        )


@dataclass(frozen=True)
class CostsSettings:
    """Fee policy used by the fee-aware entry gate.

    Built-in defaults are Kraken futures Tier 1. A taxonomy ``costs`` block or a
    config.ini ``[costs]`` block overrides them; config.ini has the highest
    precedence. Exit fees are assumed taker unless a maker exit is explicitly
    documented by the caller.
    """

    venue: str = "futures"
    tier: int = 1
    entry_order_type: str = "taker"
    maker_pct: float = KRAKEN_FUTURES_TIER_1.maker_pct
    taker_pct: float = KRAKEN_FUTURES_TIER_1.taker_pct
    min_edge_multiple: float = 1.5
    min_sl_atr_multiple: float = 0.25
    max_sl_atr_multiple: float = 12.0

    def fee_pct(self, order_type: str | None = None) -> float:
        """Return the fee in percent for an order type (defaults to entry type)."""
        normalized = str(order_type or self.entry_order_type).strip().lower()
        if normalized not in COST_ORDER_TYPES:
            raise TaxonomyConfigError("order_type must be 'taker' or 'maker'")
        return self.maker_pct if normalized == "maker" else self.taker_pct

    def round_trip_pct(self) -> float:
        """Entry fee plus the assumed taker exit fee, in percent."""
        return self.fee_pct(self.entry_order_type) + self.fee_pct("taker")

    @classmethod
    def from_mapping(
        cls,
        data: Mapping[str, Any] | None,
        base: "CostsSettings | None" = None,
    ) -> "CostsSettings":
        """Validate a ``costs`` block on top of ``base`` (defaults when absent)."""
        base = base or cls()
        if data is None:
            return base
        if not isinstance(data, Mapping):
            raise TaxonomyConfigError("costs must be a mapping")
        unknown = set(data) - COST_FIELDS
        if unknown:
            raise TaxonomyConfigError(
                f"costs has unknown keys: {', '.join(sorted(map(str, unknown)))}"
            )

        venue = _string(data.get("venue", base.venue), "costs.venue").lower()
        if venue not in COST_VENUES:
            raise TaxonomyConfigError(
                f"costs.venue must be one of {'|'.join(COST_VENUES)}"
            )
        tier_raw = data.get("tier", base.tier)
        if isinstance(tier_raw, bool) or not isinstance(tier_raw, int) or tier_raw < 1:
            raise TaxonomyConfigError("costs.tier must be a positive integer")
        tier = tier_raw
        entry_order_type = _string(
            data.get("entry_order_type", base.entry_order_type),
            "costs.entry_order_type",
        ).lower()
        if entry_order_type not in COST_ORDER_TYPES:
            raise TaxonomyConfigError(
                f"costs.entry_order_type must be one of {'|'.join(COST_ORDER_TYPES)}"
            )

        maker_pct = (
            _number(data["maker_pct"], "costs.maker_pct", minimum=0.0)
            if "maker_pct" in data
            else base.maker_pct
        )
        taker_pct = (
            _number(data["taker_pct"], "costs.taker_pct", minimum=0.0)
            if "taker_pct" in data
            else base.taker_pct
        )
        if venue != base.venue or tier != base.tier:
            # Venue/tier changed without explicit rates: follow the Kraken
            # constants for the new venue/tier instead of the previous rates.
            try:
                rate = get_fee_rate("kraken", venue, tier)
            except ValueError as exc:
                raise TaxonomyConfigError(str(exc)) from exc
            if "maker_pct" not in data:
                maker_pct = rate.maker_pct
            if "taker_pct" not in data:
                taker_pct = rate.taker_pct

        min_edge_multiple = _number(
            data.get("min_edge_multiple", base.min_edge_multiple),
            "costs.min_edge_multiple",
            minimum=0.0,
            allow_zero=False,
        )
        min_sl_atr_multiple = _number(
            data.get("min_sl_atr_multiple", base.min_sl_atr_multiple),
            "costs.min_sl_atr_multiple",
            minimum=0.0,
            allow_zero=False,
        )
        max_sl_atr_multiple = _number(
            data.get("max_sl_atr_multiple", base.max_sl_atr_multiple),
            "costs.max_sl_atr_multiple",
            minimum=0.0,
            allow_zero=False,
        )
        if max_sl_atr_multiple <= min_sl_atr_multiple:
            raise TaxonomyConfigError(
                "costs.max_sl_atr_multiple must be greater than costs.min_sl_atr_multiple"
            )
        return cls(
            venue=venue,
            tier=tier,
            entry_order_type=entry_order_type,
            maker_pct=maker_pct,
            taker_pct=taker_pct,
            min_edge_multiple=min_edge_multiple,
            min_sl_atr_multiple=min_sl_atr_multiple,
            max_sl_atr_multiple=max_sl_atr_multiple,
        )


EXPOSURE_MODE_SCALE_DOWN = "scale_down"
EXPOSURE_MODE_REJECT = "reject"
EXPOSURE_MODES: tuple[str, ...] = (EXPOSURE_MODE_SCALE_DOWN, EXPOSURE_MODE_REJECT)


@dataclass(frozen=True)
class RiskSettings:
    """Risk-budget position sizing policy (v2, flag-gated).

    The **risk level is the tuning knob**; position size is a consequence of
    it and the validated stop distance::

        size_pct = risk_per_trade_pct / sl_distance_pct

    Both inputs are percentages, so a 0.25% risk on a 2% stop gives
    ``0.25 / 2.0 = 0.125`` (12.5% of capital). There is **no fixed per-trade
    size cap** by default: the real brakes are the portfolio exposure ceiling
    (``max_portfolio_exposure_pct``) and the account limits (spot 100%,
    futures ``max_leverage``). An optional canonical per-pair ``max_stake``
    (quote currency) still acts as an explicit safety cap when configured.

    ``risk_per_trade_pct`` is the global default. ``pair_overrides`` maps a
    symbol to its own risk level (for example a volatile pair such as
    ``BEAMX/USDT`` at 0.10-0.15%). ``regime_risk_multipliers`` scales the
    effective risk per regime (for example ``high_vol: 0.5``); an absent or
    ``1.0`` multiplier is neutral.

    ``max_portfolio_exposure_pct`` is a percentage of capital. When absent it
    follows the trading mode: 100% on spot and 200% on futures.

    A taxonomy ``risk`` block or a config.ini ``[risk_sizing]`` block overrides
    the defaults; config.ini has the highest precedence.
    """

    risk_per_trade_pct: float = 0.25
    min_size_pct: float = 0.1
    max_leverage: float = 1.0
    max_portfolio_exposure_pct: float | None = None
    exposure_mode: str = EXPOSURE_MODE_SCALE_DOWN
    pair_overrides: Mapping[str, float] = field(default_factory=dict)
    regime_risk_multipliers: Mapping[str, float] = field(default_factory=dict)

    _FIELDS: ClassVar[tuple[str, ...]] = (
        "risk_per_trade_pct",
        "min_size_pct",
        "max_leverage",
        "max_portfolio_exposure_pct",
        "exposure_mode",
        "pair_overrides",
        "regime_risk_multipliers",
    )

    @classmethod
    def from_mapping(
        cls,
        data: Mapping[str, Any] | None,
        base: "RiskSettings | None" = None,
    ) -> "RiskSettings":
        """Validate a ``risk`` block on top of ``base`` (defaults when absent)."""
        base = base or cls()
        if data is None:
            return base
        if not isinstance(data, Mapping):
            raise TaxonomyConfigError("risk must be a mapping")
        unknown = set(data) - set(cls._FIELDS)
        if unknown:
            raise TaxonomyConfigError(
                f"risk has unknown keys: {', '.join(sorted(map(str, unknown)))}"
            )
        risk_per_trade_pct = _number(
            data.get("risk_per_trade_pct", base.risk_per_trade_pct),
            "risk.risk_per_trade_pct",
            minimum=0.0,
            allow_zero=False,
        )
        min_size_pct = _number(
            data.get("min_size_pct", base.min_size_pct),
            "risk.min_size_pct",
            minimum=0.0,
            allow_zero=True,
        )
        max_leverage = _number(
            data.get("max_leverage", base.max_leverage),
            "risk.max_leverage",
            minimum=1.0,
            allow_zero=True,
        )
        exposure_raw = data.get("max_portfolio_exposure_pct", base.max_portfolio_exposure_pct)
        max_portfolio_exposure_pct = (
            None
            if exposure_raw is None
            else _number(
                exposure_raw,
                "risk.max_portfolio_exposure_pct",
                minimum=0.0,
                allow_zero=False,
            )
        )
        exposure_mode = _string(
            data.get("exposure_mode", base.exposure_mode),
            "risk.exposure_mode",
        ).lower()
        if exposure_mode not in EXPOSURE_MODES:
            raise TaxonomyConfigError(
                f"risk.exposure_mode must be one of {'|'.join(EXPOSURE_MODES)}"
            )
        pair_overrides = _parse_risk_pair_overrides(data.get("pair_overrides"))
        regime_multipliers = _parse_regime_multipliers(
            data.get("regime_risk_multipliers")
        )
        return cls(
            risk_per_trade_pct=risk_per_trade_pct,
            min_size_pct=min_size_pct,
            max_leverage=max_leverage,
            max_portfolio_exposure_pct=max_portfolio_exposure_pct,
            exposure_mode=exposure_mode,
            pair_overrides=pair_overrides,
            regime_risk_multipliers=regime_multipliers,
        )

    def min_size_fraction(self) -> float:
        """Return the minimum tradable size as a fraction of capital."""
        return self.min_size_pct / 100.0

    def portfolio_exposure_pct(self, trading_mode: str = "spot") -> float:
        """Return the effective portfolio ceiling in percent of capital."""
        if self.max_portfolio_exposure_pct is not None:
            return self.max_portfolio_exposure_pct
        return 200.0 if str(trading_mode).strip().lower() == "futures" else 100.0

    def risk_for_symbol(self, symbol: str | None = None) -> float:
        """Return the global risk level or the pair's configured override."""
        if not symbol:
            return self.risk_per_trade_pct
        compact = symbol.replace("/", "").split(":", 1)[0].upper()
        for key, value in self.pair_overrides.items():
            key_symbol = key.split(":", 1)[0]
            if key_symbol == symbol or key_symbol.replace("/", "").upper() == compact:
                return value
        return self.risk_per_trade_pct

    def regime_multiplier(self, regime: str | None = None) -> float:
        """Return the regime risk multiplier; absent/unknown is neutral (1.0)."""
        if not regime:
            return 1.0
        normalized = str(regime).strip().lower().replace("-", "_").replace(" ", "_")
        return self.regime_risk_multipliers.get(normalized, 1.0)

    def effective_risk_per_trade_pct(
        self,
        symbol: str | None = None,
        regime: str | None = None,
    ) -> float:
        """Return the risk level after pair override and regime multiplier."""
        return self.risk_for_symbol(symbol) * self.regime_multiplier(regime)

    def as_dict(self) -> dict[str, Any]:
        """Return the validated settings as a plain mapping."""
        return {
            "risk_per_trade_pct": float(self.risk_per_trade_pct),
            "min_size_pct": float(self.min_size_pct),
            "max_leverage": float(self.max_leverage),
            "max_portfolio_exposure_pct": self.max_portfolio_exposure_pct,
            "exposure_mode": self.exposure_mode,
            "pair_overrides": dict(self.pair_overrides),
            "regime_risk_multipliers": dict(self.regime_risk_multipliers),
        }


def _parse_risk_pair_overrides(data: Any) -> dict[str, float]:
    """Parse the optional per-pair risk levels inside the ``risk`` block."""
    if data is None:
        return {}
    if not isinstance(data, Mapping):
        raise TaxonomyConfigError("risk.pair_overrides must be a mapping")
    result: dict[str, float] = {}
    for symbol, value in data.items():
        name = _string(symbol, "risk.pair_overrides key")
        result[name] = _number(
            value,
            f"risk.pair_overrides.{name}",
            minimum=0.0,
            allow_zero=False,
        )
    return result


def _parse_regime_multipliers(data: Any) -> dict[str, float]:
    """Parse the optional per-regime risk multipliers inside ``risk``."""
    if data is None:
        return {}
    if not isinstance(data, Mapping):
        raise TaxonomyConfigError("risk.regime_risk_multipliers must be a mapping")
    unknown = [regime for regime in data if regime not in REGIMES]
    if unknown:
        raise TaxonomyConfigError(
            "risk.regime_risk_multipliers has unknown regimes: "
            f"{', '.join(map(str, unknown))}"
        )
    result: dict[str, float] = {}
    for regime, value in data.items():
        result[str(regime)] = _number(
            value,
            f"risk.regime_risk_multipliers.{regime}",
            minimum=0.0,
            allow_zero=False,
        )
    return result


@dataclass(frozen=True)
class RegimeTaxonomyConfig:
    """Validated taxonomy document used by the snapshot, prompt and JEV gate."""

    version: str = "1"
    enabled_timeframes: tuple[str, ...] = DEFAULT_ENABLED_TIMEFRAMES
    regime_thresholds: RegimeThresholds = field(default_factory=RegimeThresholds)
    regime_setups: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    setups: Mapping[str, SetupDefinition] = field(default_factory=dict)
    pair_overrides: Mapping[str, Mapping[str, float]] = field(default_factory=dict)
    jev: JevSettings = field(default_factory=JevSettings)
    costs: CostsSettings = field(default_factory=CostsSettings)
    risk: RiskSettings = field(default_factory=RiskSettings)
    source_path: str | None = None

    def timeframe_enabled(self, timeframe: str | None) -> bool:
        """True when the taxonomy may be applied to this timeframe."""
        return str(timeframe).strip().lower() in self.enabled_timeframes

    def allowed_setups(self, regime: str) -> tuple[str, ...]:
        """Allowed setup families for a regime (empty when no-trade)."""
        return tuple(self.regime_setups.get(regime, ()))

    def pair_override(self, symbol: str | None) -> Mapping[str, float]:
        """Return the flat execution overrides for a signal symbol/pair.

        W3 writes these per-pair fields for the Freqtrade strategy; the regime
        snapshot does not consume them. Both ``BTC/USDT:USDT`` and ``BTC/USDT``
        keys resolve to the same signal symbol.
        """
        if not symbol:
            return {}
        compact = symbol.replace("/", "").split(":", 1)[0].upper()
        for key, value in self.pair_overrides.items():
            key_symbol = key.split(":", 1)[0]
            if key_symbol == symbol or key_symbol.replace("/", "").upper() == compact:
                return value
        return {}

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON/YAML-compatible representation of the document."""
        return {
            "version": self.version,
            "enabled_timeframes": list(self.enabled_timeframes),
            "regime_thresholds": self.regime_thresholds.as_dict(),
            "regime_setups": {key: list(value) for key, value in self.regime_setups.items()},
            "setups": {name: setup.as_dict() for name, setup in self.setups.items()},
            "pair_overrides": {
                pair: dict(override)
                for pair, override in self.pair_overrides.items()
            },
            "jev": {
                "bias_short_only": self.jev.bias_short_only,
                "log_decisions": self.jev.log_decisions,
                "log_path": self.jev.log_path,
            },
            "costs": {
                "venue": self.costs.venue,
                "tier": self.costs.tier,
                "entry_order_type": self.costs.entry_order_type,
                "maker_pct": self.costs.maker_pct,
                "taker_pct": self.costs.taker_pct,
                "min_edge_multiple": self.costs.min_edge_multiple,
                "min_sl_atr_multiple": self.costs.min_sl_atr_multiple,
                "max_sl_atr_multiple": self.costs.max_sl_atr_multiple,
            },
            "risk": self.risk.as_dict(),
        }


# --------------------------------------------------------------------- defaults
def _default_setup(
    name: str,
    side: str,
    description: str,
    entry: str,
    invalidation: str,
    regimes: tuple[str, ...],
    max_hold_bars: int,
    sl_atr: float | None = None,
    tp_atr: float | None = None,
    trail_atr: float | None = None,
    be_at_r: float | None = None,
    tp_mode: str = TP_MODE_ATR_MULTIPLE,
    tp_submode: str | None = None,
) -> SetupDefinition:
    """Build one built-in setup definition (mirrors the W3 canonical file)."""
    return SetupDefinition(
        name=name,
        side=side,
        description=description,
        entry=entry,
        invalidation=invalidation,
        regimes=regimes,
        max_hold_bars=max_hold_bars,
        sl_atr=sl_atr,
        tp_atr=tp_atr,
        trail_atr=trail_atr,
        be_at_r=be_at_r,
        tp_mode=tp_mode,
        tp_submode=tp_submode,
    )


DEFAULT_SETUPS: Mapping[str, SetupDefinition] = {
    "trend_pullback": _default_setup(
        "trend_pullback",
        "long",
        "Orderly pullback to SMA20/structure support in an uptrend with a rejection close.",
        "Pullback into SMA20/support with a rejection close.",
        "Close below the pullback low.",
        ("trend_up",),
        24,
        sl_atr=2.5,
        tp_atr=4.8,
    ),
    "breakout_retest": _default_setup(
        "breakout_retest",
        "long",
        "Break of a 20-bar high/range then a held retest of the broken level.",
        "Held retest of the broken 20-bar high/range level.",
        "Close back below the broken level.",
        ("trend_up",),
        24,
    ),
    "momentum_continuation": _default_setup(
        "momentum_continuation",
        "long",
        "Higher-high/higher-low continuation without more than 2 ATR extension from SMA20.",
        "Continuation after a higher high and higher low.",
        "Lower low.",
        ("trend_up",),
        12,
        sl_atr=2.5,
        tp_atr=4.0,
    ),
    "rally_failure": _default_setup(
        "rally_failure",
        "short",
        "Rally into SMA20/resistance rejected with a lower high in a downtrend.",
        "Rejection at SMA20/resistance with a lower high.",
        "Close above the swing high.",
        ("trend_down",),
        24,
    ),
    "breakdown_retest": _default_setup(
        "breakdown_retest",
        "short",
        "Support break then a rejected retest of the broken level.",
        "Rejected retest of the broken support level.",
        "Close back above the broken level.",
        ("trend_down",),
        24,
    ),
    "momentum_short": _default_setup(
        "momentum_short",
        "short",
        "Lower-high/lower-low continuation without more than 2 ATR extension.",
        "Continuation after a lower high and lower low.",
        "Higher high.",
        ("trend_down",),
        12,
        sl_atr=2.5,
        tp_atr=4.5,
    ),
    "range_fade_extreme": _default_setup(
        "range_fade_extreme",
        "both",
        "Fade a wick beyond the range boundary that closes back inside.",
        "Wick beyond the range boundary closing back inside.",
        "Close beyond the range boundary.",
        ("range",),
        12,
        sl_atr=2.0,
        tp_atr=4.5,
        tp_mode=TP_MODE_STRUCTURE,
    ),
    "range_mean_reversion": _default_setup(
        "range_mean_reversion",
        "both",
        "Return toward mid/VWAP after an excursion.",
        "Excursion away from mid/VWAP with a reversal close.",
        "Break of the range.",
        ("range",),
        12,
        sl_atr=2.0,
        tp_atr=4.5,
        tp_mode=TP_MODE_STRUCTURE,
    ),
}

DEFAULT_REGIME_SETUPS: Mapping[str, tuple[str, ...]] = {
    "trend_up": ("trend_pullback", "breakout_retest", "momentum_continuation"),
    "trend_down": ("rally_failure", "breakdown_retest", "momentum_short"),
    "range": ("range_fade_extreme", "range_mean_reversion"),
    "low_vol": (),
    "high_vol": (),
    "unclear": (),
}


# ------------------------------------------------------------------- validation
def _number(
    value: Any,
    label: str,
    minimum: float | None = None,
    allow_zero: bool = True,
) -> float:
    """Validate a finite number against an optional lower bound."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TaxonomyConfigError(f"{label} must be a number")
    number = float(value)
    if not math.isfinite(number):
        raise TaxonomyConfigError(f"{label} must be finite")
    if minimum is not None:
        if allow_zero and number < minimum:
            raise TaxonomyConfigError(f"{label} must be >= {minimum}")
        if not allow_zero and number <= minimum:
            raise TaxonomyConfigError(f"{label} must be > {minimum}")
    return number


def _optional_number(
    value: Any,
    label: str,
    allow_zero: bool,
) -> float | None:
    """Validate an optional number; ``None``/absence stays ``None``."""
    if value is None:
        return None
    return _number(value, label, minimum=0.0, allow_zero=allow_zero)


def _positive_int(value: Any, label: str) -> int:
    """Validate a strictly positive integer."""
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise TaxonomyConfigError(f"{label} must be a positive integer")
    return value


def _tp_mode(value: Any, label: str) -> str:
    """Validate the target-geometry mode enum (default applied by the caller)."""
    if not isinstance(value, str) or not value.strip():
        raise TaxonomyConfigError(
            f"{label} must be one of {'|'.join(VALID_TP_MODES)}"
        )
    mode = value.strip().lower()
    if mode not in VALID_TP_MODES:
        raise TaxonomyConfigError(
            f"{label} must be one of {'|'.join(VALID_TP_MODES)}, got {value!r}"
        )
    return mode


def _optional_tp_submode(value: Any, label: str) -> str | None:
    """Validate the optional structural target sub-mode enum."""
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise TaxonomyConfigError(
            f"{label} must be one of {'|'.join(VALID_TP_SUBMODES)} or null"
        )
    submode = value.strip().lower()
    if submode not in VALID_TP_SUBMODES:
        raise TaxonomyConfigError(
            f"{label} must be one of {'|'.join(VALID_TP_SUBMODES)}, got {value!r}"
        )
    return submode


def _boolean(value: Any, label: str) -> bool:
    """Validate a boolean flag."""
    if not isinstance(value, bool):
        raise TaxonomyConfigError(f"{label} must be a boolean")
    return value


def _string(value: Any, label: str) -> str:
    """Validate a non-empty string."""
    if not isinstance(value, str) or not value.strip():
        raise TaxonomyConfigError(f"{label} must be a non-empty string")
    return value.strip()


def _string_tuple(value: Any, label: str) -> tuple[str, ...]:
    """Validate a list of unique non-empty strings."""
    if isinstance(value, str) or not isinstance(value, (list, tuple)):
        raise TaxonomyConfigError(f"{label} must be a list of strings")
    result: list[str] = []
    for item in value:
        text = _string(item, label)
        if text not in result:
            result.append(text)
    return tuple(result)


def _regime_tuple(value: Any, label: str) -> tuple[str, ...]:
    """Validate a list of regime names."""
    regimes = _string_tuple(value, label)
    unknown = [regime for regime in regimes if regime not in REGIMES]
    if unknown:
        raise TaxonomyConfigError(
            f"{label} has unknown regimes: {', '.join(unknown)}"
        )
    return regimes


def _parse_setups(data: Any, base: Mapping[str, SetupDefinition]) -> dict[str, SetupDefinition]:
    """Parse the ``setups`` section, merging overrides into the built-ins."""
    if data is None:
        return dict(base)
    if not isinstance(data, Mapping):
        raise TaxonomyConfigError("setups must be a mapping")
    setups = dict(base)
    for name, definition in data.items():
        setups[name] = SetupDefinition.from_mapping(name, definition, setups.get(name))
    return setups


def _contract_setup_names() -> frozenset[str]:
    """Return the setup labels the signal contract can carry.

    Imported lazily: ``src.signal_service.__init__`` imports the analyzer,
    which imports this module, so a top-level import would be circular. By the
    time a document is parsed the package is fully loaded.
    """
    from src.signal_service.contract import ALLOWED_SETUPS  # pylint: disable=import-outside-toplevel

    return frozenset(ALLOWED_SETUPS) - {"none"}


def _validate_contract_setups(setups: Mapping[str, SetupDefinition]) -> None:
    """Reject setup names the signal contract enum cannot emit."""
    unknown = sorted(name for name in setups if name not in _contract_setup_names())
    if unknown:
        raise TaxonomyConfigError(
            "setups not allowed by the signal contract: " + ", ".join(unknown)
        )


def _parse_regime_setups(
    data: Any,
    base: Mapping[str, tuple[str, ...]],
    setups: Mapping[str, SetupDefinition],
    label: str = "regime_setups",
) -> dict[str, tuple[str, ...]]:
    """Parse the regime -> allowed setups mapping."""
    if data is None:
        return {regime: tuple(base.get(regime, ())) for regime in REGIMES}
    if not isinstance(data, Mapping):
        raise TaxonomyConfigError(f"{label} must be a mapping")
    unknown = [regime for regime in data if regime not in REGIMES]
    if unknown:
        raise TaxonomyConfigError(
            f"{label} has unknown regimes: {', '.join(map(str, unknown))}"
        )
    result = {regime: tuple(base.get(regime, ())) for regime in REGIMES}
    for regime, names in data.items():
        allowed = _string_tuple(names, f"{label}.{regime}")
        missing = [name for name in allowed if name not in setups]
        if missing:
            raise TaxonomyConfigError(
                f"{label}.{regime} references unknown setups: {', '.join(missing)}"
            )
        result[regime] = allowed
    return result


def _parse_pair_overrides(data: Any) -> dict[str, dict[str, float]]:
    """Parse the W3 flat per-pair execution overrides."""
    if data is None:
        return {}
    if not isinstance(data, Mapping):
        raise TaxonomyConfigError("pair_overrides must be a mapping")
    overrides: dict[str, dict[str, float]] = {}
    for pair, raw in data.items():
        pair_name = _string(pair, "pair_overrides key")
        if raw is None:
            overrides[pair_name] = {}
            continue
        if not isinstance(raw, Mapping):
            raise TaxonomyConfigError(f"pair_overrides.{pair_name} must be a mapping")
        unknown = set(raw) - PAIR_OVERRIDE_FIELDS
        if unknown:
            raise TaxonomyConfigError(
                f"pair_overrides.{pair_name} has unknown keys: "
                f"{', '.join(sorted(map(str, unknown)))}"
            )
        values: dict[str, float] = {}
        for field_name, value in raw.items():
            allow_zero = field_name in ("trail_atr", "be_at_r")
            values[field_name] = _number(
                value,
                f"pair_overrides.{pair_name}.{field_name}",
                minimum=0.0,
                allow_zero=allow_zero,
            )
        overrides[pair_name] = values
    return overrides


def parse_taxonomy_config(
    document: Any,
    source_path: str | None = None,
) -> RegimeTaxonomyConfig:
    """Validate a parsed document and return the taxonomy config."""
    if not isinstance(document, Mapping):
        raise TaxonomyConfigError("taxonomy document must be a mapping")
    known = {
        "version",
        "enabled_timeframes",
        "regime_thresholds",
        "regime_setups",
        "setups",
        "pair_overrides",
        "jev",
        "costs",
        "risk",
    }
    unknown = set(document) - known
    if unknown:
        raise TaxonomyConfigError(
            f"taxonomy document has unknown keys: {', '.join(sorted(map(str, unknown)))}"
        )

    version_raw = document.get("version", "1")
    version = str(version_raw).strip() if isinstance(version_raw, (str, int, float)) else ""
    if not version:
        raise TaxonomyConfigError("version must be a non-empty string or number")

    enabled_raw = document.get("enabled_timeframes", DEFAULT_ENABLED_TIMEFRAMES)
    enabled = tuple(item.lower() for item in _string_tuple(enabled_raw, "enabled_timeframes"))
    if not enabled:
        raise TaxonomyConfigError("enabled_timeframes must enable at least one timeframe")

    thresholds = RegimeThresholds.from_mapping(document.get("regime_thresholds"))
    setups = _parse_setups(document.get("setups"), DEFAULT_SETUPS)
    _validate_contract_setups(setups)
    regime_setups = _parse_regime_setups(document.get("regime_setups"), DEFAULT_REGIME_SETUPS, setups)
    pair_overrides = _parse_pair_overrides(document.get("pair_overrides"))
    jev = JevSettings.from_mapping(document.get("jev"))
    costs = CostsSettings.from_mapping(document.get("costs"))
    risk = RiskSettings.from_mapping(document.get("risk"))

    return RegimeTaxonomyConfig(
        version=version,
        enabled_timeframes=enabled,
        regime_thresholds=thresholds,
        regime_setups=regime_setups,
        setups=setups,
        pair_overrides=pair_overrides,
        jev=jev,
        costs=costs,
        risk=risk,
        source_path=source_path,
    )


def resolve_costs_config(
    taxonomy_config: RegimeTaxonomyConfig | None = None,
    overrides: Mapping[str, Any] | None = None,
) -> CostsSettings:
    """Resolve effective costs with precedence config.ini > taxonomy > defaults."""
    taxonomy = taxonomy_config or default_taxonomy_config()
    return CostsSettings.from_mapping(overrides, base=taxonomy.costs)


def format_costs_summary(costs: CostsSettings) -> str:
    """Return one-line startup summary of the effective cost policy."""
    return (
        f"venue={costs.venue} tier={costs.tier} "
        f"entry={costs.entry_order_type} "
        f"maker={costs.maker_pct:g}%/side taker={costs.taker_pct:g}%/side "
        f"round_trip={costs.round_trip_pct():g}% "
        f"min_edge={costs.min_edge_multiple:g}x "
        f"sl_atr=[{costs.min_sl_atr_multiple:g},{costs.max_sl_atr_multiple:g}]x"
    )


def resolve_risk_config(
    taxonomy_config: RegimeTaxonomyConfig | None = None,
    overrides: Mapping[str, Any] | None = None,
) -> RiskSettings:
    """Resolve effective sizing with precedence config.ini > taxonomy > defaults."""
    taxonomy = taxonomy_config or default_taxonomy_config()
    return RiskSettings.from_mapping(overrides, base=taxonomy.risk)


def format_risk_summary(risk: RiskSettings) -> str:
    """Return one-line startup summary of the effective sizing policy."""
    exposure = (
        "spot=100%/futures=200%"
        if risk.max_portfolio_exposure_pct is None
        else f"{risk.max_portfolio_exposure_pct:g}%"
    )
    pairs = (
        ",".join(f"{symbol}={value:g}%" for symbol, value in sorted(risk.pair_overrides.items()))
        or "none"
    )
    multipliers = (
        ",".join(
            f"{regime}={value:g}"
            for regime, value in sorted(risk.regime_risk_multipliers.items())
        )
        or "none"
    )
    return (
        f"risk_per_trade={risk.risk_per_trade_pct:g}% "
        f"pair_risk={pairs} "
        f"regime_multipliers={multipliers} "
        f"min_size={risk.min_size_pct:g}% "
        f"max_leverage={risk.max_leverage:g}x "
        f"portfolio_exposure={exposure} "
        f"exposure_mode={risk.exposure_mode}"
    )


def default_taxonomy_config() -> RegimeTaxonomyConfig:
    """Return the built-in configuration (current production behaviour)."""
    return parse_taxonomy_config({})


# ----------------------------------------------------------------------- loader
def resolve_taxonomy_config_path(path: str | None = None) -> str:
    """Resolve the config path: explicit argument, env var, canonical default."""
    if path:
        return os.path.abspath(os.path.expanduser(path))
    for env_name in (TAXONOMY_CONFIG_ENV, LEGACY_TAXONOMY_CONFIG_ENV):
        value = os.environ.get(env_name)
        if value:
            return os.path.abspath(os.path.expanduser(value))
    return DEFAULT_TAXONOMY_CONFIG_PATH


def _parse_text(text: str, path: str) -> Any:
    """Parse YAML, falling back to JSON when PyYAML is unavailable."""
    if path.lower().endswith(".json"):
        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            raise TaxonomyConfigError(f"{path}: invalid JSON: {exc}") from exc
    try:
        import yaml  # pylint: disable=import-outside-toplevel
    except ImportError:
        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            raise TaxonomyConfigError(
                f"{path}: PyYAML is unavailable and the document is not valid JSON: {exc}"
            ) from exc
    try:
        return yaml.safe_load(text)
    except yaml.YAMLError as exc:  # type: ignore[attr-defined]
        raise TaxonomyConfigError(f"{path}: invalid YAML: {exc}") from exc


@lru_cache(maxsize=16)
def _load_file(path: str, mtime_ns: int, size: int) -> RegimeTaxonomyConfig:
    """Load and validate one file; the cache key includes mtime and size."""
    del mtime_ns, size
    try:
        with open(path, "r", encoding="utf-8") as handle:
            text = handle.read()
    except OSError as exc:
        raise TaxonomyConfigError(f"{path}: cannot read config: {exc}") from exc
    if not text.strip():
        raise TaxonomyConfigError(f"{path}: config file is empty")
    document = _parse_text(text, path)
    return parse_taxonomy_config(document, source_path=path)


def load_taxonomy_config(path: str | None = None) -> RegimeTaxonomyConfig:
    """Load the taxonomy config, defaulting to the built-ins when absent.

    An explicit ``path`` (or env var) that does not exist still falls back to
    the built-in defaults, matching the "absent config reproduces current
    behaviour" requirement. A file that exists but is invalid raises
    :class:`TaxonomyConfigError`.
    """
    resolved = resolve_taxonomy_config_path(path)
    try:
        stat = os.stat(resolved)
    except OSError:
        return default_taxonomy_config()
    return _load_file(resolved, stat.st_mtime_ns, stat.st_size)


def get_taxonomy_config(path: str | None = None) -> RegimeTaxonomyConfig:
    """Alias for :func:`load_taxonomy_config` used by call sites."""
    return load_taxonomy_config(path)


# ------------------------------------------------------------ output contract
def build_signal_taxonomy_output_contract(
    config: RegimeTaxonomyConfig,
    timeframe: str | None = "1h",
) -> str:
    """Build the optional-taxonomy addendum of the signal output contract.

    The setup labels come from the effective ``regime_setups`` mapping, so the
    prompt can never ask for a setup the config does not allow. Regimes and
    htf alignments are the fixed contract enums.
    """
    labels: list[str] = []
    for regime in REGIMES:
        for name in config.allowed_setups(regime):
            if name not in labels:
                labels.append(name)
    if not labels:
        labels.append("none")
    elif "none" not in labels:
        labels.append("none")
    label = str(timeframe).strip().lower() if timeframe else "1h"
    return (
        f'- Optional {label} taxonomy keys inside "analysis": "setup" '
        f"({'|'.join(labels)}), "
        f'"regime" ({"|".join(REGIMES)}), '
        '"htf_alignment" (aligned|counter|neutral). Omit a key or use null when '
        "unknown; never invent a label outside these lists.\n"
    )


# --------------------------------------------------------------------- rendering
def _sentence(text: str) -> str:
    """Strip and ensure one terminal period on a narrative field."""
    value = text.strip()
    if value and value[-1] not in ".!?":
        value += "."
    return value


def _setup_line(setup: SetupDefinition) -> str:
    """Render one setup line for both schema styles.

    The canonical W3 file uses standalone capitalized sentences; the built-in
    fallback keeps the original lowercase fragment style. Supporting both
    avoids rewriting either source.
    """
    description = setup.description.strip()
    invalidation = _sentence(setup.invalidation)
    if description[:1].isupper():
        prefix = f"{setup.side} — " if setup.side in ("long", "short") else ""
        return f"- {setup.name}: {prefix}{_sentence(description)} {invalidation}".rstrip()
    prefix = f"{setup.side}, " if setup.side in ("long", "short") else ""
    return f"- {setup.name}: {prefix}{description}; {invalidation}"


def _side_annotation(setups: list[SetupDefinition]) -> str:
    """Describe the sides of an allowed family list for the prompt."""
    sides = {setup.side for setup in setups}
    if sides == {"long"}:
        return " (long only)"
    if sides == {"short"}:
        return " (short; on spot these are no-trade because it cannot short)"
    return ""


def render_setup_taxonomy(config: RegimeTaxonomyConfig, timeframe: str | None) -> str:
    """Render the setup taxonomy prompt block for an enabled timeframe.

    Returns an empty string when the timeframe is disabled, so callers can
    append the result unconditionally. The text is generated from the config,
    which means the descriptions, regime mapping and setup list are no longer
    hard-coded in the template manager.
    """
    if not config.timeframe_enabled(timeframe):
        return ""
    label = str(timeframe).strip().lower()
    lines = [
        f"=== Setup Taxonomy ({label}) ===",
        (
            "The REGIME SNAPSHOT block supplied in Trading Context is authoritative. "
            "Use its regime, htf and allowed_setups values as given; do not recompute "
            "ADX/choppiness/ATR or refetch candles, and never infer a regime from a "
            "future candle or from the week's realized return (no look-ahead)."
        ),
        "Regime -> allowed setups:",
    ]

    empty_regimes: list[str] = []
    for regime in REGIMES:
        names = config.allowed_setups(regime)
        if not names:
            empty_regimes.append(regime)
            continue
        setups = [config.setups[name] for name in names]
        lines.append(f"- {regime}: {', '.join(names)}{_side_annotation(setups)}.")
    if empty_regimes:
        lines.append(
            f"- {', '.join(empty_regimes)}: none — HOLD (tradeable=no). "
            "Do not invent a setup for these regimes."
        )

    lines.append(f"Setup definitions (classify on the latest closed {label} candle):")
    rendered: set[str] = set()
    for regime in REGIMES:
        for name in config.allowed_setups(regime):
            if name in rendered:
                continue
            rendered.add(name)
            lines.append(_setup_line(config.setups[name]))
    lines.append(
        "- none: no actionable setup (HOLD/CLOSE/UPDATE, or tradeable=no)."
    )
    lines.append(
        f"htf_alignment (daily/weekly vs the {label} regime): aligned = same direction; "
        "counter = opposite; neutral = no clear higher-timeframe trend. A counter setup "
        "needs exceptional confluence and reduced size; a neutral one needs explicit confirmation."
    )
    exit_reference = "4h" if label == "1h" else "higher-timeframe"
    lines.append(
        f"This is a {label} decision: the execution layer applies the {label} ATR-scaled "
        f"bracket, so propose structural levels and do not carry over {exit_reference} exit "
        "distances. Emit setup, regime and htf_alignment in the decision JSON using exactly "
        "these labels; omit them only when unknown, and never emit a label outside these lists."
    )
    return "\n".join(lines)
