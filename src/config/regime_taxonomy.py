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
from dataclasses import dataclass, field, replace
from functools import lru_cache
from typing import Any, ClassVar, Mapping

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

DEFAULT_ENABLED_TIMEFRAMES: tuple[str, ...] = ("1h",)
DEFAULT_JEV_LOG_PATH = os.path.join(
    os.path.dirname(DEFAULT_TAXONOMY_CONFIG_PATH), "jev_decisions.jsonl"
)


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
    sl_atr: float
    tp_atr: float
    max_hold_bars: int
    trail_atr: float
    be_at_r: float

    _FIELDS: ClassVar[tuple[str, ...]] = (
        "side",
        "description",
        "entry",
        "invalidation",
        "regimes",
        "sl_atr",
        "tp_atr",
        "max_hold_bars",
        "trail_atr",
        "be_at_r",
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
        if base is None and set(cls._FIELDS) - set(data):
            missing = ", ".join(sorted(set(cls._FIELDS) - set(data)))  # type: ignore[attr-defined]
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
            sl_atr=_number(
                data.get("sl_atr", base.sl_atr if base else None),
                f"setups.{name}.sl_atr",
                minimum=0.0,
                allow_zero=False,
            ),
            tp_atr=_number(
                data.get("tp_atr", base.tp_atr if base else None),
                f"setups.{name}.tp_atr",
                minimum=0.0,
                allow_zero=False,
            ),
            max_hold_bars=_positive_int(
                data.get("max_hold_bars", base.max_hold_bars if base else None),
                f"setups.{name}.max_hold_bars",
            ),
            trail_atr=_number(
                data.get("trail_atr", base.trail_atr if base else 0.0),
                f"setups.{name}.trail_atr",
                minimum=0.0,
                allow_zero=True,
            ),
            be_at_r=_number(
                data.get("be_at_r", base.be_at_r if base else 0.0),
                f"setups.{name}.be_at_r",
                minimum=0.0,
                allow_zero=True,
            ),
        )

    def as_dict(self) -> dict[str, Any]:
        """Return the setup as a JSON/YAML-compatible mapping."""
        return {
            "side": self.side,
            "description": self.description,
            "entry": self.entry,
            "invalidation": self.invalidation,
            "regimes": list(self.regimes),
            "sl_atr": self.sl_atr,
            "tp_atr": self.tp_atr,
            "max_hold_bars": self.max_hold_bars,
            "trail_atr": self.trail_atr,
            "be_at_r": self.be_at_r,
        }


@dataclass(frozen=True)
class JevSettings:
    """JEV gate policy that belongs to the taxonomy document."""

    bias_short_only: bool = True
    log_decisions: bool = False
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
class PairOverride:
    """Per-pair overrides: thresholds and/or allowed setup families."""

    regime_thresholds: RegimeThresholds | None = None
    regime_setups: Mapping[str, tuple[str, ...]] | None = None


@dataclass(frozen=True)
class RegimeTaxonomyConfig:
    """Validated taxonomy document used by the snapshot, prompt and JEV gate."""

    version: str = "1.0"
    enabled_timeframes: tuple[str, ...] = DEFAULT_ENABLED_TIMEFRAMES
    regime_thresholds: RegimeThresholds = field(default_factory=RegimeThresholds)
    regime_setups: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    setups: Mapping[str, SetupDefinition] = field(default_factory=dict)
    pair_overrides: Mapping[str, PairOverride] = field(default_factory=dict)
    jev: JevSettings = field(default_factory=JevSettings)
    source_path: str | None = None

    def timeframe_enabled(self, timeframe: str | None) -> bool:
        """True when the taxonomy may be applied to this timeframe."""
        return str(timeframe).strip().lower() in self.enabled_timeframes

    def allowed_setups(self, regime: str) -> tuple[str, ...]:
        """Allowed setup families for a regime (empty when no-trade)."""
        return tuple(self.regime_setups.get(regime, ()))

    def for_symbol(self, symbol: str | None) -> "RegimeTaxonomyConfig":
        """Return the effective config with this pair's overrides applied."""
        if not symbol:
            return self
        override = self.pair_overrides.get(symbol)
        if override is None:
            # Accept the compact BTCUSDT form as well as BTC/USDT.
            compact = symbol.replace("/", "").upper()
            for key, candidate in self.pair_overrides.items():
                if key.replace("/", "").upper() == compact:
                    override = candidate
                    break
        if override is None:
            return self
        thresholds = override.regime_thresholds or self.regime_thresholds
        regime_setups = dict(self.regime_setups)
        if override.regime_setups:
            regime_setups.update(override.regime_setups)
        return replace(
            self,
            regime_thresholds=thresholds,
            regime_setups=regime_setups,
        )

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON/YAML-compatible representation of the document."""
        return {
            "version": self.version,
            "enabled_timeframes": list(self.enabled_timeframes),
            "regime_thresholds": self.regime_thresholds.as_dict(),
            "regime_setups": {key: list(value) for key, value in self.regime_setups.items()},
            "setups": {name: setup.as_dict() for name, setup in self.setups.items()},
            "pair_overrides": {
                pair: {
                    "regime_thresholds": (
                        override.regime_thresholds.as_dict()
                        if override.regime_thresholds
                        else None
                    ),
                    "regime_setups": (
                        {key: list(value) for key, value in override.regime_setups.items()}
                        if override.regime_setups
                        else None
                    ),
                }
                for pair, override in self.pair_overrides.items()
            },
            "jev": {
                "bias_short_only": self.jev.bias_short_only,
                "log_decisions": self.jev.log_decisions,
                "log_path": self.jev.log_path,
            },
        }


# --------------------------------------------------------------------- defaults
def _default_setup(
    name: str,
    side: str,
    description: str,
    invalidation: str,
    regimes: tuple[str, ...],
    sl_atr: float,
    tp_atr: float,
    max_hold_bars: int,
    trail_atr: float = 0.0,
    be_at_r: float = 0.0,
) -> SetupDefinition:
    """Build one built-in setup definition (entry mirrors description)."""
    return SetupDefinition(
        name=name,
        side=side,
        description=description,
        entry=description,
        invalidation=invalidation,
        regimes=regimes,
        sl_atr=sl_atr,
        tp_atr=tp_atr,
        max_hold_bars=max_hold_bars,
        trail_atr=trail_atr,
        be_at_r=be_at_r,
    )


DEFAULT_SETUPS: Mapping[str, SetupDefinition] = {
    "trend_pullback": _default_setup(
        "trend_pullback",
        "long",
        "orderly pullback to SMA20/structure support in an uptrend with a rejection close",
        "invalidated below the pullback low",
        ("trend_up",),
        2.5,
        4.5,
        24,
    ),
    "breakout_retest": _default_setup(
        "breakout_retest",
        "long",
        "break of a 20-bar high/range then a held retest of the broken level",
        "invalidated back below that level",
        ("trend_up",),
        2.5,
        4.5,
        24,
    ),
    "momentum_continuation": _default_setup(
        "momentum_continuation",
        "long",
        "higher-high/higher-low continuation without >2 ATR extension from SMA20",
        "invalidated by a lower low",
        ("trend_up",),
        2.5,
        4.0,
        12,
    ),
    "rally_failure": _default_setup(
        "rally_failure",
        "short",
        "rally into SMA20/resistance rejected with a lower high in a downtrend",
        "invalidated above the swing high",
        ("trend_down",),
        2.5,
        4.5,
        24,
    ),
    "breakdown_retest": _default_setup(
        "breakdown_retest",
        "short",
        "support break then a rejected retest of the broken level",
        "invalidated back above it",
        ("trend_down",),
        2.5,
        4.5,
        24,
    ),
    "momentum_short": _default_setup(
        "momentum_short",
        "short",
        "lower-high/lower-low continuation without >2 ATR extension",
        "invalidated by a higher high",
        ("trend_down",),
        2.5,
        4.0,
        12,
    ),
    "range_fade_extreme": _default_setup(
        "range_fade_extreme",
        "both",
        "fade a wick beyond the range boundary that closes back inside",
        "invalidated beyond the boundary",
        ("range",),
        2.0,
        2.5,
        10,
    ),
    "range_mean_reversion": _default_setup(
        "range_mean_reversion",
        "both",
        "return toward mid/VWAP after an excursion",
        "invalidated by a break of the range",
        ("range",),
        2.0,
        2.5,
        10,
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


def _positive_int(value: Any, label: str) -> int:
    """Validate a strictly positive integer."""
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise TaxonomyConfigError(f"{label} must be a positive integer")
    return value


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


def _parse_regime_setups(
    data: Any,
    base: Mapping[str, tuple[str, ...]],
    setups: Mapping[str, SetupDefinition],
    label: str = "regime_setups",
    complete: bool = True,
) -> dict[str, tuple[str, ...]]:
    """Parse the regime -> allowed setups mapping.

    ``complete`` fills every regime from ``base`` (the root section); a pair
    override passes False so only the regimes named by the override are
    returned and the others keep their effective value.
    """
    if data is None:
        if not complete:
            return {}
        return {regime: tuple(base.get(regime, ())) for regime in REGIMES}
    if not isinstance(data, Mapping):
        raise TaxonomyConfigError(f"{label} must be a mapping")
    unknown = [regime for regime in data if regime not in REGIMES]
    if unknown:
        raise TaxonomyConfigError(
            f"{label} has unknown regimes: {', '.join(map(str, unknown))}"
        )
    result = (
        {regime: tuple(base.get(regime, ())) for regime in REGIMES}
        if complete
        else {}
    )
    for regime, names in data.items():
        allowed = _string_tuple(names, f"{label}.{regime}")
        missing = [name for name in allowed if name not in setups]
        if missing:
            raise TaxonomyConfigError(
                f"{label}.{regime} references unknown setups: {', '.join(missing)}"
            )
        result[regime] = allowed
    return result


def _parse_pair_overrides(
    data: Any,
    thresholds: RegimeThresholds,
    setups: Mapping[str, SetupDefinition],
) -> dict[str, PairOverride]:
    """Parse per-pair threshold/setup overrides."""
    if data is None:
        return {}
    if not isinstance(data, Mapping):
        raise TaxonomyConfigError("pair_overrides must be a mapping")
    overrides: dict[str, PairOverride] = {}
    for pair, raw in data.items():
        pair_name = _string(pair, "pair_overrides key")
        if raw is None:
            overrides[pair_name] = PairOverride()
            continue
        if not isinstance(raw, Mapping):
            raise TaxonomyConfigError(f"pair_overrides.{pair_name} must be a mapping")
        unknown = set(raw) - {"regime_thresholds", "regime_setups"}
        if unknown:
            raise TaxonomyConfigError(
                f"pair_overrides.{pair_name} has unknown keys: "
                f"{', '.join(sorted(map(str, unknown)))}"
            )
        override_thresholds = None
        if raw.get("regime_thresholds") is not None:
            override_thresholds = RegimeThresholds.from_mapping(
                raw["regime_thresholds"],
                thresholds,
                f"pair_overrides.{pair_name}.regime_thresholds",
            )
        override_setups = None
        if raw.get("regime_setups") is not None:
            override_setups = _parse_regime_setups(
                raw["regime_setups"],
                {},
                setups,
                f"pair_overrides.{pair_name}.regime_setups",
                complete=False,
            )
        overrides[pair_name] = PairOverride(
            regime_thresholds=override_thresholds,
            regime_setups=override_setups,
        )
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
    }
    unknown = set(document) - known
    if unknown:
        raise TaxonomyConfigError(
            f"taxonomy document has unknown keys: {', '.join(sorted(map(str, unknown)))}"
        )

    version_raw = document.get("version", "1.0")
    version = str(version_raw).strip() if isinstance(version_raw, (str, int, float)) else ""
    if not version:
        raise TaxonomyConfigError("version must be a non-empty string or number")

    enabled_raw = document.get("enabled_timeframes", DEFAULT_ENABLED_TIMEFRAMES)
    enabled = tuple(item.lower() for item in _string_tuple(enabled_raw, "enabled_timeframes"))
    if not enabled:
        raise TaxonomyConfigError("enabled_timeframes must enable at least one timeframe")

    thresholds = RegimeThresholds.from_mapping(document.get("regime_thresholds"))
    setups = _parse_setups(document.get("setups"), DEFAULT_SETUPS)
    regime_setups = _parse_regime_setups(document.get("regime_setups"), DEFAULT_REGIME_SETUPS, setups)
    pair_overrides = _parse_pair_overrides(document.get("pair_overrides"), thresholds, setups)
    jev = JevSettings.from_mapping(document.get("jev"))

    return RegimeTaxonomyConfig(
        version=version,
        enabled_timeframes=enabled,
        regime_thresholds=thresholds,
        regime_setups=regime_setups,
        setups=setups,
        pair_overrides=pair_overrides,
        jev=jev,
        source_path=source_path,
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


# --------------------------------------------------------------------- rendering
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
            setup = config.setups[name]
            prefix = f"{setup.side}, " if setup.side in ("long", "short") else ""
            lines.append(
                f"- {name}: {prefix}{setup.description}; {setup.invalidation}."
            )
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
