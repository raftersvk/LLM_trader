"""JEV gate: typed decision parsing, no-look-ahead state builder and modes.

The gate is a filter, never an executor. It supports four modes:

* ``off``       no JEV call at all (default per PLAN Q6);
* ``pre``       JEV decides whether the LLM is worth calling at all;
* ``post``      JEV vetoes a non-HOLD LLM decision;
* ``pre_post``  both checks, each with its own JEV call.

HOLD is always a legitimate decision and is never blocked: the post gate does
not even spend a JEV call on it.

The state builder only uses candles that are fully closed at the reference
time. No news, no future candle and no Freqtrade position ever enters the v1
state.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence

from src.signal_service.contract import iso_z, timeframe_seconds
from src.signal_service.jev_client import JevClient, JevError, JevResponse

logger = logging.getLogger(__name__)

ALLOWED_BIAS = frozenset({"long", "short", "none"})
ALLOWED_REGIME = frozenset({"trend", "range", "high_vol", "low_vol"})
DEFAULT_TRADEABILITY_THRESHOLD = 0.5


class JevMode(str, Enum):
    """Supported gate modes."""

    OFF = "off"
    PRE = "pre"
    POST = "post"
    PRE_POST = "pre_post"


def parse_mode(value: Any) -> JevMode:
    """Parse a mode name, raising ValueError on unknown values."""
    if isinstance(value, JevMode):
        return value
    text = str(value).strip().lower()
    for mode in JevMode:
        if mode.value == text:
            return mode
    raise ValueError(f"Unknown JEV mode: {value!r} (expected off|pre|post|pre_post)")


# --------------------------------------------------------------------- state
def _ohlcv(row: Sequence[Any]) -> tuple[float, float, float, float, float]:
    """Normalize one candle row to (open, high, low, close, volume)."""
    values = list(row)
    if len(values) >= 6:
        values = values[-5:]
    if len(values) < 5:
        raise ValueError(f"candle row needs 5 values, got {len(values)}")
    return tuple(float(v) for v in values)  # type: ignore[return-value]


def _mean(values: Sequence[float]) -> Optional[float]:
    """Arithmetic mean, or None for an empty series (NumPy-safe)."""
    if values is None or len(values) == 0:
        return None
    return sum(values) / len(values)


def _round(value: Optional[float], digits: int = 6) -> Optional[float]:
    """Round optional floats, mapping non-finite values to None.

    The exchange layer fills missing OHLCV values with NaN, so one gap in the
    ATR or volume window would otherwise put NaN into the state payload and
    make its JSON serialization invalid (bare NaN literal). Non-finite values
    become None so the state stays valid JSON.
    """
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return round(number, digits)


def closed_candles(
    candles: Sequence[Sequence[Any]],
    timestamps: Sequence[datetime],
    timeframe: str,
    as_of: Optional[datetime] = None,
) -> List[tuple[Sequence[Any], datetime]]:
    """Keep candles whose close time is at or before ``as_of``.

    This is the single no-look-ahead guard of the v1 state: a candle that is
    still forming at ``as_of`` is never used.
    """
    if len(candles) != len(timestamps):
        raise ValueError("candles and timestamps must have the same length")
    cutoff = as_of or datetime.now(timezone.utc)
    period = timedelta(seconds=timeframe_seconds(timeframe))
    kept: List[tuple[Sequence[Any], datetime]] = []
    for row, open_at in zip(candles, timestamps):
        open_utc = open_at if open_at.tzinfo else open_at.replace(tzinfo=timezone.utc)
        if open_utc + period <= cutoff:
            kept.append((row, open_utc))
    return kept


def build_state(
    symbol: str,
    timeframe: str,
    candles: Sequence[Sequence[Any]],
    timestamps: Optional[Sequence[datetime]] = None,
    as_of: Optional[datetime] = None,
) -> Dict[str, Any]:
    """Build the JEV state from closed candles only.

    Args:
        symbol: Trading pair, e.g. "BTC/USDT".
        timeframe: Candle timeframe label.
        candles: OHLCV rows, with or without a leading timestamp column.
        timestamps: Candle open times. When provided, every candle that is not
            fully closed at ``as_of`` is discarded. When omitted, all rows are
            assumed to be closed candles (caller responsibility).
        as_of: Reference time; defaults to the current UTC time.

    Returns:
        A JSON-serializable state dict for JEV.

    Raises:
        ValueError: when fewer than two closed candles are available.
    """
    if timestamps is not None:
        selected = closed_candles(candles, timestamps, timeframe, as_of)
        rows = [row for row, _open_at in selected]
        times = [open_at for _row, open_at in selected]
    else:
        rows = list(candles)
        times = None

    normalized = [_ohlcv(row) for row in rows]
    if len(normalized) < 2:
        raise ValueError("JEV state needs at least two closed candles")

    closes = [bar[3] for bar in normalized]
    highs = [bar[1] for bar in normalized]
    lows = [bar[2] for bar in normalized]
    volumes = [bar[4] for bar in normalized]
    last_close = closes[-1]

    def _return(periods: int) -> Optional[float]:
        if len(closes) <= periods:
            return None
        return last_close / closes[-1 - periods] - 1.0

    def _sma(periods: int) -> Optional[float]:
        if len(closes) < periods:
            return None
        return sum(closes[-periods:]) / periods

    true_ranges: List[float] = []
    for index in range(1, len(normalized)):
        high = highs[index]
        low = lows[index]
        previous_close = closes[index - 1]
        true_ranges.append(max(high - low, abs(high - previous_close), abs(low - previous_close)))
    atr = _mean(true_ranges[-14:])
    average_volume = _mean(volumes[-20:])

    sma_fast = _sma(9)
    sma_slow = _sma(21)

    state: Dict[str, Any] = {
        "symbol": symbol,
        "timeframe": timeframe,
        "as_of": iso_z(as_of or datetime.now(timezone.utc)),
        "candles_used": len(normalized),
        "last_close": _round(last_close),
        "last_candle_close_time": (
            iso_z(times[-1] + timedelta(seconds=timeframe_seconds(timeframe)))
            if times
            else None
        ),
        "returns": {
            "1": _round(_return(1)),
            "4": _round(_return(4)),
            "24": _round(_return(24)),
        },
        "last_range_pct": _round((highs[-1] - lows[-1]) / last_close if last_close else None),
        "atr_pct": _round(atr / last_close if atr and last_close else None),
        "volume_ratio": _round(
            volumes[-1] / average_volume if average_volume and volumes[-1] >= 0 else None
        ),
        "trend": {
            "sma_9": _round(sma_fast),
            "sma_21": _round(sma_slow),
            "sma_spread_pct": _round(
                sma_fast / sma_slow - 1.0 if sma_fast is not None and sma_slow else None
            ),
        },
        "recent_closes": [_round(value) for value in closes[-8:]],
    }
    return state


# ------------------------------------------------------------------ questions
def build_questions() -> Dict[str, Dict[str, Any]]:
    """Return the four typed JEV questions used by the gate."""
    return {
        "tradeability": {
            "type": "noul",
            "instructions": (
                "Answer yes if a short-term directional trade on this market "
                "is likely to be worth taking right now based on the state."
            ),
            "criteria": (
                "Illiquid, choppy or exhausted conditions are not tradeable. "
                "Only the provided state may be used."
            ),
        },
        "bias": {
            "type": "choice",
            "instructions": "What is the most likely short-term directional bias?",
            "criteria": {
                "long": "Upward pressure is the most likely near-term outcome.",
                "short": "Downward pressure is the most likely near-term outcome.",
                "none": "No directional edge is visible right now.",
            },
        },
        "regime": {
            "type": "choice",
            "instructions": "Which market regime best describes the provided state?",
            "criteria": {
                "trend": "A directional move with persistence.",
                "range": "Range-bound oscillation without persistence.",
                "high_vol": "Directional or erratic moves with unusually high volatility.",
                "low_vol": "Compressed volatility with very small ranges.",
            },
        },
        "confidence": {
            "type": "score",
            "instructions": "How strong is the evidence for your bias and tradeability answers?",
            "criteria": [
                "No usable evidence; answers would be a guess.",
                "Weak evidence with heavily conflicting signals.",
                "Moderate evidence with some conflicting signals.",
                "Strong evidence with mostly consistent signals.",
                "Very strong, consistent evidence across the state.",
            ],
        },
    }


# --------------------------------------------------------------------- signal
@dataclass
class JevSignal:
    """Typed, contract-compatible view of one JEV answer set."""

    model: str
    tradeability: Optional[bool] = None
    tradeability_probability: Optional[float] = None
    bias: Optional[str] = None
    bias_confidence: Optional[float] = None
    regime: Optional[str] = None
    regime_confidence: Optional[float] = None
    confidence: Optional[float] = None
    score_raw: Optional[float] = None
    latency_ms: Optional[int] = None

    def to_payload(self) -> Dict[str, Any]:
        """Return only the keys allowed by the shared `jev` contract object."""
        payload: Dict[str, Any] = {"model": self.model}
        if self.tradeability is not None:
            payload["tradeability"] = self.tradeability
        if self.bias is not None:
            payload["bias"] = self.bias
        if self.regime is not None:
            payload["regime"] = self.regime
        return payload


def _normalize_score(answer: Dict[str, Any]) -> tuple[Optional[float], Optional[float]]:
    """Normalize a score answer to [0, 1]; returns (normalized, raw)."""
    raw_value = answer.get("score")
    if raw_value is None:
        return None, None
    try:
        raw = float(raw_value)
    except (TypeError, ValueError):
        return None, None
    legend = answer.get("legend")
    levels = 0
    if isinstance(legend, dict):
        levels = len(legend)
    elif isinstance(legend, list):
        levels = len(legend)
    if levels >= 2:
        normalized = raw / (levels - 1)
    elif 0.0 <= raw <= 1.0:
        normalized = raw
    else:
        normalized = raw / 10.0
    return max(0.0, min(1.0, normalized)), raw


def _choice_value(answer: Dict[str, Any], allowed: frozenset[str], field_name: str) -> Optional[str]:
    """Extract a validated choice value, lowercased."""
    raw = answer.get("choice")
    if raw is None:
        return None
    text = str(raw).strip().lower()
    if text not in allowed:
        logger.warning("JEV returned an unexpected %s choice: %r", field_name, raw)
        return None
    return text


def _probability(answer: Dict[str, Any]) -> Optional[float]:
    """Extract a noul probability from either documented field name."""
    for key in ("noul", "probability"):
        if answer.get(key) is not None:
            try:
                return max(0.0, min(1.0, float(answer[key])))
            except (TypeError, ValueError):
                return None
    return None


def parse_jev_signal(
    response: JevResponse,
    tradeability_threshold: float = DEFAULT_TRADEABILITY_THRESHOLD,
) -> JevSignal:
    """Convert raw JEV answers into a typed, validated JevSignal."""
    tradeability_answer = response.answer("tradeability")
    bias_answer = response.answer("bias")
    regime_answer = response.answer("regime")
    confidence_answer = response.answer("confidence")

    probability = _probability(tradeability_answer)
    tradeability = None if probability is None else probability >= tradeability_threshold
    confidence, score_raw = _normalize_score(confidence_answer)

    return JevSignal(
        model=response.model,
        tradeability=tradeability,
        tradeability_probability=probability,
        bias=_choice_value(bias_answer, ALLOWED_BIAS, "bias"),
        bias_confidence=_as_optional_float(bias_answer.get("confidence")),
        regime=_choice_value(regime_answer, ALLOWED_REGIME, "regime"),
        regime_confidence=_as_optional_float(regime_answer.get("confidence")),
        confidence=confidence,
        score_raw=score_raw,
        latency_ms=response.latency_ms,
    )


def _as_optional_float(value: Any) -> Optional[float]:
    """Coerce a value to float when possible."""
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


# ----------------------------------------------------------------------- gate
@dataclass
class GateResult:
    """Result of one gate check."""

    allowed: bool
    reason: str
    consulted: bool = False
    signal: Optional[JevSignal] = None
    error: Optional[str] = None


class JevGate:
    """Applies the configured JEV mode to a state and/or an LLM decision."""

    def __init__(
        self,
        client: Optional[JevClient] = None,
        mode: Any = JevMode.OFF,
        tradeability_threshold: float = DEFAULT_TRADEABILITY_THRESHOLD,
        min_confidence: Optional[float] = None,
    ) -> None:
        """Bind a client and mode.

        Args:
            client: JevClient instance; required for any mode except ``off``.
            mode: off|pre|post|pre_post.
            tradeability_threshold: noul probability needed to consider the
                market tradeable.
            min_confidence: Optional score threshold below which a non-HOLD
                decision is vetoed.
        """
        self.client = client
        self.mode = parse_mode(mode)
        self.tradeability_threshold = tradeability_threshold
        self.min_confidence = min_confidence
        self.last_signal: Optional[JevSignal] = None
        self.last_error: Optional[str] = None

    @property
    def active(self) -> bool:
        """True when the mode consults JEV at all."""
        return self.mode is not JevMode.OFF

    def uses_pre(self) -> bool:
        """True when the mode runs the pre-LLM gate."""
        return self.mode in (JevMode.PRE, JevMode.PRE_POST)

    def uses_post(self) -> bool:
        """True when the mode runs the post-LLM gate."""
        return self.mode in (JevMode.POST, JevMode.PRE_POST)

    async def _consult(self, state: Dict[str, Any]) -> GateResult:
        """Call JEV once and parse the answer set.

        Any JEV failure fails open: the gate is a filter and must not silently
        turn an outage into a blocked decision. The failure is logged and kept
        in `last_error`.
        """
        if self.client is None:
            return GateResult(
                allowed=True,
                reason="JEV client not configured (fail-open)",
                consulted=False,
                error="no client",
            )
        try:
            response = await self.client.ask(state, build_questions())
            signal = parse_jev_signal(response, self.tradeability_threshold)
        except JevError as exc:
            self.last_error = str(exc)
            logger.warning("JEV unavailable, failing open: %s", exc)
            return GateResult(
                allowed=True,
                reason=f"JEV unavailable (fail-open): {exc}",
                consulted=True,
                error=str(exc),
            )
        except Exception as exc:  # pylint: disable=broad-exception-caught
            # Transport bugs and parser failures (e.g. a malformed payload
            # raising TypeError/AttributeError) must not abort the analysis:
            # the documented fail-open contract covers every failure mode.
            detail = f"{type(exc).__name__}: {exc}"
            self.last_error = detail
            logger.warning("JEV gate error, failing open: %s", detail)
            return GateResult(
                allowed=True,
                reason=f"JEV error (fail-open): {detail}",
                consulted=True,
                error=detail,
            )
        self.last_signal = signal
        return GateResult(allowed=True, reason="JEV consulted", consulted=True, signal=signal)

    async def check_pre(self, state: Dict[str, Any]) -> GateResult:
        """Decide whether the LLM should be called at all."""
        if not self.uses_pre():
            return GateResult(allowed=True, reason="JEV pre-gate disabled", consulted=False)
        result = await self._consult(state)
        if not result.consulted or result.error is not None:
            return result
        signal = result.signal
        assert signal is not None  # consulted implies parsed signal
        if signal.tradeability is False:
            probability = signal.tradeability_probability
            detail = "unknown" if probability is None else f"{probability:.3f}"
            return GateResult(
                allowed=False,
                reason=(
                    "JEV pre-gate: not tradeable "
                    f"(noul={detail} < {self.tradeability_threshold})"
                ),
                consulted=True,
                signal=signal,
            )
        if signal.bias == "none":
            return GateResult(
                allowed=False,
                reason="JEV pre-gate: bias none",
                consulted=True,
                signal=signal,
            )
        return GateResult(allowed=True, reason="JEV pre-gate: tradeable", consulted=True, signal=signal)

    async def check_post(self, state: Dict[str, Any], action: str) -> GateResult:
        """Veto a non-HOLD LLM decision. HOLD is never blocked."""
        if not self.uses_post():
            return GateResult(allowed=True, reason="JEV post-gate disabled", consulted=False)
        normalized_action = str(action).upper()
        if normalized_action not in ("BUY", "SELL"):
            return GateResult(
                allowed=True,
                reason="HOLD exempt from JEV veto",
                consulted=False,
            )
        result = await self._consult(state)
        if not result.consulted or result.error is not None:
            return result
        signal = result.signal
        assert signal is not None  # consulted implies parsed signal
        if signal.tradeability is False:
            probability = signal.tradeability_probability
            detail = "unknown" if probability is None else f"{probability:.3f}"
            return GateResult(
                allowed=False,
                reason=(
                    "JEV veto: not tradeable "
                    f"(noul={detail} < {self.tradeability_threshold})"
                ),
                consulted=True,
                signal=signal,
            )
        if normalized_action == "BUY" and signal.bias == "short":
            return GateResult(
                allowed=False,
                reason="JEV veto: short bias conflicts with BUY",
                consulted=True,
                signal=signal,
            )
        if normalized_action == "SELL" and signal.bias == "long":
            return GateResult(
                allowed=False,
                reason="JEV veto: long bias conflicts with SELL",
                consulted=True,
                signal=signal,
            )
        if (
            self.min_confidence is not None
            and signal.confidence is not None
            and signal.confidence < self.min_confidence
        ):
            return GateResult(
                allowed=False,
                reason=(
                    "JEV veto: confidence "
                    f"{signal.confidence:.2f} < {self.min_confidence:.2f}"
                ),
                consulted=True,
                signal=signal,
            )
        return GateResult(
            allowed=True,
            reason="JEV post-gate: decision approved",
            consulted=True,
            signal=signal,
        )
