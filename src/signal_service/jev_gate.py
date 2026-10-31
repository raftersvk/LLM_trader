"""JEV gate: typed decision parsing, no-look-ahead state builder and modes.

The gate is a filter, never an executor. It supports four modes:

* ``off``       no JEV call at all (default per PLAN Q6);
* ``pre``       JEV is consulted for logging and decision enrichment only;
* ``post``      the short-only bias filter vetoes a non-HOLD LLM decision;
* ``pre_post``  both checks, each with its own JEV call.

The production policy is **bias only, asymmetric short** (post-OOS decision):
a SELL is admitted only when ``jev.bias == "short"``; a BUY is never filtered
by the JEV bias. The noul tradeability probability is kept in the payload and
logged, but it is never a threshold, the ``trend_down`` short exemption and
the regime gate are retired. HOLD is always a legitimate decision and is never
blocked: the post gate does not even spend a JEV call on it.

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
#: Only used to derive the advisory ``tradeability`` boolean carried in the
#: payload. The gate never filters on it: noul is logged, not enforced.
DEFAULT_TRADEABILITY_THRESHOLD = 0.5
#: Default question set. v1 is the production wording and must stay the
#: default so every existing run is byte-comparable.
DEFAULT_QUESTION_SET = "v1"
QUESTION_SETS = ("v1", "v2")
#: Calibration context lookback and magnitude reference used by v2 states.
CALIBRATION_LOOKBACK_HOURS = 200
CALIBRATION_VOL_HOURS = 24
CALIBRATION_MOVE_THRESHOLD_PCT = 1.0


def normalize_question_set(value: Any) -> str:
    """Return a validated question-set name (``v1`` or ``v2``)."""
    text = str(value).strip().lower()
    if text not in QUESTION_SETS:
        raise ValueError(
            f"Unknown JEV question set: {value!r} (expected {'|'.join(QUESTION_SETS)})"
        )
    return text


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


def _calibration_context(closes: Sequence[float]) -> Dict[str, Any]:
    """Return the optional v2 reference block computed from past closes only.

    The block is a small, documented summary of how this symbol has behaved
    before ``as_of``; it never reads a future candle.

    Formula (all candles are already closed and ordered oldest first):

    * ``hourly_returns`` are simple returns ``close[i] / close[i-1] - 1`` for
      adjacent closed candles with a non-zero previous close. Only the last
      ``CALIBRATION_LOOKBACK_HOURS`` (200) returns are used.
    * ``base_rate_move_1pct`` is the share of those returns whose absolute
      value is at least ``CALIBRATION_MOVE_THRESHOLD_PCT`` percent.
    * ``realized_vol_24h_pct`` is 100 times the population standard deviation
      of the last ``CALIBRATION_VOL_HOURS`` (24) hourly simple returns, i.e.
      the one-day dispersion of hourly returns in percent, not an annualised
      volatility.

    A zero previous close is skipped instead of raising ``ZeroDivisionError``.
    Fields that cannot be computed from the available history are ``None``.
    """
    returns: List[float] = []
    for index in range(1, len(closes)):
        previous = closes[index - 1]
        if previous == 0.0:
            continue
        returns.append(closes[index] / previous - 1.0)

    window = returns[-CALIBRATION_LOOKBACK_HOURS:]
    threshold = CALIBRATION_MOVE_THRESHOLD_PCT / 100.0
    base_rate = (
        sum(1 for value in window if abs(value) >= threshold) / len(window)
        if window
        else None
    )
    vol_window = returns[-CALIBRATION_VOL_HOURS:]
    realized_vol_pct = None
    if len(vol_window) >= 2:
        mean = sum(vol_window) / len(vol_window)
        variance = sum((value - mean) ** 2 for value in vol_window) / len(vol_window)
        realized_vol_pct = 100.0 * math.sqrt(variance)

    return {
        "lookback_hours": CALIBRATION_LOOKBACK_HOURS,
        "hourly_returns_used": len(window),
        "move_threshold_pct": CALIBRATION_MOVE_THRESHOLD_PCT,
        "base_rate_move_1pct": _round(base_rate, 4),
        "vol_hours": CALIBRATION_VOL_HOURS,
        "vol_returns_used": len(vol_window),
        "realized_vol_24h_pct": _round(realized_vol_pct, 6),
    }


def build_state(
    symbol: str,
    timeframe: str,
    candles: Sequence[Sequence[Any]],
    timestamps: Optional[Sequence[datetime]] = None,
    as_of: Optional[datetime] = None,
    calibration_context: bool = False,
    regime_snapshot: Optional[Dict[str, Any]] = None,
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
        calibration_context: Add the optional v2 reference block computed from
            past closes only. ``False`` keeps the v1 state byte-comparable.
        regime_snapshot: Optional deterministic regime snapshot. When omitted
            the v1 state stays byte-comparable; when supplied its trend,
            volatility, HTF and relative-strength fields are added for the
            regime-aware gate. The state is not part of the public contract.

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
        reference = closes[-1 - periods]
        # A corrupt candle with a zero close must not raise ZeroDivisionError:
        # the state builder has to return None for that horizon instead.
        if reference == 0.0:
            return None
        return last_close / reference - 1.0

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
    if regime_snapshot:
        _apply_regime_snapshot(state, regime_snapshot)
    if calibration_context:
        state["calibration_context"] = _calibration_context(closes)
    return state


def _apply_regime_snapshot(state: Dict[str, Any], snapshot: Dict[str, Any]) -> None:
    """Add the deterministic regime fields to the JEV state (non-contractual)."""
    state["adx"] = _round(snapshot.get("adx"))
    state["choppiness"] = _round(snapshot.get("choppiness"))
    state["htf_alignment"] = snapshot.get("htf_alignment")
    state["extension_atr"] = _round(snapshot.get("extension_atr"))
    state["rs_btc_7d"] = _round(snapshot.get("rs_btc_7d"))
    state["rs_btc_30d"] = _round(snapshot.get("rs_btc_30d"))
    state["deterministic_regime"] = snapshot.get("regime")
    state["regime_confidence"] = _round(snapshot.get("confidence"))
    state["no_trade"] = bool(snapshot.get("no_trade"))
    if snapshot.get("allowed_setups") is not None:
        state["allowed_setups"] = list(snapshot.get("allowed_setups") or [])


# ------------------------------------------------------------------ questions
def _build_questions_v1() -> Dict[str, Dict[str, Any]]:
    """Return the production v1 question set used by the gate."""
    return {
        "tradeability": {
            "type": "noul",
            "instructions": (
                "Answer yes if a short-term directional trade on this market "
                "is likely to be worth taking right now based on the state."
            ),
            # Decision 3.0 requires a noul criteria object with exactly the
            # ``true`` and ``false`` keys; a plain string is rejected with
            # ``invalid_question`` and no noul answer.
            "criteria": {
                "true": (
                    "A short-term directional trade is likely to be worth "
                    "taking based only on the provided state."
                ),
                "false": (
                    "Illiquid, choppy or exhausted conditions make a trade "
                    "not worth taking right now."
                ),
            },
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


def _build_questions_v2() -> Dict[str, Dict[str, Any]]:
    """Return the experiment question set separating magnitude and direction.

    The tradeability question asks only whether a large move is likely,
    regardless of its sign. The bias question asks only for direction and is
    explicitly independent of magnitude. Regime and confidence are unchanged
    from v1 so the only difference is the two questions under test.
    """
    questions = _build_questions_v1()
    questions["tradeability"] = {
        "type": "noul",
        "instructions": (
            "Answer yes if the absolute price change over the next hour is "
            "likely to be at least 1.0 percent, regardless of direction."
        ),
        "criteria": {
            "true": (
                "An absolute price change of at least 1.0 percent over the "
                "next hour is likely, no matter which way price moves."
            ),
            "false": (
                "An absolute price change of at least 1.0 percent over the "
                "next hour is unlikely."
            ),
        },
    }
    questions["bias"] = {
        "type": "choice",
        "instructions": (
            "Direction only: which way is price most likely to move over the "
            "next hour? This is independent of how large the move is."
        ),
        "criteria": {
            "long": "Upward direction is more likely than downward, whatever the size.",
            "short": "Downward direction is more likely than upward, whatever the size.",
            "none": "There is no directional edge, whatever the potential size.",
        },
    }
    return questions


def build_questions(question_set: str = DEFAULT_QUESTION_SET) -> Dict[str, Dict[str, Any]]:
    """Return a typed JEV question set.

    Args:
        question_set: ``v1`` (default, production wording) or ``v2`` (separated
            magnitude/direction experiment). An unknown value raises
            ``ValueError`` rather than silently falling back.
    """
    if normalize_question_set(question_set) == "v2":
        return _build_questions_v2()
    return _build_questions_v1()


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
    """Normalize a score answer to [0, 1]; returns (normalized, raw).

    Decision 3.0 returns the expected level (e.g. ``1.69``) plus a ``legend``
    of 2 to 10 levels starting at 0. The level is offset by the legend's
    lowest index and divided by its span, so the top level always maps to
    1.0; a legacy answer without a legend falls back to the previous
    heuristics.
    """
    raw_value = answer.get("score")
    if raw_value is None:
        return None, None
    try:
        raw = float(raw_value)
    except (TypeError, ValueError):
        return None, None
    legend = answer.get("legend")
    levels = 0
    lowest_index: Optional[float] = None
    highest_index: Optional[float] = None
    if isinstance(legend, dict):
        levels = len(legend)
        try:
            numeric_keys = [float(key) for key in legend]
        except (TypeError, ValueError):
            numeric_keys = []
        if numeric_keys:
            lowest_index = min(numeric_keys)
            highest_index = max(numeric_keys)
    elif isinstance(legend, list):
        levels = len(legend)
    if (
        lowest_index is not None
        and highest_index is not None
        and highest_index > lowest_index
    ):
        normalized = (raw - lowest_index) / (highest_index - lowest_index)
    elif levels >= 2:
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
        min_confidence: Optional[float] = None,
    ) -> None:
        """Bind a client and mode.

        Args:
            client: JevClient instance; required for any mode except ``off``.
            mode: off|pre|post|pre_post.
            min_confidence: Optional score threshold below which a non-HOLD
                decision is vetoed.
        """
        self.client = client
        self.mode = parse_mode(mode)
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

    @staticmethod
    def _log_consultation(scope: str, state: Dict[str, Any], signal: JevSignal) -> None:
        """Log the advisory noul/bias values; they are never thresholds.

        The post-OOS policy only filters shorts on ``bias == "short"``. noul is
        kept in the log (and in the payload) so a later study can measure it,
        but it must not gate the decision.
        """
        probability = signal.tradeability_probability
        detail = "unknown" if probability is None else f"{probability:.3f}"
        logger.info(
            "JEV %s %s: noul=%s bias=%s regime=%s (advisory only)",
            scope,
            state.get("symbol", "?"),
            detail,
            signal.bias,
            signal.regime,
        )

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
            signal = parse_jev_signal(response)
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
        """Consult JEV before the LLM call without blocking on noul or regime.

        The pre-gate has no side, so it cannot apply the short-only bias rule.
        It only enriches the decision and logs noul; any filtering happens in
        the post-gate and in the adapter risk layer once the side is known.
        """
        if not self.uses_pre():
            return GateResult(allowed=True, reason="JEV pre-gate disabled", consulted=False)
        result = await self._consult(state)
        if not result.consulted or result.error is not None:
            return result
        signal = result.signal
        assert signal is not None  # consulted implies parsed signal
        self._log_consultation("pre-gate", state, signal)
        return GateResult(
            allowed=True,
            reason="JEV pre-gate: consulted (advisory only)",
            consulted=True,
            signal=signal,
        )

    async def check_post(self, state: Dict[str, Any], action: str) -> GateResult:
        """Veto a non-HOLD LLM decision. HOLD is never blocked.

        Policy: bias only, asymmetric short. A SELL is admitted only when the
        parsed JEV bias is ``short``; a BUY is never filtered by the bias. A
        missing/unparseable bias fails open, and noul is only logged.
        """
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
        self._log_consultation("post-gate", state, signal)
        if (
            normalized_action == "SELL"
            and signal.bias is not None
            and signal.bias != "short"
        ):
            return GateResult(
                allowed=False,
                reason=(
                    "JEV veto: SELL requires short bias "
                    f"(bias={signal.bias})"
                ),
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
