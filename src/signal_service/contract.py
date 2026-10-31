"""Signal contract v1.

Implements the shared contract defined in trading-stack/contracts/signal.schema.json.
The signal is the immutable boundary between the LLM brain and execution.
The full raw LLM response is NEVER stored or logged here: only `reasoning`
is kept, plus a sha256 fingerprint of the raw response for audit.
"""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Literal, Optional

SIGNAL_SCHEMA_VERSION = "1.0"
#: Schema versions a reader must accept. 1.1 adds UPDATE/CLOSE position records
#: on a separate channel; 1.0 entry records are unchanged.
SIGNAL_SCHEMA_VERSIONS = frozenset({"1.0", "1.1"})

#: Position-related actions (v1.1). They never enter a new position.
POSITION_ACTIONS = frozenset({"UPDATE", "CLOSE"})
POSITION_SIDES = frozenset({"long", "short"})

SUPPORTED_TIMEFRAMES = {
    "1m": 60,
    "5m": 300,
    "15m": 900,
    "30m": 1800,
    "1h": 3600,
    "4h": 14400,
    "1d": 86400,
}

#: Signal actions. v1.0 uses BUY/SELL/HOLD; v1.1 adds UPDATE/CLOSE position
#: records, which are written to a separate updates channel.
Action = Literal["BUY", "SELL", "HOLD", "UPDATE", "CLOSE"]

ALLOWED_ACTIONS = frozenset({"BUY", "SELL", "HOLD", "UPDATE", "CLOSE"})

ALLOWED_SOURCES = frozenset({"llm_trader", "fenixai", "manual", "jev"})

# HIGH/MEDIUM/LOW -> float in [0, 1]. MODERATE is accepted as a synonym of
# MEDIUM because providers occasionally emit it; it must never invalidate an
# otherwise usable signal.
CONFIDENCE_MAP: Dict[str, float] = {
    "HIGH": 0.90,
    "MEDIUM": 0.65,
    "MODERATE": 0.65,
    "LOW": 0.35,
}

_SYMBOL_PATTERN = re.compile(r"^[A-Z0-9]+/[A-Z0-9]+$")
_HASH_PATTERN = re.compile(r"^sha256:[a-f0-9]{64}$")


class SignalValidationError(ValueError):
    """Raised when a signal does not conform to the signal contract."""


def utc_now_iso() -> str:
    """Return the current UTC time as an ISO-8601 string with Z suffix."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def iso_z(dt: datetime) -> str:
    """Format a datetime as an ISO-8601 string with Z suffix."""
    return dt.astimezone(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def parse_iso(ts: str) -> datetime:
    """Parse an ISO-8601 timestamp into a timezone-aware UTC datetime."""
    parsed = datetime.fromisoformat(ts.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def timeframe_seconds(timeframe: str) -> int:
    """Convert a timeframe label to a number of seconds."""
    if timeframe not in SUPPORTED_TIMEFRAMES:
        raise SignalValidationError(f"Unsupported timeframe: {timeframe!r}")
    return SUPPORTED_TIMEFRAMES[timeframe]


def confidence_to_float(confidence: Any) -> float:
    """Map HIGH/MEDIUM/MODERATE/LOW (or a 0-100 numeric) to a float in [0, 1].

    A numeric score is preserved instead of being collapsed to a label: it is
    clamped to 0-100, then rounded to the nearest 5 percentage points with
    half-up rounding. Only values strictly between 0 and 1 are treated as an
    already-normalized fraction; 1 and 1.0 are valid 0-100 scores (a raw score
    of 1 is minimum confidence, not 100%). For example, 51 and 52 map to 0.50,
    53 to 0.55, 57 to 0.55 and 58 to 0.60. Labels remain the fallback and keep
    their historical mapping (HIGH=0.90, MEDIUM=0.65, MODERATE=0.65, LOW=0.35).
    MODERATE is treated as an exact synonym of MEDIUM. Label matching is
    case-insensitive and surrounding whitespace is ignored. Non-finite scores
    (NaN, Infinity) are rejected with :class:`SignalValidationError` rather
    than clamped into a valid range.

    Args:
        confidence: "HIGH", "MEDIUM", "MODERATE", "LOW" or a numeric 0-100
            value.

    Returns:
        Confidence as a float between 0 and 1.

    Raises:
        SignalValidationError: for labels or numbers that are not valid
            confidence values, including non-finite scores.
    """
    if isinstance(confidence, (int, float)):
        raw_100 = float(confidence)
        if not math.isfinite(raw_100):
            raise SignalValidationError(f"Invalid confidence value: {confidence!r}")
        # 1 and 1.0 are valid 0-100 scores (the extractor forwards raw scores),
        # so only a strictly fractional value is rescaled.
        if 0.0 < raw_100 < 1.0:
            raw_100 *= 100.0
        raw_100 = max(0.0, min(100.0, raw_100))
        # Add half a step before truncating to round half-up (e.g. 52.5 -> 55).
        rounded = int(raw_100 / 5.0 + 0.5) * 5
        return rounded / 100.0
    text = str(confidence).strip().upper()
    if text in CONFIDENCE_MAP:
        return CONFIDENCE_MAP[text]
    try:
        value = float(text)
        return confidence_to_float(value)
    except ValueError as exc:
        raise SignalValidationError(f"Invalid confidence value: {confidence!r}") from exc


def build_decision_id(symbol: str, timeframe: str, candle_open_time: str) -> str:
    """Build a deterministic decision id from symbol, timeframe and candle open time.

    Two runs over the same candle produce the same id, enabling idempotency
    and deduplication.
    """
    token = f"{symbol.replace('/', '-')}-{timeframe}-{candle_open_time}".lower()
    return token


def hash_raw_response(raw_response: str) -> str:
    """Return the sha256 fingerprint of a raw LLM response.

    The raw response itself must never be persisted by the signal service.
    """
    return f"sha256:{hashlib.sha256(raw_response.encode('utf-8')).hexdigest()}"


def _check_common_fields(
    symbol: str,
    timeframe: str,
    action: str,
    confidence: float,
    position_size_pct: float,
    reasoning: str,
    source: str,
) -> None:
    if not _SYMBOL_PATTERN.match(symbol):
        raise SignalValidationError(f"Invalid symbol format: {symbol!r}")
    if timeframe not in SUPPORTED_TIMEFRAMES:
        raise SignalValidationError(f"Unsupported timeframe: {timeframe!r}")
    if action not in ALLOWED_ACTIONS:
        raise SignalValidationError(f"Invalid action: {action!r}")
    if source not in ALLOWED_SOURCES:
        raise SignalValidationError(f"Invalid source: {source!r}")
    if not isinstance(confidence, float) or not 0.0 <= confidence <= 1.0:
        raise SignalValidationError("Confidence must be a float in [0, 1]")
    if not isinstance(position_size_pct, (int, float)) or not 0.0 <= position_size_pct <= 1.0:
        raise SignalValidationError("position_size_pct must be a number in [0, 1]")
    if len(reasoning) > 8000:
        raise SignalValidationError("reasoning exceeds 8000 characters")


@dataclass
class Signal:
    """Immutable trading signal matching contracts/signal.schema.json v1.0."""

    schema_version: str
    decision_id: str
    source: str
    symbol: str
    timeframe: str
    candle_open_time: str
    candle_close_time: str
    created_at: str
    expires_at: str
    action: str
    confidence: float
    risk_approved: bool
    sl: Optional[float]
    tp: Optional[float]
    position_size_pct: float
    reasoning: str
    model: str
    provider: str
    prompt_version: Optional[str] = None
    latency_ms: Optional[int] = None
    raw_response_hash: Optional[str] = None
    jev: Optional[Dict[str, Any]] = field(default=None)
    # v1.1 position-update fields (omitted on v1.0 entry records).
    # position_id is market-scoped: "<market>:<trade_id>" (spot|futures),
    # because the two Freqtrade databases have separate trade-id sequences.
    position_id: Optional[str] = None
    position_side: Optional[str] = None
    revision: Optional[int] = None
    based_on_entry: Optional[str] = None
    close_reason: Optional[str] = None
    source_price: Optional[float] = None

    def validate(self) -> None:
        """Validate the signal against the contract invariants.

        Raises:
            SignalValidationError: when any invariant is violated.
        """
        if self.schema_version not in SIGNAL_SCHEMA_VERSIONS:
            raise SignalValidationError(
                f"schema_version must be one of {sorted(SIGNAL_SCHEMA_VERSIONS)!r}, "
                f"got {self.schema_version!r}"
            )
        if not self.decision_id:
            raise SignalValidationError("decision_id must not be empty")
        _check_common_fields(
            self.symbol, self.timeframe, self.action,
            self.confidence, self.position_size_pct, self.reasoning, self.source,
        )
        if self.model is None or len(self.model) < 1:
            raise SignalValidationError("model is required")
        if self.provider is None or len(self.provider) < 1:
            raise SignalValidationError("provider is required")

        if self.raw_response_hash is not None and not _HASH_PATTERN.match(self.raw_response_hash):
            raise SignalValidationError("raw_response_hash must match 'sha256:<64 hex chars>'")

        open_at = parse_iso(self.candle_open_time)
        tf_seconds = timeframe_seconds(self.timeframe)
        expected_close = open_at + timedelta(seconds=tf_seconds)
        if parse_iso(self.candle_close_time) != expected_close:
            raise SignalValidationError("candle_close_time must equal candle_open_time + timeframe")
        if parse_iso(self.expires_at) != expected_close + timedelta(seconds=tf_seconds):
            raise SignalValidationError("expires_at must equal candle_close_time + timeframe")

        for name, value in (("sl", self.sl), ("tp", self.tp)):
            if value is None:
                continue
            # json.loads accepts NaN/Infinity: an infinite level must never be
            # persisted (an infinite TP would disable the consumer's exit).
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value <= 0
            ):
                raise SignalValidationError(f"{name} must be a positive finite number or null")

        if self.action in ("BUY", "SELL"):
            if self.sl is None or self.tp is None:
                raise SignalValidationError(f"{self.action} requires non-null sl and tp")
        # HOLD may carry null sl/tp (or valid numbers); it never triggers an
        # entry in Freqtrade. BUY/SELL require non-null sl/tp (see above).

        if self.action in POSITION_ACTIONS:
            self._validate_position_action()

        if self.jev is not None:
            if not isinstance(self.jev, dict) or set(self.jev) - {
                "model", "tradeability", "bias", "regime"
            }:
                raise SignalValidationError("jev payload has unexpected keys")

    def _validate_position_action(self) -> None:
        """Validate the v1.1 UPDATE/CLOSE invariants."""
        if self.schema_version != "1.1":
            raise SignalValidationError(
                f"{self.action} requires schema_version '1.1', got {self.schema_version!r}"
            )
        if not isinstance(self.position_id, str) or not self.position_id:
            # The contract shape is a string; position_sync stringifies the
            # integer Freqtrade trade_id before the producer writes a record.
            raise SignalValidationError(f"{self.action} requires a non-empty string position_id")
        if self.position_side not in POSITION_SIDES:
            raise SignalValidationError(
                f"{self.action} requires position_side in {sorted(POSITION_SIDES)!r}"
            )
        if not self.based_on_entry:
            raise SignalValidationError(f"{self.action} requires based_on_entry")
        if not isinstance(self.revision, int) or isinstance(self.revision, bool) or self.revision < 1:
            raise SignalValidationError(f"{self.action} requires revision >= 1")
        if self.source_price is not None and (
            not isinstance(self.source_price, (int, float)) or self.source_price <= 0
        ):
            raise SignalValidationError("source_price must be a positive number or null")
        if self.action == "UPDATE":
            if self.sl is None and self.tp is None:
                raise SignalValidationError("UPDATE requires a new sl and/or tp")
        if self.action == "CLOSE":
            if self.close_reason is None or not str(self.close_reason).strip():
                raise SignalValidationError("CLOSE requires close_reason")
            if self.sl is not None or self.tp is not None:
                raise SignalValidationError("CLOSE must not carry sl/tp")

    def is_expired(self, now: Optional[datetime] = None) -> bool:
        """Return True when the signal is past its expiry time.

        Args:
            now: Evaluation time; defaults to current UTC time.
        """
        return parse_iso(self.expires_at) <= (now or datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a JSON-compatible dict (schema-valid shape).

        Optional fields are omitted (not emitted as null) when unset, because
        the shared schema types them strictly (e.g. latency_ms: integer).
        """
        data = {
            "schema_version": self.schema_version,
            "decision_id": self.decision_id,
            "source": self.source,
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "candle_open_time": self.candle_open_time,
            "candle_close_time": self.candle_close_time,
            "created_at": self.created_at,
            "expires_at": self.expires_at,
            "action": self.action,
            "confidence": self.confidence,
            "risk_approved": self.risk_approved,
            "sl": self.sl,
            "tp": self.tp,
            "position_size_pct": self.position_size_pct,
            "reasoning": self.reasoning,
            "model": self.model,
            "provider": self.provider,
        }
        optional = (
            ("prompt_version", self.prompt_version),
            ("latency_ms", self.latency_ms),
            ("raw_response_hash", self.raw_response_hash),
            ("jev", self.jev),
            ("position_id", self.position_id),
            ("position_side", self.position_side),
            ("revision", self.revision),
            ("based_on_entry", self.based_on_entry),
            ("close_reason", self.close_reason),
            ("source_price", self.source_price),
        )
        for key, value in optional:
            if value is not None:
                data[key] = value
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Signal":
        """Build a signal from a dict and validate it.

        Raises:
            SignalValidationError: on any missing key or contract violation.
        """
        required = {
            "schema_version", "decision_id", "source", "symbol", "timeframe",
            "candle_open_time", "candle_close_time", "created_at", "expires_at",
            "action", "confidence", "risk_approved", "sl", "tp", "position_size_pct",
            "reasoning", "model", "provider",
        }
        if required - set(data):
            missing = ", ".join(sorted(required - set(data)))
            raise SignalValidationError(f"missing required fields: {missing}")
        allowed = required | {
            "prompt_version", "latency_ms", "raw_response_hash", "jev",
            "position_id", "position_side", "revision", "based_on_entry",
            "close_reason", "source_price",
        }
        extra = set(data) - allowed
        if extra:
            raise SignalValidationError(f"unexpected fields: {', '.join(sorted(extra))}")
        signal = cls(
            schema_version=data["schema_version"],
            decision_id=data["decision_id"],
            source=data["source"],
            symbol=data["symbol"],
            timeframe=data["timeframe"],
            candle_open_time=data["candle_open_time"],
            candle_close_time=data["candle_close_time"],
            created_at=data["created_at"],
            expires_at=data["expires_at"],
            action=data["action"],
            confidence=float(data["confidence"]),
            risk_approved=bool(data["risk_approved"]),
            sl=data["sl"],
            tp=data["tp"],
            position_size_pct=float(data["position_size_pct"]),
            reasoning=data["reasoning"],
            model=data["model"],
            provider=data["provider"],
            prompt_version=data.get("prompt_version"),
            latency_ms=data.get("latency_ms"),
            raw_response_hash=data.get("raw_response_hash"),
            jev=data.get("jev"),
            position_id=data.get("position_id"),
            position_side=data.get("position_side"),
            revision=data.get("revision"),
            based_on_entry=data.get("based_on_entry"),
            close_reason=data.get("close_reason"),
            source_price=data.get("source_price"),
        )
        signal.validate()
        return signal


@dataclass
class SignalDraft:
    """Intermediate extraction result before the final Signal is built."""

    action: str
    confidence_raw: Any
    sl: Optional[float]
    tp: Optional[float]
    position_size: Optional[float]
    reasoning: str
    model: str = "unknown"
    provider: str = "unknown"
    prompt_version: Optional[str] = None
    latency_ms: Optional[int] = None
    raw_response_hash: Optional[str] = None
    jev: Optional[Dict[str, Any]] = None