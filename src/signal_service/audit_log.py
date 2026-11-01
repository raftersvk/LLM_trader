"""Structured JSONL audit log for every signal-service analysis.

The log is independent of the JEV gate: one record is appended for each
analysis even when JEV is disabled (``--jev-mode off``) or the gate does not
consult the model (e.g. a HOLD decision). It carries the final contract signal
plus the advisory JEV answer set and the gate outcome, so a decision can be
replayed and judged a posteriori.

The file never contains the raw LLM response, prompts, headers or API keys:
only the fields listed in :func:`build_analysis_record`.
"""

from __future__ import annotations

import json
import logging
import math
import os
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from src.signal_service.contract import Signal, iso_z

logger = logging.getLogger(__name__)

#: Label of the shipped post-OOS decision policy. It is recorded on every
#: line, including lines where the gate was not used, so an audit can group
#: records by policy without inspecting code.
GATE_POLICY = "bias_short_only"


def _optional_float(value: Any) -> Optional[float]:
    """Return ``value`` as a finite float when possible, otherwise None."""
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def _gate_mode(gate: Any) -> str:
    """Return the configured mode label of a gate (``off`` when absent)."""
    mode = getattr(gate, "mode", None)
    value = getattr(mode, "value", mode)
    text = str(value).strip().lower() if value is not None else ""
    return text or "off"


def _jev_values(signal: Signal, gate_signal: Any) -> Dict[str, Any]:
    """Build the advisory JEV block from the gate signal or the payload."""
    payload = signal.jev if isinstance(signal.jev, dict) else {}

    def _value(attribute: str, key: str) -> Any:
        if gate_signal is not None:
            return getattr(gate_signal, attribute, None)
        return payload.get(key)

    tradeability = _optional_float(_value("tradeability_probability", "tradeability_probability"))
    if tradeability is None and gate_signal is None:
        # Older payloads may only carry the boolean. Keep it as a diagnostic
        # rather than dropping the field, but never treat it as the raw noul.
        tradeability = payload.get("tradeability")
    return {
        "tradeability": tradeability,
        "bias": _value("bias", "bias"),
        "regime": _value("regime", "regime"),
        "bias_confidence": _optional_float(_value("bias_confidence", "bias_confidence")),
        "regime_confidence": _optional_float(
            _value("regime_confidence", "regime_confidence")
        ),
        "tradeability_confidence": _optional_float(
            _value("tradeability_confidence", "tradeability_confidence")
        ),
        "confidence": _optional_float(_value("confidence", "confidence")),
    }


def _gate_values(gate: Any, gate_result: Any) -> Dict[str, Any]:
    """Describe whether the gate ran and why it allowed or vetoed the signal."""
    if gate_result is not None:
        return {
            "policy": GATE_POLICY,
            "used": bool(getattr(gate_result, "consulted", False)),
            "allowed": bool(getattr(gate_result, "allowed", True)),
            "reason": str(getattr(gate_result, "reason", "") or ""),
        }
    return {
        "policy": GATE_POLICY,
        "used": False,
        "allowed": True,
        "reason": f"JEV gate not used (mode={_gate_mode(gate)})",
    }


def build_analysis_record(
    signal: Signal,
    gate_result: Any = None,
    gate: Any = None,
    now: Optional[datetime] = None,
) -> Dict[str, Any]:
    """Build one audit record for a final contract signal.

    Args:
        signal: The contract signal produced by the analysis.
        gate_result: Optional ``GateResult`` from the pre/post gate. ``None``
            means the gate was not consulted (disabled mode, HOLD, update).
        gate: Optional gate instance, used only to report its mode.
        now: Write timestamp; defaults to the current UTC time.

    Returns:
        A JSON-serializable mapping with the audit fields.
    """
    gate_signal = getattr(gate_result, "signal", None)
    record: Dict[str, Any] = {
        "ts": iso_z(now or datetime.now(timezone.utc)),
        "symbol": signal.symbol,
        "timeframe": signal.timeframe,
        "decision_id": signal.decision_id,
        "candle_close_time": signal.candle_close_time,
        "signal_action": signal.action,
        "signal_confidence": signal.confidence,
        "risk_approved": signal.risk_approved,
        "jev": _jev_values(signal, gate_signal),
        "gate": _gate_values(gate, gate_result),
    }
    # The taxonomy fields are optional contract fields: only emit them when the
    # analysis actually resolved a value, keeping older records unchanged.
    for field_name in ("setup", "regime", "htf_alignment"):
        value = getattr(signal, field_name, None)
        if value is not None:
            record[field_name] = value
    return record


class DecisionAuditLog:
    """Append-only JSONL writer for analysis audit records."""

    def __init__(self, log_path: str, enabled: bool = True) -> None:
        """Bind the destination file.

        Args:
            log_path: JSONL path. Parent directories are created on first write.
            enabled: When False, :meth:`record` is a no-op.
        """
        self.log_path = log_path
        self.enabled = bool(enabled)

    @classmethod
    def from_config(cls, config: Any) -> Optional["DecisionAuditLog"]:
        """Build a logger from a ``RegimeTaxonomyConfig``, or None when disabled."""
        if config is None or not getattr(getattr(config, "jev", None), "log_decisions", False):
            return None
        return cls(config.jev.log_path, enabled=True)

    def record(
        self,
        signal: Signal,
        gate_result: Any = None,
        gate: Any = None,
        now: Optional[datetime] = None,
    ) -> Optional[Dict[str, Any]]:
        """Append one record and return it, or None when logging is disabled.

        Write failures are logged and swallowed: the audit log is diagnostic
        and must never lose or abort a trading signal.
        """
        if not self.enabled:
            return None
        record = build_analysis_record(signal, gate_result=gate_result, gate=gate, now=now)
        try:
            directory = os.path.dirname(self.log_path)
            if directory:
                os.makedirs(directory, exist_ok=True)
            with open(self.log_path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
        except OSError as exc:
            logger.warning("Cannot write JEV decision audit log %s: %s", self.log_path, exc)
            return None
        return record
