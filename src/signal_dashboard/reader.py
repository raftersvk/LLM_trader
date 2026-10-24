"""Read-only data access for the standalone signal dashboard.

Nothing in this module writes to disk. Malformed input is skipped and
counted, never raised: the dashboard must survive partial writes, corrupt
lines and missing directories.
"""

from __future__ import annotations

import json
import logging
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

#: Timeframe -> seconds, used for the default heartbeat staleness window.
_TIMEFRAME_UNITS = {
    "s": 1,
    "m": 60,
    "h": 3600,
    "d": 86400,
    "w": 604800,
}
_TIMEFRAME_RE = re.compile(r"^(\d+)([smhdw])$")
_HEARTBEAT_RE = re.compile(r"^heartbeat_(.+)_([0-9]+[smhdw])\.json$")

#: Fallback staleness window when neither --stale-after nor a timeframe exist.
DEFAULT_STALE_SECONDS = 7200

#: Minimal signal fields required for a row to be considered displayable.
_REQUIRED_SIGNAL_FIELDS = (
    "decision_id",
    "symbol",
    "timeframe",
    "candle_open_time",
    "action",
)


def utc_now() -> datetime:
    """Return the current time as a timezone-aware UTC datetime."""
    return datetime.now(timezone.utc)


def utc_now_iso() -> str:
    """Return the current UTC time as an ISO-8601 string with Z suffix."""
    return utc_now().isoformat(timespec="seconds").replace("+00:00", "Z")


def default_signals_dir() -> str:
    """Return the signals directory, preferring the signal service helper.

    ``src.signal_service.storage.default_signals_dir`` is the source of truth
    when that package is installed. The dashboard must also start on a
    checkout where the signal service is absent, so the identical fallback
    (``LLM_TRADER_SIGNALS_DIR`` or ``<repo>/data/signals``) is used then.
    """
    try:
        from src.signal_service.storage import (  # noqa: PLC0415
            default_signals_dir as upstream_default,
        )
    except Exception:  # pragma: no cover - depends on checkout layout
        upstream_default = None
    if upstream_default is not None:
        try:
            return upstream_default()
        except Exception:  # pragma: no cover - defensive
            logger.warning("signal_service.default_signals_dir() failed", exc_info=True)

    project_root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    return os.environ.get(
        "LLM_TRADER_SIGNALS_DIR",
        os.path.join(project_root, "data", "signals"),
    )


def parse_timestamp(value: Any) -> Optional[datetime]:
    """Parse an ISO-8601 timestamp as UTC, returning None when invalid."""
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip()
    if text.endswith(("Z", "z")):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except (TypeError, ValueError):
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def timeframe_seconds(timeframe: Any) -> Optional[int]:
    """Convert a timeframe label such as ``1h`` or ``15m`` to seconds."""
    if not isinstance(timeframe, str):
        return None
    match = _TIMEFRAME_RE.match(timeframe.strip().lower())
    if not match:
        return None
    return int(match.group(1)) * _TIMEFRAME_UNITS[match.group(2)]


def _as_float(value: Any) -> Optional[float]:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _as_int(value: Any) -> Optional[int]:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return None


def _age_seconds(timestamp: Any, now: datetime) -> Optional[int]:
    parsed = parse_timestamp(timestamp)
    if parsed is None:
        return None
    return int((now - parsed).total_seconds())


def _jev_fields(jev: Any) -> Tuple[Optional[Dict[str, Any]], Optional[bool], Optional[str]]:
    """Return (raw jev, derived veto flag, derived reason) for a signal."""
    if not isinstance(jev, dict):
        return None, None, None
    veto = jev.get("veto")
    veto_flag: Optional[bool] = veto if isinstance(veto, bool) else None
    if veto_flag is None:
        tradeability = jev.get("tradeability")
        if isinstance(tradeability, str):
            veto_flag = tradeability.strip().lower() in {
                "veto",
                "blocked",
                "no_trade",
                "untradeable",
            }
    reason = jev.get("reason")
    if not isinstance(reason, str):
        reason = None
    return jev, veto_flag, reason


def _normalise_signal(row: Any, now: datetime) -> Optional[Dict[str, Any]]:
    """Normalise one decoded JSONL object, or return None when unusable."""
    if not isinstance(row, dict):
        return None
    if any(not row.get(field) for field in _REQUIRED_SIGNAL_FIELDS):
        return None

    jev, jev_veto, jev_reason = _jev_fields(row.get("jev"))
    expires_at = row.get("expires_at")
    expires_dt = parse_timestamp(expires_at)
    created_at = row.get("created_at")
    age_source = created_at if parse_timestamp(created_at) is not None else row.get(
        "candle_open_time"
    )

    return {
        "decision_id": row.get("decision_id"),
        "symbol": row.get("symbol"),
        "timeframe": row.get("timeframe"),
        "candle_open_time": row.get("candle_open_time"),
        "candle_close_time": row.get("candle_close_time"),
        "created_at": created_at,
        "expires_at": expires_at,
        "action": row.get("action"),
        "confidence": _as_float(row.get("confidence")),
        "risk_approved": row.get("risk_approved"),
        "sl": _as_float(row.get("sl")),
        "tp": _as_float(row.get("tp")),
        "position_size_pct": _as_float(row.get("position_size_pct")),
        "model": row.get("model"),
        "provider": row.get("provider"),
        "latency_ms": _as_int(row.get("latency_ms")),
        "reasoning": row.get("reasoning"),
        "jev": jev,
        "jev_veto": jev_veto,
        "jev_reason": jev_reason,
        "age_seconds": _age_seconds(age_source, now),
        "expired": expires_dt is not None and expires_dt <= now,
    }


def _signal_sort_key(signal: Dict[str, Any]) -> Tuple[datetime, datetime]:
    created = parse_timestamp(signal.get("created_at"))
    candle = parse_timestamp(signal.get("candle_open_time"))
    floor = datetime.min.replace(tzinfo=timezone.utc)
    return (created or candle or floor, candle or floor)


def _signal_files(directory: Path) -> List[Path]:
    if not directory.is_dir():
        return []
    try:
        return sorted(path for path in directory.glob("*.jsonl") if path.is_file())
    except OSError:  # pragma: no cover - defensive
        logger.warning("Cannot list signal files in %s", directory, exc_info=True)
        return []


def read_signals(
    signals_dir: str,
    now: Optional[datetime] = None,
) -> Dict[str, Any]:
    """Read every JSONL signal file, skipping bad lines.

    Returns a dict with ``signals`` (newest first), ``files``,
    ``skipped_lines`` and ``exists``.
    """
    now = now or utc_now()
    directory = Path(signals_dir)
    result: Dict[str, Any] = {
        "signals": [],
        "files": 0,
        "skipped_lines": 0,
        "exists": directory.is_dir(),
    }
    for path in _signal_files(directory):
        result["files"] += 1
        try:
            handle = path.open("r", encoding="utf-8", errors="replace")
        except OSError:
            logger.warning("Cannot read signal file %s", path, exc_info=True)
            continue
        with handle:
            for line_number, line in enumerate(handle, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    row = json.loads(stripped)
                except (json.JSONDecodeError, TypeError, ValueError):
                    result["skipped_lines"] += 1
                    logger.warning(
                        "Skipping malformed JSONL line %d in %s",
                        line_number,
                        path,
                    )
                    continue
                signal = _normalise_signal(row, now)
                if signal is None:
                    result["skipped_lines"] += 1
                    logger.warning(
                        "Skipping incomplete signal line %d in %s",
                        line_number,
                        path,
                    )
                    continue
                result["signals"].append(signal)

    result["signals"].sort(key=_signal_sort_key, reverse=True)
    return result


def _heartbeat_files(directory: Path) -> List[Path]:
    if not directory.is_dir():
        return []
    paths: List[Path] = []
    try:
        paths.extend(
            sorted(
                path
                for path in directory.glob("heartbeat_*.json")
                # heartbeat_summary.json (aggregated pair map) shares the
                # prefix but is not a per-pair heartbeat: only accept the
                # heartbeat_<SYMBOL>_<TF>.json shape.
                if path.is_file() and _HEARTBEAT_RE.match(path.name)
            )
        )
        legacy = directory / "heartbeat.json"
        if legacy.is_file():
            paths.append(legacy)
    except OSError:  # pragma: no cover - defensive
        logger.warning("Cannot list heartbeat files in %s", directory, exc_info=True)
    return paths


def _heartbeat_identity(
    path: Path,
    payload: Dict[str, Any],
) -> Tuple[Optional[str], Optional[str], bool]:
    legacy = path.name == "heartbeat.json"
    symbol = payload.get("symbol")
    timeframe = payload.get("timeframe")
    symbol = symbol if isinstance(symbol, str) and symbol else None
    timeframe = timeframe if isinstance(timeframe, str) and timeframe else None

    if not legacy:
        match = _HEARTBEAT_RE.match(path.name)
        if match:
            token, name_timeframe = match.groups()
            if symbol is None:
                symbol = token.replace("_", "/")
            if timeframe is None:
                timeframe = name_timeframe
    return symbol, timeframe, legacy


def read_heartbeats(
    signals_dir: str,
    stale_after: Optional[float] = None,
    now: Optional[datetime] = None,
) -> Dict[str, Any]:
    """Read per-pair heartbeat files plus the legacy shared heartbeat.json.

    ``stale`` is ``age_seconds > stale_after`` where ``stale_after`` defaults
    to two times the parsed timeframe (falling back to two hours when the
    timeframe is unknown). An unreadable heartbeat timestamp is treated as
    stale because freshness cannot be proven.
    """
    now = now or utc_now()
    directory = Path(signals_dir)
    result: Dict[str, Any] = {
        "heartbeats": [],
        "files": 0,
        "skipped_heartbeats": 0,
        "exists": directory.is_dir(),
    }
    for path in _heartbeat_files(directory):
        result["files"] += 1
        try:
            raw = path.read_text(encoding="utf-8", errors="replace")
            payload = json.loads(raw)
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            result["skipped_heartbeats"] += 1
            logger.warning("Skipping unreadable heartbeat file %s", path)
            continue
        if not isinstance(payload, dict):
            result["skipped_heartbeats"] += 1
            logger.warning("Skipping non-object heartbeat file %s", path)
            continue

        symbol, timeframe, legacy = _heartbeat_identity(path, payload)
        tf_seconds = timeframe_seconds(timeframe)
        if stale_after is None:
            effective_stale = (
                tf_seconds * 2 if tf_seconds is not None else DEFAULT_STALE_SECONDS
            )
        else:
            effective_stale = float(stale_after)

        heartbeat_at = payload.get("heartbeat_at")
        age = _age_seconds(heartbeat_at, now)
        stale = age is None or age > effective_stale

        result["heartbeats"].append(
            {
                "symbol": symbol,
                "timeframe": timeframe,
                "candle_open_time": payload.get("candle_open_time"),
                "heartbeat_at": heartbeat_at,
                "age_seconds": age,
                "stale": stale,
                "legacy": legacy,
                "stale_after_seconds": effective_stale,
                "file": path.name,
            }
        )

    result["heartbeats"].sort(
        key=lambda item: (
            item["symbol"] or "",
            item["timeframe"] or "",
            item["file"],
        )
    )
    return result


def _heartbeat_is_newer(candidate: Dict[str, Any], current: Dict[str, Any]) -> bool:
    candidate_age = candidate.get("age_seconds")
    current_age = current.get("age_seconds")
    if candidate_age is None:
        return False
    if current_age is None:
        return True
    return candidate_age < current_age


def read_pairs(
    signals_dir: str,
    stale_after: Optional[float] = None,
    now: Optional[datetime] = None,
) -> List[Dict[str, Any]]:
    """Build one freshness summary per (symbol, timeframe)."""
    now = now or utc_now()
    signals = read_signals(signals_dir, now=now)["signals"]
    heartbeats = read_heartbeats(
        signals_dir, stale_after=stale_after, now=now
    )["heartbeats"]

    groups: Dict[Tuple[str, str], Dict[str, Any]] = {}

    def group(symbol: str, timeframe: str) -> Dict[str, Any]:
        key = (symbol, timeframe)
        if key not in groups:
            groups[key] = {
                "symbol": symbol,
                "timeframe": timeframe,
                "last_action": None,
                "last_confidence": None,
                "last_candle_open_time": None,
                "last_signal_created_at": None,
                "last_signal_expired": None,
                "heartbeat_present": False,
                "heartbeat_at": None,
                "heartbeat_age_seconds": None,
                "heartbeat_candle_open_time": None,
                "heartbeat_legacy": False,
                "stale": True,
            }
        return groups[key]

    for signal in signals:
        symbol, timeframe = signal.get("symbol"), signal.get("timeframe")
        if not symbol or not timeframe:
            continue
        entry = group(symbol, timeframe)
        if entry["last_action"] is None:
            # read_signals() is already newest first, so the first row wins.
            entry["last_action"] = signal.get("action")
            entry["last_confidence"] = signal.get("confidence")
            entry["last_candle_open_time"] = signal.get("candle_open_time")
            entry["last_signal_created_at"] = signal.get("created_at")
            entry["last_signal_expired"] = signal.get("expired")

    for heartbeat in heartbeats:
        symbol, timeframe = heartbeat.get("symbol"), heartbeat.get("timeframe")
        if not symbol or not timeframe:
            continue
        entry = group(symbol, timeframe)
        if not entry["heartbeat_present"] or _heartbeat_is_newer(
            heartbeat, {"age_seconds": entry["heartbeat_age_seconds"]}
        ):
            entry["heartbeat_present"] = True
            entry["heartbeat_at"] = heartbeat.get("heartbeat_at")
            entry["heartbeat_age_seconds"] = heartbeat.get("age_seconds")
            entry["heartbeat_candle_open_time"] = heartbeat.get("candle_open_time")
            entry["heartbeat_legacy"] = heartbeat.get("legacy", False)
            entry["stale"] = heartbeat.get("stale", True)

    return sorted(groups.values(), key=lambda item: (item["symbol"], item["timeframe"]))


def build_health(
    signals_dir: str,
    now: Optional[datetime] = None,
) -> Dict[str, Any]:
    """Return the /api/health payload."""
    now = now or utc_now()
    signal_data = read_signals(signals_dir, now=now)
    heartbeat_data = read_heartbeats(signals_dir, now=now)
    return {
        "status": "ok" if signal_data["exists"] else "missing",
        "signals_dir": os.path.abspath(signals_dir),
        "counts": {
            "signal_files": signal_data["files"],
            "signals": len(signal_data["signals"]),
            "heartbeat_files": heartbeat_data["files"],
            "heartbeats": len(heartbeat_data["heartbeats"]),
            "skipped_lines": signal_data["skipped_lines"],
            "skipped_heartbeats": heartbeat_data["skipped_heartbeats"],
            "skipped": (
                signal_data["skipped_lines"]
                + heartbeat_data["skipped_heartbeats"]
            ),
        },
        "generated_at": now.isoformat(timespec="seconds").replace("+00:00", "Z"),
    }
