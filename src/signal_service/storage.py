"""Append-only JSONL signal storage.

One JSONL file per (symbol, timeframe). Writes are guarded by fcntl.flock,
flushed and fsynced. Signals are deduplicated by
(symbol, timeframe, candle_open_time). Invalid lines skipped on read never
corrupt the file or break future append-only writes.
"""

from __future__ import annotations

import fcntl
import json
import logging
import os
from datetime import datetime, timezone
from typing import List, Optional, Set, Tuple

from src.signal_service.contract import Signal, utc_now_iso

DedupKey = Tuple[str, str, str]

logger = logging.getLogger(__name__)


class SignalStorage:
    """Append-only JSONL storage with locking, deduplication and TTL checks."""

    def __init__(self, path: str) -> None:
        """Open (or create) the JSONL file at `path`.

        Args:
            path: Absolute path of the JSONL file, e.g.
                /home/djelinux/TradingBots/trading-stack/runtime/signals/BTC_USDT_1h.jsonl
        """
        self.path = path
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        self._dedup_keys: Set[DedupKey] = set()
        self._dedup_loaded = False

    # ------------------------------------------------------------------ dedup
    def _dedup_key_for(self, signal: Signal) -> DedupKey:
        return (signal.symbol, signal.timeframe, signal.candle_open_time)

    def _load_dedup_keys(self) -> None:
        """Scan the existing file and collect dedup keys, skipping bad lines."""
        keys: Set[DedupKey] = set()
        if os.path.exists(self.path):
            with open(self.path, "r", encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, start=1):
                    stripped = line.strip()
                    if not stripped:
                        continue
                    try:
                        data = json.loads(stripped)
                        key = (data["symbol"], data["timeframe"], data["candle_open_time"])
                        keys.add(key)
                    except (json.JSONDecodeError, KeyError, TypeError):
                        # Never let a corrupted line break the storage: log and skip.
                        logger.warning(
                            "Skipping invalid JSONL line %d in %s", line_number, self.path
                        )
        self._dedup_keys = keys
        self._dedup_loaded = True

    def is_duplicate(self, signal: Signal) -> bool:
        """Return True when a signal already exists for the same candle."""
        if not self._dedup_loaded:
            self._load_dedup_keys()
        return self._dedup_key_for(signal) in self._dedup_keys

    # ------------------------------------------------------------------ write
    def append(self, signal: Signal) -> bool:
        """Append one signal to the JSONL file, or skip if already stored.

        Args:
            signal: Validated signal to persist.

        Returns:
            True when the signal was appended, False when it was a duplicate.

        Raises:
            SignalValidationError: when the signal violates the contract.
        """
        signal.validate()
        if self.is_duplicate(signal):
            logger.info(
                "Duplicate signal for (%s, %s, %s): not appended",
                signal.symbol, signal.timeframe, signal.candle_open_time,
            )
            return False

        line = json.dumps(signal.to_dict(), separators=(",", ":"), sort_keys=False)
        with open(self.path, "a", encoding="utf-8") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                handle.write(line + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        self._dedup_keys.add(self._dedup_key_for(signal))
        return True

    # ------------------------------------------------------------------- read
    def load_all(self) -> List[Signal]:
        """Load every stored signal, skipping invalid lines.

        Invalid lines are reported but never rewritten: the file stays
        append-only and uncorrupted.
        """
        signals: List[Signal] = []
        if not os.path.exists(self.path):
            return signals
        with open(self.path, "r", encoding="utf-8") as handle:
            for line in handle:
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    signals.append(Signal.from_dict(json.loads(stripped)))
                except (json.JSONDecodeError, ValueError) as exc:
                    logger.warning("Invalid signal line in %s: %s", self.path, exc)
        return signals

    def latest_for_candle(self, symbol: str, timeframe: str, candle_open_time: str) -> Optional[Signal]:
        """Return the stored signal for a given candle, if any."""
        for signal in self.load_all():
            if (
                signal.symbol == symbol
                and signal.timeframe == timeframe
                and signal.candle_open_time == candle_open_time
            ):
                return signal
        return None

    @staticmethod
    def is_ttl_expired(signal: Signal, now: Optional[datetime] = None) -> bool:
        """Return True when the signal is past its expires_at timestamp."""
        return signal.is_expired(now or datetime.now(timezone.utc))


def default_signals_dir() -> str:
    """Return the runtime signals directory of the trading stack."""
    return "/home/djelinux/TradingBots/trading-stack/runtime/signals"


def signal_file_path(signals_dir: str, symbol: str, timeframe: str) -> str:
    """Build the output file path for a (symbol, timeframe) couple."""
    token = f"{symbol.replace('/', '_')}_{timeframe}.jsonl"
    return os.path.join(signals_dir, token)


def heartbeat_path(signals_dir: str) -> str:
    """Build the heartbeat file path."""
    return os.path.join(signals_dir, "heartbeat.json")


def write_heartbeat(signals_dir: str, symbol: str, timeframe: str, candle_open_time: str) -> None:
    """Write the per-pair heartbeat with a flush+fsync guarantee."""
    os.makedirs(signals_dir, exist_ok=True)
    payload = {
        "symbol": symbol,
        "timeframe": timeframe,
        "candle_open_time": candle_open_time,
        "heartbeat_at": utc_now_iso(),
    }
    with open(heartbeat_path(signals_dir), "w", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            handle.write(json.dumps(payload, separators=(",", ":")) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)