"""Read/write the read-only position snapshot consumed by the signal service.

``position_sync`` (a separate process) polls Freqtrade's REST API and writes
``runtime/positions.json`` atomically. The producer only ever reads it, so the
LLM service needs no Freqtrade credentials and never touches the exchange.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

POSITIONS_SCHEMA_VERSION = "1.0"


@dataclass
class PositionSnapshot:
    """One open position as reported by Freqtrade."""

    position_id: str
    symbol: str
    side: str  # "long" | "short"
    entry_price: float
    amount: float = 0.0
    stake_amount: float = 0.0
    open_date: Optional[str] = None
    leverage: float = 1.0
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    unrealized_pnl_pct: Optional[float] = None
    current_price: Optional[float] = None
    enter_tag: Optional[str] = None
    market: Optional[str] = None

    def validate(self) -> None:
        """Validate the snapshot invariants."""
        if not self.position_id:
            raise ValueError("position_id is required")
        if not self.symbol:
            raise ValueError("symbol is required")
        if self.side not in ("long", "short"):
            raise ValueError(f"invalid side: {self.side!r}")
        if self.entry_price <= 0:
            raise ValueError("entry_price must be positive")

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a JSON-compatible dict."""
        data: Dict[str, Any] = {
            "position_id": self.position_id,
            "symbol": self.symbol,
            "side": self.side,
            "entry_price": self.entry_price,
            "amount": self.amount,
            "stake_amount": self.stake_amount,
            "leverage": self.leverage,
        }
        for key in (
            "open_date", "stop_loss", "take_profit", "unrealized_pnl_pct",
            "current_price", "enter_tag", "market",
        ):
            value = getattr(self, key)
            if value is not None:
                data[key] = value
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PositionSnapshot":
        """Build a snapshot from a dict."""
        if not isinstance(data, dict):
            raise ValueError("position must be an object")
        return cls(
            position_id=str(data.get("position_id", "")),
            symbol=str(data.get("symbol", "")),
            side=str(data.get("side", "")),
            entry_price=_as_float(data.get("entry_price")),
            amount=_as_float(data.get("amount"), 0.0),
            stake_amount=_as_float(data.get("stake_amount"), 0.0),
            open_date=data.get("open_date"),
            leverage=_as_float(data.get("leverage"), 1.0),
            stop_loss=_as_optional_float(data.get("stop_loss")),
            take_profit=_as_optional_float(data.get("take_profit")),
            unrealized_pnl_pct=_as_optional_float(data.get("unrealized_pnl_pct")),
            current_price=_as_optional_float(data.get("current_price")),
            enter_tag=data.get("enter_tag"),
            market=data.get("market"),
        )


@dataclass
class PositionsSnapshot:
    """All open positions reported by Freqtrade at one point in time."""

    updated_at: str = ""
    positions: List[PositionSnapshot] = field(default_factory=list)

    def for_symbol(self, symbol: str) -> List[PositionSnapshot]:
        """Return the positions whose contract symbol matches ``symbol``.

        ``BTC/USDT:USDT`` matches a requested ``BTC/USDT``; the contract stays on
        the spot-style symbol.
        """
        wanted = contract_symbol(symbol)
        return [p for p in self.positions if contract_symbol(p.symbol) == wanted]

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to the on-disk payload."""
        return {
            "schema_version": POSITIONS_SCHEMA_VERSION,
            "updated_at": self.updated_at,
            "positions": [p.to_dict() for p in self.positions],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PositionsSnapshot":
        """Build the snapshot from the on-disk payload."""
        positions = []
        for item in data.get("positions", []) or []:
            snapshot = PositionSnapshot.from_dict(item)
            snapshot.validate()
            positions.append(snapshot)
        return cls(updated_at=str(data.get("updated_at", "")), positions=positions)


def contract_symbol(symbol: str) -> str:
    """Map a futures exchange pair to the contract symbol (``BTC/USDT:USDT``)."""
    if not symbol:
        return symbol
    return symbol.split(":", 1)[0]


def load_positions(path: str) -> PositionsSnapshot:
    """Load the positions snapshot; missing/corrupt files yield an empty one.

    A missing or malformed snapshot must never stop the producer: it simply
    means "no known positions", i.e. the stateless entry behaviour.
    """
    if not path or not os.path.exists(path):
        return PositionsSnapshot()
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (json.JSONDecodeError, OSError):
        return PositionsSnapshot()
    if not isinstance(data, dict):
        return PositionsSnapshot()
    try:
        return PositionsSnapshot.from_dict(data)
    except ValueError:
        return PositionsSnapshot()


def write_positions_atomic(path: str, snapshot: PositionsSnapshot) -> None:
    """Write the snapshot atomically (temp file + fsync + os.replace)."""
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    payload = json.dumps(snapshot.to_dict(), separators=(",", ":"), sort_keys=False)
    fd, tmp_path = tempfile.mkstemp(prefix=".positions-", suffix=".tmp", dir=directory or None)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(payload + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def utc_now_iso() -> str:
    """Return the current UTC time as an ISO-8601 string with Z suffix."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_optional_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def format_position_context(
    position: "PositionSnapshot",
    stop_loss: float,
    take_profit: Optional[float],
    metrics: Optional[Dict[str, Any]] = None,
) -> str:
    """Build the compact position-management block injected into the prompt.

    Shared by the live position-aware service and the position replay so both
    describe a position identically. Entry/SL/TP alone are not enough to decide
    whether to trail or close, so current price, unrealised PnL in R, MFE/MAE,
    time in trade and ATR are included when the caller supplies them.
    """
    take_profit_text = f"{take_profit:g}" if take_profit is not None else "unknown"
    lines = [
        "### Open position to manage",
        f"symbol={position.symbol} side={position.side} "
        f"entry={position.entry_price:g} amount={position.amount:g} "
        f"stop_loss={stop_loss:g} take_profit={take_profit_text} "
        f"opened={position.open_date or 'unknown'}",
    ]
    metrics = metrics or {}
    if metrics.get("current_price") is not None:
        lines.append(f"current_price={metrics['current_price']:g}")
    if metrics.get("pnl_R") is not None:
        lines.append(f"unrealised_pnl_R={metrics['pnl_R']:+.2f}")
    if metrics.get("mfe_R") is not None and metrics.get("mae_R") is not None:
        lines.append(f"mfe_R={metrics['mfe_R']:+.2f} mae_R={metrics['mae_R']:+.2f}")
    if metrics.get("hours_in_trade") is not None:
        lines.append(f"hours_in_trade={metrics['hours_in_trade']:.1f}")
    if metrics.get("atr") is not None:
        pct = metrics.get("atr_pct")
        lines.append(
            f"atr={metrics['atr']:g}" + (f" ({pct:.2f}% of price)" if pct is not None else "")
        )
    lines.append(
        "If you would adjust this position, answer UPDATE with a new stop_loss "
        "and/or take_profit (tighten the stop, improve the target, or move it to "
        "break-even). If you would close it, answer CLOSE. Do not answer BUY or "
        "SELL for an already open position."
    )
    return "\n".join(lines)
