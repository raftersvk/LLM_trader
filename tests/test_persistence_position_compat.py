"""Backward-compatible position persistence around ``conditions_at_entry``."""
import json
from datetime import datetime, timezone
from unittest.mock import MagicMock

from src.managers.persistence_manager import PersistenceManager
from src.trading.data_models import Position


def _legacy_position_json() -> dict:
    """State written by the pre-merge fork: no ``conditions_at_entry`` key."""
    return {
        "entry_price": 100.0,
        "stop_loss": 95.0,
        "take_profit": 110.0,
        "size": 1.5,
        "entry_time": "2026-03-01T00:00:00+00:00",
        "confidence": "HIGH",
        "direction": "LONG",
        "symbol": "BTC/USDC",
        "confluence_factors": [["trend", 2]],
        "entry_fee": 0.1,
        "quote_amount": 150.0,
        "size_pct": 0.02,
    }


def _position(conditions_at_entry=None) -> Position:
    return Position(
        entry_price=100.0,
        stop_loss=95.0,
        take_profit=110.0,
        size=1.5,
        entry_time=datetime(2026, 3, 1, tzinfo=timezone.utc),
        confidence="HIGH",
        direction="LONG",
        symbol="BTC/USDC",
        conditions_at_entry=conditions_at_entry,
    )


def test_load_legacy_position_without_conditions(tmp_path):
    """A pre-merge positions.json must load instead of being treated as flat."""
    manager = PersistenceManager(MagicMock(), data_dir=str(tmp_path))
    manager.positions_file.write_text(json.dumps(_legacy_position_json()), encoding="utf-8")

    position = manager.load_position()

    assert position is not None
    assert position.symbol == "BTC/USDC"
    assert position.direction == "LONG"
    assert position.conditions_at_entry is None


def test_load_position_with_null_conditions(tmp_path):
    """A persisted null conditions snapshot loads as None, not a crash."""
    data = _legacy_position_json()
    data["conditions_at_entry"] = None
    manager = PersistenceManager(MagicMock(), data_dir=str(tmp_path))
    manager.positions_file.write_text(json.dumps(data), encoding="utf-8")

    position = manager.load_position()

    assert position is not None
    assert position.conditions_at_entry is None


def test_save_position_without_conditions_round_trips(tmp_path):
    """Saving a Position whose conditions are None must not raise."""
    manager = PersistenceManager(MagicMock(), data_dir=str(tmp_path))

    manager.save_position(_position(conditions_at_entry=None))

    payload = json.loads(manager.positions_file.read_text(encoding="utf-8"))
    assert payload["conditions_at_entry"] is None

    manager._position_cache_valid = False
    restored = manager.load_position()
    assert restored is not None
    assert restored.conditions_at_entry is None
