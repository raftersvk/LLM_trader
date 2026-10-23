"""Tests for the portfolio dashboard router file loading."""
from types import SimpleNamespace

from src.dashboard.routers.portfolio import PortfolioRouter


class _StubLogger:
    """Collects error log calls for assertions."""

    def __init__(self) -> None:
        self.errors = []

    def error(self, message, *args, **kwargs) -> None:  # pylint: disable=unused-argument
        """Record an error message."""
        self.errors.append(message % args if args else message)


def _build_router(logger) -> PortfolioRouter:
    """Build a PortfolioRouter with stubbed dependencies."""
    config = SimpleNamespace(CRYPTO_PAIRS=[])
    return PortfolioRouter(
        config=config,
        logger=logger,
        dashboard_state=SimpleNamespace(),
        engine_registry=None,
        vector_memory_registry=None,
    )


def test_load_json_file_returns_none_when_missing(tmp_path):
    """A missing file is skipped without logging."""
    logger = _StubLogger()
    router = _build_router(logger)

    assert router._load_json_file(tmp_path / "missing.json") is None
    assert logger.errors == []


def test_load_json_file_returns_none_on_truncated_json(tmp_path):
    """A truncated JSON file is logged and skipped instead of raising."""
    logger = _StubLogger()
    router = _build_router(logger)
    truncated = tmp_path / "statistics.json"
    truncated.write_text('{"total_pnl_quote": 12.0, "total_trades":', encoding="utf-8")

    assert router._load_json_file(truncated) is None
    assert len(logger.errors) == 1
    assert "statistics.json" in logger.errors[0]


def test_load_json_file_returns_none_on_non_utf8_bytes(tmp_path):
    """A byte-corrupted file raises UnicodeDecodeError and must be skipped."""
    logger = _StubLogger()
    router = _build_router(logger)
    corrupted = tmp_path / "previous_response.json"
    corrupted.write_bytes(b"\xff\xfe\x00\x01not-utf8")

    assert router._load_json_file(corrupted) is None
    assert len(logger.errors) == 1
    assert "previous_response.json" in logger.errors[0]


def test_load_json_file_returns_data_for_valid_json(tmp_path):
    """Valid JSON is returned unchanged."""
    logger = _StubLogger()
    router = _build_router(logger)
    valid = tmp_path / "positions.json"
    valid.write_text('{"symbol": "BTC/USDT"}', encoding="utf-8")

    assert router._load_json_file(valid) == {"symbol": "BTC/USDT"}
    assert logger.errors == []
