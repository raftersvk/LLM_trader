"""FastAPI application for the standalone, read-only signal dashboard.

Routes are strictly read-only: there are no POST/PUT/PATCH/DELETE handlers.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse

from src.signal_dashboard import reader

logger = logging.getLogger(__name__)

_STATIC_DIR = Path(__file__).resolve().parent / "static"
_INDEX_FILE = _STATIC_DIR / "index.html"


def _load_page() -> str:
    try:
        return _INDEX_FILE.read_text(encoding="utf-8")
    except OSError:  # pragma: no cover - packaged file should exist
        logger.error("Dashboard page missing at %s", _INDEX_FILE, exc_info=True)
        return (
            "<!doctype html><html><body><h1>Signal dashboard</h1>"
            "<p>The static page could not be loaded.</p></body></html>"
        )


def create_app(
    signals_dir: Optional[str] = None,
    limit: int = 50,
    stale_after: Optional[float] = None,
) -> FastAPI:
    """Create the read-only dashboard app.

    Args:
        signals_dir: Directory holding ``*.jsonl`` and ``heartbeat*.json``.
            Defaults to the signal service's default signals directory.
        limit: Default number of signals returned by /api/signals.
        stale_after: Heartbeat staleness threshold in seconds. When None the
            reader uses two times the heartbeat's parsed timeframe.
    """
    resolved_dir = os.path.abspath(signals_dir or reader.default_signals_dir())
    default_limit = max(0, int(limit))

    app = FastAPI(
        title="LLM Trader Signal Dashboard",
        description="Read-only view of the signal chain between brain and execution.",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )
    app.state.signals_dir = resolved_dir
    app.state.limit = default_limit
    app.state.stale_after = stale_after
    page = _load_page()

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    def index() -> str:
        return page

    @app.get("/api/health")
    def health() -> dict:
        return reader.build_health(app.state.signals_dir)

    @app.get("/api/signals")
    def signals(
        limit: Optional[int] = Query(default=None, ge=0),
        symbol: Optional[str] = Query(default=None),
        timeframe: Optional[str] = Query(default=None),
    ) -> List[dict]:
        effective_limit = app.state.limit if limit is None else limit
        rows = reader.read_signals(app.state.signals_dir)["signals"]
        if symbol:
            rows = [row for row in rows if row.get("symbol") == symbol]
        if timeframe:
            rows = [row for row in rows if row.get("timeframe") == timeframe]
        return rows[:effective_limit]

    @app.get("/api/heartbeats")
    def heartbeats() -> List[dict]:
        return reader.read_heartbeats(
            app.state.signals_dir,
            stale_after=app.state.stale_after,
        )["heartbeats"]

    @app.get("/api/pairs")
    def pairs() -> List[dict]:
        return reader.read_pairs(
            app.state.signals_dir,
            stale_after=app.state.stale_after,
        )

    return app
