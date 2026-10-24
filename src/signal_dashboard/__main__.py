"""Command-line entry point: ``python -m src.signal_dashboard``.

Serves the read-only signal dashboard. No config.ini, API key or exchange
connection is required.
"""

from __future__ import annotations

import argparse
import logging
from typing import List, Optional

import uvicorn

from src.signal_dashboard.app import create_app
from src.signal_dashboard.reader import default_signals_dir


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="python -m src.signal_dashboard",
        description="Read-only web view of the LLM_trader signal chain.",
    )
    parser.add_argument(
        "--signals-dir",
        default=None,
        help="Directory with *.jsonl signals and heartbeat files "
        "(default: LLM_TRADER_SIGNALS_DIR or <repo>/data/signals)",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Bind host")
    parser.add_argument("--port", type=int, default=8010, help="Bind port")
    parser.add_argument(
        "--limit",
        type=int,
        default=50,
        help="Default number of signals returned by /api/signals",
    )
    parser.add_argument(
        "--stale-after",
        type=float,
        default=None,
        help="Heartbeat staleness threshold in seconds "
        "(default: 2x the heartbeat timeframe)",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> None:
    """Parse arguments and run the read-only server."""
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    signals_dir = args.signals_dir or default_signals_dir()
    app = create_app(
        signals_dir=signals_dir,
        limit=args.limit,
        stale_after=args.stale_after,
    )
    logging.getLogger(__name__).info(
        "Serving read-only signal dashboard for %s on http://%s:%d",
        signals_dir,
        args.host,
        args.port,
    )
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
