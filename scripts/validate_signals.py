#!/usr/bin/env python3
"""Validate signal JSONL files against contracts/signal.schema.json.

Thin wrapper around `python -m src.signal_service validate`, so the schema
loading and validation rules live in exactly one place.

Usage:
    python scripts/validate_signals.py [--dir DIR] [--file FILE] [--schema PATH]

Exits 0 when every line is a schema-valid signal, 1 otherwise (2 on setup
errors such as a missing schema).
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.signal_service.__main__ import DEFAULT_SCHEMA_PATH, DEFAULT_SIGNALS_DIR  # noqa: E402


def main() -> int:
    """Translate the legacy flags and delegate to the CLI validator."""
    parser = argparse.ArgumentParser(description="Validate signal JSONL files")
    parser.add_argument("--dir", default=DEFAULT_SIGNALS_DIR)
    parser.add_argument("--file", default=None)
    parser.add_argument("--schema", default=DEFAULT_SCHEMA_PATH)
    args = parser.parse_args()

    from src.signal_service.__main__ import main as cli_main

    argv = ["validate", "--input", args.dir, "--schema", args.schema]
    if args.file:
        argv += ["--file", args.file]
    return cli_main(argv)


if __name__ == "__main__":
    sys.exit(main())
