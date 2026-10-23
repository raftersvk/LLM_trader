#!/usr/bin/env python3
"""Validate signal JSONL files against contracts/signal.schema.json.

Usage:
    python scripts/validate_signals.py [--dir DIR] [--file FILE]

Exits 0 when every line is a schema-valid signal, 1 otherwise.
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SCHEMA_PATH = "/home/djelinux/TradingBots/trading-stack/contracts/signal.schema.json"
DEFAULT_DIR = "/home/djelinux/TradingBots/trading-stack/runtime/signals"


def iter_files(directive_file, directive_dir):
    if directive_file:
        return [directive_file]
    if os.path.isdir(directive_dir):
        return sorted(
            os.path.join(directive_dir, name)
            for name in os.listdir(directive_dir)
            if name.endswith(".jsonl")
        )
    print(f"input directory not found: {directive_dir}")
    return []


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate signal JSONL files")
    parser.add_argument("--dir", default=DEFAULT_DIR)
    parser.add_argument("--file", default=None)
    args = parser.parse_args()

    from src.signal_service.contract import Signal

    schema = None
    with open(SCHEMA_PATH, "r", encoding="utf-8") as handle:
        schema = json.load(handle)
    validator = None
    try:
        import jsonschema
        validator = jsonschema.Draft202012Validator(schema)
    except ImportError:
        print("jsonschema not installed: structural (contract) validation only")

    total_valid = 0
    total_failures = 0
    for path in iter_files(args.file, args.dir):
        valid = failures = 0
        with open(path, "r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    data = json.loads(stripped)
                    signal = Signal.from_dict(data)
                    if validator is not None:
                        validator.validate(signal.to_dict())
                    valid += 1
                except (json.JSONDecodeError, ValueError) as exc:
                    print(f"{path}:{line_number}: INVALID: {exc}")
                    failures += 1
        total_valid += valid
        total_failures += failures
        print(f"{path}: ok={valid} errors={failures}")
    print(f"total: ok={total_valid} failures={total_failures}")
    return 0 if total_failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())