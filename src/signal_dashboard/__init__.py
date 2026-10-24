"""Standalone, read-only web view of the signal chain.

This package only reads JSONL signal files and heartbeat files. It never
imports the trading bot, exchange clients, risk manager or signal writer.
"""

from src.signal_dashboard.app import create_app

__all__ = ["create_app"]
