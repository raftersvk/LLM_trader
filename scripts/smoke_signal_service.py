#!/usr/bin/env python3
"""Smoke test for the signal service with a canned (offline) analysis engine.

Produces a few validated signals in a temp directory and re-validates them
end-to-end (contract + JSONL + dedup). No network, no keys, no orders.
"""

import asyncio
import json
import logging
import os
import sys
import tempfile
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.signal_service.analyzer_adapter import PairAnalyzer
from src.signal_service.contract import iso_z
from src.signal_service.storage import SignalStorage

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("smoke_signal_service")

SCHEMA_PATH = "/home/djelinux/TradingBots/trading-stack/contracts/signal.schema.json"

CANNED_RESPONSE = (
    '{"signal": "BUY", "confidence": "HIGH", "stop_loss": 67000.0, '
    '"take_profit": 70000.0, "position_size": 5, '
    '"reasoning": "Trend + breakout with volume confirmation."}'
)


class FakeModelManager:
    """Minimal model manager stub honouring the use_chart ablation flag."""

    def supports_image_analysis(self, *_args, **_kwargs) -> bool:
        return False


class FakeRagEngine:
    """Minimal RAG stub returning empty market context."""

    async def retrieve_context(self, *_args, **_kwargs):
        return ("", {})

    context_builder = type("CB", (), {"get_latest_article_urls": lambda self: {}})()


class FakeEngine:
    """Canned AnalysisEngine stand-in exposing the same hooks as the real one."""

    class Context:
        current_price = 68000.0
        ohlcv_candles = None
        timeframe = "1h"

    def __init__(self, raw_response: str) -> None:
        self.last_llm_response = raw_response
        self.model_manager = FakeModelManager()
        self.rag_engine = FakeRagEngine()
        self.context = self.Context()

    async def analyze_market(self, **_kwargs):
        self.last_response = True
        return {
            "model": "deepseek-v4.1-flash",
            "provider": "opencode-go",
            "raw_response": self.last_llm_response,
        }


def _validate_with_jsonschema(signals_dir: str) -> None:
    import jsonschema

    with open(SCHEMA_PATH, "r", encoding="utf-8") as handle:
        schema = json.load(handle)
    validator = jsonschema.Draft202012Validator(schema)
    path = os.path.join(signals_dir, "BTC_USDT_1h.jsonl")
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            validator.validate(json.loads(line))
    logger.info("All lines validated against the shared JSON schema")


def main() -> int:
    """Run the offline smoke test."""
    with tempfile.TemporaryDirectory() as signals_dir:
        engine = FakeEngine(CANNED_RESPONSE)
        analyzer = PairAnalyzer(
            engine=engine, symbol="BTC/USDT", timeframe="1h",
            use_news=False, use_chart=False,
        )
        storage = SignalStorage(os.path.join(signals_dir, "BTC_USDT_1h.jsonl"))

        signal = asyncio.run(analyzer.analyze())
        appended = storage.append(signal)
        assert appended, "first signal must be appended"
        duplicated = storage.append(signal)
        assert not duplicated, "dedup must reject the second identical signal"

        loaded = storage.load_all()
        assert len(loaded) == 1, f"expected 1 stored signal, got {len(loaded)}"
        stored = loaded[0]
        assert stored.action == "BUY", "smoke response should map to BUY"
        assert stored.risk_approved is True, "coherent BUY should be risk-approved"
        assert stored.decision_id == signal.decision_id

        _validate_with_jsonschema(signals_dir)

        # HOLD fallback on engine failure
        broken = FakeEngine(None)
        broken.analyze_market = None  # type: ignore[assignment]
        broken_analyzer = PairAnalyzer(engine=broken, symbol="BTC/USDT", use_news=False)
        hold = asyncio.run(broken_analyzer.analyze())
        assert hold.action == "HOLD", "engine failure must fall back to HOLD"
        assert hold.raw_response_hash is None

        # Expired signal is detected (TTL)
        candle_open = datetime.now(timezone.utc) - timedelta(hours=4)
        storage.append(stored.__class__(
            **{
                **stored.to_dict(),
                "candle_open_time": iso_z(candle_open),
                "candle_close_time": iso_z(candle_open + timedelta(hours=1)),
                "expires_at": iso_z(candle_open + timedelta(hours=2)),
                "decision_id": f"btc-usdt-1h-{iso_z(candle_open).lower()}",
            }
        ))
        expired = [s for s in storage.load_all() if s.is_expired()]
        assert expired, "past candle signal must be flagged expired"

    print("SMOKE OK: contract, storage, dedup, schema validation, HOLD fallback, TTL")
    return 0


if __name__ == "__main__":
    sys.exit(main())