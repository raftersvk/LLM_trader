#!/usr/bin/env python3
"""Smoke test for the signal service with a canned (offline) analysis engine.

Produces a few validated signals in a temp directory and re-validates them
end-to-end (contract + JSONL + dedup). No network, no keys, no orders.
"""

import asyncio
import json
import logging
import os
import re
import sys
import tempfile
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.signal_service.analyzer_adapter import PairAnalyzer
from src.signal_service.contract import iso_z
from src.signal_service.storage import SignalStorage

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("smoke_signal_service")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
#: Shared contract from the sibling trading-stack checkout (optional for smoke).
SCHEMA_PATH = os.environ.get(
    "SIGNAL_SCHEMA_PATH",
    os.path.join(os.path.dirname(PROJECT_ROOT), "trading-stack", "contracts", "signal.schema.json"),
)

# Fenced like real model output, so the engine's JSON extraction path runs.
def _configure_offline_runtime() -> str:
    """Let config-dependent imports load without real keys.

    The extractor import pulls src.trading, which loads the config singleton.
    Point it at a temp empty keys file and the committed config example so the
    smoke stays offline and needs no secrets.

    Returns:
        Path of the temp keys file to clean up, or "" when untouched.
    """
    temp_keys = ""
    if not os.environ.get("LLM_TRADER_KEYS_ENV"):
        handle = tempfile.NamedTemporaryFile("w", suffix=".env", delete=False)
        handle.close()
        temp_keys = handle.name
        os.environ["LLM_TRADER_KEYS_ENV"] = temp_keys
    if not os.environ.get("LLM_TRADER_CONFIG_INI"):
        example = os.path.join(PROJECT_ROOT, "config", "config.ini.example")
        if os.path.isfile(example):
            os.environ["LLM_TRADER_CONFIG_INI"] = example
    return temp_keys


CANNED_RESPONSE = (
    "```json\n"
    '{"signal": "BUY", "confidence": "HIGH", "stop_loss": 67000.0, '
    '"take_profit": 70000.0, "position_size": 5, '
    '"reasoning": "Trend + breakout with volume confirmation."}\n'
    "```"
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


class MinimalJsonParser:
    """Offline stand-in for UnifiedParser.extract_json_block (no config, no keys)."""

    _BLOCK = re.compile(r"```json\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)

    def extract_json_block(self, text, unwrap_key=None):
        match = self._BLOCK.search(text)
        if not match:
            return None
        try:
            data = json.loads(match.group(1))
        except json.JSONDecodeError:
            return None
        if unwrap_key and isinstance(data.get(unwrap_key), dict):
            return data[unwrap_key]
        return data


class FakeResultProcessor:
    """Stand-in for AnalysisResultProcessor, exposing a parser like the bot."""

    def __init__(self) -> None:
        self.unified_parser = MinimalJsonParser()


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
        self.result_processor = FakeResultProcessor()

    async def analyze_market(self, **_kwargs):
        self.last_response = True
        return {
            "model": "deepseek-v4.1-flash",
            "provider": "opencode-go",
            "raw_response": self.last_llm_response,
        }


def _validate_with_jsonschema(signals_dir: str) -> None:
    """Validate against the shared schema when both it and jsonschema exist."""
    if not os.path.isfile(SCHEMA_PATH):
        logger.warning(
            "Shared schema not found at %s: skipping JSON schema validation "
            "(set SIGNAL_SCHEMA_PATH to enable it)", SCHEMA_PATH,
        )
        return
    try:
        import jsonschema
    except ImportError:
        logger.warning("jsonschema not installed: skipping JSON schema validation")
        return

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
    temp_keys = _configure_offline_runtime()
    try:
        return _run_smoke()
    finally:
        if temp_keys:
            try:
                os.unlink(temp_keys)
            except OSError:
                pass


def _run_smoke() -> int:
    """Run the smoke assertions with an offline config already in place."""
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