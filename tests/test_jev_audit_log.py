"""Tests for the per-analysis JEV audit log.

The log must be written for every analysis, including runs where the JEV gate
is disabled. It carries the final contract signal, the advisory JEV answers and
the gate outcome, and never the raw LLM response or an API key.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import pytest

from src.config.regime_taxonomy import (
    DEFAULT_JEV_LOG_PATH,
    load_taxonomy_config,
)
from src.signal_service.analyzer_adapter import PairAnalyzer
from src.signal_service.audit_log import DecisionAuditLog, build_analysis_record
from src.signal_service.contract import build_decision_id
from src.signal_service.jev_client import JevResponse
from src.signal_service.jev_gate import JevGate

SYMBOL = "BTC/USDT"
BASE = datetime(2026, 1, 1, tzinfo=timezone.utc)


class FakeContext:
    """Minimal analysis context used by the adapter."""

    def __init__(self, candles: List[List[float]], timestamps: List[datetime]) -> None:
        self.ohlcv_candles = candles
        self.timestamps = timestamps
        self.current_price = candles[-1][3]


class FakeEngine:
    """Minimal AnalysisEngine surface used by PairAnalyzer."""

    def __init__(self, result: Dict[str, Any], candles: List[List[float]], timestamps: List[datetime]) -> None:
        self._result = result
        self.last_llm_response = json.dumps({
            "signal": result.get("recommendation", "HOLD"),
            "confidence": result.get("confidence", "LOW"),
            "stop_loss": result.get("stop_loss"),
            "take_profit": result.get("take_profit"),
            "position_size": result.get("position_size"),
            "reasoning": result.get("reasoning", ""),
        })
        self.context = FakeContext(candles, timestamps)
        self.calls = 0

    async def analyze_market(self, **_kwargs: Any) -> Dict[str, Any]:
        self.calls += 1
        return dict(self._result)


class FakeJevClient:
    """Return one canned JevResponse and record calls."""

    def __init__(self, response: JevResponse) -> None:
        self.response = response
        self.calls: List[Dict[str, Any]] = []

    async def ask(self, state: Dict[str, Any], questions: Dict[str, Dict[str, Any]]) -> JevResponse:
        self.calls.append(state)
        return self.response


def _write(path: Any, text: str) -> str:
    path.write_text(text, encoding="utf-8")
    return str(path)


def _candles(count: int = 6) -> List[List[float]]:
    rows = []
    for index in range(count):
        close = 100.0 + index
        rows.append([close - 1, close + 1, close - 2, close, 10.0 + index])
    return rows


def _timestamps(count: int = 6) -> List[datetime]:
    return [BASE + timedelta(hours=index) for index in range(count)]


def _engine_result(action: str = "BUY", **extra: Any) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "recommendation": action,
        "confidence": "HIGH",
        "stop_loss": 102.0,
        "take_profit": 114.0,
        "position_size": 0.05,
        "reasoning": "test reasoning",
        "model": "deepseek-v4.1-flash",
        "provider": "opencode-go",
    }
    payload.update(extra)
    return payload


def _jev_response(
    tradeable: float = 0.82,
    bias: str = "long",
    regime: str = "trend",
    tradeability_confidence: float = 0.66,
) -> JevResponse:
    return JevResponse(
        model="jev-test",
        answers={
            "tradeability": {
                "type": "noul",
                "noul": tradeable,
                "confidence": tradeability_confidence,
            },
            "bias": {"type": "choice", "choice": bias, "confidence": 0.8},
            "regime": {"type": "choice", "choice": regime, "confidence": 0.7},
            "confidence": {
                "type": "score",
                "score": 3.0,
                "legend": {"0": "none", "1": "weak", "2": "fair", "3": "strong", "4": "very strong"},
            },
        },
        latency_ms=120,
    )


def _analyzer(
    action: str = "BUY",
    gate: Optional[JevGate] = None,
    log_path: Optional[str] = None,
    taxonomy_config: Any = None,
    bias_short_only: bool = True,
    **extra: Any,
) -> PairAnalyzer:
    engine = FakeEngine(_engine_result(action, **extra), _candles(), _timestamps())
    return PairAnalyzer(
        engine=engine,
        symbol=SYMBOL,
        timeframe="1h",
        jev_gate=gate,
        taxonomy_config=taxonomy_config,
        bias_short_only=bias_short_only,
        decision_log=DecisionAuditLog(log_path) if log_path is not None else None,
    )


def _read_records(path: Any) -> List[Dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_absent_taxonomy_uses_the_runtime_logs_fallback():
    """The built-in audit path is repo-local, not the external W3 checkout."""
    config = load_taxonomy_config("/nonexistent/regime_taxonomy.yaml")

    assert config.jev.log_decisions is True
    assert config.jev.log_path == DEFAULT_JEV_LOG_PATH
    assert config.jev.log_path.replace("\\", "/").endswith(
        "runtime/logs/jev_decisions.jsonl"
    )


def test_audit_logger_is_disabled_unless_log_decisions_is_true(tmp_path):
    disabled = load_taxonomy_config(
        _write(
            tmp_path / "disabled_regime_taxonomy.yaml",
            "jev:\n  log_decisions: false\n",
        )
    )
    assert DecisionAuditLog.from_config(disabled) is None

    enabled = load_taxonomy_config(
        _write(
            tmp_path / "regime_taxonomy.yaml",
            f"jev:\n  log_decisions: true\n  log_path: {tmp_path / 'decisions.jsonl'}\n",
        )
    )
    logger = DecisionAuditLog.from_config(enabled)
    assert logger is not None
    assert logger.log_path == str(tmp_path / "decisions.jsonl")


@pytest.mark.asyncio
async def test_log_is_written_when_the_jev_gate_is_not_used(tmp_path):
    """Mode off still logs the decision, with an explicit unused-gate block."""
    path = tmp_path / "jev_decisions.jsonl"
    analyzer = _analyzer(gate=None, log_path=str(path))

    signal = await analyzer.analyze(candle_open_time="2026-01-01T05:00:00Z")

    records = _read_records(path)
    assert len(records) == 1
    record = records[0]
    assert record["symbol"] == SYMBOL
    assert record["timeframe"] == "1h"
    assert record["decision_id"] == build_decision_id(
        SYMBOL, "1h", "2026-01-01T05:00:00Z"
    )
    assert record["decision_id"] == signal.decision_id
    assert record["candle_close_time"] == "2026-01-01T06:00:00Z"
    assert record["signal_action"] == "BUY"
    assert record["signal_confidence"] == pytest.approx(0.9)
    assert record["risk_approved"] is True
    assert record["ts"].endswith("Z")
    assert record["jev"] == {
        "tradeability": None,
        "bias": None,
        "regime": None,
        "bias_confidence": None,
        "regime_confidence": None,
        "tradeability_confidence": None,
        "confidence": None,
    }
    assert record["gate"] == {
        "policy": "not_configured",
        "bias_short_only": None,
        "min_confidence": None,
        "used": False,
        "allowed": True,
        "reason": "JEV gate not used (mode=off)",
    }
    # Optional taxonomy fields stay absent when the model emitted none.
    assert "setup" not in record
    assert "regime" not in record
    assert "htf_alignment" not in record


@pytest.mark.asyncio
async def test_log_records_gate_veto_and_all_jev_confidences(tmp_path):
    """A consulted gate records noul, choices, confidences and the veto reason."""
    path = tmp_path / "jev_decisions.jsonl"
    client = FakeJevClient(_jev_response(tradeable=0.82, bias="long", regime="trend"))
    gate = JevGate(client=client, mode="post", bias_short_only=True)
    analyzer = _analyzer(action="SELL", gate=gate, log_path=str(path))

    signal = await analyzer.analyze(candle_open_time="2026-01-01T05:00:00Z")

    records = _read_records(path)
    assert len(records) == 1
    record = records[0]
    assert record["decision_id"] == signal.decision_id
    assert record["signal_action"] == "SELL"
    assert record["risk_approved"] is False
    assert record["jev"] == {
        "tradeability": pytest.approx(0.82),
        "bias": "long",
        "regime": "trend",
        "bias_confidence": pytest.approx(0.8),
        "regime_confidence": pytest.approx(0.7),
        "tradeability_confidence": pytest.approx(0.66),
        "confidence": pytest.approx(0.75),
    }
    assert record["gate"]["policy"] == "bias_short_only"
    assert record["gate"]["bias_short_only"] is True
    assert record["gate"]["min_confidence"] is None
    assert record["gate"]["used"] is True
    assert record["gate"]["allowed"] is False
    assert "SELL requires short bias" in record["gate"]["reason"]


@pytest.mark.asyncio
async def test_gate_policy_reflects_effective_bias_short_only_false(tmp_path):
    """The W3 taxonomy disables the short veto: the record must say so."""
    path = tmp_path / "jev_decisions.jsonl"
    client = FakeJevClient(_jev_response(bias="long"))
    gate = JevGate(
        client=client,
        mode="post",
        bias_short_only=False,
        min_confidence=0.5,
    )
    analyzer = _analyzer(
        action="SELL",
        gate=gate,
        log_path=str(path),
        bias_short_only=False,
    )

    await analyzer.analyze(candle_open_time="2026-01-01T05:00:00Z")

    gate_block = _read_records(path)[0]["gate"]
    assert gate_block["policy"] == "bias_filter_disabled+min_confidence"
    assert gate_block["bias_short_only"] is False
    assert gate_block["min_confidence"] == pytest.approx(0.5)
    assert gate_block["allowed"] is True


def test_log_carries_optional_taxonomy_fields_when_present():
    from src.signal_service.contract import Signal

    signal = Signal(
        schema_version="1.2",
        decision_id=build_decision_id(SYMBOL, "1h", "2026-01-01T05:00:00Z"),
        source="llm_trader",
        symbol=SYMBOL,
        timeframe="1h",
        candle_open_time="2026-01-01T05:00:00Z",
        candle_close_time="2026-01-01T06:00:00Z",
        created_at="2026-01-01T06:01:00Z",
        expires_at="2026-01-01T07:00:00Z",
        action="BUY",
        confidence=0.9,
        risk_approved=True,
        sl=102.0,
        tp=114.0,
        position_size_pct=0.05,
        reasoning="reasoning",
        model="model",
        provider="provider",
        setup="trend_pullback",
        regime="trend_up",
        htf_alignment="aligned",
    )

    record = build_analysis_record(signal)

    assert record["setup"] == "trend_pullback"
    assert record["regime"] == "trend_up"
    assert record["htf_alignment"] == "aligned"


@pytest.mark.asyncio
async def test_logged_candle_close_never_reads_a_future_candle(tmp_path):
    """The audit timestamp is the analysed candle close, not wall-clock data."""
    path = tmp_path / "jev_decisions.jsonl"
    analyzer = _analyzer(log_path=str(path))

    await analyzer.analyze(candle_open_time="2026-01-01T05:00:00Z")

    record = _read_records(path)[0]
    assert record["candle_close_time"] == "2026-01-01T06:00:00Z"
    assert record["candle_close_time"] <= record["ts"]


def test_record_contains_no_raw_response_or_secret_fields():
    from src.signal_service.contract import Signal

    signal = Signal(
        schema_version="1.0",
        decision_id=build_decision_id(SYMBOL, "1h", "2026-01-01T05:00:00Z"),
        source="llm_trader",
        symbol=SYMBOL,
        timeframe="1h",
        candle_open_time="2026-01-01T05:00:00Z",
        candle_close_time="2026-01-01T06:00:00Z",
        created_at="2026-01-01T06:01:00Z",
        expires_at="2026-01-01T07:00:00Z",
        action="HOLD",
        confidence=0.0,
        risk_approved=False,
        sl=None,
        tp=None,
        position_size_pct=0.05,
        reasoning="reasoning",
        model="model",
        provider="provider",
        prompt_version="v1",
    )
    record = build_analysis_record(signal)

    flattened = json.dumps(record).lower()
    assert "raw_response" not in flattened
    assert "api_key" not in flattened
    assert "authorization" not in flattened
    assert "reasoning" not in record
