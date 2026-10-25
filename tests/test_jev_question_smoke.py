"""Tests for the JEV question-set smoke checker (no provider calls)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any, Dict

from src.signal_service.jev_client import JevResponse

_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

_SPEC = importlib.util.spec_from_file_location(
    "jev_question_smoke",
    _SCRIPTS / "jev_question_smoke.py",
)
assert _SPEC is not None and _SPEC.loader is not None
smoke = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = smoke
_SPEC.loader.exec_module(smoke)


def _response(**overrides: Any) -> JevResponse:
    answers: Dict[str, Dict[str, Any]] = {
        "tradeability": {"type": "noul", "noul": 0.62},
        "bias": {"type": "choice", "choice": "long", "confidence": 0.6},
        "regime": {"type": "choice", "choice": "trend", "confidence": 0.6},
        "confidence": {"type": "score", "score": 2.0, "legend": {"0": "a", "1": "b", "2": "c"}},
    }
    answers.update(overrides)
    return JevResponse(model="test", answers=answers, latency_ms=10)


def test_check_response_accepts_a_valid_answer_set():
    assert smoke._check_response(_response()) == []


def test_check_response_rejects_missing_and_invalid_fields():
    missing = _response()
    del missing.answers["bias"]
    assert any("missing answers" in problem for problem in smoke._check_response(missing))

    bad_noul = _response(tradeability={"type": "noul", "noul": 1.4})
    assert any("outside [0, 1]" in problem for problem in smoke._check_response(bad_noul))

    bad_bias = _response(bias={"type": "choice", "choice": "sideways"})
    assert any("bias.choice" in problem for problem in smoke._check_response(bad_bias))


def test_check_response_detects_invalid_question_marker_anywhere():
    response = _response(
        tradeability={
            "type": "noul",
            "noul": 0.5,
            "note": {"error": "invalid_question: criteria must be an object"},
        }
    )
    assert any(
        "invalid_question" in problem for problem in smoke._check_response(response)
    )


def test_discover_symbols_skips_futures(tmp_path):
    (tmp_path / "BTC_USDT_1h.csv").write_text("time,open,high,low,close,volume\n")
    (tmp_path / "BTC_USDT_1h_futures.csv").write_text("time,open,high,low,close,volume\n")
    (tmp_path / "NEXO_USDT_1h.csv").write_text("time,open,high,low,close,volume\n")

    assert smoke._discover_symbols(str(tmp_path)) == ["BTC/USDT", "NEXO/USDT"]
