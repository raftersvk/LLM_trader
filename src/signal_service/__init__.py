"""Signal service: deterministic, signal-only producer for Trading Stack.

The LLM never places orders. This package emits immutable, schema-valid
signals consumed by Freqtrade through JSONL files.
"""

from src.signal_service.contract import (
    ALLOWED_ACTIONS,
    CONFIDENCE_MAP,
    SIGNAL_SCHEMA_VERSION,
    Signal,
    SignalValidationError,
    build_decision_id,
    confidence_to_float,
    hash_raw_response,
)
from src.signal_service.storage import SignalStorage
from src.signal_service.analyzer_adapter import PairAnalyzer, ReplayDataInjector
from src.signal_service.service import SignalService

__all__ = [
    "ALLOWED_ACTIONS",
    "CONFIDENCE_MAP",
    "SIGNAL_SCHEMA_VERSION",
    "Action",
    "Signal",
    "SignalValidationError",
    "build_decision_id",
    "confidence_to_float",
    "hash_raw_response",
    "SignalStorage",
    "PairAnalyzer",
    "ReplayDataInjector",
    "SignalService",
]