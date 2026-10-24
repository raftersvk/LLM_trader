"""Tests for the signal contract v1 (contracts/signal.schema.json)."""

import pytest

from src.signal_service.contract import (
    ALLOWED_ACTIONS,
    CONFIDENCE_MAP,
    SIGNAL_SCHEMA_VERSION,
    Signal,
    SignalValidationError,
    build_decision_id,
    confidence_to_float,
    hash_raw_response,
    parse_iso,
)


CANDLE_OPEN = "2026-10-23T14:00:00Z"


def make_signal_kwargs(**overrides):
    base = dict(
        schema_version=SIGNAL_SCHEMA_VERSION,
        decision_id=build_decision_id("BTC/USDT", "1h", CANDLE_OPEN),
        source="llm_trader",
        symbol="BTC/USDT",
        timeframe="1h",
        candle_open_time=CANDLE_OPEN,
        candle_close_time="2026-10-23T15:00:00Z",
        created_at="2026-10-23T15:00:02Z",
        expires_at="2026-10-23T16:00:00Z",
        action="BUY",
        confidence=0.9,
        risk_approved=True,
        sl=67000.0,
        tp=70000.0,
        position_size_pct=0.05,
        reasoning="Trend + breakout with volume confirmation.",
        model="deepseek-v4.1-flash",
        provider="opencode-go",
    )
    base.update(overrides)
    return base


def round_trip(signal):
    return Signal.from_dict(signal.to_dict())


def test_build_decision_id_is_deterministic():
    first = build_decision_id("BTC/USDT", "1h", CANDLE_OPEN)
    second = build_decision_id("BTC/USDT", "1h", CANDLE_OPEN)
    other = build_decision_id("BTC/USDT", "1h", "2026-10-23T15:00:00Z")
    assert first == second
    assert first.startswith("btc-usdt-1h-")
    assert first != other


def test_confidence_mapping():
    assert confidence_to_float("HIGH") == CONFIDENCE_MAP["HIGH"]
    assert confidence_to_float("MEDIUM") == CONFIDENCE_MAP["MEDIUM"]
    assert confidence_to_float("LOW") == CONFIDENCE_MAP["LOW"]
    assert 0.0 <= confidence_to_float("HIGH") <= 1.0
    assert confidence_to_float(85) == 0.85
    assert confidence_to_float(0.95) == 0.95
    with pytest.raises(SignalValidationError):
        confidence_to_float("NONSENSE")


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        pytest.param(51, 0.50, id="51-down"),
        pytest.param(52, 0.50, id="52-down"),
        pytest.param(53, 0.55, id="53-up"),
        pytest.param(55, 0.55, id="55-exact"),
        pytest.param(57, 0.55, id="57-down"),
        pytest.param(58, 0.60, id="58-up"),
        pytest.param(68, 0.70, id="68-up"),
        pytest.param(100, 1.00, id="100-max"),
        pytest.param(0, 0.00, id="0-min"),
    ],
)
def test_numeric_confidence_rounds_to_nearest_five_percent(raw, expected):
    """A numeric score keeps its precision, rounded to the nearest 5% half-up."""
    assert confidence_to_float(raw) == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        pytest.param(-5, 0.0, id="below-zero-clamped"),
        pytest.param(140, 1.0, id="above-hundred-clamped"),
        pytest.param(0.6, 0.60, id="fraction-input"),
        pytest.param("57", 0.55, id="numeric-string"),
        pytest.param(1, 0.0, id="integer-one-is-a-low-score"),
        pytest.param(1.0, 0.0, id="float-one-is-a-low-score"),
        pytest.param(2, 0.0, id="two-is-a-low-score"),
        pytest.param(0.99, 1.0, id="just-below-one-is-a-fraction"),
    ],
)
def test_numeric_confidence_clamps_and_accepts_fractions(raw, expected):
    assert confidence_to_float(raw) == expected


def test_hash_raw_response_format_and_determinism():
    raw = "raw llm response text"
    assert hash_raw_response(raw) == hash_raw_response(raw)
    assert hash_raw_response(raw).startswith("sha256:")
    assert len(hash_raw_response(raw)) == len("sha256:") + 64


def test_buy_requires_sl_tp_and_hold_may_have_nulls():
    buy = Signal(**make_signal_kwargs())
    buy.validate()
    assert round_trip(buy).to_dict() == buy.to_dict()

    with pytest.raises(SignalValidationError):
        Signal(**make_signal_kwargs(sl=None)).validate()
    with pytest.raises(SignalValidationError):
        Signal(**make_signal_kwargs(tp=None)).validate()

    hold = Signal(**make_signal_kwargs(
        action="HOLD", sl=None, tp=None, risk_approved=False, confidence=0.0,
    ))
    hold.validate()
    assert round_trip(hold).to_dict() == hold.to_dict()


def test_invalid_sl_tp_values_rejected():
    with pytest.raises(SignalValidationError):
        Signal(**make_signal_kwargs(sl=-1.0)).validate()
    with pytest.raises(SignalValidationError):
        Signal(**make_signal_kwargs(tp=0.0)).validate()


def test_candle_time_invariants():
    with pytest.raises(SignalValidationError):
        Signal(**make_signal_kwargs(candle_close_time="2026-10-23T16:00:00Z")).validate()
    with pytest.raises(SignalValidationError):
        Signal(**make_signal_kwargs(expires_at="2026-10-23T17:00:00Z")).validate()


def test_no_extra_fields_allowed():
    extra = make_signal_kwargs()
    extra["secret_field"] = "oops"
    with pytest.raises(SignalValidationError):
        Signal.from_dict(extra)


def test_missing_required_field_rejected():
    incomplete = make_signal_kwargs()
    del incomplete["reasoning"]
    with pytest.raises(SignalValidationError):
        Signal.from_dict(incomplete)


def test_is_expired_false_after_one_timeframe():
    signal = Signal(**make_signal_kwargs())
    at_close = parse_iso("2026-10-23T15:59:00Z")
    at_expiry = parse_iso("2026-10-23T16:01:00Z")
    assert not signal.is_expired(at_close)
    assert signal.is_expired(at_expiry)


def test_schema_version_is_locked():
    signal = Signal(**make_signal_kwargs())
    assert signal.schema_version == SIGNAL_SCHEMA_VERSION == "1.0"
    assert ALLOWED_ACTIONS == {"BUY", "SELL", "HOLD"}


def test_raw_response_hash_pattern_validated():
    signal = Signal(**make_signal_kwargs(raw_response_hash="sha256:" + "a" * 64))
    signal.validate()
    with pytest.raises(SignalValidationError):
        Signal(**make_signal_kwargs(raw_response_hash="md5:" + "a" * 32)).validate()


def test_jev_payload_shape():
    signal = Signal(**make_signal_kwargs(jev={
        "model": "jev-1.13-free", "tradeability": True, "bias": "long",
    }))
    signal.validate()
    assert round_trip(signal).jev == signal.jev
    with pytest.raises(SignalValidationError):
        Signal(**make_signal_kwargs(jev={"unexpected": 1})).validate()

def test_package_all_names_are_defined():
    """Every __all__ entry must exist, otherwise star-import raises AttributeError."""
    from typing import get_args

    import src.signal_service as package

    missing = [name for name in package.__all__ if not hasattr(package, name)]
    assert missing == []
    assert set(get_args(package.Action)) == ALLOWED_ACTIONS


def test_star_import_exposes_every_name():
    """`from src.signal_service import *` must not raise AttributeError."""
    import src.signal_service as package

    namespace: dict = {}
    exec("from src.signal_service import *", namespace)  # pylint: disable=exec-used

    exported = {name for name in namespace if not name.startswith("__")}
    assert exported == set(package.__all__)
