"""Tests for the signal contract v1 (contracts/signal.schema.json)."""

import pytest

from src.signal_service.contract import (
    ALLOWED_ACTIONS,
    ALLOWED_HTF_ALIGNMENTS,
    ALLOWED_REGIMES,
    ALLOWED_SETUPS,
    CONFIDENCE_MAP,
    SIGNAL_SCHEMA_VERSION,
    SIGNAL_SCHEMA_VERSIONS,
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


def test_moderate_is_a_medium_synonym():
    """MODERATE is accepted and maps exactly like MEDIUM (production 2026-10-29)."""
    assert CONFIDENCE_MAP["MODERATE"] == CONFIDENCE_MAP["MEDIUM"] == 0.65
    assert confidence_to_float("MODERATE") == 0.65


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        pytest.param("MODERATE", 0.65, id="moderate-exact"),
        pytest.param("moderate", 0.65, id="moderate-lower"),
        pytest.param("Moderate", 0.65, id="moderate-title"),
        pytest.param("  MODERATE \n", 0.65, id="moderate-whitespace"),
        pytest.param("high", 0.90, id="high-lower-non-regression"),
        pytest.param(" medium ", 0.65, id="medium-whitespace-non-regression"),
        pytest.param("Low\t", 0.35, id="low-whitespace-non-regression"),
    ],
)
def test_label_confidence_is_case_and_whitespace_insensitive(raw, expected):
    """Labels are normalized before lookup; numeric scores are untouched."""
    assert confidence_to_float(raw) == expected



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


@pytest.mark.parametrize(
    "raw",
    [
        pytest.param(float("nan"), id="nan-float"),
        pytest.param(float("inf"), id="positive-infinity"),
        pytest.param(float("-inf"), id="negative-infinity"),
        pytest.param("NaN", id="nan-string"),
        pytest.param("Infinity", id="infinity-string"),
    ],
)
def test_non_finite_confidence_is_rejected(raw):
    """NaN and Infinity must never be clamped into a valid high confidence."""
    with pytest.raises(SignalValidationError):
        confidence_to_float(raw)


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
    # v1.1 adds UPDATE/CLOSE position records on a separate channel; entry
    # signals stay 1.0 and keep only BUY/SELL/HOLD.
    assert ALLOWED_ACTIONS == {"BUY", "SELL", "HOLD", "UPDATE", "CLOSE"}


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


def test_v12_taxonomy_fields_round_trip():
    """v1.2 adds an optional, additive 1h taxonomy to entry records."""
    signal = Signal(**make_signal_kwargs(
        schema_version="1.2",
        setup="trend_pullback",
        regime="trend_up",
        htf_alignment="aligned",
    ))
    signal.validate()
    payload = signal.to_dict()
    assert payload["schema_version"] == "1.2"
    assert payload["setup"] == "trend_pullback"
    assert payload["regime"] == "trend_up"
    assert payload["htf_alignment"] == "aligned"
    assert round_trip(signal).to_dict() == payload


def test_v10_and_v11_readers_keep_the_old_shape():
    """A record without taxonomy stays byte-for-byte the v1.0/v1.1 shape."""
    for version in ("1.0", "1.1"):
        signal = Signal(**make_signal_kwargs(schema_version=version))
        signal.validate()
        payload = signal.to_dict()
        assert "setup" not in payload
        assert "regime" not in payload
        assert "htf_alignment" not in payload
        assert round_trip(signal).schema_version == version


def test_v12_is_accepted_without_forcing_it():
    assert "1.2" in SIGNAL_SCHEMA_VERSIONS
    assert SIGNAL_SCHEMA_VERSION == "1.0"


@pytest.mark.parametrize(("field", "bad"), [
    ("setup", "trend"),
    ("regime", "bull"),
    ("htf_alignment", "up"),
])
def test_v12_taxonomy_rejects_unknown_values(field, bad):
    with pytest.raises(SignalValidationError):
        Signal(**make_signal_kwargs(**{field: bad})).validate()


@pytest.mark.parametrize("bad", [1, ["trend_pullback"], {"setup": "trend_pullback"}, True])
def test_v12_taxonomy_rejects_non_string_values_without_typeerror(bad):
    """A non-string enum must be a SignalValidationError, not an unhashable TypeError."""
    with pytest.raises(SignalValidationError):
        Signal(**make_signal_kwargs(setup=bad)).validate()


def test_v12_taxonomy_accepts_every_contract_value():
    for setup in ALLOWED_SETUPS:
        Signal(**make_signal_kwargs(setup=setup)).validate()
    for regime in ALLOWED_REGIMES:
        Signal(**make_signal_kwargs(regime=regime)).validate()
    for alignment in ALLOWED_HTF_ALIGNMENTS:
        Signal(**make_signal_kwargs(htf_alignment=alignment)).validate()


def test_v12_jev_confidence_numbers_are_optional_and_bounded():
    jev = {
        "model": "jev-1.13-free",
        "tradeability": True,
        "bias": "long",
        "regime": "trend",
        "regime_confidence": 0.72,
        "bias_confidence": 0.61,
        "tradeability_probability": 0.83,
        "confidence": 0.75,
    }
    signal = Signal(**make_signal_kwargs(jev=jev))
    signal.validate()
    assert round_trip(signal).jev == jev
    for key in ("regime_confidence", "bias_confidence", "tradeability_probability", "confidence"):
        with pytest.raises(SignalValidationError):
            Signal(**make_signal_kwargs(jev={"model": "jev", key: 1.5})).validate()
        with pytest.raises(SignalValidationError):
            Signal(**make_signal_kwargs(jev={"model": "jev", key: True})).validate()

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
