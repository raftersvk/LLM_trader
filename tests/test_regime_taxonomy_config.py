"""Tests for the external regime/taxonomy configuration.

The config is a pure document: these tests never touch the network, a model or
an exchange. They cover the absent/valid/invalid loader paths, the thresholds
used by the regime snapshot, the rendered taxonomy and the 1h/4h gate.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from src.analyzer.prompts.template_manager import TemplateManager
from src.config.regime_taxonomy import (
    DEFAULT_SETUPS,
    DEFAULT_TAXONOMY_CONFIG_PATH,
    NO_TRADE_REGIMES,
    REGIMES,
    RegimeThresholds,
    TaxonomyConfigError,
    default_taxonomy_config,
    load_taxonomy_config,
    parse_taxonomy_config,
    render_setup_taxonomy,
    resolve_taxonomy_config_path,
)
from src.signal_service.jev_client import JevResponse
from src.signal_service.jev_gate import JevGate
from src.signal_service.regime_snapshot import build_regime_snapshot
from src.utils.timeframe_validator import TimeframeValidator
from tests.conftest import make_config, null_logger

DEFAULT_TREND_SETUPS = ["trend_pullback", "breakout_retest", "momentum_continuation"]


def _trend_up_inputs() -> dict[str, Any]:
    return {
        "technical_data": {
            "adx": 32.0,
            "plus_di": 31.0,
            "minus_di": 11.0,
            "choppiness": 30.0,
            "atr": 2.0,
            "atr_percent": 2.0,
            "sma_20": 100.0,
            "supertrend_direction": 1,
        },
        "long_term_data": {
            "daily_adx": 30.0,
            "daily_plus_di": 28.0,
            "daily_minus_di": 12.0,
        },
        "weekly_macro": {"weekly_macro_trend": {"trend_direction": "Bullish"}},
        "closes_1h": [99.0, 100.0, 101.0, 103.0, 105.0],
    }


def _write(path, text: str):
    path.write_text(text, encoding="utf-8")
    return str(path)


# --------------------------------------------------------------------- absent
def test_absent_config_returns_the_current_defaults(tmp_path):
    config = load_taxonomy_config(str(tmp_path / "missing.yaml"))

    assert config.source_path is None
    assert config.version == "1.0"
    assert config.enabled_timeframes == ("1h",)
    assert config.regime_thresholds == RegimeThresholds()
    assert config.regime_thresholds.as_dict() == {
        "adx_trend": 25.0,
        "adx_weak": 20.0,
        "range_choppiness": 61.8,
        "high_vol_atr_pct_floor": 3.0,
        "high_vol_atr_multiple": 2.5,
        "low_vol_atr_multiple": 0.55,
        "low_vol_choppiness": 55.0,
    }
    assert config.allowed_setups("trend_up") == tuple(DEFAULT_TREND_SETUPS)
    for regime in NO_TRADE_REGIMES:
        assert config.allowed_setups(regime) == ()
    assert config.jev.bias_short_only is True
    assert config.jev.log_decisions is False


def test_default_path_is_canonical_and_env_overrides_it(tmp_path, monkeypatch):
    monkeypatch.delenv("TAXONOMY_CONFIG_PATH", raising=False)
    monkeypatch.delenv("REGIME_TAXONOMY_CONFIG", raising=False)
    assert resolve_taxonomy_config_path() == DEFAULT_TAXONOMY_CONFIG_PATH

    path = _write(tmp_path / "custom.yaml", "version: '2.0'\nenabled_timeframes: [1h]\n")
    monkeypatch.setenv("TAXONOMY_CONFIG_PATH", path)
    assert resolve_taxonomy_config_path() == path
    assert load_taxonomy_config().version == "2.0"


def test_explicit_path_wins_over_environment(tmp_path, monkeypatch):
    env_path = _write(tmp_path / "env.yaml", "version: '9.9'\n")
    explicit_path = _write(tmp_path / "explicit.yaml", "version: '3.0'\n")
    monkeypatch.setenv("TAXONOMY_CONFIG_PATH", env_path)

    assert load_taxonomy_config(explicit_path).version == "3.0"


# ---------------------------------------------------------------------- valid
def test_valid_yaml_overrides_thresholds_setups_and_jev(tmp_path):
    path = _write(
        tmp_path / "regime_taxonomy.yaml",
        """
version: "1.1"
enabled_timeframes: [1h]
regime_thresholds:
  adx_trend: 30.0
  adx_weak: 22.0
  range_choppiness: 60.0
  high_vol_atr_pct_floor: 4.0
  high_vol_atr_multiple: 3.0
  low_vol_atr_multiple: 0.4
  low_vol_choppiness: 50.0
regime_setups:
  trend_up: [trend_pullback]
setups:
  trend_pullback:
    side: long
    description: custom pullback description
    entry: custom entry
    invalidation: custom invalidation
    regimes: [trend_up]
    sl_atr: 2.0
    tp_atr: 3.5
    max_hold_bars: 16
    trail_atr: 0.5
    be_at_r: 1.0
pair_overrides:
  BTC/USDT:
    regime_thresholds:
      adx_trend: 40.0
      adx_weak: 30.0
jev:
  bias_short_only: false
  log_decisions: true
  log_path: /tmp/jev_decisions.jsonl
""",
    )

    config = load_taxonomy_config(path)

    assert config.source_path == path
    assert config.version == "1.1"
    assert config.regime_thresholds.adx_trend == 30.0
    assert config.regime_thresholds.low_vol_choppiness == 50.0
    assert config.allowed_setups("trend_up") == ("trend_pullback",)
    setup = config.setups["trend_pullback"]
    assert setup.description == "custom pullback description"
    assert setup.sl_atr == 2.0
    assert setup.max_hold_bars == 16
    assert config.pair_overrides["BTC/USDT"].regime_thresholds.adx_trend == 40.0
    assert config.jev.bias_short_only is False
    assert config.jev.log_decisions is True
    assert config.jev.log_path == "/tmp/jev_decisions.jsonl"


def test_example_config_matches_the_built_in_defaults():
    path = Path(__file__).resolve().parents[1] / "config" / "regime_taxonomy.example.yaml"

    example = load_taxonomy_config(str(path))

    assert example.as_dict() == default_taxonomy_config().as_dict()


def test_json_document_is_accepted_as_fallback(tmp_path):
    path = _write(
        tmp_path / "regime_taxonomy.json",
        """
{
  "version": "1.2",
  "enabled_timeframes": ["1h"],
  "regime_thresholds": {"adx_trend": 28.0},
  "regime_setups": {"trend_up": ["momentum_continuation"]},
  "jev": {"bias_short_only": false}
}
""",
    )

    config = load_taxonomy_config(path)

    assert config.version == "1.2"
    assert config.regime_thresholds.adx_trend == 28.0
    assert config.regime_thresholds.adx_weak == 20.0
    assert config.allowed_setups("trend_up") == ("momentum_continuation",)
    assert config.jev.bias_short_only is False


# -------------------------------------------------------------------- invalid
@pytest.mark.parametrize(
    "document, needle",
    [
        ("version: '1.0'\nregime_thresholds:\n  adx_trend: nope\n", "adx_trend must be a number"),
        ("version: '1.0'\nregime_thresholds:\n  unknown_cutoff: 1\n", "unknown keys"),
        ("version: '1.0'\nregime_thresholds:\n  adx_trend: 10\n  adx_weak: 20\n", "adx_trend"),
        ("version: '1.0'\nregime_setups:\n  trend_up: [does_not_exist]\n", "unknown setups"),
        ("version: '1.0'\nregime_setups:\n  sideways: []\n", "unknown regimes"),
        ("version: '1.0'\nenabled_timeframes: []\n", "at least one timeframe"),
        ("version: '1.0'\nsetups:\n  trend_pullback:\n    side: sideways\n", "side must be one of"),
        ("version: '1.0'\njev:\n  bias_short_only: maybe\n", "must be a boolean"),
        ("version: '1.0'\nunexpected_section: true\n", "unknown keys"),
        ("regime_thresholds: [not-a-mapping\n", "invalid YAML"),
        ("", "empty"),
    ],
)
def test_invalid_configs_raise_clear_errors(tmp_path, document, needle):
    path = _write(tmp_path / "invalid.yaml", document)

    with pytest.raises(TaxonomyConfigError, match=needle):
        load_taxonomy_config(path)


def test_non_mapping_document_is_rejected():
    with pytest.raises(TaxonomyConfigError, match="must be a mapping"):
        parse_taxonomy_config(["not", "a", "mapping"])


# ------------------------------------------------------------------ thresholds
def test_snapshot_thresholds_come_from_the_config():
    default = build_regime_snapshot(**_trend_up_inputs(), timeframe="1h")
    assert default["regime"] == "trend_up"

    strict = parse_taxonomy_config({"regime_thresholds": {"adx_trend": 35.0}})
    configured = build_regime_snapshot(
        **_trend_up_inputs(), timeframe="1h", taxonomy_config=strict
    )
    assert configured["regime"] != "trend_up"
    assert configured["allowed_setups"] == []


def test_snapshot_allowed_setups_come_from_the_config():
    config = parse_taxonomy_config(
        {
            "regime_setups": {"trend_up": ["momentum_continuation"]},
        }
    )

    snapshot = build_regime_snapshot(
        **_trend_up_inputs(), timeframe="1h", taxonomy_config=config
    )

    assert snapshot["regime"] == "trend_up"
    assert snapshot["allowed_setups"] == ["momentum_continuation"]


def test_pair_overrides_apply_only_to_the_matching_symbol():
    config = parse_taxonomy_config(
        {
            "pair_overrides": {
                "BTC/USDT": {"regime_thresholds": {"adx_trend": 50.0, "adx_weak": 35.0}},
            }
        }
    )

    btc = build_regime_snapshot(
        **_trend_up_inputs(), timeframe="1h", symbol="BTC/USDT", taxonomy_config=config
    )
    eth = build_regime_snapshot(
        **_trend_up_inputs(), timeframe="1h", symbol="ETH/USDT", taxonomy_config=config
    )

    assert eth["regime"] == "trend_up"
    assert eth["allowed_setups"] == DEFAULT_TREND_SETUPS
    assert btc["regime"] == "range"


def test_partial_pair_override_keeps_other_regime_families():
    config = parse_taxonomy_config(
        {"pair_overrides": {"BTC/USDT": {"regime_setups": {"trend_up": []}}}}
    )

    effective = config.for_symbol("BTC/USDT")

    assert effective.allowed_setups("trend_up") == ()
    assert effective.allowed_setups("trend_down") == (
        "rally_failure",
        "breakdown_retest",
        "momentum_short",
    )


def test_compact_pair_override_key_matches_the_slash_symbol():
    config = parse_taxonomy_config(
        {"pair_overrides": {"BTCUSDT": {"regime_setups": {"trend_up": []}}}}
    )

    snapshot = build_regime_snapshot(
        **_trend_up_inputs(), timeframe="1h", symbol="BTC/USDT", taxonomy_config=config
    )

    assert snapshot["regime"] == "trend_up"
    assert snapshot["allowed_setups"] == []
    assert snapshot["no_trade"] is True


# ------------------------------------------------------------------- taxonomy
def test_default_taxonomy_renders_the_current_prompt_text():
    block = render_setup_taxonomy(default_taxonomy_config(), "1h")

    assert block.startswith("=== Setup Taxonomy (1h) ===")
    assert "The REGIME SNAPSHOT block supplied in Trading Context is authoritative." in block
    assert "- trend_up: trend_pullback, breakout_retest, momentum_continuation (long only)." in block
    assert "- trend_down: rally_failure, breakdown_retest, momentum_short (short; on spot" in block
    assert "- range: range_fade_extreme, range_mean_reversion." in block
    assert "- low_vol, high_vol, unclear: none — HOLD (tradeable=no)." in block
    for setup in DEFAULT_SETUPS.values():
        assert setup.description in block
    assert "- none: no actionable setup" in block


def test_custom_description_is_rendered_from_the_config():
    config = parse_taxonomy_config(
        {
            "setups": {
                "trend_pullback": {
                    "description": "CUSTOM PULLBACK DESCRIPTION",
                }
            }
        }
    )

    block = render_setup_taxonomy(config, "1h")

    assert "CUSTOM PULLBACK DESCRIPTION" in block
    assert "orderly pullback to SMA20/structure support" not in block


def test_one_hour_is_enabled_and_four_hours_is_disabled():
    config = default_taxonomy_config()

    assert config.timeframe_enabled("1h") is True
    assert config.timeframe_enabled("4h") is False
    assert render_setup_taxonomy(config, "1h") != ""
    assert render_setup_taxonomy(config, "4h") == ""

    snapshot = build_regime_snapshot(**_trend_up_inputs(), timeframe="4h", taxonomy_config=config)
    assert snapshot["regime"] == "trend_up"
    assert snapshot["allowed_setups"] == []
    assert snapshot["no_trade"] is True


def test_four_hours_can_be_enabled_explicitly():
    config = parse_taxonomy_config({"enabled_timeframes": ["1h", "4h"]})

    block = render_setup_taxonomy(config, "4h")

    assert block.startswith("=== Setup Taxonomy (4h) ===")
    assert "latest closed 4h candle" in block
    assert "This is a 4h decision" in block


def test_template_manager_renders_the_config_taxonomy_only_on_1h():
    config = parse_taxonomy_config(
        {"setups": {"trend_pullback": {"description": "CUSTOM TAXONOMY DESCRIPTION"}}}
    )
    manager = TemplateManager(
        config=make_config(),
        logger=null_logger(),
        timeframe_validator=TimeframeValidator(),
        taxonomy_config=config,
    )

    one_hour = manager.build_decision_rules(timeframe="1h", regime_snapshot_available=True)
    four_hours = manager.build_decision_rules(timeframe="4h", regime_snapshot_available=True)

    assert "CUSTOM TAXONOMY DESCRIPTION" in one_hour
    assert "=== Setup Taxonomy (1h) ===" in one_hour
    assert "Setup Taxonomy" not in four_hours

    one_hour_format = manager.build_response_template(
        timeframe="1h", regime_snapshot_available=True
    )
    four_hours_format = manager.build_response_template(
        timeframe="4h", regime_snapshot_available=True
    )
    assert '"setup": "none"' in one_hour_format
    assert '"setup": "none"' not in four_hours_format


def test_all_regimes_have_a_definition_even_when_the_file_is_partial():
    config = load_taxonomy_config("/nonexistent/config.yaml")

    assert set(config.regime_setups) == set(REGIMES)
    assert set(config.setups) == set(DEFAULT_SETUPS)
    for regime in REGIMES:
        for name in config.allowed_setups(regime):
            assert name in config.setups


# ------------------------------------------------------------------------ jev
def test_jev_section_defaults_reproduce_the_asymmetric_short_policy(tmp_path):
    config = load_taxonomy_config(str(tmp_path / "missing.yaml"))

    assert config.jev.bias_short_only is True
    assert config.jev.log_decisions is False


@pytest.mark.asyncio
async def test_jev_bias_short_only_can_be_disabled_from_config(tmp_path):
    config = load_taxonomy_config(
        _write(
            tmp_path / "jev.yaml",
            """
jev:
  bias_short_only: false
  log_decisions: true
  log_path: {log_path}
""".format(log_path=tmp_path / "decisions.jsonl"),
        )
    )

    class _Client:
        async def ask(self, state, questions):
            return JevResponse(
                model="jev-test",
                answers={
                    "bias": {"type": "choice", "choice": "long", "confidence": 0.8},
                    "tradeability": {"type": "noul", "noul": 0.9},
                },
            )

    gate = JevGate(
        client=_Client(),
        mode="post",
        bias_short_only=config.jev.bias_short_only,
        log_decisions=config.jev.log_decisions,
        log_path=config.jev.log_path,
    )

    result = await gate.check_post({"symbol": "BTC/USDT"}, "SELL")

    assert result.allowed is True
    assert (tmp_path / "decisions.jsonl").read_text(encoding="utf-8").strip()

