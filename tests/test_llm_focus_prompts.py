"""Tests for the v2 prompt/schema contract and TemplateManager isolation."""

from types import SimpleNamespace

import pytest

from src.llm_focus.prompts import (
    V2_ANALYSIS_SCHEMA,
    V2_FORBIDDEN_FIELDS,
    V2_REQUIRED_FIELDS,
    build_v2_analysis_steps,
    build_v2_decision_rules,
    build_v2_response_template,
    build_v2_system_prompt,
    to_v1_compatible,
    validate_v2_payload,
)


def _valid_payload():
    return {
        "analysis": {
            "action": "BUY",
            "setup": "TREND_PULLBACK",
            "entry_thesis": "Pullback held above the 50 SMA with volume support.",
            "invalidation": "Close below the swing low invalidates the thesis.",
            "confidence": 78,
            "reasoning": "Trend and momentum align.",
            "stop_loss": 95_000.0,
            "take_profit": 104_000.0,
            "news": ["ETF flow positive"],
        }
    }


def _template_config(*, v2=False):
    return SimpleNamespace(
        STOP_LOSS_TYPE="soft",
        STOP_LOSS_CHECK_INTERVAL="1h",
        TAKE_PROFIT_TYPE="soft",
        TAKE_PROFIT_CHECK_INTERVAL="1h",
        RESEARCH_TEAM_ENABLED=False,
        PROVIDER="googleai",
        MARKET_TYPE="spot",
        ENTRY_ORDER_TYPE="market",
        MODEL_VERBOSITY="medium",
        LLM_FOCUS_V2=v2,
    )


class TestV2Schema:
    def test_schema_keeps_qualitative_fields(self):
        properties = V2_ANALYSIS_SCHEMA["properties"]
        for field in V2_REQUIRED_FIELDS:
            assert field in properties
        assert "news" in properties

    def test_schema_exposes_llm_levels_and_forbids_regime_size(self):
        properties = V2_ANALYSIS_SCHEMA["properties"]
        assert "stop_loss" in properties
        assert "take_profit" in properties
        for field in V2_FORBIDDEN_FIELDS:
            assert field not in properties
        assert V2_ANALYSIS_SCHEMA["additionalProperties"] is False

    def test_valid_payload_has_no_issues(self):
        assert validate_v2_payload(_valid_payload()) == ()

    def test_missing_required_field(self):
        payload = _valid_payload()
        del payload["analysis"]["setup"]
        issues = validate_v2_payload(payload)
        assert any("setup" in issue for issue in issues)

    def test_forbidden_size_and_regime_fields_are_rejected(self):
        payload = _valid_payload()
        payload["analysis"]["regime"] = "trend_up"
        payload["analysis"]["position_size_pct"] = 0.2
        issues = validate_v2_payload(payload)
        assert any("regime" in issue for issue in issues)
        assert any("position_size_pct" in issue for issue in issues)

    def test_buy_requires_both_llm_levels(self):
        payload = _valid_payload()
        payload["analysis"]["stop_loss"] = None
        issues = validate_v2_payload(payload)
        assert any("stop_loss" in issue for issue in issues)

        del payload["analysis"]["take_profit"]
        issues = validate_v2_payload(payload)
        assert any("take_profit" in issue for issue in issues)

    def test_hold_may_use_null_levels(self):
        payload = _valid_payload()
        payload["analysis"]["action"] = "HOLD"
        payload["analysis"]["stop_loss"] = None
        payload["analysis"]["take_profit"] = None
        assert validate_v2_payload(payload) == ()

    def test_invalid_action_and_confidence(self):
        payload = _valid_payload()
        payload["analysis"]["action"] = "MAYBE"
        payload["analysis"]["confidence"] = 140
        issues = validate_v2_payload(payload)
        assert any("invalid action" in issue for issue in issues)
        assert any("between 0 and 100" in issue for issue in issues)


class TestV2Mapping:
    def test_to_v1_compatible_maps_action_and_levels(self):
        mapped = to_v1_compatible(_valid_payload())["analysis"]
        assert mapped["signal"] == "BUY"
        assert mapped["setup"] == "TREND_PULLBACK"
        assert mapped["entry_thesis"]
        assert mapped["invalidation"]
        assert mapped["stop_loss"] == 95_000.0
        assert mapped["take_profit"] == 104_000.0
        for forbidden in ("position_size", "position_size_pct", "regime", "htf"):
            assert forbidden not in mapped

    def test_to_v1_compatible_fills_execution_fields(self):
        mapped = to_v1_compatible(
            _valid_payload(), entry_price=100_000.0, position_size=0.03
        )["analysis"]
        assert mapped["entry_price"] == 100_000.0
        assert mapped["position_size"] == 0.03
        assert mapped["risk_reward_ratio"] == pytest.approx(
            (104_000.0 - 100_000.0) / (100_000.0 - 95_000.0)
        )

    def test_to_v1_compatible_derives_short_risk_reward(self):
        payload = _valid_payload()
        payload["analysis"].update(
            {"action": "SELL", "stop_loss": 105_000.0, "take_profit": 90_000.0}
        )
        mapped = to_v1_compatible(payload, entry_price=100_000.0)["analysis"]
        assert mapped["risk_reward_ratio"] == pytest.approx(
            (100_000.0 - 90_000.0) / (105_000.0 - 100_000.0)
        )

    def test_v2_template_contract(self):
        template = build_v2_response_template()
        for field in (
            "action",
            "setup",
            "entry_thesis",
            "invalidation",
            "confidence",
            "reasoning",
            "stop_loss",
            "take_profit",
            "news",
        ):
            assert f'"{field}"' in template
        assert '"position_size"' not in template
        assert "policy C" in template
        assert "NEVER replaced" in template

    def test_v2_template_json_mode_emits_one_object(self):
        template = build_v2_response_template(json_object_output=True)
        assert "ONE JSON object and NOTHING outside it" in template
        assert '"narrative"' in template
        assert '"action"' in template
        assert "Structure your analysis before JSON" not in template
        assert "Then output JSON" not in template

    def test_v2_prompt_text_states_atr_relative_stop_band(self):
        template = build_v2_response_template()
        rules = build_v2_decision_rules()
        system_prompt = build_v2_system_prompt("BTC/USDT")
        steps = build_v2_analysis_steps("BTC/USDT")
        for text in (template, rules, system_prompt, steps):
            assert "0.5-10%" not in text
        assert "ATR-relative stop band" in template
        assert "min_sl_atr_multiple" in template
        assert "ATR-relative stop band" in rules

    def test_v2_steps_request_levels_without_regime_self_report(self):
        steps = build_v2_analysis_steps("BTC/USDT")
        assert "deterministic P0-A layer already owns regime" in steps
        assert "absolute stop-loss and take-profit" in steps
        assert "invalid levels reject" in steps
        assert "R/R CALCULATION" not in steps


class TestV2SystemPrompt:
    def test_v2_system_prompt_has_decision_boundary(self):
        prompt = build_v2_system_prompt("BTC/USDT", last_analysis_time="2026-11-02 12:00:00")
        assert "V2 Decision Boundary" in prompt
        assert "deterministic P0-A" in prompt
        assert "Do NOT self-report regime/HTF" in prompt
        assert "policy C" in prompt
        assert "absolute stop-loss/take-profit" in prompt
        assert "never replaced by ATR targets" in prompt
        assert "Last analysis: 2026-11-02 12:00:00 UTC" in prompt

    def test_v2_decision_rules_request_levels_but_not_size(self):
        rules = build_v2_decision_rules(confidence_threshold=82)
        assert "confidence >= 82" in rules
        assert "JEV veto" in rules
        assert "provide absolute stop-loss and take-profit" in rules.lower()
        assert "never replaced by ATR targets" in rules
        assert "POSITION SIZING FORMULA" not in rules


class TestTemplateManagerIsolation:
    @staticmethod
    def _manager(config):
        from src.analyzer.prompts.template_manager import TemplateManager

        return TemplateManager(config=config)

    def test_default_off_keeps_v1_output_contract(self):
        manager = self._manager(_template_config(v2=False))
        template = manager.build_response_template()
        system_prompt = manager.build_system_prompt("BTC/USDT")
        steps = manager.build_analysis_steps("BTC/USDT")
        rules = manager.build_decision_rules()

        assert "## Response Format" in template
        assert "V2 Decision Boundary" not in system_prompt
        assert "Analysis Steps (LLM-focus v2)" not in steps
        assert "Decision Rules (LLM-focus v2)" not in rules

    def test_flag_on_switches_to_v2_contract(self):
        manager = self._manager(_template_config(v2=True))
        template = manager.build_response_template()
        system_prompt = manager.build_system_prompt("BTC/USDT")
        steps = manager.build_analysis_steps("BTC/USDT")
        rules = manager.build_decision_rules()

        assert '"action"' in template
        assert '"setup"' in template
        assert '"entry_thesis"' in template
        assert '"invalidation"' in template
        assert '"stop_loss"' in template
        assert '"take_profit"' in template
        assert '"position_size"' not in template
        assert "POSITION SIZING FORMULA" not in template
        assert "V2 Decision Boundary" in system_prompt
        assert "Classify regime first" not in system_prompt
        assert "Analysis Steps (LLM-focus v2)" in steps
        assert "Decision Rules (LLM-focus v2)" in rules

    def test_flag_on_with_thresholds_uses_confidence_guidance(self):
        manager = self._manager(_template_config(v2=True))
        template = manager.build_response_template(
            dynamic_thresholds={"confidence_threshold": 82}
        )
        assert ">= 82" in template

    def test_flag_on_json_mode_provider_uses_single_object(self):
        config = _template_config(v2=True)
        config.PROVIDER = "deepseek"
        template = self._manager(config).build_response_template()
        assert "ONE JSON object and NOTHING outside it" in template
        assert '"narrative"' in template
        assert "Structure your analysis before JSON" not in template

    def test_flag_on_json_object_flag_uses_single_object(self):
        manager = self._manager(_template_config(v2=True))
        template = manager.build_response_template(json_object_output=True)
        assert "ONE JSON object and NOTHING outside it" in template
        assert '"narrative"' in template
