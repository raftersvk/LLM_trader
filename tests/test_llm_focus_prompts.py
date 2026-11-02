"""Tests for the v2 prompt/schema contract and TemplateManager isolation."""

from types import SimpleNamespace

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

    def test_schema_forbids_risk_and_regime_fields(self):
        properties = V2_ANALYSIS_SCHEMA["properties"]
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

    def test_forbidden_risk_fields_are_rejected(self):
        payload = _valid_payload()
        payload["analysis"]["stop_loss"] = 95_000
        payload["analysis"]["position_size_pct"] = 0.2
        issues = validate_v2_payload(payload)
        assert any("stop_loss" in issue for issue in issues)
        assert any("position_size_pct" in issue for issue in issues)

    def test_invalid_action_and_confidence(self):
        payload = _valid_payload()
        payload["analysis"]["action"] = "MAYBE"
        payload["analysis"]["confidence"] = 140
        issues = validate_v2_payload(payload)
        assert any("invalid action" in issue for issue in issues)
        assert any("between 0 and 100" in issue for issue in issues)


class TestV2Mapping:
    def test_to_v1_compatible_maps_action_to_signal(self):
        mapped = to_v1_compatible(_valid_payload())["analysis"]
        assert mapped["signal"] == "BUY"
        assert mapped["setup"] == "TREND_PULLBACK"
        assert mapped["entry_thesis"]
        assert mapped["invalidation"]
        for forbidden in ("stop_loss", "take_profit", "position_size", "position_size_pct"):
            assert forbidden not in mapped

    def test_v2_template_contract(self):
        template = build_v2_response_template()
        for field in ("action", "setup", "entry_thesis", "invalidation", "confidence", "reasoning", "news"):
            assert f'"{field}"' in template
        assert '"stop_loss"' not in template
        assert '"take_profit"' not in template
        assert '"position_size"' not in template
        assert "POSITION SIZING FORMULA" not in template

    def test_v2_steps_do_not_request_regime_self_report(self):
        steps = build_v2_analysis_steps("BTC/USDT")
        assert "deterministic P0-A layer already owns regime" in steps
        assert "Do not output absolute stop-loss or take-profit prices" in steps
        assert "R/R CALCULATION" not in steps


class TestV2SystemPrompt:
    def test_v2_system_prompt_has_decision_boundary(self):
        prompt = build_v2_system_prompt("BTC/USDT", last_analysis_time="2026-11-02 12:00:00")
        assert "V2 Decision Boundary" in prompt
        assert "deterministic P0-A" in prompt
        assert "Do NOT self-report regime/HTF" in prompt
        assert "Last analysis: 2026-11-02 12:00:00 UTC" in prompt

    def test_v2_decision_rules_do_not_request_size_or_sl_tp(self):
        rules = build_v2_decision_rules(confidence_threshold=82)
        assert "confidence >= 82" in rules
        assert "JEV veto" in rules
        assert "do not emit absolute stop-loss" in rules.lower()
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
        assert '"stop_loss"' not in template
        assert '"take_profit"' not in template
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
