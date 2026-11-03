"""Tests for LLM-focus v2 feature flags and default-off isolation."""

from types import SimpleNamespace
from unittest.mock import MagicMock

from src.llm_focus.flags import (
    LLM_FOCUS_V2,
    event_positions_v2_enabled,
    flag_enabled,
    llm_focus_v2_enabled,
    shortlist_v2_enabled,
)


class TestFlagDefaults:
    def test_none_config_is_off(self):
        assert not flag_enabled(None, LLM_FOCUS_V2)
        assert not llm_focus_v2_enabled(None)
        assert not shortlist_v2_enabled(None)
        assert not event_positions_v2_enabled(None)

    def test_empty_mapping_is_off(self):
        assert not llm_focus_v2_enabled({})
        assert not shortlist_v2_enabled({})
        assert not event_positions_v2_enabled({})

    def test_missing_attribute_is_off(self):
        assert not llm_focus_v2_enabled(SimpleNamespace())
        assert not shortlist_v2_enabled(SimpleNamespace())
        assert not event_positions_v2_enabled(SimpleNamespace())

    def test_magicmock_values_are_not_treated_as_truthy(self):
        config = MagicMock()
        config.get_config.return_value = MagicMock()
        assert not flag_enabled(config, LLM_FOCUS_V2)


class TestFlagLookup:
    def test_uppercase_attribute(self):
        config = SimpleNamespace(LLM_FOCUS_V2=True, SHORTLIST_V2=False)
        assert llm_focus_v2_enabled(config)
        assert not shortlist_v2_enabled(config)

    def test_flat_mapping(self):
        assert llm_focus_v2_enabled({"llm_focus_v2": "true"})
        assert shortlist_v2_enabled({"SHORTLIST_V2": "yes"})
        assert event_positions_v2_enabled({"event_positions_v2": 1})

    def test_nested_section_mapping(self):
        config = {"llm_focus": {"llm_focus_v2": "on", "shortlist_v2": "off"}}
        assert llm_focus_v2_enabled(config)
        assert not shortlist_v2_enabled(config)

    def test_get_config_lookup(self):
        config = MagicMock()
        config.get_config.return_value = "true"
        assert llm_focus_v2_enabled(config)
        call_args = config.get_config.call_args.args
        assert call_args[:2] == ("llm_focus", "llm_focus_v2")

    def test_string_false_values(self):
        for value in ("false", "0", "no", "off", ""):
            assert not llm_focus_v2_enabled({"llm_focus_v2": value})

    def test_unknown_value_uses_default(self):
        assert not flag_enabled(SimpleNamespace(LLM_FOCUS_V2=object()), LLM_FOCUS_V2)
        assert flag_enabled(SimpleNamespace(LLM_FOCUS_V2=object()), LLM_FOCUS_V2, default=True)
