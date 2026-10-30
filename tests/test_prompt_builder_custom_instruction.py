"""Regression tests for PromptBuilder custom instruction handling.

Passing a tuple (for example a RAG ``(text, urls)`` result) used to raise
``TypeError`` inside ``build_prompt``'s ``str.join`` and silently degraded the
analysis to HOLD without calling the LLM.
"""

from unittest.mock import MagicMock

from src.analyzer.analysis_context import AnalysisContext
from src.analyzer.analysis_engine import _normalize_market_context
from src.analyzer.prompts.prompt_builder import PromptBuilder
from src.analyzer.prompts.untrusted_context import EXTERNAL_CONTEXT_HEADER
from src.signal_service.analyzer_adapter import SIGNAL_OUTPUT_CONTRACT


def _build_prompt_builder() -> PromptBuilder:
    """Build a PromptBuilder whose sections are empty except custom instructions."""
    builder = PromptBuilder(
        timeframe="1h",
        format_utils=MagicMock(),
        template_manager=MagicMock(),
        overview_formatter=MagicMock(),
        long_term_formatter=MagicMock(),
        technical_formatter=MagicMock(),
        market_formatter=MagicMock(),
    )
    for method in (
        "build_trading_context",
        "build_sentiment_section",
        "build_coin_details_section",
        "build_market_data_section",
        "build_market_period_metrics_section",
    ):
        setattr(builder, method, MagicMock(return_value=""))
    builder.technical_analysis_formatter.format_technical_analysis.return_value = ""
    return builder


def _context() -> AnalysisContext:
    return AnalysisContext(symbol="BTC/USDT")


def test_tuple_instructions_are_flattened_without_type_error():
    """A tuple of strings must render instead of raising TypeError."""
    builder = _build_prompt_builder()
    builder.add_custom_instruction(("News one", "News two"))

    prompt = builder.build_prompt(_context())

    assert builder.custom_instructions == ["News one", "News two"]
    assert EXTERNAL_CONTEXT_HEADER in prompt
    assert "News one\nNews two" in prompt


def test_list_instructions_are_flattened_without_type_error():
    """Lists are flattened the same way as tuples."""
    builder = _build_prompt_builder()
    builder.add_custom_instruction(["First", "Second", "Third"])

    prompt = builder.build_prompt(_context())

    assert builder.custom_instructions == ["First", "Second", "Third"]
    assert "First\nSecond\nThird" in prompt


def test_plain_string_behavior_is_unchanged():
    """A plain string is still stored as a single instruction."""
    builder = _build_prompt_builder()
    builder.add_custom_instruction("Single instruction")

    prompt = builder.build_prompt(_context())

    assert builder.custom_instructions == ["Single instruction"]
    assert "Single instruction" in prompt


def test_empty_sequence_adds_no_instructions():
    """An empty sequence must be accepted and add nothing."""
    builder = _build_prompt_builder()
    builder.add_custom_instruction(())

    prompt = builder.build_prompt(_context())

    assert builder.custom_instructions == []
    assert EXTERNAL_CONTEXT_HEADER not in prompt
    assert isinstance(prompt, str)


def test_normalize_market_context_flattens_sequence():
    """AnalysisEngine normalization converts tuple context to text."""
    assert _normalize_market_context(("line one", "line two")) == "line one\nline two"
    assert _normalize_market_context("plain context") == "plain context"
    assert _normalize_market_context("") == ""


def test_normalize_market_context_handles_news_ablation_tuple():
    """The ``(text, urls)`` return shape keeps the text and drops the URLs.

    Covers both the news-disabled ablation stub ``("", {})`` and a populated
    URLs mapping, which is the shape that used to reach ``PromptBuilder`` and
    degrade the analysis to HOLD.
    """
    text = "News headline one\nNews headline two"
    normalized = _normalize_market_context(
        (text, {"article": "https://example.com", "other": "https://example.org"})
    )

    assert normalized == text
    assert "example.com" not in normalized
    assert "example.org" not in normalized
    assert _normalize_market_context(("", {})) == ""


def test_normalize_market_context_drops_non_text_values():
    """No usable text means the analyzer must skip the custom instruction."""
    assert _normalize_market_context(None) == ""
    assert _normalize_market_context((None, {})) == ""
    assert _normalize_market_context(123) == ""
    assert _normalize_market_context({"article": "https://example.com"}) == ""


def _build_system_prompt_builder() -> PromptBuilder:
    """PromptBuilder whose system-prompt sections are deterministic strings."""
    builder = _build_prompt_builder()
    builder.config = MagicMock(MODEL_VERBOSITY="medium")
    builder.template_manager.build_system_prompt.return_value = "BASE SYSTEM"
    builder.template_manager.build_analysis_steps.return_value = "STEPS"
    builder.template_manager.build_decision_rules.return_value = "RULES"
    builder.template_manager.build_response_template.return_value = "RESPONSE TEMPLATE"
    builder._has_retested_support_resistance = lambda: False
    builder._calculate_period_candles = lambda: {}
    return builder


def test_signal_system_suffix_is_appended_after_the_response_template():
    """The strict contract must be the LAST block of the assembled system prompt."""
    builder = _build_system_prompt_builder()

    system_prompt = builder.build_system_prompt(
        "BTC/USDT", _context(), system_prompt_suffix=SIGNAL_OUTPUT_CONTRACT
    )

    assert system_prompt.rstrip().endswith(SIGNAL_OUTPUT_CONTRACT.strip())
    # It must come after the shared response template, not before it.
    assert system_prompt.index("RESPONSE TEMPLATE") < system_prompt.index(
        SIGNAL_OUTPUT_CONTRACT.strip()
    )


def test_signal_system_suffix_is_never_added_to_the_user_prompt():
    """The contract belongs to the system role, never to user/RAG context."""
    builder = _build_system_prompt_builder()

    user_prompt = builder.build_prompt(_context())

    assert SIGNAL_OUTPUT_CONTRACT not in user_prompt


def test_system_prompt_is_unchanged_when_no_suffix_is_given():
    """The live bot passes no suffix, so its system prompt stays byte-identical."""
    builder = _build_system_prompt_builder()

    system_prompt = builder.build_system_prompt("BTC/USDT", _context())

    assert SIGNAL_OUTPUT_CONTRACT not in system_prompt
    assert system_prompt.endswith("RESPONSE TEMPLATE")
