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


def test_normalize_market_context_drops_urls_payload():
    """A (text, urls) return keeps only the text part."""
    assert _normalize_market_context(("News text", {"article": "https://example.com"})) == "News text"
    assert _normalize_market_context(("", {})) == ""


def test_normalize_market_context_drops_non_text_values():
    """No usable text means the analyzer must skip the custom instruction."""
    assert _normalize_market_context(None) == ""
    assert _normalize_market_context((None, {})) == ""
    assert _normalize_market_context(123) == ""
    assert _normalize_market_context({"article": "https://example.com"}) == ""
