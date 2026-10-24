"""The weekend concept is neutralized for 24/7 crypto markets.

``is_weekend`` fields and parameters remain in upstream-owned structures so
future upstream syncs stay conflict-free; the single neutral helper always
reports False, so weekend-specific segments never render in production.
"""

from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from src.analyzer.prompts.prompt_builder import PromptBuilder
from src.analyzer.prompts.template_manager import TemplateManager
from src.utils.indicator_classifier import build_context_string_from_technical_data
from src.utils.market_calendar import is_weekend


@pytest.mark.parametrize(
    "moment",
    [
        datetime(2026, 9, 12, 12, 0, tzinfo=timezone.utc),  # Saturday
        datetime(2026, 9, 13, 12, 0, tzinfo=timezone.utc),  # Sunday
        datetime(2026, 9, 14, 12, 0, tzinfo=timezone.utc),  # Monday
        datetime(2026, 9, 16, 12, 0, tzinfo=timezone.utc),  # Wednesday
    ],
    ids=["saturday", "sunday", "monday", "wednesday"],
)
def test_is_weekend_is_always_false(moment: datetime) -> None:
    """Both weekend and weekday moments must report a non-weekend market."""
    assert is_weekend(moment) is False


def test_is_weekend_defaults_to_false() -> None:
    """Call sites that omit the reference time still get False."""
    assert is_weekend() is False


def test_context_string_never_contains_a_weekend_segment() -> None:
    """The real context-builder path emits no Weekend segment."""
    technical_data = {"adx": 25, "rsi": 55, "atr_percent": 2.0}

    context = build_context_string_from_technical_data(technical_data)

    assert "Weekend" not in context
    assert "Weekend Low Volume" not in context


SATURDAY = datetime(2026, 9, 12, 12, 0, tzinfo=timezone.utc)
SUNDAY = datetime(2026, 9, 13, 12, 0, tzinfo=timezone.utc)
MONDAY = datetime(2026, 9, 14, 12, 0, tzinfo=timezone.utc)
WEDNESDAY = datetime(2026, 9, 16, 12, 0, tzinfo=timezone.utc)


def _trading_context(now: datetime) -> str:
    """Render the real Trading Context section for a given clock."""
    builder = PromptBuilder(
        timeframe="1h",
        format_utils=MagicMock(),
        template_manager=TemplateManager(
            config=MagicMock(), logger=MagicMock(), timeframe_validator=None
        ),
        overview_formatter=MagicMock(),
        long_term_formatter=MagicMock(),
        technical_formatter=MagicMock(),
        market_formatter=MagicMock(),
        config=MagicMock(),
    )
    context = SimpleNamespace(symbol="BTC/USDT", current_price=68000.0)
    return builder.build_trading_context(context, now=now)


@pytest.mark.parametrize("moment", [SATURDAY, SUNDAY], ids=["saturday", "sunday"])
def test_trading_context_has_no_weekend_behaviour_claims(moment: datetime) -> None:
    """Stock-market weekend artifacts must never reach a 24/7 crypto prompt."""
    text = _trading_context(moment)

    assert "WEEKEND MODE" not in text
    assert "CME/Service Open" not in text
    assert "Weekly Open" not in text
    # The factual UTC day line is timestamp information, not a market claim.
    assert "Current Day:" in text


def _without_time_lines(text: str) -> str:
    """Drop lines that legitimately depend on the wall clock."""
    return "\n".join(
        line
        for line in text.splitlines()
        if not line.startswith(("- Current Day:", "- Analysis Time:"))
    )


def test_weekdays_produce_identical_prompt_structure_apart_from_timestamp() -> None:
    """Two weekdays may only differ in their time-derived lines."""
    assert _without_time_lines(_trading_context(MONDAY)) == _without_time_lines(
        _trading_context(WEDNESDAY)
    )
