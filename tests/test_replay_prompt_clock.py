"""Regression tests: the prompt clock must be frozen to the analysed candle.

Replay truncates OHLCV to closed candles but, before this fix, the prompt still
used ``datetime.now()`` for Current Day / Analysis Time / Next Candle Close and
the deterministic time check. A historical candle was therefore described with
today's date: a temporal look-ahead leak. Replay must also stop attaching the
current cached news URLs.
"""

from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from src.analyzer.analysis_context import AnalysisContext
from src.analyzer.analysis_engine import AnalysisEngine
from src.analyzer.prompts.prompt_builder import PromptBuilder
from src.analyzer.prompts.template_manager import TemplateManager
from src.signal_service.analyzer_adapter import PairAnalyzer, ReplayDataInjector
from src.utils.timeframe_validator import TimeframeValidator

AS_OF = datetime(2026, 10, 21, 0, 0, 0, tzinfo=timezone.utc)
AS_OF_TEXT = "2026-10-21 00:00:00"


def _today() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def _prompt_builder() -> PromptBuilder:
    return PromptBuilder(
        timeframe="1h",
        format_utils=MagicMock(),
        template_manager=MagicMock(),
        overview_formatter=MagicMock(),
        long_term_formatter=MagicMock(),
        technical_formatter=MagicMock(),
        market_formatter=MagicMock(),
        config=MagicMock(),
    )


def _blank_sections(builder: PromptBuilder) -> PromptBuilder:
    """Stub every prompt section except the one under test."""
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


def test_trading_context_uses_injected_as_of_not_today():
    """A historical as_of renders that date and never today's date."""
    context = SimpleNamespace(symbol="BTC/USDT", current_price=68000.0)

    text = _prompt_builder().build_trading_context(context, now=AS_OF)

    assert AS_OF_TEXT in text
    assert "Current Day: Wednesday" in text
    assert _today() not in text


def test_trading_context_default_still_uses_now():
    """Omitting as_of keeps the live wall-clock behaviour."""
    context = SimpleNamespace(symbol="BTC/USDT", current_price=68000.0)

    text = _prompt_builder().build_trading_context(context)

    assert _today() in text


def test_deterministic_time_check_uses_injected_as_of():
    """The system prompt's DETERMINISTIC TIME CHECK honours as_of."""
    manager = TemplateManager(config=MagicMock(), logger=MagicMock(), timeframe_validator=None)

    prompt = manager.build_system_prompt(
        "BTC/USDT", previous_response="prior analysis", analysis_as_of=AS_OF,
    )

    assert AS_OF_TEXT in prompt
    assert _today() not in prompt


def test_deterministic_time_check_default_still_uses_now():
    manager = TemplateManager(config=MagicMock(), logger=MagicMock(), timeframe_validator=None)

    prompt = manager.build_system_prompt("BTC/USDT", previous_response="prior analysis")

    assert _today() in prompt


def test_prompt_builder_threads_as_of_into_trading_context():
    builder = _blank_sections(_prompt_builder())
    builder.analysis_as_of = AS_OF
    context = AnalysisContext(symbol="BTC/USDT")

    builder.build_prompt(context)

    builder.build_trading_context.assert_called_once_with(context, now=AS_OF)


class _ClockCollector:
    def __init__(self) -> None:
        self.article_urls = {"current-headline": "https://example.invalid/now"}

    async def collect_data(self, *_args, **_kwargs):
        return {"success": True, "errors": [], "market_context": "", "article_urls": {}}


class _ClockRagEngine:
    async def retrieve_context(self, *_args, **_kwargs):
        return ("", {})

    def get_latest_article_urls_snapshot(self):
        return {"current-headline": "https://example.invalid/now"}


class _ClockEngine:
    """Minimal AnalysisEngine surface used by PairAnalyzer."""

    last_finish_reason = None

    def __init__(self) -> None:
        self.model_manager = MagicMock()
        self.model_manager.supports_image_analysis.return_value = False
        self.rag_engine = _ClockRagEngine()
        self.data_collector = _ClockCollector()
        self.context = SimpleNamespace(
            current_price=105.0,
            ohlcv_candles=None,
            timestamps=None,
            timeframe="1h",
        )
        self.analysis_as_of = None
        self.observed_as_of = None
        self.article_urls = {}

    def set_analysis_as_of(self, as_of: datetime) -> None:
        self.analysis_as_of = as_of

    async def analyze_market(self, **_kwargs):
        self.observed_as_of = self.analysis_as_of
        # Mirror production: collector URLs + RAG snapshot feed the prompt metadata.
        self.article_urls = dict(getattr(self.data_collector, "article_urls", {}) or {})
        self.article_urls.update(self.rag_engine.get_latest_article_urls_snapshot())
        return {}


def _engine_config() -> MagicMock:
    config = MagicMock()
    config.TIMEFRAME = "1h"
    config.CANDLE_LIMIT = 200
    config.PROMPT_MAX_TOKENS = 200000
    config.MODEL_VERBOSITY = "low"
    config.STOP_LOSS_TYPE = "soft"
    config.TAKE_PROFIT_TYPE = "soft"
    config.STOP_LOSS_CHECK_INTERVAL = "1m"
    config.TAKE_PROFIT_CHECK_INTERVAL = "1m"
    return config


def _real_prompt_builder(config: MagicMock) -> PromptBuilder:
    """Real PromptBuilder/TemplateManager; only dumb formatters are stubbed."""
    template_manager = TemplateManager(
        config=config, logger=MagicMock(), timeframe_validator=TimeframeValidator,
    )
    builder = PromptBuilder(
        timeframe="1h",
        format_utils=MagicMock(),
        template_manager=template_manager,
        overview_formatter=MagicMock(),
        long_term_formatter=MagicMock(),
        technical_formatter=MagicMock(),
        market_formatter=MagicMock(),
        config=config,
        timeframe_validator=TimeframeValidator,
    )
    builder.technical_analysis_formatter.format_technical_analysis.return_value = ""
    return builder


@pytest.mark.asyncio
async def test_real_analysis_engine_prompt_uses_injected_clock():
    """The real engine seam must carry analysis_as_of into both prompt sections.

    Removing the assignment in AnalysisEngine._generate_ai_analysis (or the
    analysis_as_of kwarg in PromptBuilder.build_system_prompt) would make this
    test fall back to the wall clock and fail.
    """
    config = _engine_config()
    builder = _real_prompt_builder(config)
    engine = AnalysisEngine(
        MagicMock(), MagicMock(), MagicMock(), MagicMock(), config, prompt_builder=builder,
    )
    engine.token_counter = SimpleNamespace(count_tokens=lambda text: len(text) // 4)
    engine.symbol = "BTC/USDT"
    engine.context = AnalysisContext(symbol="BTC/USDT", timeframe="1h")
    engine.model_manager.describe_provider_and_model.return_value = ("provider", "model")
    engine.set_analysis_as_of(AS_OF)

    captured: dict[str, str] = {}

    async def fake_execute(system_prompt, prompt, provider, model, chart_image=None):
        captured["system"] = system_prompt
        captured["user"] = prompt
        _ = (provider, model, chart_image)
        return {}

    engine._execute_ai_request = fake_execute  # type: ignore[method-assign]

    await engine._generate_ai_analysis(
        provider=None,
        model=None,
        previous_response="prior analysis",
        precomputed_chart=(None, False),
    )

    assert AS_OF_TEXT in captured["user"]
    assert "### DETERMINISTIC TIME CHECK" in captured["system"]
    assert AS_OF_TEXT in captured["system"]
    assert _today() not in captured["user"]
    assert _today() not in captured["system"]


@pytest.mark.asyncio
async def test_replay_passes_candle_close_to_engine_and_drops_article_urls():
    """Replay freezes the prompt clock at the candle close and leaks no URLs."""
    engine = _ClockEngine()
    analyzer = PairAnalyzer(engine=engine, symbol="BTC/USDT", timeframe="1h")
    timestamps = [datetime(2026, 10, 21, 0, 0, tzinfo=timezone.utc)]
    candles = np.array([[0.0, 100.0, 110.0, 90.0, 105.0, 1000.0]])
    injector = ReplayDataInjector(analyzer, candles, timestamps)

    injector.install()
    try:
        await analyzer.analyze(candle_open_time="2026-10-21T00:00:00Z")
        assert engine.observed_as_of == datetime(2026, 10, 21, 1, 0, tzinfo=timezone.utc)
        assert engine.article_urls == {}
        assert engine.rag_engine.get_latest_article_urls_snapshot() == {}
    finally:
        injector.restore()

    # Live behaviour is restored: the cached URLs are visible again.
    assert engine.rag_engine.get_latest_article_urls_snapshot() != {}


@pytest.mark.asyncio
async def test_replay_restore_restores_article_url_state():
    """restore() must undo every article-URL mutation made by install()."""
    timestamps = [datetime(2026, 10, 21, 0, 0, tzinfo=timezone.utc)]
    candles = np.array([[0.0, 100.0, 110.0, 90.0, 105.0, 1000.0]])

    engine = _ClockEngine()
    original_engine_urls = {"current": "https://example.invalid/now"}
    original_collector_urls = {"collector": "https://example.invalid/collector"}
    engine.article_urls = original_engine_urls
    engine.data_collector.article_urls = original_collector_urls
    injector = ReplayDataInjector(
        PairAnalyzer(engine=engine, symbol="BTC/USDT", timeframe="1h"), candles, timestamps,
    )

    injector.install()
    assert engine.article_urls == {}
    assert engine.data_collector.article_urls == {}
    injector.restore()
    assert engine.article_urls is original_engine_urls
    assert engine.data_collector.article_urls is original_collector_urls

    # A collector that had no article_urls attribute must not keep the added {}.
    engine_without_attr = _ClockEngine()
    del engine_without_attr.data_collector.article_urls
    injector = ReplayDataInjector(
        PairAnalyzer(engine=engine_without_attr, symbol="BTC/USDT", timeframe="1h"),
        candles,
        timestamps,
    )

    injector.install()
    assert engine_without_attr.data_collector.article_urls == {}
    injector.restore()
    assert not hasattr(engine_without_attr.data_collector, "article_urls")


@pytest.mark.asyncio
async def test_live_analysis_uses_candle_close_clock_not_wall_clock():
    """Live analyses for a closed candle also use that candle's close."""
    engine = _ClockEngine()
    analyzer = PairAnalyzer(engine=engine, symbol="BTC/USDT", timeframe="1h")

    await analyzer.analyze(candle_open_time="2026-10-21T00:00:00Z")

    assert engine.observed_as_of == datetime(2026, 10, 21, 1, 0, tzinfo=timezone.utc)


@pytest.mark.asyncio
async def test_default_analysis_clock_is_now():
    """With no candle supplied the clock falls back to the current time."""
    engine = _ClockEngine()
    analyzer = PairAnalyzer(engine=engine, symbol="BTC/USDT", timeframe="1h")

    before = datetime.now(timezone.utc)
    await analyzer.analyze()
    after = datetime.now(timezone.utc)

    assert before <= engine.observed_as_of <= after
