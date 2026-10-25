"""Regression tests for ProviderOrchestrator chart routing."""
import io
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.managers.provider_orchestrator import ProviderOrchestrator
from src.managers.provider_types import ProviderClients
from src.platforms.ai_providers.litellm_client import LitellmAIClient


LITELLM_CONFIG = {
    "provider": "litellm",
    "base_url": "https://opencode.ai/inference/go/openai/v1",
    "api_key": "test-key",
    "model": "deepseek-v4.1-flash",
}


def _orchestrator(clients: ProviderClients) -> ProviderOrchestrator:
    config = SimpleNamespace(
        GOOGLE_STUDIO_MODEL="gemini-test",
        OPENROUTER_BASE_MODEL="openrouter/test",
        LITELLM_MODEL="deepseek-v4.1-flash",
        BLOCKRUN_MODEL="blockrun-test",
        VISION_MODEL="vision-test",
        get_model_config=lambda model: {"model": model},
    )
    return ProviderOrchestrator(logger=MagicMock(), config=config, clients=clients)


@pytest.mark.asyncio
async def test_invoke_litellm_forwards_chart_image():
    """invoke() with provider=litellm must route chart=True through image analysis."""
    client = LitellmAIClient(config=LITELLM_CONFIG, logger=MagicMock())
    orchestrator = _orchestrator(ProviderClients(litellm=client))
    captured = {}
    response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content="chart analyzed"),
                finish_reason="stop",
            )
        ],
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        id="response-id",
        model="deepseek-v4.1-flash",
    )

    async def fake_acompletion(**kwargs):
        captured.update(kwargs)
        return response

    with patch(
        "src.platforms.ai_providers.litellm_client.acompletion",
        new=fake_acompletion,
    ):
        result = await orchestrator.invoke(
            "litellm",
            [{"role": "user", "content": "Analyze this chart"}],
            chart=True,
            chart_image=io.BytesIO(b"fake-png-bytes"),
        )

    assert result.success is True
    assert captured["messages"][0]["content"][1]["type"] == "image_url"
    assert captured["messages"][0]["content"][1]["image_url"]["url"].startswith(
        "data:image/png;base64,"
    )


@pytest.mark.asyncio
async def test_invoke_litellm_without_chart_stays_text_only():
    """Without chart=True the litellm invocation must send plain text."""
    client = LitellmAIClient(config=LITELLM_CONFIG, logger=MagicMock())
    orchestrator = _orchestrator(ProviderClients(litellm=client))
    captured = {}
    response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content="ok"),
                finish_reason="stop",
            )
        ],
        usage=None,
        id="response-id",
        model="deepseek-v4.1-flash",
    )

    async def fake_acompletion(**kwargs):
        captured.update(kwargs)
        return response

    with patch(
        "src.platforms.ai_providers.litellm_client.acompletion",
        new=fake_acompletion,
    ):
        await orchestrator.invoke(
            "litellm",
            [{"role": "user", "content": "hello"}],
        )

    assert captured["messages"][0]["content"] == "hello"


def test_supports_chart_all_with_litellm_only():
    """supports_chart('all') must be True when only the litellm stack is available."""
    client = LitellmAIClient(config=LITELLM_CONFIG, logger=MagicMock())
    orchestrator = _orchestrator(ProviderClients(litellm=client))

    assert orchestrator.supports_chart("all") is True


def test_supports_chart_all_with_vision_litellm_only():
    """The vision_litellm fallback alone must also satisfy supports_chart('all')."""
    client = LitellmAIClient(config=LITELLM_CONFIG, logger=MagicMock())
    orchestrator = _orchestrator(ProviderClients(vision_litellm=client))

    assert orchestrator.supports_chart("all") is True


@pytest.mark.asyncio
async def test_get_chart_response_all_routes_to_litellm_fallback():
    """get_chart_response('all') must route to litellm when it is the only client."""
    client = LitellmAIClient(config=LITELLM_CONFIG, logger=MagicMock())
    orchestrator = _orchestrator(ProviderClients(litellm=client))
    captured = {}
    response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content="chart analyzed"),
                finish_reason="stop",
            )
        ],
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        id="response-id",
        model="deepseek-v4.1-flash",
    )

    async def fake_acompletion(**kwargs):
        captured.update(kwargs)
        return response

    with patch(
        "src.platforms.ai_providers.litellm_client.acompletion",
        new=fake_acompletion,
    ):
        result = await orchestrator.get_chart_response(
            "all",
            [{"role": "user", "content": "Analyze this chart"}],
            io.BytesIO(b"fake-png-bytes"),
        )

    assert result.success is True
    assert result.provider == "litellm"
    assert captured["messages"][0]["content"][1]["type"] == "image_url"
    assert captured["messages"][0]["content"][1]["image_url"]["url"].startswith(
        "data:image/png;base64,"
    )


def test_supports_chart_all_without_providers():
    """With no vision-capable provider available, supports_chart('all') is False."""
    orchestrator = _orchestrator(ProviderClients())

    assert orchestrator.supports_chart("all") is False

@pytest.mark.asyncio
async def test_get_chart_response_all_prefers_the_vision_endpoint():
    """A configured vision client must serve chart calls instead of the text client."""
    text_client = LitellmAIClient(config=LITELLM_CONFIG, logger=MagicMock())
    vision_config = {
        "provider": "litellm",
        "base_url": "https://vision.example/v1",
        "api_key": "vision-key",
        "model": "vision-test",
    }
    vision_client = LitellmAIClient(config=vision_config, logger=MagicMock())
    orchestrator = _orchestrator(
        ProviderClients(litellm=text_client, vision_litellm=vision_client)
    )
    captured = {}
    response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content="vision chart analyzed"),
                finish_reason="stop",
            )
        ],
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        id="vision-response-id",
        model="vision-test",
    )

    async def fake_acompletion(**kwargs):
        captured.update(kwargs)
        return response

    with patch(
        "src.platforms.ai_providers.litellm_client.acompletion",
        new=fake_acompletion,
    ):
        result = await orchestrator.get_chart_response(
            "all",
            [{"role": "user", "content": "Analyze this chart"}],
            io.BytesIO(b"fake-png-bytes"),
        )

    assert result.success is True
    assert result.provider == "vision_litellm"
    assert captured["base_url"] == "https://vision.example/v1"
    assert captured["api_key"] == "vision-key"
    assert captured["model"] == "vision-test"
    assert captured["messages"][0]["content"][1]["type"] == "image_url"


def _response(text: str) -> SimpleNamespace:
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=text),
                finish_reason="stop",
            )
        ],
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        id="response-id",
        model="vision-test",
    )


@pytest.mark.asyncio
async def test_explicit_litellm_chart_prefers_vision_endpoint():
    """VISION_* must also apply when provider = litellm (the usual fork setting)."""
    text_client = LitellmAIClient(config=LITELLM_CONFIG, logger=MagicMock())
    vision_client = LitellmAIClient(
        config={
            "provider": "litellm",
            "base_url": "https://vision.example/v1",
            "api_key": "vision-key",
            "model": "vision-test",
        },
        logger=MagicMock(),
    )
    orchestrator = _orchestrator(
        ProviderClients(litellm=text_client, vision_litellm=vision_client)
    )
    captured = {}

    async def fake_acompletion(**kwargs):
        captured.update(kwargs)
        return _response("vision analyzed")

    with patch(
        "src.platforms.ai_providers.litellm_client.acompletion", new=fake_acompletion
    ):
        result = await orchestrator.get_chart_response(
            "litellm", [{"role": "user", "content": "chart"}], io.BytesIO(b"png")
        )

    assert result.provider == "vision_litellm"
    assert captured["base_url"] == "https://vision.example/v1"


@pytest.mark.asyncio
async def test_chart_falls_back_to_text_litellm_when_vision_fails():
    """A vision failure must continue down the chain instead of returning early."""
    text_client = LitellmAIClient(config=LITELLM_CONFIG, logger=MagicMock())
    vision_client = LitellmAIClient(
        config={
            "provider": "litellm",
            "base_url": "https://vision.example/v1",
            "api_key": "vision-key",
            "model": "vision-test",
        },
        logger=MagicMock(),
    )
    orchestrator = _orchestrator(
        ProviderClients(litellm=text_client, vision_litellm=vision_client)
    )
    calls = []

    async def fake_acompletion(**kwargs):
        calls.append(kwargs.get("base_url"))
        if kwargs.get("base_url") == "https://vision.example/v1":
            raise RuntimeError("vision endpoint down")
        return _response("text endpoint analyzed")

    with patch(
        "src.platforms.ai_providers.litellm_client.acompletion", new=fake_acompletion
    ):
        result = await orchestrator.get_chart_response(
            "all", [{"role": "user", "content": "chart"}], io.BytesIO(b"png")
        )

    assert result.success is True
    assert result.provider == "litellm"
    assert calls == [
        "https://vision.example/v1",
        LITELLM_CONFIG["base_url"],
    ]


@pytest.mark.asyncio
async def test_vision_chart_goes_through_the_prompt_budget():
    """The vision path must be budgeted like every other provider."""
    from src.utils.prompt_budget import PromptBudgetPolicy

    token_counter = MagicMock()
    token_counter.count_tokens.return_value = 10
    policy = PromptBudgetPolicy(
        token_counter=token_counter,
        logger=MagicMock(),
        context_windows={},
        brain_max_tokens=5000,
        request_max_tokens=10000,
    )
    config = SimpleNamespace(
        GOOGLE_STUDIO_MODEL="gemini-test",
        OPENROUTER_BASE_MODEL="openrouter/test",
        LITELLM_MODEL="deepseek-v4.1-flash",
        BLOCKRUN_MODEL="blockrun-test",
        VISION_MODEL="vision-test",
        LITELLM_CONTEXT_WINDOW=1_000_000,
        VISION_CONTEXT_WINDOW=200_000,
        get_model_config=lambda model: {"model": model, "max_tokens": 2048},
    )
    vision_client = LitellmAIClient(
        config={
            "provider": "litellm",
            "base_url": "https://vision.example/v1",
            "api_key": "vision-key",
            "model": "vision-test",
        },
        logger=MagicMock(),
    )
    orchestrator = ProviderOrchestrator(
        logger=MagicMock(),
        config=config,
        clients=ProviderClients(vision_litellm=vision_client),
        prompt_budget=policy,
    )
    assert "vision_litellm/vision-test" in policy.context_windows

    with patch(
        "src.platforms.ai_providers.litellm_client.acompletion",
        new=AsyncMock(return_value=_response("budgeted")),
    ):
        result = await orchestrator.get_chart_response(
            "all", [{"role": "user", "content": "chart"}], io.BytesIO(b"png")
        )

    assert result.success is True
    assert result.provider == "vision_litellm"


def _budget_orchestrator(context_windows, config_overrides):
    from src.utils.prompt_budget import PromptBudgetPolicy

    token_counter = MagicMock()
    token_counter.count_tokens.return_value = 10
    policy = PromptBudgetPolicy(
        token_counter=token_counter,
        logger=MagicMock(),
        context_windows=context_windows,
        brain_max_tokens=5000,
        request_max_tokens=10000,
    )
    config = SimpleNamespace(
        GOOGLE_STUDIO_MODEL="gemini-test",
        OPENROUTER_BASE_MODEL="openrouter/test",
        LITELLM_MODEL="deepseek-v4.1-flash",
        BLOCKRUN_MODEL="blockrun-test",
        VISION_MODEL="vision-test",
        get_model_config=lambda model: {"model": model, "max_tokens": 2048},
        **config_overrides,
    )
    vision_client = LitellmAIClient(
        config={
            "provider": "litellm",
            "base_url": "https://vision.example/v1",
            "api_key": "vision-key",
            "model": "vision-test",
        },
        logger=MagicMock(),
    )
    orchestrator = ProviderOrchestrator(
        logger=MagicMock(),
        config=config,
        clients=ProviderClients(vision_litellm=vision_client),
        prompt_budget=policy,
    )
    return orchestrator, policy


def test_configured_context_window_is_used_and_exact_entry_wins():
    """The configured key is authoritative; exact [model_context_windows] wins."""
    orchestrator, policy = _budget_orchestrator(
        {"vision_litellm/vision-test": 8192},
        {"LITELLM_CONTEXT_WINDOW": 1_000_000, "VISION_CONTEXT_WINDOW": 200_000},
    )

    assert policy.context_windows["vision_litellm/vision-test"] == 8192
    assert policy.context_windows["litellm/deepseek-v4.1-flash"] == 1_000_000


@pytest.mark.asyncio
async def test_unknown_vision_capacity_is_refused():
    """Without a configured window the budget must refuse, not guess."""
    orchestrator, policy = _budget_orchestrator({}, {})

    assert "vision_litellm/vision-test" not in policy.context_windows
    with patch(
        "src.platforms.ai_providers.litellm_client.acompletion",
        new=AsyncMock(return_value=_response("should not run")),
    ):
        result = await orchestrator.get_chart_response(
            "all", [{"role": "user", "content": "chart"}], io.BytesIO(b"png")
        )

    assert result.success is False
    assert "Context window is not configured" in (result.error or "")


@pytest.mark.asyncio
async def test_invoke_forwards_response_format_without_mutating_provider_config():
    """The signal path's response_format reaches the request but never the shared config."""
    client = LitellmAIClient(config=LITELLM_CONFIG, logger=MagicMock())
    orchestrator = _orchestrator(ProviderClients(litellm=client))
    captured = []
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="{}"), finish_reason="stop")],
        usage=None,
        id="response-id",
        model="deepseek-v4.1-flash",
    )

    async def fake_acompletion(**kwargs):
        captured.append(dict(kwargs))
        return response

    with patch("src.platforms.ai_providers.litellm_client.acompletion", new=fake_acompletion):
        await orchestrator.invoke(
            "litellm",
            [{"role": "user", "content": "hi"}],
            response_format={"type": "json_object"},
        )
        await orchestrator.invoke("litellm", [{"role": "user", "content": "hi"}])

    assert captured[0]["response_format"] == {"type": "json_object"}
    assert "response_format" not in captured[1]
    assert "response_format" not in orchestrator.get_metadata("litellm").config
