"""Regression tests for ProviderOrchestrator chart routing."""
import io
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

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