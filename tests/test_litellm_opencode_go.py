"""Tests for the OpenCode Go LiteLLM integration."""
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from src.factories.litellm_factory import LiteLLMFactory
from src.platforms.ai_providers.litellm_client import LitellmAIClient


OPENCODE_HEADERS = {
    "User-Agent": "tradingbot/0.1",
    "x-opencode-session": "tradingbot",
}


def _client() -> LitellmAIClient:
    return LitellmAIClient(
        config={
            "provider": "litellm",
            "base_url": "https://opencode.ai/inference/go/openai/v1",
            "api_key": "test-key",
            "model": "openai/deepseek-v4.1-flash",
            "extra_headers": OPENCODE_HEADERS,
        },
        logger=MagicMock(),
    )


def test_client_stores_extra_headers():
    """OpenCode Go headers must reach LiteLLM requests."""
    client = _client()

    assert client.litellm_kwargs["extra_headers"] == OPENCODE_HEADERS


@pytest.mark.asyncio
async def test_chart_analysis_sends_image_and_headers():
    """Chart analysis must send the image payload, not the original messages."""
    client = _client()
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
        model="openai/deepseek-v4.1-flash",
    )

    async def fake_acompletion(**kwargs):
        captured.update(kwargs)
        return response

    with patch(
        "src.platforms.ai_providers.litellm_client.acompletion",
        new=fake_acompletion,
    ):
        result = await client.chat_completion_with_chart_analysis(
            "openai/deepseek-v4.1-flash",
            [{"role": "user", "content": "Analyze this chart"}],
            b"fake-png-bytes",
            {"temperature": 0.0, "max_tokens": 128},
        )

    assert result is not None
    assert result.choices[0].message.content == "chart analyzed"
    assert result.choices[0].finish_reason == "stop"
    assert captured["extra_headers"] == OPENCODE_HEADERS
    assert captured["messages"][0]["content"][1]["type"] == "image_url"
    assert captured["messages"][0]["content"][1]["image_url"]["url"].startswith(
        "data:image/png;base64,"
    )


def test_response_preserves_finish_reason_and_none_content():
    """Truncated responses must not be reported as a normal stop."""
    client = _client()
    response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=None),
                finish_reason="length",
            )
        ],
        usage=None,
        id="response-id",
        model="openai/deepseek-v4.1-flash",
    )

    result = client._create_response_from_litellm(response)

    assert result.choices[0].finish_reason == "length"
    assert result.choices[0].message.content == ""


def test_factory_adds_opencode_headers_without_minimax_fallback():
    """Factory uses only LITELLM_API_KEY and forwards OpenCode headers."""
    config = SimpleNamespace(
        LITELLM_BASE_URL="https://opencode.ai/inference/go/openai/v1",
        LITELLM_MODEL="openai/deepseek-v4.1-flash",
        LITELLM_API_KEY="test-key",
        LITELLM_USER_AGENT="tradingbot/0.1",
        LITELLM_SESSION_ID="tradingbot",
        MINIMAX_API_KEY="unrelated-minimax-key",
    )
    factory = LiteLLMFactory(logger=MagicMock(), config=config)

    provider_config = factory._get_litellm_config()

    assert provider_config is not None
    assert provider_config["extra_headers"] == OPENCODE_HEADERS


def test_factory_does_not_fall_back_to_minimax_key():
    """A missing LITELLM_API_KEY must not silently use the MiniMax key."""
    config = SimpleNamespace(
        LITELLM_BASE_URL="https://opencode.ai/inference/go/openai/v1",
        LITELLM_MODEL="openai/deepseek-v4.1-flash",
        LITELLM_API_KEY=None,
        LITELLM_USER_AGENT="tradingbot/0.1",
        LITELLM_SESSION_ID="tradingbot",
        MINIMAX_API_KEY="unrelated-minimax-key",
    )
    factory = LiteLLMFactory(logger=MagicMock(), config=config)

    assert factory._get_litellm_config() is None
