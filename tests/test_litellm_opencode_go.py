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
            "model": "deepseek-v4.1-flash",
            "extra_headers": OPENCODE_HEADERS,
        },
        logger=MagicMock(),
    )


def test_client_stores_extra_headers():
    """OpenCode Go headers must reach LiteLLM requests."""
    client = _client()

    assert client.litellm_kwargs["extra_headers"] == OPENCODE_HEADERS
    assert client.litellm_kwargs["extra_headers"]["x-opencode-session"] == "tradingbot"


def test_litellm_provider_forces_openai_routing():
    """Bare model ids need explicit OpenAI routing (Go rejects prefixed ids)."""
    client = _client()

    assert client.litellm_kwargs["custom_llm_provider"] == "openai"
    # The configured model id is forwarded unchanged.
    assert client._get_full_model_name(client.default_model) == "deepseek-v4.1-flash"


def test_non_litellm_provider_keeps_its_own_routing():
    """Other providers must not inherit the OpenAI routing override."""
    client = LitellmAIClient(
        config={
            "provider": "openrouter",
            "base_url": "https://openrouter.ai/api/v1",
            "api_key": "test-key",
            "model": "google/gemini-3-flash-preview",
        },
        logger=MagicMock(),
    )

    assert "custom_llm_provider" not in client.litellm_kwargs


@pytest.mark.asyncio
async def test_openrouter_request_keeps_provider_prefix():
    """OpenRouter models must reach LiteLLM prefixed, or routing breaks."""
    client = LitellmAIClient(
        config={
            "provider": "openrouter",
            "base_url": "https://openrouter.ai/api/v1",
            "api_key": "test-key",
            "model": "google/gemini-3-flash-preview",
        },
        logger=MagicMock(),
    )
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
        model="google/gemini-3-flash-preview",
    )

    async def fake_acompletion(**kwargs):
        captured.update(kwargs)
        return response

    with patch(
        "src.platforms.ai_providers.litellm_client.acompletion",
        new=fake_acompletion,
    ):
        await client.chat_completion(
            "deepseek/deepseek-r1:free",
            [{"role": "user", "content": "hello"}],
            {},
        )

    assert captured["model"] == "openrouter/deepseek/deepseek-r1:free"
    assert captured["base_url"] == "https://openrouter.ai/api/v1"
    assert "custom_llm_provider" not in captured


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
        model="deepseek-v4.1-flash",
    )

    async def fake_acompletion(**kwargs):
        captured.update(kwargs)
        return response

    with patch(
        "src.platforms.ai_providers.litellm_client.acompletion",
        new=fake_acompletion,
    ):
        result = await client.chat_completion_with_chart_analysis(
            "deepseek-v4.1-flash",
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
        model="deepseek-v4.1-flash",
    )

    result = client._create_response_from_litellm(response)

    assert result.choices[0].finish_reason == "length"
    assert result.choices[0].message.content == ""


def test_factory_adds_opencode_headers_without_minimax_fallback():
    """Factory uses only LITELLM_API_KEY and forwards OpenCode headers."""
    config = SimpleNamespace(
        LITELLM_BASE_URL="https://opencode.ai/inference/go/openai/v1",
        LITELLM_MODEL="deepseek-v4.1-flash",
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
        LITELLM_MODEL="deepseek-v4.1-flash",
        LITELLM_API_KEY=None,
        LITELLM_USER_AGENT="tradingbot/0.1",
        LITELLM_SESSION_ID="tradingbot",
        MINIMAX_API_KEY="unrelated-minimax-key",
    )
    factory = LiteLLMFactory(logger=MagicMock(), config=config)

    assert factory._get_litellm_config() is None
