"""Tests for forwarding the dedicated vision LiteLLM client through ProviderFactory."""
from types import SimpleNamespace
from unittest.mock import MagicMock

from src.factories.provider_factory import ProviderFactory
from src.managers.provider_types import ProviderClients


def _config(**overrides):
    values = {
        "GOOGLE_STUDIO_API_KEY": None,
        "GOOGLE_STUDIO_PAID_API_KEY": None,
        "GOOGLE_STUDIO_MODEL": "gemini-1.5-flash",
        "OPENROUTER_API_KEY": None,
        "BLOCKRUN_WALLET_KEY": None,
        "LITELLM_BASE_URL": "https://opencode.ai/inference/go/openai/v1",
        "LITELLM_MODEL": "deepseek-v4.1-flash",
        "LITELLM_API_KEY": "text-key",
        "LITELLM_USER_AGENT": "tradingbot/0.1",
        "LITELLM_SESSION_ID": "tradingbot",
        "VISION_PROVIDER": "litellm",
        "VISION_BASE_URL": "https://vision.example/v1",
        "VISION_API_KEY": "vision-key",
        "VISION_MODEL": "openai/gpt-4o-mini",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_create_all_clients_forwards_vision_litellm():
    """The vision client must reach ProviderClients instead of being dropped."""
    factory = ProviderFactory(logger=MagicMock(), config=_config())

    clients = factory.create_all_clients()

    assert clients["vision_litellm"] is not None
    assert ProviderClients.from_factory_dict(clients).vision_litellm is clients["vision_litellm"]


def test_vision_client_uses_vision_overrides():
    """VISION_BASE_URL/VISION_MODEL/VISION_API_KEY configure the vision client."""
    factory = ProviderFactory(logger=MagicMock(), config=_config())

    vision_client = factory.create_all_clients()["vision_litellm"]

    assert vision_client.base_url == "https://vision.example/v1"
    assert vision_client.default_model == "openai/gpt-4o-mini"
    assert vision_client.api_key == "vision-key"


def test_vision_litellm_is_none_without_vision_provider():
    """No VISION_PROVIDER means no vision client, without breaking the dict."""
    factory = ProviderFactory(logger=MagicMock(), config=_config(VISION_PROVIDER=None))

    clients = factory.create_all_clients()

    assert "vision_litellm" in clients
    assert clients["vision_litellm"] is None
