"""
Factory to create a unified LiteLLM client.
Replaces specific factories with a single multi-provider client.
"""
from typing import Dict, Any, Optional

from src.logger.logger import Logger
from src.platforms.ai_providers.litellm_client import LitellmAIClient


class LiteLLMFactory:
    """Factory to create a unified LiteLLM client."""

    def __init__(self, logger: Logger, config: Any):
        """
        Initialize the factory.

        Args:
            logger: Logger instance
            config: Configuration (ConfigProtocol)
        """
        self.logger = logger
        self.config = config

    def create_litellm_client(self, provider: str) -> Optional[LitellmAIClient]:
        """
        Create a LiteLLM client for the specified provider.

        Args:
            provider: Provider name (ollama, googleai, openrouter, blockrun)

        Returns:
            LitellmAIClient or None if not configured
        """
        config = self._get_provider_config(provider)
        if not config:
            self.logger.debug("No configuration for provider: %s", provider)
            return None

        client = LitellmAIClient(config=config, logger=self.logger)
        self.logger.debug("LiteLLM client created for provider: %s", provider)
        return client

    def create_vision_client(self) -> Optional[LitellmAIClient]:
        """
        Create a dedicated LiteLLM client for vision analysis.

        Uses VISION_PROVIDER, VISION_BASE_URL, VISION_API_KEY, VISION_MODEL from config.

        Returns:
            LitellmAIClient or None if vision not configured
        """
        vision_provider = getattr(self.config, 'VISION_PROVIDER', None)
        if not vision_provider:
            self.logger.debug("No vision provider configured")
            return None

        config = self._get_provider_config(vision_provider)
        if not config:
            self.logger.debug("No configuration for vision provider: %s", vision_provider)
            return None

        base_url = getattr(self.config, 'VISION_BASE_URL', None)
        api_key = getattr(self.config, 'VISION_API_KEY', None)
        model = getattr(self.config, 'VISION_MODEL', None)

        if base_url:
            config["base_url"] = base_url
        if api_key:
            config["api_key"] = api_key
        if model:
            config["model"] = model

        client = LitellmAIClient(config=config, logger=self.logger)
        self.logger.debug("Vision LiteLLM client created for provider: %s", vision_provider)
        return client

    def _get_provider_config(self, provider: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve config for a specific provider.

        Args:
            provider: Provider name

        Returns:
            Config dict or None
        """
        if provider == "googleai":
            return self._get_google_config()
        elif provider == "openrouter":
            return self._get_openrouter_config()
        elif provider == "blockrun":
            return self._get_blockrun_config()
        elif provider == "litellm":
            return self._get_litellm_config()

        self.logger.warning("Unknown provider: %s", provider)
        return None

    def _get_litellm_config(self) -> Optional[Dict[str, Any]]:
        """
        LiteLLM configuration for generic OpenAI-compatible APIs.

        Uses LITELLM_BASE_URL, LITELLM_MODEL, and LITELLM_API_KEY from config.
        Supports MiniMax, OpenAI-compatible endpoints, etc.
        """
        base_url = getattr(self.config, "LITELLM_BASE_URL", None)
        if not base_url:
            self.logger.debug("No base URL configured for litellm provider")
            return None

        model = getattr(self.config, "LITELLM_MODEL", None)
        if not model:
            self.logger.debug("No model configured for litellm provider")
            return None

        api_key = getattr(self.config, "LITELLM_API_KEY", None)
        if not api_key:
            api_key = getattr(self.config, "MINIMAX_API_KEY", None)
        if not api_key:
            self.logger.debug("No API key configured for litellm provider")
            return None

        self.logger.debug("LiteLLM config - base_url: %s, model: %s, api_key_prefix: %s",
                         base_url, model, api_key[:8] if api_key else None)

        return {
            "provider": "litellm",
            "base_url": base_url,
            "model": model,
            "api_key": api_key,
        }

    def _get_google_config(self) -> Optional[Dict[str, Any]]:
        """
        Google AI Studio configuration via LiteLLM.

        Uses vertexai or gemini via LiteLLM.
        """
        api_key = getattr(self.config, "GOOGLE_STUDIO_API_KEY", None)
        if not api_key:
            self.logger.debug("No Google API key configured")
            return None

        model = getattr(self.config, "GOOGLE_STUDIO_MODEL", "gemini-1.5-flash")

        return {
            "provider": "gemini",  # LiteLLM uses "gemini" for Google
            "model": model,
            "api_key": api_key,
        }

    def _get_openrouter_config(self) -> Optional[Dict[str, Any]]:
        """
        OpenRouter configuration via LiteLLM.
        """
        api_key = getattr(self.config, "OPENROUTER_API_KEY", None)
        if not api_key:
            self.logger.debug("No OpenRouter API key configured")
            return None

        base_url = getattr(self.config, "OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
        model = getattr(self.config, "OPENROUTER_BASE_MODEL", "google/gemma-4-31b-it:free")

        return {
            "provider": "openrouter",
            "base_url": base_url,
            "model": model,
            "api_key": api_key,
        }

    def _get_blockrun_config(self) -> Optional[Dict[str, Any]]:
        """
        BlockRun configuration via LiteLLM.

        BlockRun uses an OpenAI-compatible API with wallet key.
        """
        wallet_key = getattr(self.config, "BLOCKRUN_WALLET_KEY", None)
        if not wallet_key:
            self.logger.debug("No BlockRun wallet key configured")
            return None

        base_url = getattr(
            self.config,
            "BLOCKRUN_BASE_URL",
            "https://blockrun.ai/api",
        )
        model = getattr(self.config, "BLOCKRUN_MODEL", "openai/gpt-4o")

        return {
            "provider": "openai",  # BlockRun is OpenAI-compatible
            "base_url": base_url,
            "model": model,
            "api_key": wallet_key,
        }

    def get_all_clients(self) -> Dict[str, Optional[LitellmAIClient]]:
        """
        Create all configured clients.

        Returns:
            Dict with clients for each provider
        """
        providers = ["googleai", "openrouter", "blockrun", "litellm"]
        clients = {}

        for provider in providers:
            client = self.create_litellm_client(provider)
            clients[provider] = client

        vision_client = self.create_vision_client()
        if vision_client:
            clients["vision_litellm"] = vision_client

        return clients
