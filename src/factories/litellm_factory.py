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

    def _get_provider_config(self, provider: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve config for a specific provider.

        Args:
            provider: Provider name

        Returns:
            Config dict or None
        """
        if provider == "ollama":
            return self._get_ollama_config()
        elif provider == "googleai":
            return self._get_google_config()
        elif provider == "openrouter":
            return self._get_openrouter_config()
        elif provider == "blockrun":
            return self._get_blockrun_config()

        self.logger.warning("Unknown provider: %s", provider)
        return None

    def _get_ollama_config(self) -> Optional[Dict[str, Any]]:
        """
        Ollama configuration (OpenAI-compatible local API).

        Uses http://localhost:11434/v1
        """
        base_url = getattr(self.config, "LM_STUDIO_BASE_URL", None)
        if not base_url:
            # Default value if not configured
            base_url = "http://localhost:11434/v1"

        model = getattr(self.config, "LM_STUDIO_MODEL", "gemma4:31b-cloud")

        # For Ollama/LM Studio local, we use OpenAI-compatible API
        # No API key needed
        return {
            "provider": "ollama",
            "base_url": base_url,
            "model": model,
            "api_key": "ollama",  # Dummy key, local doesn't need it
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
        providers = ["ollama", "googleai", "openrouter", "blockrun"]
        clients = {}

        for provider in providers:
            client = self.create_litellm_client(provider)
            clients[provider] = client

        return clients
