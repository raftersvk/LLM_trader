"""
Factory pour créer un client LiteLLM unifié.
Remplace les factories spécifiques par un client unique multi-provider.
"""
from typing import Dict, Any, Optional

from src.logger.logger import Logger
from src.platforms.ai_providers.litellm_client import LitellmAIClient


class LiteLLMFactory:
    """Factory pour créer un client LiteLLM unifié."""

    def __init__(self, logger: Logger, config: Any):
        """
        Initialise la factory.

        Args:
            logger: Logger instance
            config: Configuration (ConfigProtocol)
        """
        self.logger = logger
        self.config = config

    def create_litellm_client(self, provider: str) -> Optional[LitellmAIClient]:
        """
        Crée un client LiteLLM pour le provider spécifié.

        Args:
            provider: Nom du provider (ollama, googleai, openrouter, blockrun)

        Returns:
            LitellmAIClient ou None si non configuré
        """
        config = self._get_provider_config(provider)
        if not config:
            self.logger.debug("Aucune configuration pour le provider: %s", provider)
            return None

        client = LitellmAIClient(config=config, logger=self.logger)
        self.logger.debug("Client LiteLLM créé pour le provider: %s", provider)
        return client

    def _get_provider_config(self, provider: str) -> Optional[Dict[str, Any]]:
        """
        Récupère la configuration pour un provider spécifique.

        Args:
            provider: Nom du provider

        Returns:
            Dict de configuration ou None
        """
        if provider == "ollama":
            return self._get_ollama_config()
        elif provider == "googleai":
            return self._get_google_config()
        elif provider == "openrouter":
            return self._get_openrouter_config()
        elif provider == "blockrun":
            return self._get_blockrun_config()

        self.logger.warning("Provider inconnu: %s", provider)
        return None

    def _get_ollama_config(self) -> Optional[Dict[str, Any]]:
        """
        Configuration pour Ollama (API OpenAI-compatible locale).

        Utilise l'URL http://localhost:11434/v1
        """
        base_url = getattr(self.config, "LM_STUDIO_BASE_URL", None)
        if not base_url:
            # Valeur par défaut si non configurée
            base_url = "http://localhost:11434/v1"

        model = getattr(self.config, "LM_STUDIO_MODEL", "gemma4:31b-cloud")

        # Pour Ollama/LM Studio local, on utilise l'API OpenAI-compatible
        # Pas besoin de API key
        return {
            "provider": "ollama",
            "base_url": base_url,
            "model": model,
            "api_key": "ollama",  # Dummy key, pas utilisé pour local
        }

    def _get_google_config(self) -> Optional[Dict[str, Any]]:
        """
        Configuration pour Google AI Studio via LiteLLM.

        Utilise vertexai ou gemini via LiteLLM.
        """
        api_key = getattr(self.config, "GOOGLE_STUDIO_API_KEY", None)
        if not api_key:
            self.logger.debug("Aucune API key Google configurée")
            return None

        model = getattr(self.config, "GOOGLE_STUDIO_MODEL", "gemini-1.5-flash")

        return {
            "provider": "gemini",  # LiteLLM utilise "gemini" pour Google
            "model": model,
            "api_key": api_key,
        }

    def _get_openrouter_config(self) -> Optional[Dict[str, Any]]:
        """
        Configuration pour OpenRouter via LiteLLM.
        """
        api_key = getattr(self.config, "OPENROUTER_API_KEY", None)
        if not api_key:
            self.logger.debug("Aucune API key OpenRouter configurée")
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
        Configuration pour BlockRun via LiteLLM.

        BlockRun utilise une API OpenAI-compatible avec wallet key.
        """
        wallet_key = getattr(self.config, "BLOCKRUN_WALLET_KEY", None)
        if not wallet_key:
            self.logger.debug("Aucune wallet key BlockRun configurée")
            return None

        base_url = getattr(
            self.config,
            "BLOCKRUN_BASE_URL",
            "https://blockrun.ai/api",
        )
        model = getattr(self.config, "BLOCKRUN_MODEL", "openai/gpt-4o")

        return {
            "provider": "openai",  # BlockRun est OpenAI-compatible
            "base_url": base_url,
            "model": model,
            "api_key": wallet_key,
        }

    def get_all_clients(self) -> Dict[str, Optional[LitellmAIClient]]:
        """
        Crée tous les clients configurés.

        Returns:
            Dict avec les clients pour chaque provider
        """
        providers = ["ollama", "googleai", "openrouter", "blockrun"]
        clients = {}

        for provider in providers:
            client = self.create_litellm_client(provider)
            clients[provider] = client

        return clients
