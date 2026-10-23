"""Factory for creating AI provider clients based on configuration."""
from typing import Optional, Tuple, TYPE_CHECKING

from src.logger.logger import Logger
from src.factories.litellm_factory import LiteLLMFactory

from src.platforms.ai_providers.google import GoogleAIClient

if TYPE_CHECKING:
    from src.config.protocol import ConfigProtocol


class ProviderFactory:
    """
    Factory for creating AI provider client instances based on configuration.

    Centralizes provider instantiation logic and handles API key validation.
    Supports multiple providers: Google AI Studio (free/paid), OpenRouter, LiteLLM (generic).

    Usage:
        factory = ProviderFactory(logger, config)
        google_client, google_paid_client = factory.create_google_clients()
        openrouter_client = litellm_factory.create_litellm_client("openrouter")
        litellm_client = litellm_factory.create_litellm_client("litellm")
    """

    def __init__(self, logger: Logger, config: "ConfigProtocol"):
        """
        Initialize the provider factory.

        Args:
            logger: Logger instance for logging
            config: ConfigProtocol instance for configuration access
        """
        self.logger = logger
        self.config = config
        self.litellm_factory = LiteLLMFactory(logger, config)

    def create_google_clients(self) -> Tuple[Optional[GoogleAIClient], Optional[GoogleAIClient]]:
        """
        Create Google AI clients (free tier and optional paid tier).
        
        Returns:
            Tuple of (google_client, google_paid_client). Both can be None if API keys not configured.
        """
        google_client: Optional[GoogleAIClient] = None
        google_paid_client: Optional[GoogleAIClient] = None

        # Initialize Google AI client if API key is available
        if self.config.GOOGLE_STUDIO_API_KEY:
            google_client = GoogleAIClient(
                api_key=self.config.GOOGLE_STUDIO_API_KEY,
                model=self.config.GOOGLE_STUDIO_MODEL,
                logger=self.logger
            )
            self.logger.debug("Google AI client initialized")

            # Initialize paid client if paid API key is available
            if self.config.GOOGLE_STUDIO_PAID_API_KEY:
                google_paid_client = GoogleAIClient(
                    api_key=self.config.GOOGLE_STUDIO_PAID_API_KEY,
                    model=self.config.GOOGLE_STUDIO_MODEL,
                    logger=self.logger
                )
                self.logger.debug("Google AI paid client initialized as fallback for overloaded free tier")

        return google_client, google_paid_client

    # === Specific methods (deprecated - kept for compatibility) ===

    def create_openrouter_client(self) -> None:
        """Deprecated - use LiteLLM."""
        return None

    def create_lmstudio_client(self) -> None:
        """Deprecated - use LiteLLM."""
        return None

    def create_blockrun_client(self) -> None:
        """Deprecated - use LiteLLM."""
        return None

    def create_all_clients(self) -> dict:
        """
        Create all available AI provider clients based on configuration.

        Returns:
            Dictionary with keys: 'google', 'google_paid', 'openrouter', 'litellm',
            'blockrun', 'vision_litellm'.
        """
        google_client, google_paid_client = self.create_google_clients()

        # Create all clients via LiteLLM factory
        litellm_clients = self.litellm_factory.get_all_clients()

        return {
            'google': google_client,
            'google_paid': google_paid_client,
            'openrouter': litellm_clients.get('openrouter'),
            'litellm': litellm_clients.get('litellm'),
            'blockrun': litellm_clients.get('blockrun'),
            'vision_litellm': litellm_clients.get('vision_litellm')
        }
