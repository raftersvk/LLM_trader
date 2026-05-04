"""
LiteLLM Unified Client - Replaces specific SDKs with a standard OpenAI interface.

Supports: Ollama, OpenAI, Anthropic, Google, OpenRouter, etc.
via a single unified client.
"""
import io
import base64
from typing import Optional, Dict, Any, List, Union

import litellm
from litellm import acompletion

from src.logger.logger import Logger
from src.platforms.ai_providers.base import BaseAIClient
from src.platforms.ai_providers.response_models import ChatResponseModel, UsageModel, ChoiceModel, MessageModel
from src.utils.decorators import retry_api_call

litellm.drop_params = True  # Ignore unsupported params by some providers


class LitellmAIClient(BaseAIClient):
    """Unified client using LiteLLM for all LLM providers."""

    def __init__(self, config: Dict[str, Any], logger: Logger) -> None:
        """
        Initialize the LiteLLM client.

        Args:
            config: Provider config with keys:
                - provider: provider name (ollama, openai, anthropic, google, openrouter...)
                - base_url: Base URL (optional, e.g. http://localhost:11434/v1)
                - api_key: API key (optional for local)
                - model: Default model
            logger: Logger instance
        """
        super().__init__(logger)
        self.config = config
        self.provider = config.get("provider", "openai")
        self.base_url = config.get("base_url")
        self.api_key = config.get("api_key")
        self.default_model = config.get("model", "default")

        # LiteLLM configuration for this client
        self.litellm_kwargs = {}
        if self.base_url:
            self.litellm_kwargs["base_url"] = self.base_url
        if self.api_key:
            self.litellm_kwargs["api_key"] = self.api_key
        elif self.base_url:
            # For custom URLs (Ollama/LM Studio), LiteLLM requires an api_key
            # We use a dummy value
            self.litellm_kwargs["api_key"] = "ollama"

        self.logger.debug(
            "LitellmAIClient initialized - provider: %s, model: %s, base_url: %s, api_key_set: %s",
            self.provider,
            self.default_model,
            self.base_url,
            bool(self.api_key),
        )
        if self.api_key:
            self.logger.debug("API key prefix: %s", self.api_key[:8])

    async def _initialize_client(self) -> None:
        """LiteLLM does not need specific async initialization."""
        self.logger.debug("LitellmAIClient ready - no initialization required")

    async def close(self) -> None:
        """LiteLLM does not need specific cleanup."""
        self.logger.debug("LitellmAIClient - closing (no action required)")

    @retry_api_call(max_retries=3, initial_delay=1, backoff_factor=2, max_delay=30)
    async def chat_completion(
        self, model: str, messages: List[Dict[str, Any]], model_config: Dict[str, Any]
    ) -> Optional[ChatResponseModel]:
        """
        Send a chat completion request via LiteLLM.

        Args:
            model: Model name (e.g. gemma4:31b-cloud, gemini-flash, claude-sonnet)
            messages: List of OpenAI-style messages
            model_config: Model parameters (temperature, max_tokens, etc.)

        Returns:
            ChatResponseModel or None on error
        """
        full_model = self._get_full_model_name(model)
        params = self._build_litellm_params(model_config)

        self.logger.debug(
            "LiteLLM request - model: %s, provider: %s, messages: %d, api_key_set: %s, base_url: %s",
            full_model,
            self.provider,
            len(messages),
            bool(self.api_key),
            self.base_url,
        )

        try:
            response = await acompletion(
                model=full_model,
                messages=messages,
                **params,
                **self.litellm_kwargs,
            )
            self.logger.debug("LiteLLM response received successfully")
            return self._create_response_from_litellm(response)
        except Exception as e:
            self.logger.error("Erreur LiteLLM request: %s", str(e), exc_info=True)
            error_response = self._handle_litellm_exception(e, full_model)
            if error_response:
                return error_response
            raise

    @retry_api_call(max_retries=3, initial_delay=1, backoff_factor=2, max_delay=30)
    async def chat_completion_with_chart_analysis(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        chart_image: Union[io.BytesIO, bytes, str],
        model_config: Dict[str, Any],
    ) -> Optional[ChatResponseModel]:
        """
        Send a request with image for chart analysis.

        LiteLLM handles multimodal requests (text + image) for providers
        that support it (Gemini, GPT-4o, Claude 3, etc.).

        Args:
            model: Model name
            messages: List of OpenAI-style messages
            chart_image: Chart image (BytesIO, bytes, or path)

        Returns:
            ChatResponseModel or None on error
        """
        full_model = self._get_full_model_name(model)
        params = self._build_litellm_params(model_config)

            # Prepare image data
        img_data = self.process_chart_image(chart_image)
        img_b64 = base64.b64encode(img_data).decode("utf-8")

        # Construire les messages avec l'image
        messages_with_image = []
        for msg in messages:
            if msg.get("role") == "user":
                content = msg.get("content", "")
                if isinstance(content, str):
                    content = [
                        {"type": "text", "text": content},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
                    ]
                messages_with_image.append({"role": "user", "content": content})
            else:
                messages_with_image.append(msg)

        if not any(m.get("role") == "user" for m in messages_with_image):
            messages_with_image.append({
                "role": "user",
                "content": [
                    {"type": "text", "text": "Analyze this chart."},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
                ],
            })

        self.logger.debug("LiteLLM chart analysis - model: %s, provider: %s", full_model, self.provider)

        try:
            self.logger.debug(
                "LiteLLM request - model: %s, provider: %s, messages: %d, params: %s",
                full_model,
                self.provider,
                len(messages),
                params,
            )
            response = await acompletion(
                model=full_model,
                messages=messages,
                **params,
                **self.litellm_kwargs,
            )
            self.logger.debug("LiteLLM chart analysis response received")
            return self._create_response_from_litellm(response)
        except Exception as e:
            self.logger.error("Erreur LiteLLM chart analysis: %s", str(e), exc_info=True)
            error_response = self._handle_litellm_exception(e, full_model)
            if error_response:
                return error_response
            raise

    @retry_api_call(max_retries=3, initial_delay=1, backoff_factor=2, max_delay=30)
    async def stream_chat_completion(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        model_config: Dict[str, Any],
        callback=None,
    ) -> Optional[ChatResponseModel]:
        """
        Send a streaming chat completion request via LiteLLM.

        Args:
            model: Model name
            messages: List of OpenAI-style messages
            model_config: Model parameters
            callback: Function called for each streaming fragment

        Returns:
            ChatResponseModel or None on error
        """
        full_model = self._get_full_model_name(model)
        params = self._build_litellm_params(model_config)
        params["stream"] = True

        self.logger.debug("LiteLLM streaming request - model: %s, provider: %s", full_model, self.provider)

        try:
            response = await acompletion(
                model=full_model,
                messages=messages,
                **params,
                **self.litellm_kwargs,
            )

            complete_content = ""
            async for chunk in response:
                content = self._extract_chunk_content(chunk)
                if content:
                    complete_content += content
                    if callback:
                        await callback(content)

            self.logger.debug("LiteLLM streaming response complete")
            return self.create_response(
                content=complete_content,
                usage=UsageModel(prompt_tokens=0, completion_tokens=0, total_tokens=0),
            )
        except Exception as e:
            self.logger.error("Erreur LiteLLM streaming request: %s", str(e), exc_info=True)
            if not locals().get('complete_content', ''):
                raise
            return self.create_response(
                content=locals().get('complete_content', ''),
                usage=UsageModel(prompt_tokens=0, completion_tokens=0, total_tokens=0),
            )

    def _get_full_model_name(self, model: str) -> str:
        """Return the model name as-is from config.

        LiteLLM expects the model identifier as configured by the user.
        """
        return model if model else self.default_model

    def _build_litellm_params(self, model_config: Dict[str, Any]) -> Dict[str, Any]:
        """Build LiteLLM params from model config."""
        params: Dict[str, Any] = {}
        if "temperature" in model_config:
            params["temperature"] = model_config["temperature"]
        if "max_tokens" in model_config:
            params["max_tokens"] = model_config["max_tokens"]
        if "top_p" in model_config:
            params["top_p"] = model_config["top_p"]
        if "frequency_penalty" in model_config:
            params["frequency_penalty"] = model_config["frequency_penalty"]
        if "presence_penalty" in model_config:
            params["presence_penalty"] = model_config["presence_penalty"]

        params["timeout"] = model_config.get("timeout", 30)
        # base_url is in litellm_kwargs
        # base_url already in litellm_kwargs, not added here
        return params

    def _create_response_from_litellm(self, response) -> ChatResponseModel:
        """Convert LiteLLM response to standard ChatResponseModel."""
        if hasattr(response, "choices") and response.choices:
            choice = response.choices[0]
            if hasattr(choice, "message"):
                content = choice.message.content if choice.message else ""
            elif hasattr(choice, "text"):
                content = choice.text
            else:
                content = str(choice)
        elif hasattr(response, "text"):
            content = response.text
        elif hasattr(response, "content"):
            content = response.content if isinstance(response.content, str) else str(response.content)
        else:
            content = str(response)

        usage = UsageModel(prompt_tokens=0, completion_tokens=0, total_tokens=0)
        if hasattr(response, "usage") and response.usage:
            usage = UsageModel(
                prompt_tokens=getattr(response.usage, "prompt_tokens", 0) or 0,
                completion_tokens=getattr(response.usage, "completion_tokens", 0) or 0,
                total_tokens=getattr(response.usage, "total_tokens", 0) or 0,
            )

        choice_model = ChoiceModel(
            message=MessageModel(role="assistant", content=content),
            finish_reason="stop",
            index=0,
        )
        return ChatResponseModel(
            id=getattr(response, "id", "litellm-response"),
            created=getattr(response, "created", 0),
            model=getattr(response, "model", self.default_model),
            choices=[choice_model],
            usage=usage,
        )

    def _extract_chunk_content(self, chunk) -> Optional[str]:
        """Extrait le contenu d'un chunk de streaming LiteLLM."""
        try:
            if hasattr(chunk, "choices") and chunk.choices:
                choice = chunk.choices[0]
                if hasattr(choice, "delta"):
                    delta = choice.delta
                    if hasattr(delta, "content") and delta.content:
                        return delta.content
                    elif hasattr(delta, "text") and delta.text:
                        return delta.text
            elif hasattr(chunk, "text") and chunk.text:
                return chunk.text
            elif hasattr(chunk, "content") and chunk.content:
                content = chunk.content
                return content if isinstance(content, str) else str(content)
        except Exception as e:
            self.logger.debug("Erreur extraction chunk streaming: %s", e)
        return None

    def _handle_litellm_exception(self, exception: Exception, model: str) -> Optional[ChatResponseModel]:
        """Handle LiteLLM exceptions and convert to standardized responses."""
        error_message = str(exception)
        error_message_lower = error_message.lower()

        if "rate_limit" in error_message_lower or "429" in error_message:
            self.logger.error("Rate limit LiteLLM [%s]: %s", model, error_message)
            return ChatResponseModel.from_error(f"rate_limit: {error_message}")
        if "timeout" in error_message_lower:
            self.logger.error("Timeout LiteLLM [%s]: %s", model, error_message)
            return ChatResponseModel.from_error(f"timeout: {error_message}")
        if any(err in error_message_lower for err in ["connection", "refused", "unavailable", "503", "502", "500"]):
            self.logger.error("Service indisponible LiteLLM [%s]: %s", model, error_message)
            return ChatResponseModel.from_error(f"service_unavailable: {error_message}")
        if any(err in error_message_lower for err in ["not found", "404", "unknown model", "does not exist"]):
            self.logger.error("LiteLLM model not found [%s]: %s", model, error_message)
            return ChatResponseModel.from_error(f"model_not_found: {error_message}")
        if any(err in error_message_lower for err in ["auth", "api_key", "401", "403"]):
            self.logger.error("Erreur auth LiteLLM [%s]: %s", model, error_message)
            return ChatResponseModel.from_error(f"authentication: {error_message}")

        result = self.handle_common_errors(exception)
        if result:
            return result

        self.logger.error("Unhandled LiteLLM error [%s]: %s", model, error_message)
        return None

    @property
    def supports_streaming(self) -> bool:
        return True

    @property
    def supports_images(self) -> bool:
        return True
