"""
LiteLLM Unified Client - Remplace les SDK spécifiques par une interface standard OpenAI.

Supporte: Ollama, OpenAI, Anthropic, Google, OpenRouter, etc.
via un seul client unifié.
"""
import io
import base64
import logging
from typing import Optional, Dict, Any, List, Union

import litellm
from litellm import acompletion

from src.logger.logger import Logger
from src.platforms.ai_providers.base import BaseAIClient
from src.platforms.ai_providers.response_models import ChatResponseModel, UsageModel, ChoiceModel, MessageModel
from src.utils.decorators import retry_api_call

litellm.drop_params = True  # Ignore les paramètres non supportés par certains providers


class LitellmAIClient(BaseAIClient):
    """Client unifié utilisant LiteLLM pour tous les providers LLM."""

    def __init__(self, config: Dict[str, Any], logger: Logger) -> None:
        """
        Initialise le client LiteLLM.

        Args:
            config: Configuration du provider avec clés:
                - provider: nom du provider (ollama, openai, anthropic, google, openrouter...)
                - base_url: URL de base (optionnel, ex: http://localhost:11434/v1)
                - api_key: Clé API (optionnel pour local)
                - model: Modèle par défaut
            logger: Logger instance
        """
        super().__init__(logger)
        self.config = config
        self.provider = config.get("provider", "openai")
        self.base_url = config.get("base_url")
        self.api_key = config.get("api_key")
        self.default_model = config.get("model", "default")

        # Configuration LiteLLM pour ce client
        self.litellm_kwargs = {
            "base_url": self.base_url,
        }
        if self.api_key:
            self.litellm_kwargs["api_key"] = self.api_key
        elif self.base_url:
            # Pour les URLs custom (Ollama/LM Studio), LiteLLM requiert une api_key
            # On utilise une valeur factice
            self.litellm_kwargs["api_key"] = "ollama"

        self.logger.debug(
            "LitellmAIClient initialisé - provider: %s, model: %s, base_url: %s",
            self.provider,
            self.default_model,
            self.base_url,
        )

    async def _initialize_client(self) -> None:
        """LiteLLM n'a pas besoin d'initialisation asynchrone spécifique."""
        self.logger.debug("LitellmAIClient prêt - aucune initialisation requise")

    async def close(self) -> None:
        """LiteLLM n'a pas besoin de nettoyage spécifique."""
        self.logger.debug("LitellmAIClient - fermeture (aucune action requise)")

    @retry_api_call(max_retries=3, initial_delay=1, backoff_factor=2, max_delay=30)
    async def chat_completion(
        self, model: str, messages: List[Dict[str, Any]], model_config: Dict[str, Any]
    ) -> Optional[ChatResponseModel]:
        """
        Envoie une requête de chat completion via LiteLLM.

        Args:
            model: Nom du modèle (ex: gemma4:31b-cloud, gemini-flash, claude-sonnet)
            messages: Liste de messages OpenAI-style
            model_config: Paramètres du modèle (temperature, max_tokens, etc.)

        Returns:
            ChatResponseModel ou None en cas d'erreur
        """
        full_model = self._get_full_model_name(model)
        params = self._build_litellm_params(model_config)

        self.logger.debug(
            "LiteLLM request - model: %s, provider: %s, messages: %d",
            full_model,
            self.provider,
            len(messages),
        )

        try:
            response = await acompletion(
                model=full_model,
                messages=messages,
                **params,
                **self.litellm_kwargs,
            )
            self.logger.debug("LiteLLM response reçue avec succès")
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
        Envoie une requête avec image pour l'analyse de graphique.

        LiteLLM gère les requêtes multimodales (texte + image) pour les
        providers qui le supportent (Gemini, GPT-4o, Claude 3, etc.).

        Args:
            model: Nom du modèle
            messages: Liste de messages OpenAI-style
            chart_image: Image du graphique (BytesIO, bytes, ou path)
            model_config: Paramètres du modèle

        Returns:
            ChatResponseModel ou None en cas d'erreur
        """
        full_model = self._get_full_model_name(model)
        params = self._build_litellm_params(model_config)

        # Préparer les données de l'image
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
            response = await acompletion(
                model=full_model,
                messages=messages_with_image,
                **params,
                **self.litellm_kwargs,
            )
            self.logger.debug("LiteLLM chart analysis response reçue")
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
        Envoie une requête de chat completion en streaming via LiteLLM.

        Args:
            model: Nom du modèle
            messages: Liste de messages OpenAI-style
            model_config: Paramètres du modèle
            callback: Fonction appelée pour chaque fragment de streaming

        Returns:
            ChatResponseModel ou None en cas d'erreur
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

            self.logger.debug("LiteLLM streaming response complète")
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
        """Construit le nom complet du modèle pour LiteLLM.
        
        Pour les URLs locales (Ollama/LM Studio), on utilise l'API OpenAI-compatible
        avec prefixe openai/ pour utiliser le transport OpenAI standard de LiteLLM.
        """
        if not model:
            model = self.default_model

        if self.provider == "openrouter":
            return f"openrouter/{model}"
        # URLs locales -> openai/ prefix avec base_url
        if self.base_url:
            return f"openai/{model}"
        if self.provider == "ollama":
            return f"ollama/{model}"
        return f"{self.provider}/{model}"

    def _build_litellm_params(self, model_config: Dict[str, Any]) -> Dict[str, Any]:
        """Construit les paramètres LiteLLM à partir de la config du modèle."""
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
        # base_url déjà dans litellm_kwargs, non ajouté ici
        return params

    def _create_response_from_litellm(self, response) -> ChatResponseModel:
        """Convertit une réponse LiteLLM en ChatResponseModel standard."""
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
        """Gère les exceptions LiteLLM et les convertit en réponses standardisées."""
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
            self.logger.error("Modèle non trouvé LiteLLM [%s]: %s", model, error_message)
            return ChatResponseModel.from_error(f"model_not_found: {error_message}")
        if any(err in error_message_lower for err in ["auth", "api_key", "401", "403"]):
            self.logger.error("Erreur auth LiteLLM [%s]: %s", model, error_message)
            return ChatResponseModel.from_error(f"authentication: {error_message}")

        result = self.handle_common_errors(exception)
        if result:
            return result

        self.logger.error("Erreur LiteLLM non gérée [%s]: %s", model, error_message)
        return None

    @property
    def supports_streaming(self) -> bool:
        return True

    @property
    def supports_images(self) -> bool:
        return True
