"""Router for dashboard monitoring and news endpoints."""
import asyncio
import json
from pathlib import Path
from typing import Dict, Any, Optional

from fastapi import APIRouter
from src.utils.token_counter import CostStorage

NEWS_FILES = ("crypto_news.json", "news_cache/recent_news.json")

class MonitorRouter:
    """Handles endpoints for system monitoring and news."""
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    def __init__(self, config, logger, dashboard_state, analysis_engine, rag_engine, engine_registry=None):
        self.router = APIRouter(prefix="/api/monitor", tags=["monitor"])
        self.config = config
        self.logger = logger
        self.dashboard_state = dashboard_state
        self.analysis_engine = analysis_engine
        self.rag_engine = rag_engine
        self.engine_registry = engine_registry

        self.router.add_api_route("/last_prompt", self.get_last_prompt, methods=["GET"])
        self.router.add_api_route("/last_response", self.get_last_response, methods=["GET"])
        self.router.add_api_route("/system_prompt", self.get_system_prompt, methods=["GET"])
        self.router.add_api_route("/costs", self.get_api_costs, methods=["GET"])
        self.router.add_api_route("/news", self.get_news, methods=["GET"])
        self.router.add_api_route("/pairs", self.get_available_pairs, methods=["GET"])
        self.router.add_api_route("/status/countdown", self.get_countdown, methods=["GET"])
        self.router.add_api_route("/status/all", self.get_all_pairs_status, methods=["GET"])

    def _get_engine_for_symbol(self, symbol: Optional[str]):
        """Get analysis engine for a specific symbol via registry."""
        if self.engine_registry and symbol:
            engine = self.engine_registry.get(symbol)
            if engine:
                return engine
        return self.analysis_engine

    def _load_prev_response_sync(self, symbol: Optional[str] = None) -> Dict[str, Any]:
        """Helper to load previous response data synchronously."""
        data_dir = self.config.DATA_DIR
        if symbol:
            safe_symbol = symbol.replace("/", "_").replace("-", "_")
            pair_path = Path(data_dir) / "trading" / f"pair_{safe_symbol}" / "previous_response.json"
            if pair_path.exists():
                try:
                    with open(pair_path, "r", encoding="utf-8") as file:
                        return json.load(file)
                except Exception:
                    self.logger.error("Error loading previous response for %s", symbol, exc_info=True)
                return {}
        path = Path(data_dir) / "trading" / "previous_response.json"
        if path.exists():
            try:
                with open(path, "r", encoding="utf-8") as file:
                    return json.load(file)
            except Exception:
                self.logger.error("Error loading previous response", exc_info=True)
        return {}

    async def get_last_prompt(self, symbol: Optional[str] = None) -> Dict[str, Any]:
        """Get the last prompt sent to the LLM."""
        engine = self._get_engine_for_symbol(symbol)
        if engine and hasattr(engine, 'last_generated_prompt') and engine.last_generated_prompt:
            return {
                "prompt": engine.last_generated_prompt,
                "source": "memory",
                "timestamp": getattr(engine, 'last_prompt_timestamp', None)
            }
        data = await asyncio.to_thread(self._load_prev_response_sync, symbol)
        prompt = data.get("prompt")
        if prompt:
            return {
                "prompt": prompt,
                "source": "disk",
                "timestamp": data.get("timestamp", "unknown")
            }
        return {"prompt": "No prompt generated yet.", "source": None}

    async def get_last_response(self, symbol: Optional[str] = None) -> Dict[str, Any]:
        """Get the last response received from the LLM."""
        engine = self._get_engine_for_symbol(symbol)
        if engine and hasattr(engine, 'last_llm_response') and engine.last_llm_response:
            return {
                "response": engine.last_llm_response,
                "source": "memory",
                "timestamp": getattr(engine, 'last_response_timestamp', None)
            }
        data = await asyncio.to_thread(self._load_prev_response_sync, symbol)
        if data:
            response = data.get("response", {})
            text_analysis = response.get("text_analysis", "No analysis available")
            return {
                "response": text_analysis,
                "source": "disk",
                "timestamp": data.get("timestamp"),
                "indicators": {k: v for k, v in response.items() if k != "text_analysis"}
            }
        return {"response": "No response received yet."}

    async def get_system_prompt(self, symbol: Optional[str] = None) -> Dict[str, Any]:
        """Get the last system prompt (contains brain context and trading rules)."""
        engine = self._get_engine_for_symbol(symbol)
        system_prompt = getattr(engine, 'last_system_prompt', None) if engine else None
        if system_prompt:
            has_brain = "TRADING BRAIN" in system_prompt
            return {
                "system_prompt": system_prompt,
                "source": "memory",
                "has_brain_context": has_brain
            }
        return {"system_prompt": "No system prompt generated yet.", "source": None, "has_brain_context": False}

    async def get_api_costs(self) -> Dict[str, Any]:
        """Get current API cost tracking data."""
        cached = self.dashboard_state.get_cached("costs", ttl_seconds=30.0)
        if cached:
            return cached
        storage = await asyncio.to_thread(CostStorage)
        openrouter_cost = storage.get_provider_costs("openrouter").total_cost
        google_cost = storage.get_provider_costs("google").total_cost
        total = openrouter_cost + google_cost
        result = {
            "costs_by_provider": {
                "openrouter": openrouter_cost,
                "google": google_cost
            },
            "total_session_cost": total,
            "last_request_cost": None,
            "formatted_total": f"${total:.6f}" if total > 0 else "Free"
        }
        self.dashboard_state.set_cached("costs", result)
        return result

    def _read_news_file_sync(self, file_path: Path) -> list:
        """Synchronously read and parse a news JSON file."""
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data.get("articles", data) if isinstance(data, dict) else data

    async def get_news(self) -> Dict[str, Any]:
        """Get cached news articles from RAG engine or disk."""
        cached = self.dashboard_state.get_cached("news", ttl_seconds=3600.0)
        if cached is not None:
            return {"articles": cached, "count": len(cached)}
        articles = []
        if self.rag_engine:
            news_manager = self.rag_engine.news_manager
            if news_manager:
                articles = news_manager.news_database
        if not articles:
            data_dir = self.config.DATA_DIR
            for news_path in NEWS_FILES:
                news_file = Path(data_dir) / news_path
                if news_file.exists():
                    try:
                        parsed_articles = await asyncio.to_thread(self._read_news_file_sync, news_file)
                        articles = parsed_articles
                        if articles:
                            break
                    except Exception:
                        self.logger.error("Failed to load news from %s", news_path, exc_info=True)
        self.dashboard_state.set_cached("news", articles)
        return {"articles": articles, "count": len(articles)}

    async def get_available_pairs(self) -> Dict[str, Any]:
        """Get the list of configured trading pairs."""
        return {"pairs": self.config.CRYPTO_PAIRS}

    async def get_countdown(self, symbol: Optional[str] = None) -> Dict[str, Any]:
        """Get countdown timer for a specific pair or the first configured pair."""
        return self.dashboard_state.get_countdown_data(symbol)

    async def get_all_pairs_status(self) -> Dict[str, Any]:
        """Get status of all configured pairs."""
        pairs_state = self.dashboard_state.get_all_pairs_state()
        return {
            "pairs": pairs_state,
            "total_pairs": len(self.config.CRYPTO_PAIRS),
            "active_pairs": sum(1 for state in pairs_state.values() if state.get("is_running", False))
        }