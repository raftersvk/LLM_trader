"""Router for fetching dashboard visualizations and charts."""
import base64
from datetime import datetime
from typing import Dict, Any, Optional

from fastapi import APIRouter


class VisualsRouter:
    """Handles endpoints related to charts and visuals."""
    def __init__(self, analysis_engine, engine_registry=None):
        self.router = APIRouter(prefix="/api/visuals", tags=["visuals"])
        self.analysis_engine = analysis_engine
        self.engine_registry = engine_registry

        self.router.add_api_route("/charts/latest", self.get_latest_chart, methods=["GET"])

    def _get_engine_for_symbol(self, symbol: Optional[str]):
        """Get analysis engine for a specific symbol via registry."""
        if self.engine_registry and symbol:
            engine = self.engine_registry.get(symbol)
            if engine:
                return engine
        return self.analysis_engine

    async def get_latest_chart(self, symbol: Optional[str] = None) -> Dict[str, Any]:
        """Get the latest generated analysis chart as base64-encoded JSON."""
        engine = self._get_engine_for_symbol(symbol)
        last_chart_buffer = engine.last_chart_buffer if engine else None
        if last_chart_buffer:
            last_chart_buffer.seek(0)
            chart_bytes = last_chart_buffer.getvalue()
            if chart_bytes:
                chart_base64 = base64.b64encode(chart_bytes).decode("utf-8")
                return {
                    "chart_base64": chart_base64,
                    "timestamp": datetime.now().isoformat()
                }
            return {
                "error": "Chart buffer exists but is empty",
                "debug": {"buffer_exists": True, "bytes_length": 0}
            }

        try:
            has_buffer_attr = True
            _ = engine.last_chart_buffer
        except AttributeError:
            has_buffer_attr = False

        return {
            "error": "No chart generated recently.",
            "debug": {
                "analysis_engine_exists": engine is not None,
                "has_buffer_attr": has_buffer_attr,
                "buffer_value": str(type(last_chart_buffer)) if last_chart_buffer else "None"
            }
        }
