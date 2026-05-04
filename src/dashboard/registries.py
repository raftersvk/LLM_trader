"""Registries for per-pair service lookup.

Provides symbol-keyed dictionaries to look up AnalysisEngine and
VectorMemoryService instances that are created per trading pair by
the BotDependencyFactory.
"""
from typing import Dict, Optional, Any


class AnalysisEngineRegistry:
    """Registry mapping symbol -> AnalysisEngine instance."""

    def __init__(self) -> None:
        self._engines: Dict[str, Any] = {}

    def register(self, symbol: str, engine: Any) -> None:
        self._engines[symbol] = engine

    def get(self, symbol: Optional[str]) -> Optional[Any]:
        if not symbol:
            return next(iter(self._engines.values())) if self._engines else None
        return self._engines.get(symbol)

    def all_symbols(self) -> list[str]:
        return list(self._engines.keys())

    def items(self) -> Dict[str, Any]:
        return dict(self._engines)


class VectorMemoryRegistry:
    """Registry mapping symbol -> VectorMemoryService instance."""

    def __init__(self) -> None:
        self._memories: Dict[str, Any] = {}

    def register(self, symbol: str, vector_memory: Any) -> None:
        self._memories[symbol] = vector_memory

    def get(self, symbol: Optional[str]) -> Optional[Any]:
        if not symbol:
            return next(iter(self._memories.values())) if self._memories else None
        return self._memories.get(symbol)

    def all_symbols(self) -> list[str]:
        return list(self._memories.keys())

    def items(self) -> Dict[str, Any]:
        return dict(self._memories)
