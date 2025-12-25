import os
from typing import Any, Dict

# Baseline RAG
from .rag_system import EnhancedRAGSystem


class BaseEngineInterface:
    """Minimal interface for RAG engines used by chat apps."""

    def query(self, user_query: str, use_hybrid_search: bool = True, use_reranking: bool = True) -> Dict[str, Any]:
        raise NotImplementedError

    def clear_conversation_history(self) -> None:
        raise NotImplementedError

    def get_system_stats(self) -> Dict[str, Any]:
        raise NotImplementedError


class BaselineEngine(BaseEngineInterface):
    def __init__(self) -> None:
        self._engine = EnhancedRAGSystem()

    def query(self, user_query: str, use_hybrid_search: bool = True, use_reranking: bool = True) -> Dict[str, Any]:
        return self._engine.query(user_query, use_hybrid_search=use_hybrid_search, use_reranking=use_reranking)

    def clear_conversation_history(self) -> None:
        self._engine.clear_conversation_history()

    def get_system_stats(self) -> Dict[str, Any]:
        return self._engine.get_system_stats()


class GraphRAGEngine(BaseEngineInterface):
    def __init__(self) -> None:
        try:
            import graphrag  # noqa: F401
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                "GraphRAG is not installed. Please add 'graphrag' to requirements and configure it."
            ) from exc
        # TODO: initialize actual GraphRAG pipeline/graph index here
        self._not_ready_reason = (
            "GraphRAG adapter is a placeholder. Configure GraphRAG project/index paths and initialization."
        )

    def query(self, user_query: str, use_hybrid_search: bool = True, use_reranking: bool = True) -> Dict[str, Any]:
        return {
            "answer": f"GraphRAG is not yet configured. {self._not_ready_reason}",
            "sources": [],
            "search_results": [],
        }

    def clear_conversation_history(self) -> None:
        return None

    def get_system_stats(self) -> Dict[str, Any]:
        return {"engine": "graphrag", "configured": False}


class LightRAGEngine(BaseEngineInterface):
    def __init__(self) -> None:
        try:
            import lightrag  # noqa: F401
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                "LightRAG is not installed. Please add 'lightrag' to requirements and configure it."
            ) from exc
        # TODO: initialize actual LightRAG components here
        self._not_ready_reason = (
            "LightRAG adapter is a placeholder. Configure corpus ingestion and retrieval pipeline."
        )

    def query(self, user_query: str, use_hybrid_search: bool = True, use_reranking: bool = True) -> Dict[str, Any]:
        return {
            "answer": f"LightRAG is not yet configured. {self._not_ready_reason}",
            "sources": [],
            "search_results": [],
        }

    def clear_conversation_history(self) -> None:
        return None

    def get_system_stats(self) -> Dict[str, Any]:
        return {"engine": "lightrag", "configured": False}


class RAGFactory:
    """Factory for creating RAG engines by name."""

    @staticmethod
    def create_rag_system(engine_name: str = "baseline") -> BaseEngineInterface:
        engine = engine_name.strip().lower()
        if engine in ("baseline", "default"):
            return BaselineEngine()
        if engine in ("graphrag", "graph"):
            return GraphRAGEngine()
        if engine in ("lightrag", "light"):
            return LightRAGEngine()
        return BaselineEngine()


def get_rag_engine(engine_name: str | None = None) -> BaseEngineInterface:
    engine_name = (engine_name or os.getenv("RAG_ENGINE", "baseline")).strip().lower()
    if engine_name in ("baseline", "default"):
        return BaselineEngine()
    if engine_name in ("graphrag", "graph"):
        return GraphRAGEngine()
    if engine_name in ("lightrag", "light"):
        return LightRAGEngine()
    # Fallback to baseline if unknown value
    return BaselineEngine()
