import logging
from dataclasses import dataclass
from pathlib import Path

from langchain_core.embeddings import Embeddings

from app.config import settings
from app.rag.errors import KnowledgeBaseError
from app.rag.store import open_store

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RetrievedChunk:
    text: str
    source: str
    section: str
    distance: float  # lower = closer match


def retrieve(
    question: str,
    persist_dir: Path,
    embeddings: Embeddings,
    k: int | None = None,
    max_distance: float | None = None,
) -> list[RetrievedChunk]:
    """Return the closest chunks within max_distance. An empty list means 'no evidence'."""
    question = (question or "").strip()
    if not question:
        return []
    if len(question) > settings.max_question_chars:
        raise KnowledgeBaseError("invalid_input", "Question is too long.")

    k = k or settings.rag_top_k
    threshold = settings.rag_max_distance if max_distance is None else max_distance

    try:
        store = open_store(persist_dir, embeddings)
        if not store.get(limit=1)["ids"]:
            raise KnowledgeBaseError("not_indexed", "The knowledge base has not been indexed yet.")
        results = store.similarity_search_with_score(question, k=k)
    except KnowledgeBaseError:
        raise
    except Exception as error:
        # Log the error type only: messages could contain private text.
        logger.error("Knowledge base search failed: %s", type(error).__name__)
        raise KnowledgeBaseError(
            "search_failed", "The knowledge base is temporarily unavailable."
        ) from None

    return [
        RetrievedChunk(
            text=document.page_content,
            source=document.metadata.get("source", "unknown"),
            section=document.metadata.get("section", ""),
            distance=float(score),
        )
        for document, score in results
        if score <= threshold
    ]