import logging
import os
from pathlib import Path

# Privacy default: ask Chroma not to send anonymous usage telemetry.
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

from langchain_chroma import Chroma  # noqa: E402
from langchain_core.embeddings import Embeddings  # noqa: E402
from langchain_openai import OpenAIEmbeddings  # noqa: E402

from app.config import settings  # noqa: E402
from app.rag.chunker import split_documents  # noqa: E402
from app.rag.errors import KnowledgeBaseError  # noqa: E402
from app.rag.loader import load_markdown_documents  # noqa: E402

logger = logging.getLogger(__name__)

COLLECTION_NAME = "knowledge_base"
COLLECTION_METADATA = {"hnsw:space": "cosine"}  # distance 0 = identical text


def make_embeddings() -> Embeddings:
    if settings.openai_api_key is None:
        raise KnowledgeBaseError("not_configured", "The assistant is not configured yet.")
    return OpenAIEmbeddings(
        model=settings.openai_embedding_model,
        api_key=settings.openai_api_key.get_secret_value(),
        timeout=settings.openai_timeout_seconds,
        max_retries=settings.openai_max_retries,
    )


def open_store(persist_dir: Path, embeddings: Embeddings) -> Chroma:
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(persist_dir),
        collection_metadata=COLLECTION_METADATA,
    )


def build_index(documents_dir: Path, persist_dir: Path, embeddings: Embeddings) -> int:
    """Rebuild the whole index from the Markdown files. Returns the number of chunks."""
    documents = load_markdown_documents(documents_dir)
    chunks = split_documents(documents, settings.rag_chunk_size, settings.rag_chunk_overlap)
    if not chunks:
        raise KnowledgeBaseError("kb_invalid", "The knowledge base has no text to index.")

    ids = [f"{c.metadata['source']}::{c.metadata['chunk_index']}" for c in chunks]
    try:
        open_store(persist_dir, embeddings).delete_collection()  # start fresh
        store = open_store(persist_dir, embeddings)
        store.add_documents(chunks, ids=ids)
    except Exception as error:
        logger.error("Indexing failed: %s", type(error).__name__)
        raise KnowledgeBaseError("index_failed", "Could not build the knowledge base index.") from None
    return len(chunks)