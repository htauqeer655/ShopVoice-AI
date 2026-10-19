from dataclasses import dataclass, field
from pathlib import Path

from langchain_core.embeddings import Embeddings

from app.config import settings
from app.prompts import build_grounded_system_prompt, build_grounded_user_message
from app.rag.context import format_context
from app.rag.errors import KnowledgeBaseError
from app.rag.retriever import retrieve
from app.rag.store import make_embeddings
from app.services.llm_service import LLMError, complete, validate_question

# Fixed reply when the knowledge base has nothing relevant. The model is not called.
NO_EVIDENCE_ANSWER = (
    "I'm sorry, I don't have that information. "
    "Please contact the store's support team directly. "
    "माफ़ कीजिए, यह जानकारी मेरे पास उपलब्ध नहीं है। "
    "कृपया स्टोर की सपोर्ट टीम से सीधे संपर्क करें।"
)


@dataclass(frozen=True)
class RagAnswer:
    answer: str
    sources: list[str] = field(default_factory=list)
    grounded: bool = False


def answer_question(
    question: str,
    persist_dir: Path | None = None,
    embeddings: Embeddings | None = None,
) -> RagAnswer:
    question = validate_question(question)

    try:
        if embeddings is None:
            embeddings = make_embeddings()
        chunks = retrieve(question, persist_dir or settings.chroma_dir, embeddings)
    except KnowledgeBaseError as error:
        raise LLMError(error.kind, error.user_message) from None

    if not chunks:
        return RagAnswer(answer=NO_EVIDENCE_ANSWER, sources=[], grounded=False)

    answer = complete(
        build_grounded_system_prompt(settings.business_name),
        build_grounded_user_message(format_context(chunks), question),
    )
    sources = list(dict.fromkeys(chunk.source for chunk in chunks))
    return RagAnswer(answer=answer, sources=sources, grounded=True)