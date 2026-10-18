from pathlib import Path

from langchain_core.documents import Document

from app.rag.errors import KnowledgeBaseError


def load_markdown_documents(directory: Path) -> list[Document]:
    """Read every .md file in the folder as UTF-8. Fail clearly on any problem."""
    directory = Path(directory)
    if not directory.is_dir():
        raise KnowledgeBaseError("kb_missing", f"Knowledge base folder not found: {directory}")

    paths = sorted(directory.glob("*.md"))
    if not paths:
        raise KnowledgeBaseError("kb_missing", f"No .md files found in {directory}")

    documents = []
    for path in paths:
        try:
            # utf-8-sig also accepts files that start with a hidden BOM marker
            text = path.read_text(encoding="utf-8-sig")
        except UnicodeDecodeError:
            raise KnowledgeBaseError("kb_invalid", f"{path.name} is not valid UTF-8") from None
        except OSError:
            raise KnowledgeBaseError("kb_invalid", f"{path.name} could not be read") from None
        if not text.strip():
            raise KnowledgeBaseError("kb_invalid", f"{path.name} is empty")
        documents.append(Document(page_content=text, metadata={"source": path.name}))
    return documents