from app.config import settings
from app.rag.chunker import split_documents
from app.rag.errors import KnowledgeBaseError
from app.rag.loader import load_markdown_documents


def main() -> None:
    try:
        documents = load_markdown_documents(settings.knowledge_base_dir)
    except KnowledgeBaseError as error:
        print(f"[{error.kind}] {error.user_message}")
        return

    chunks = split_documents(documents, settings.rag_chunk_size, settings.rag_chunk_overlap)
    print(f"Folder: {settings.knowledge_base_dir}")
    for document in documents:
        name = document.metadata["source"]
        count = sum(1 for c in chunks if c.metadata["source"] == name)
        print(f"  {name:<34} {len(document.page_content):>6} chars  {count:>3} chunks")
    print(f"Total: {len(documents)} files, {len(chunks)} chunks")


if __name__ == "__main__":
    main()