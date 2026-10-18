from app.config import settings
from app.rag.errors import KnowledgeBaseError
from app.rag.store import build_index, make_embeddings


def main() -> None:
    try:
        count = build_index(settings.knowledge_base_dir, settings.chroma_dir, make_embeddings())
    except KnowledgeBaseError as error:
        print(f"[{error.kind}] {error.user_message}")
        return
    print(f"Indexed {count} chunks into {settings.chroma_dir}")


if __name__ == "__main__":
    main()