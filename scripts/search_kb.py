import sys

from app.config import settings
from app.rag.errors import KnowledgeBaseError
from app.rag.retriever import retrieve
from app.rag.store import make_embeddings


def main() -> None:
    question = " ".join(sys.argv[1:]) or input("Question: ")
    try:
        results = retrieve(
            question, settings.chroma_dir, make_embeddings(), max_distance=float("inf")
        )
    except KnowledgeBaseError as error:
        print(f"[{error.kind}] {error.user_message}")
        return

    print(f"Threshold: {settings.rag_max_distance} (lower distance = closer match)")
    for result in results:
        verdict = "USED   " if result.distance <= settings.rag_max_distance else "too far"
        preview = result.text[:90].replace("\n", " ")
        print(f"{result.distance:.3f} [{verdict}] {result.source} | {result.section}")
        print(f"        {preview}")


if __name__ == "__main__":
    main()