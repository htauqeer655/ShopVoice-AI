from langchain_core.embeddings import Embeddings

from app.rag.retriever import retrieve
from app.rag.store import build_index


class ParallelVectors(Embeddings):
    """Document vector [2, 0] and query vector [1, 0] point the same way.

    Cosine distance is 0. Squared L2 distance would be 1.0. So this test fails
    if the collection is not really using the cosine metric.
    """

    def embed_documents(self, texts):
        return [[2.0, 0.0] for _ in texts]

    def embed_query(self, text):
        return [1.0, 0.0]


def test_collection_uses_cosine_distance(tmp_path):
    kb = tmp_path / "kb"
    kb.mkdir()
    (kb / "doc.md").write_text("# Doc\nHello", encoding="utf-8")
    persist = tmp_path / "chroma"

    build_index(kb, persist, ParallelVectors())
    results = retrieve("anything", persist, ParallelVectors(), k=1, max_distance=10.0)

    assert results
    assert results[0].distance < 0.01