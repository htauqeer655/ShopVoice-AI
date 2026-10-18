from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

HEADERS = [("#", "h1"), ("##", "h2"), ("###", "h3")]


def split_documents(
    documents: list[Document], chunk_size: int, chunk_overlap: int
) -> list[Document]:
    """Split by Markdown headings first, then by size. Adds source, section, chunk_index."""
    header_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=HEADERS, strip_headers=False)
    size_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )

    chunks: list[Document] = []
    for document in documents:
        sections = header_splitter.split_text(document.page_content)
        for section in sections:
            section.metadata["source"] = document.metadata["source"]
        chunks.extend(size_splitter.split_documents(sections))

    next_index: dict[str, int] = {}
    for chunk in chunks:
        source = chunk.metadata["source"]
        chunk.metadata["chunk_index"] = next_index.get(source, 0)
        next_index[source] = chunk.metadata["chunk_index"] + 1
        chunk.metadata["section"] = " > ".join(
            chunk.metadata[h] for h in ("h1", "h2", "h3") if h in chunk.metadata
        )
        for h in ("h1", "h2", "h3"):
            chunk.metadata.pop(h, None)
    return chunks