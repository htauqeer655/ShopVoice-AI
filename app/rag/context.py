from html import escape

from app.rag.retriever import RetrievedChunk


def format_context(chunks: list[RetrievedChunk]) -> str:
    """Wrap retrieved text as escaped, clearly delimited evidence.

    Escaping < and > stops a document from closing the wrapper and pretending
    to be an instruction. Retrieved text is data, never instructions.
    """
    blocks = []
    for number, chunk in enumerate(chunks, start=1):
        source = escape(chunk.source, quote=True)
        section = escape(chunk.section, quote=True)
        blocks.append(
            f'<document id="{number}" source="{source}" section="{section}">\n'
            f"{escape(chunk.text, quote=False)}\n"
            "</document>"
        )
    return "\n".join(blocks)