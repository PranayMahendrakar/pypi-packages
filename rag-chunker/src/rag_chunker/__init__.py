"""rag-chunker: split documents for retrieval at meaning boundaries.

Quick use::

    import rag_chunker
    result = rag_chunker.chunk(text, size=120, overlap=20)   # or a .md / .html path
    print(result.summary())
    for piece in result.to_list():
        embed(piece)

The chunk bodies, minus the repeated overlap, always reproduce the source text
exactly: ``result.reassemble() == result.text``.
"""
from ._core import (
    METHOD_HELP,
    METHODS,
    Chunk,
    Chunker,
    ChunkResult,
    chunk,
    chunk_documents,
)

__version__ = "0.1.0"

__all__ = [
    "Chunk",
    "ChunkResult",
    "Chunker",
    "METHODS",
    "METHOD_HELP",
    "chunk",
    "chunk_documents",
    "__version__",
]
