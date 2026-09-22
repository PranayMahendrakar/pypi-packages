"""document-memory: a persistent, searchable memory of documents and conversations.

Quick use::

    from document_memory import Memory

    with Memory("notes.db") as memory:                     # created on demand
        memory.add("Solar panels turn light into power.", source="guide.md")
        print(memory.search("solar")[0].text)

Everything lives in one SQLite file, so the memory survives restarts and several
processes can share it.  Ranking is Okapi BM25 out of the box; hand ``Memory``
an ``embed`` callable and cosine similarity is blended in with weight
:data:`VECTOR_WEIGHT`.

``memory.context(query)`` returns those memories already packed to a token
budget and formatted to paste straight into a prompt.
"""
from ._memory import (
    BUSY_TIMEOUT,
    CONTEXT_HEADER,
    VECTOR_FLOOR,
    VECTOR_WEIGHT,
    Memory,
)
from ._results import Hit, Hits, Record, Records, Turn, Turns
from ._text import B as BM25_B
from ._text import K1 as BM25_K1
from ._text import tokenize

__version__ = "0.1.0"


def open_memory(path=None, *, namespace="default", embed=None) -> Memory:
    """Open (creating on demand) the store at ``path``.

    Exactly ``Memory(path, namespace=..., embed=...)``, spelled as a verb for
    people who prefer ``document_memory.open_memory("notes.db")``.
    """
    return Memory(path, namespace=namespace, embed=embed)


__all__ = [
    "BM25_B",
    "BM25_K1",
    "BUSY_TIMEOUT",
    "CONTEXT_HEADER",
    "Hit",
    "Hits",
    "Memory",
    "Record",
    "Records",
    "Turn",
    "Turns",
    "VECTOR_FLOOR",
    "VECTOR_WEIGHT",
    "open_memory",
    "tokenize",
    "__version__",
]
