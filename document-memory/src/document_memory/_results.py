"""The objects handed back to the caller: ``Record``, ``Hit``, ``Turn``.

All three are plain dataclasses.  Every one of them answers two questions
without help: ``summary()`` prints a line a human can read, and ``to_dict()``
gives the same content as JSON-safe types.

The list wrappers (``Records``, ``Hits``, ``Turns``) are real ``list``
subclasses, so ``result == []`` and ``result[0]`` behave exactly as expected,
but they also carry a ``summary()`` that explains the whole result set.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ._store import iso

PREVIEW_CHARS = 72
"""How much of a memory's text ``summary()`` shows before it stops."""


def preview(text: str, width: int = PREVIEW_CHARS) -> str:
    """One line of ``text``, whitespace collapsed and cut to ``width`` characters."""
    flat = " ".join(str(text).split())
    if len(flat) <= width:
        return flat
    return flat[: max(0, width - 3)].rstrip() + "..."


@dataclass
class Record:
    """One stored memory, as returned by ``Memory.get`` and ``Memory.recent``.

    A ``Record`` has no score because nothing was ranked to produce it; a
    ``Hit`` is the same content plus the score that ranked it.
    """

    id: str
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    source: Optional[str] = None
    timestamp: float = 0.0
    kind: str = "document"

    @property
    def when(self) -> str:
        """The timestamp as a UTC ISO 8601 string."""
        return iso(self.timestamp)

    def summary(self) -> str:
        """One human-readable line: when, where it came from, what it says."""
        where = self.source or self.metadata.get("role") or self.kind
        return f"{self.when}  {where}  [{self.id}]  {preview(self.text)}"

    def to_dict(self) -> Dict[str, Any]:
        """JSON-safe dict of every field, plus the ISO timestamp as ``when``."""
        return {
            "id": self.id,
            "text": self.text,
            "metadata": self.metadata,
            "source": self.source,
            "timestamp": self.timestamp,
            "when": self.when,
            "kind": self.kind,
        }

    def __str__(self) -> str:  # pragma: no cover - thin wrapper
        return self.summary()


@dataclass
class Hit:
    """One search result.

    ``score`` is always between 0 and 1 and is *relative to the best candidate
    for this query*: the strongest match scores 1.0.  It is a ranking, not an
    absolute quality measure, so do not compare scores across queries - compare
    ``bm25`` instead, which is the raw Okapi BM25 value.

    ``similarity`` is the cosine similarity against the query vector, or
    ``None`` when the store has no embedding function.  ``terms`` lists the
    query terms this memory actually contained, which is usually the fastest
    way to see why something ranked where it did.
    """

    id: str
    text: str
    score: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    source: Optional[str] = None
    timestamp: float = 0.0
    kind: str = "document"
    bm25: float = 0.0
    similarity: Optional[float] = None
    terms: Tuple[str, ...] = ()

    @property
    def when(self) -> str:
        """The timestamp as a UTC ISO 8601 string."""
        return iso(self.timestamp)

    def why(self) -> str:
        """Short plain-text explanation of how this score was reached."""
        parts = [f"bm25 {self.bm25:.2f}"]
        if self.similarity is not None:
            parts.append(f"cosine {self.similarity:+.2f}")
        parts.append("matched: " + (", ".join(self.terms) if self.terms else "no query term"))
        return "; ".join(parts)

    def summary(self) -> str:
        """One human-readable line: score, origin, why, and a text preview."""
        where = self.source or self.metadata.get("role") or self.kind
        return f"{self.score:.3f}  {where}  {self.why()}  {preview(self.text)}"

    def to_dict(self) -> Dict[str, Any]:
        """JSON-safe dict of every field, plus ``when`` and ``why``."""
        return {
            "id": self.id,
            "text": self.text,
            "score": self.score,
            "metadata": self.metadata,
            "source": self.source,
            "timestamp": self.timestamp,
            "when": self.when,
            "kind": self.kind,
            "bm25": self.bm25,
            "similarity": self.similarity,
            "terms": list(self.terms),
            "why": self.why(),
        }

    def to_record(self) -> Record:
        """The same memory without the ranking information."""
        return Record(
            id=self.id,
            text=self.text,
            metadata=self.metadata,
            source=self.source,
            timestamp=self.timestamp,
            kind=self.kind,
        )

    def __str__(self) -> str:  # pragma: no cover - thin wrapper
        return self.summary()


@dataclass
class Turn:
    """One conversation turn, as stored by ``Memory.remember``."""

    id: str
    role: str
    content: str
    timestamp: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    source: Optional[str] = None

    @property
    def text(self) -> str:
        """Alias for ``content``, so a turn reads like any other memory."""
        return self.content

    @property
    def when(self) -> str:
        """The timestamp as a UTC ISO 8601 string."""
        return iso(self.timestamp)

    def summary(self) -> str:
        """One human-readable line: when, who spoke, what they said."""
        return f"{self.when}  {self.role}: {preview(self.content)}"

    def to_dict(self) -> Dict[str, Any]:
        """JSON-safe dict of every field, plus the ISO timestamp as ``when``."""
        return {
            "id": self.id,
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp,
            "when": self.when,
            "metadata": self.metadata,
            "source": self.source,
        }

    def __str__(self) -> str:  # pragma: no cover - thin wrapper
        return self.summary()


class _ResultList(list):
    """Shared behaviour for the three list wrappers."""

    _noun = "result"

    def __init__(self, items: Sequence = (), **info: Any) -> None:
        super().__init__(items)
        self.info: Dict[str, Any] = dict(info)

    def _headline(self) -> str:
        count = len(self)
        noun = self._noun if count == 1 else self._noun + "s"
        return f"{count} {noun}"

    def summary(self) -> str:
        """A headline plus one numbered line per item."""
        lines: List[str] = [self._headline()]
        for number, item in enumerate(self, start=1):
            lines.append(f"  {number:>2}. {item.summary()}")
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """JSON-safe dict: the headline info plus every item as a dict."""
        payload = dict(self.info)
        payload["count"] = len(self)
        payload["items"] = [item.to_dict() for item in self]
        return payload

    def to_list(self) -> List[Dict[str, Any]]:
        """Just the items, each as a JSON-safe dict."""
        return [item.to_dict() for item in self]

    def __str__(self) -> str:  # pragma: no cover - thin wrapper
        return self.summary()


class Records(_ResultList):
    """A list of ``Record``, newest first."""

    _noun = "memory"

    def _headline(self) -> str:
        noun = "memory" if len(self) == 1 else "memories"
        return f"{len(self)} {noun}, newest first"


class Hits(_ResultList):
    """A list of ``Hit``, best match first."""

    _noun = "hit"

    def _headline(self) -> str:
        query = self.info.get("query", "")
        mode = self.info.get("mode", "lexical BM25")
        noun = "hit" if len(self) == 1 else "hits"
        if not self:
            return f"no hits for {query!r} ({mode})"
        return f"{len(self)} {noun} for {query!r}, best first ({mode})"


class Turns(_ResultList):
    """A list of ``Turn``, oldest first."""

    _noun = "turn"

    def _headline(self) -> str:
        noun = "turn" if len(self) == 1 else "turns"
        return f"{len(self)} {noun}, oldest first"
