"""The public chunking API: :func:`chunk`, :func:`chunk_documents`, :class:`Chunker`."""
from __future__ import annotations

import copy
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from ._segment import (
    CostFn,
    Unit,
    count_words,
    expand_to_words,
    expand_with_cuts,
    overlap_start,
    pack,
    paragraph_cuts,
    sentence_cuts,
    units_from_cuts,
    word_cuts,
)
from ._semantic import analyse_shifts
from ._source import SOURCE_KINDS, Source, load_source

logger = logging.getLogger(__name__)

METHODS = ("semantic", "structural", "sentence", "fixed", "recursive")

METHOD_HELP = {
    "semantic": (
        "split where word overlap between neighbouring sentences dips; finds real "
        "topic shifts only when the topics use different words, so pass an embedding "
        "model to make it live up to its name"
    ),
    "structural": "split on markdown or HTML headings, keeping the heading trail",
    "sentence": "pack whole sentences up to size",
    "fixed": "fixed windows of size, provided for comparison",
    "recursive": "paragraph, then sentence, then word, whichever fits",
}

_count_words = count_words


def _copy_metadata(metadata: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """A private copy of ``metadata``.

    Deep, so that writing through a nested value on one chunk cannot reach back
    into the dict the caller passed in, nor into any other chunk.
    """
    if not metadata:
        return {}
    try:
        return copy.deepcopy(dict(metadata))
    except Exception:  # pragma: no cover - values that refuse to be copied
        return dict(metadata)


def _make_cost_fn(counter: Optional[Callable[[str], int]]) -> Tuple[CostFn, str]:
    """Return ``(cost_fn, unit_name)``; ``unit_name`` is what size/overlap mean."""
    if counter is None:
        return _count_words, "words"
    if not callable(counter):
        raise TypeError(
            f"counter must be a callable(str) -> int, got {type(counter).__name__}"
        )
    try:
        probe = counter("a short probe sentence")
    except Exception as exc:  # pragma: no cover - depends on the user's callable
        raise ValueError(f"counter raised {type(exc).__name__}: {exc}") from exc
    if isinstance(probe, bool) or not isinstance(probe, (int, np.integer)):
        raise TypeError(
            f"counter must return an int, got {type(probe).__name__}"
        )

    def cost(text: str) -> int:
        value = counter(text)
        return max(0, int(value))

    return cost, "tokens"


# --------------------------------------------------------------------------- #
# results
# --------------------------------------------------------------------------- #
@dataclass
class Chunk:
    """One chunk of the source text.

    ``text`` is exactly ``source_text[start:end]``. ``tokens`` is its size in the
    unit named by :attr:`ChunkResult.unit` (``"words"`` or ``"tokens"``).
    """

    text: str
    index: int
    start: int
    end: int
    tokens: int
    heading_path: Tuple[str, ...] = ()
    overlap_with: Tuple[int, ...] = ()
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def n_chars(self) -> int:
        """Length of the chunk in characters."""
        return len(self.text)

    def to_dict(self) -> Dict[str, Any]:
        """JSON-safe view of the chunk."""
        return {
            "index": self.index,
            "text": self.text,
            "start": self.start,
            "end": self.end,
            "tokens": self.tokens,
            "n_chars": len(self.text),
            "heading_path": list(self.heading_path),
            "overlap_with": list(self.overlap_with),
            "metadata": dict(self.metadata),
        }

    def __len__(self) -> int:
        return len(self.text)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        preview = self.text[:40].replace("\n", " ")
        return f"Chunk(index={self.index}, tokens={self.tokens}, text={preview!r}...)"


@dataclass
class ChunkResult:
    """The chunks plus everything needed to explain them."""

    chunks: List[Chunk]
    text: str
    method: str
    unit: str
    size: int
    overlap: int
    source_kind: str = "text"
    origin: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    n_headings: int = 0
    n_atomic_blocks: int = 0
    warnings: List[str] = field(default_factory=list)

    # -- basics ---------------------------------------------------------- #
    @property
    def n_chunks(self) -> int:
        """How many chunks were produced (always at least one)."""
        return len(self.chunks)

    @property
    def texts(self) -> List[str]:
        """The chunk texts, ready to embed."""
        return [c.text for c in self.chunks]

    @property
    def mean_size(self) -> float:
        """Average chunk size, measured in :attr:`unit`."""
        if not self.chunks:
            return 0.0
        return round(float(np.mean([c.tokens for c in self.chunks])), 1)

    @property
    def size_distribution(self) -> Dict[str, float]:
        """Min / quartiles / max / mean / std of the chunk sizes, in :attr:`unit`."""
        if not self.chunks:
            return {"count": 0, "min": 0, "p25": 0.0, "median": 0.0, "p75": 0.0,
                    "max": 0, "mean": 0.0, "std": 0.0}
        sizes = np.asarray([c.tokens for c in self.chunks], dtype=float)
        return {
            "count": int(sizes.size),
            "min": int(sizes.min()),
            "p25": round(float(np.percentile(sizes, 25)), 1),
            "median": round(float(np.percentile(sizes, 50)), 1),
            "p75": round(float(np.percentile(sizes, 75)), 1),
            "max": int(sizes.max()),
            "mean": round(float(sizes.mean()), 1),
            "std": round(float(sizes.std(ddof=0)), 1),
        }

    @property
    def n_words(self) -> int:
        """Word count of the whole source text."""
        return _count_words(self.text)

    # -- the guarantee --------------------------------------------------- #
    def reassemble(self) -> str:
        """Join the chunk bodies (dropping repeated overlap) back into the source.

        This returns :attr:`text` exactly. Nothing is dropped by chunking.
        """
        pieces: List[str] = []
        cursor = 0
        for chunk in sorted(self.chunks, key=lambda c: (c.start, c.end)):
            if chunk.end <= cursor:
                continue
            pieces.append(chunk.text[cursor - chunk.start:] if chunk.start < cursor else chunk.text)
            cursor = chunk.end
        return "".join(pieces)

    @property
    def reproduces_source(self) -> bool:
        """True when the chunks minus their overlaps rebuild the source exactly."""
        return self.reassemble() == self.text

    # -- exports --------------------------------------------------------- #
    def to_list(self) -> List[str]:
        """The chunk texts as a plain list of strings."""
        return self.texts

    def to_dict(self) -> Dict[str, Any]:
        """JSON-safe dict of the whole result, chunks included."""
        return {
            "method": self.method,
            "unit": self.unit,
            "size": self.size,
            "overlap": self.overlap,
            "n_chunks": self.n_chunks,
            "mean_size": self.mean_size,
            "size_distribution": self.size_distribution,
            "reproduces_source": self.reproduces_source,
            "source": {
                "kind": self.source_kind,
                "origin": self.origin,
                "n_chars": len(self.text),
                "n_words": _count_words(self.text),
                "n_headings": self.n_headings,
                "n_atomic_blocks": self.n_atomic_blocks,
            },
            "metadata": dict(self.metadata),
            "warnings": list(self.warnings),
            "chunks": [c.to_dict() for c in self.chunks],
        }

    def summary(self) -> str:
        """Human readable report, plain ASCII punctuation only."""
        dist = self.size_distribution
        where = f" from {self.origin}" if self.origin else ""
        lines = [
            f"rag-chunker: {self.n_chunks} chunk(s){where} "
            f"[method '{self.method}', {self.source_kind} source, "
            f"{len(self.text):,} chars]",
            f"  method: {METHOD_HELP.get(self.method, '')}",
            f"  budget: at most {self.size} {self.unit} per chunk, up to {self.overlap} "
            f"of them repeated from the previous chunk "
            + (
                "(counted by the counter you passed)"
                if self.unit == "tokens"
                else "(counted in words; pass counter= to count tokens instead)"
            ),
            f"  sizes in {self.unit}: min {dist['min']}, p25 {dist['p25']}, "
            f"median {dist['median']}, p75 {dist['p75']}, max {dist['max']}, "
            f"mean {dist['mean']}",
        ]
        if self.n_headings:
            trail = next((c.heading_path for c in self.chunks if c.heading_path), ())
            lines.append(
                f"  headings: {self.n_headings} found"
                + (f"; first trail carried: {' > '.join(trail)}" if trail else "")
            )
        if self.n_atomic_blocks:
            lines.append(
                f"  code blocks / tables kept whole: {self.n_atomic_blocks}"
            )
        if self.metadata:
            lines.append(f"  metadata on every chunk: {self.metadata}")
        for note in self.warnings:
            lines.append(f"  warning: {note}")
        lines.append(
            "  check: chunk bodies minus overlap reproduce the source exactly: "
            + ("yes" if self.reproduces_source else "NO")
        )
        lines.append(
            "  next: result.to_list() gives the chunk texts; result[0] is the first chunk"
        )
        return "\n".join(lines)

    # -- container sugar ------------------------------------------------- #
    def __len__(self) -> int:
        return len(self.chunks)

    def __iter__(self):
        return iter(self.chunks)

    def __getitem__(self, item):
        return self.chunks[item]

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"ChunkResult(n_chunks={self.n_chunks}, method={self.method!r}, "
            f"size={self.size}, overlap={self.overlap}, unit={self.unit!r})"
        )


# --------------------------------------------------------------------------- #
# the chunker
# --------------------------------------------------------------------------- #
class Chunker:
    """Reusable chunker. :func:`chunk` is this class with defaults, in one call.

    Parameters
    ----------
    size, overlap:
        The most a chunk may hold, and how much of the previous chunk it repeats,
        both in tokens when ``counter`` is given and in words otherwise. The
        overlap is part of the ``size`` budget, so a chunk stays within ``size``
        unless a single code block, table or word is larger than that on its own.
    method:
        One of ``semantic``, ``structural``, ``sentence``, ``fixed``, ``recursive``.
    counter:
        ``callable(str) -> int``; pass your tokenizer to count in real tokens.
    metadata:
        Dict copied onto every chunk produced.
    sensitivity:
        Percentile (0-100) of boundary similarities considered a topic shift,
        used by ``method="semantic"``. Higher means more splits.
    min_fill:
        Fraction of ``size`` a chunk must reach before a topic shift is allowed
        to end it, so semantic chunks do not come out tiny.
    source:
        ``auto`` (default), ``text``, ``markdown`` or ``html``.
    """

    def __init__(
        self,
        size: int = 512,
        overlap: int = 64,
        method: str = "recursive",
        embed: Optional[Callable[[Any], Any]] = None,
        counter: Optional[Callable[[str], int]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        *,
        sensitivity: float = 30.0,
        min_fill: float = 0.5,
        source: str = "auto",
    ) -> None:
        if isinstance(size, bool) or not isinstance(size, int):
            raise ValueError(f"size must be a positive integer, got {size!r}")
        if size <= 0:
            raise ValueError(
                f"size must be a positive integer, got {size}; "
                "it is the number of words (or tokens, with counter=) per chunk"
            )
        if isinstance(overlap, bool) or not isinstance(overlap, int):
            raise ValueError(f"overlap must be a non-negative integer, got {overlap!r}")
        if overlap < 0:
            raise ValueError(f"overlap must be a non-negative integer, got {overlap}")
        if overlap >= size:
            raise ValueError(
                f"overlap ({overlap}) must be smaller than size ({size}); "
                "an overlap at least as large as a chunk would repeat every chunk forever"
            )
        if method not in METHODS:
            raise ValueError(
                f"method must be one of {', '.join(METHODS)}, got {method!r}"
            )
        if source not in SOURCE_KINDS:
            raise ValueError(
                f"source must be one of {', '.join(SOURCE_KINDS)}, got {source!r}"
            )
        if metadata is not None and not isinstance(metadata, dict):
            raise TypeError(
                f"metadata must be a dict or None, got {type(metadata).__name__}"
            )
        if not 0.0 <= float(sensitivity) <= 100.0:
            raise ValueError(
                f"sensitivity must be a percentile between 0 and 100, got {sensitivity!r}"
            )
        if not 0.0 <= float(min_fill) <= 1.0:
            raise ValueError(
                f"min_fill must be a fraction between 0 and 1, got {min_fill!r}"
            )
        self.size = size
        self.overlap = overlap
        self.method = method
        self.embed = embed
        self.counter = counter
        self.metadata: Dict[str, Any] = _copy_metadata(metadata)
        self.sensitivity = float(sensitivity)
        self.min_fill = float(min_fill)
        self.source = source
        self._cost, self.unit = _make_cost_fn(counter)
        # Budget for a chunk's own text. The overlap repeated from the previous
        # chunk is added on top, so a whole chunk stays within ``size``.
        self._budget = size - overlap

    # -- internals ------------------------------------------------------- #
    def _units(self, src: Source) -> List[Unit]:
        text = src.text
        cost = self._cost
        budget = self._budget
        if self.method == "fixed":
            units = units_from_cuts(text, word_cuts(text), src.atomic, cost)
        elif self.method == "recursive":
            units = units_from_cuts(text, paragraph_cuts(src), src.atomic, cost)
            units = expand_with_cuts(
                units, text, budget, sentence_cuts(src), src.atomic, cost
            )
        else:
            units = units_from_cuts(text, sentence_cuts(src), src.atomic, cost)
        return expand_to_words(units, text, budget, cost)

    def _groups(
        self, src: Source, units: Sequence[Unit]
    ) -> Tuple[List[Tuple[int, int]], List[str]]:
        """Group indices plus any warning about a heuristic that found nothing."""
        forced = frozenset()
        soft = frozenset()
        min_fill = 0
        warnings: List[str] = []
        if self.method == "structural":
            heads = set(src.heading_offsets)
            forced = frozenset(i for i, u in enumerate(units) if u.start in heads)
            if not forced:
                warnings.append(
                    "structural chunking found no headings in this document; "
                    "falling back to packing whole sentences"
                )
        elif self.method == "semantic":
            shifts, _sims, reason = analyse_shifts(
                [src.text[u.start:u.end] for u in units],
                sensitivity=self.sensitivity,
                embed=self.embed,
            )
            soft = frozenset(shifts)
            min_fill = max(1, int(self._budget * self.min_fill))
            if reason:
                warnings.append(
                    "semantic chunking found no usable topic shift (" + reason + "); "
                    "chunks fall back to whole sentences packed to size"
                )
            elif not soft:
                warnings.append(
                    "semantic chunking found no clear topic shift; "
                    "chunks fall back to whole sentences packed to size"
                )
        for message in warnings:
            logger.info(message)
        groups = pack(units, self._budget, forced=forced, soft=soft, min_fill=min_fill)
        return groups, warnings

    def _build(self, src: Source, metadata: Dict[str, Any]) -> ChunkResult:
        text = src.text
        units = self._units(src)
        groups, warnings = self._groups(src, units)
        cores = [(units[a].start, units[b - 1].end) for a, b in groups]
        chunks: List[Chunk] = []
        for index, (a, b) in enumerate(groups):
            core_start, end = cores[index]
            start = (
                core_start
                if index == 0
                else overlap_start(
                    units,
                    a,
                    self.overlap,
                    text,
                    self._cost,
                    floor=groups[index - 1][0],
                )
            )
            body = text[start:end]
            shared = tuple(
                k
                for k in range(index)
                if cores[k][1] > start and cores[k][0] < core_start
            )
            chunks.append(
                Chunk(
                    text=body,
                    index=index,
                    start=start,
                    end=end,
                    tokens=self._cost(body),
                    heading_path=src.heading_path_at(core_start),
                    overlap_with=shared,
                    metadata=_copy_metadata(metadata),
                )
            )
        return ChunkResult(
            chunks=chunks,
            text=text,
            method=self.method,
            unit=self.unit,
            size=self.size,
            overlap=self.overlap,
            source_kind=src.kind,
            origin=src.origin,
            metadata=_copy_metadata(metadata),
            n_headings=len(src.headings),
            n_atomic_blocks=len(src.atomic),
            warnings=warnings,
        )

    # -- public ---------------------------------------------------------- #
    def chunk(self, text: Any, metadata: Optional[Dict[str, Any]] = None) -> ChunkResult:
        """Chunk one document: a string, or a path to ``.txt`` / ``.md`` / ``.html``."""
        if metadata is not None and not isinstance(metadata, dict):
            raise TypeError(
                f"metadata must be a dict or None, got {type(metadata).__name__}"
            )
        src = load_source(text, self.source)
        merged = dict(self.metadata)
        if metadata:
            merged.update(metadata)
        return self._build(src, merged)

    def chunk_documents(self, docs: Iterable[Any]) -> List[ChunkResult]:
        """Chunk many documents. Each item is text, a path, or ``{"text":..., "metadata":...}``."""
        if isinstance(docs, (str, bytes)):
            raise TypeError(
                "docs must be a list of documents, not a single string; "
                "use chunk(text) for one document"
            )
        results: List[ChunkResult] = []
        for doc in docs:
            if isinstance(doc, dict):
                if "text" not in doc:
                    raise KeyError("a document dict must have a 'text' key")
                extra = doc.get("metadata")
                if extra is not None and not isinstance(extra, dict):
                    raise TypeError("a document's 'metadata' must be a dict")
                results.append(self.chunk(doc["text"], extra))
            elif isinstance(doc, tuple) and len(doc) == 2 and isinstance(doc[1], dict):
                results.append(self.chunk(doc[0], doc[1]))
            else:
                results.append(self.chunk(doc))
        return results


# --------------------------------------------------------------------------- #
# one-line entry points
# --------------------------------------------------------------------------- #
def chunk(
    text: Any,
    *,
    size: int = 512,
    overlap: int = 64,
    method: str = "recursive",
    counter: Optional[Callable[[str], int]] = None,
    metadata: Optional[Dict[str, Any]] = None,
    embed: Optional[Callable[[Any], Any]] = None,
) -> ChunkResult:
    """Split one document into retrieval-sized chunks.

    ``text`` is a string or a path to a ``.txt``, ``.md`` or ``.html`` file.
    ``size`` and ``overlap`` count tokens when ``counter`` is given and words
    otherwise; the result says which in ``result.unit``. The overlap is counted
    inside ``size``, so a chunk holds at most ``size``.

    The default is ``recursive``: paragraph, then sentence, then word. It is
    predictable and never loses content. ``semantic`` compares word overlap between
    neighbouring sentences, which only tracks meaning when the topics happen to use
    different words; pass ``embed`` (a callable taking the sentences and returning one
    vector each) to drive it with a real embedding model instead.
    """
    return Chunker(
        size=size,
        overlap=overlap,
        method=method,
        counter=counter,
        metadata=metadata,
        embed=embed,
    ).chunk(text)


def chunk_documents(docs: Iterable[Any], **kw: Any) -> List[ChunkResult]:
    """Chunk many documents with the same settings; returns one result per document."""
    return Chunker(**kw).chunk_documents(docs)
