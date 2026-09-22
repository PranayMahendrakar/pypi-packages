"""The public behaviour: ``Deduper``, ``DedupeResult`` and the one-line helpers."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from . import _minhash, _vectors
from ._io import load_texts
from ._text import canonical_tokens, normalize

logger = logging.getLogger(__name__)

__all__ = ["DedupeResult", "Deduper", "dedupe", "find_duplicates", "similarity"]

METHODS = ("auto", "tfidf", "minhash", "embed")
KEEP_MODES = ("longest", "first", "last", "most_complete")

#: Above this many distinct texts, ``method="auto"`` switches to MinHash.
#: Above this many texts, ``method="auto"`` switches from exhaustive TF-IDF to MinHash
#: candidate generation. Measured on 2,700 mixed passages: TF-IDF took 15.2s and MinHash
#: 1.6s, and BOTH removed exactly the same 200 duplicates, because MinHash is used only
#: to pick candidate pairs and every candidate is then scored exactly. A threshold of
#: 5,000 left the default an order of magnitude slower than it needed to be with nothing
#: gained for it.
AUTO_MINHASH_ABOVE = 1000

EmbedFn = Callable[[List[str]], Any]


def _shorten(text: str, width: int = 62) -> str:
    """One-line, ASCII-punctuated preview of a text."""
    flat = " ".join(text.split())
    if len(flat) <= width:
        return flat
    return flat[: width - 3] + "..."


def _check_ngram(name: str, value: Any) -> Tuple[int, int]:
    """Validate an inclusive ``(low, high)`` n-gram range.

    A reversed range produces no features at all, which would silently turn
    every run into "no duplicates found", so it is rejected here instead.
    """
    try:
        low, high = value
        low, high = int(low), int(high)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{name} must be a (low, high) pair of ints, got {value!r}"
        ) from exc
    if low < 0 or high < 0:
        raise ValueError(f"{name} must not be negative, got ({low}, {high})")
    if low > high:
        raise ValueError(
            f"{name}=({low}, {high}) is reversed: the low end must not be above "
            f"the high end (did you mean ({high}, {low})?)"
        )
    return (low, high)


class _Union:
    """Tiny union-find over ``n`` items."""

    def __init__(self, n: int) -> None:
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


@dataclass
class DedupeResult:
    """What a dedupe run found. Indices always refer to the input positions."""

    kept: List[int]
    removed: List[int]
    groups: List[List[int]]
    pairs: List[Tuple[int, int, float]]
    texts: List[str]
    threshold: float
    method: str
    keep: str
    n_texts: int
    dropped: bool = True
    warnings: List[str] = field(default_factory=list)
    all_texts: List[str] = field(default_factory=list, repr=False)

    # -- derived -----------------------------------------------------------
    @property
    def n_removed(self) -> int:
        """How many texts were dropped (always 0 for ``find_duplicates``)."""
        return len(self.removed)

    @property
    def n_kept(self) -> int:
        """How many texts survived."""
        return len(self.kept)

    @property
    def n_groups(self) -> int:
        """How many duplicate groups were found."""
        return len(self.groups)

    @property
    def n_duplicates(self) -> int:
        """Texts that repeat something already present (group size minus one)."""
        return sum(len(group) - 1 for group in self.groups)

    @property
    def reduction(self) -> float:
        """Fraction of the input that was removed, 0.0 to 1.0."""
        if not self.n_texts:
            return 0.0
        return len(self.removed) / self.n_texts

    def group_range(self, group: Sequence[int]) -> Optional[Tuple[float, float]]:
        """The lowest and highest similarity recorded inside one group."""
        members = set(group)
        scores = [s for i, j, s in self.pairs if i in members and j in members]
        if not scores:
            return None
        return (min(scores), max(scores))

    # -- output ------------------------------------------------------------
    def summary(self, max_groups: int = 5) -> str:
        """A human-readable report of what was found, in plain ASCII."""
        head = f"semantic-dedup: {self.n_texts} text{'' if self.n_texts == 1 else 's'}"
        if not self.groups:
            lines = [
                f"{head}, no duplicates found "
                f"(method {self.method}, threshold {self.threshold:g})"
            ]
            for note in self.warnings:
                lines.append(f"  note: {note}")
            return "\n".join(lines)

        if self.dropped:
            head += (
                f", {self.n_groups} duplicate group{'' if self.n_groups == 1 else 's'}, "
                f"{self.n_removed} removed, {self.reduction * 100:.1f}% smaller"
            )
        else:
            head += (
                f", {self.n_groups} duplicate group{'' if self.n_groups == 1 else 's'}, "
                f"{self.n_duplicates} repeat{'' if self.n_duplicates == 1 else 's'} "
                "found, nothing removed"
            )
        lines = [head, f"  method {self.method}, threshold {self.threshold:g}, keep {self.keep}"]
        for note in self.warnings:
            lines.append(f"  note: {note}")

        removed = set(self.removed)
        for number, group in enumerate(self.groups[:max_groups], start=1):
            span = self.group_range(group)
            scale = "" if span is None else f", similarity {span[0]:.2f} to {span[1]:.2f}"
            lines.append(f"  group {number} ({len(group)} texts{scale})")
            for index in group:
                label = "removed" if index in removed else "kept   "
                if not self.dropped and index != group[0]:
                    label = "repeat "
                text = self.all_texts[index] if index < len(self.all_texts) else ""
                lines.append(f'    {label} [{index}] "{_shorten(text)}"')
        if self.n_groups > max_groups:
            lines.append(f"  ... and {self.n_groups - max_groups} more groups")
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """A JSON-safe dict of the result as *indices*, never the texts.

        Every index refers to a position in the input, so join the report back
        to the input you passed in to read it. ``pairs`` holds one entry per
        matching pair and can be long on a large corpus.
        """
        return {
            "n_texts": self.n_texts,
            "n_kept": self.n_kept,
            "n_removed": self.n_removed,
            "n_groups": self.n_groups,
            "n_duplicates": self.n_duplicates,
            "reduction": round(self.reduction, 6),
            "method": self.method,
            "threshold": self.threshold,
            "keep": self.keep,
            "dropped": self.dropped,
            "kept": list(self.kept),
            "removed": list(self.removed),
            "groups": [list(group) for group in self.groups],
            "pairs": [[int(i), int(j), float(s)] for i, j, s in self.pairs],
            "warnings": list(self.warnings),
        }

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"DedupeResult(n_texts={self.n_texts}, n_groups={self.n_groups}, "
            f"n_removed={self.n_removed}, method={self.method!r}, "
            f"threshold={self.threshold})"
        )


class Deduper:
    """The engine under :func:`dedupe`, with the extra knobs exposed.

    Args:
        threshold: minimum cosine similarity to call two texts duplicates, in
            ``(0, 1]``. ``1.0`` means exact matches only.
        method: ``"tfidf"``, ``"minhash"``, ``"embed"`` or ``"auto"``.
        keep: which member of a group survives - ``"longest"``, ``"first"``,
            ``"last"`` or ``"most_complete"``.
        embed: ``callable(list[str]) -> ndarray`` of one vector per text.
        word_ngram / char_ngram: inclusive n-gram ranges for the TF-IDF features.
        num_perm / random_state: MinHash size and seed (results are deterministic).
        column: which csv column or json key to read when given a file path.
    """

    def __init__(
        self,
        threshold: float = 0.82,
        method: str = "auto",
        keep: str = "longest",
        embed: Optional[EmbedFn] = None,
        word_ngram: Tuple[int, int] = (1, 2),
        char_ngram: Tuple[int, int] = (3, 4),
        num_perm: int = 128,
        random_state: int = 0,
        column: Optional[str] = None,
    ) -> None:
        if method not in METHODS:
            raise ValueError(f"method must be one of {METHODS}, got {method!r}")
        if keep not in KEEP_MODES:
            raise ValueError(f"keep must be one of {KEEP_MODES}, got {keep!r}")
        threshold = float(threshold)
        if not (0.0 < threshold <= 1.0):
            raise ValueError(f"threshold must be in (0, 1], got {threshold}")
        if method == "embed" and embed is None:
            raise ValueError("method='embed' needs embed=callable(list[str]) -> ndarray")
        if embed is not None and not callable(embed):
            raise TypeError("embed must be a callable taking list[str] and returning vectors")
        if num_perm < 8:
            raise ValueError(f"num_perm must be at least 8, got {num_perm}")
        self.threshold = threshold
        self.method = method
        self.keep = keep
        self.embed = embed
        self.word_ngram = _check_ngram("word_ngram", word_ngram)
        self.char_ngram = _check_ngram("char_ngram", char_ngram)
        self.num_perm = num_perm
        self.random_state = random_state
        self.column = column

    # -- internals ---------------------------------------------------------
    def _resolve_method(self, n_unique: int) -> str:
        if self.method != "auto":
            return self.method
        if self.embed is not None:
            return "embed"
        return "minhash" if n_unique > AUTO_MINHASH_ABOVE else "tfidf"

    def _unique_pairs(
        self, unique_texts: List[str], method: str, warnings: List[str]
    ) -> List[Tuple[int, int, float]]:
        """Similar pairs among the distinct texts, as exact cosine similarities."""
        if len(unique_texts) < 2 or self.threshold >= 1.0:
            return []

        if method == "embed":
            assert self.embed is not None  # guarded in __init__/dedupe
            vectors = np.asarray(self.embed(list(unique_texts)))
            if vectors.ndim != 2 or vectors.shape[0] != len(unique_texts):
                raise ValueError(
                    "embed must return one vector per text: expected shape "
                    f"({len(unique_texts)}, d), got {tuple(vectors.shape)}"
                )
            return _vectors.dense_pairs_above(
                _vectors.normalize_rows(vectors), self.threshold
            )

        matrix = _vectors.build_tfidf(unique_texts, self.word_ngram, self.char_ngram)
        if matrix.indices.size == 0:
            # Nothing was vectorized at all: every text is featureless (symbols
            # only) or the n-gram ranges select nothing. No pair can match, so
            # say so rather than report a confident "no duplicates found".
            warnings.append(
                "no comparable features were produced for any text "
                f"(word_ngram={tuple(self.word_ngram)}, "
                f"char_ngram={tuple(self.char_ngram)}); no pair can match"
            )
            return []
        if method == "tfidf":
            return _vectors.pairs_above(matrix, self.threshold)

        signatures = _minhash.minhash_signatures(matrix, self.num_perm, self.random_state)
        candidates, truncated = _minhash.candidate_pairs(signatures, self.threshold)
        if truncated:
            warnings.append(
                "minhash candidate cap reached; some far-apart pairs may be missing "
                "(use method='tfidf' for an exhaustive scan)"
            )
        if not candidates and len(unique_texts) <= 2000:
            found = _vectors.pairs_above(matrix, self.threshold)
            if found:
                warnings.append(
                    "minhash proposed no candidates; fell back to an exact scan"
                )
            return found
        return _vectors.candidate_similarities(matrix, candidates, self.threshold)

    def _choose_keeper(self, group: Sequence[int], texts: Sequence[str]) -> int:
        if self.keep == "first":
            return group[0]
        if self.keep == "last":
            return group[-1]
        if self.keep == "longest":
            return max(group, key=lambda i: (len(texts[i]), -i))
        token_sets: Dict[int, Set[str]] = {i: set(canonical_tokens(texts[i])) for i in group}

        def coverage(i: int) -> float:
            others = [token_sets[o] for o in group if o != i]
            scores = [
                len(token_sets[i] & other) / len(other) for other in others if other
            ]
            return sum(scores) / len(scores) if scores else 0.0

        return max(group, key=lambda i: (coverage(i), len(texts[i]), -i))

    # -- entry point -------------------------------------------------------
    def run(self, texts: Any, drop: bool = True) -> DedupeResult:
        """Analyse ``texts``; ``drop=False`` reports without removing anything."""
        items = load_texts(texts, self.column)
        n = len(items)
        warnings: List[str] = []

        if n == 0:
            return DedupeResult(
                kept=[], removed=[], groups=[], pairs=[], texts=[],
                threshold=self.threshold, method=self._resolve_method(0),
                keep=self.keep, n_texts=0, dropped=drop, warnings=warnings, all_texts=[],
            )

        # Collapse exact (post-normalization) duplicates first: it keeps the
        # similarity search small and makes threshold=1.0 meaningful.
        normalized = [normalize(text) for text in items]
        first_seen: Dict[str, int] = {}
        exact_pairs: List[Tuple[int, int, float]] = []
        unique_positions: List[int] = []
        for index, key in enumerate(normalized):
            owner = first_seen.get(key)
            if owner is None:
                first_seen[key] = index
                unique_positions.append(index)
            else:
                exact_pairs.append((owner, index, 1.0))

        unique_texts = [items[i] for i in unique_positions]
        method = self._resolve_method(len(unique_texts))
        if method == "embed" and self.embed is None:
            raise ValueError("method='embed' needs embed=callable(list[str]) -> ndarray")

        near = self._unique_pairs(unique_texts, method, warnings)
        pairs = exact_pairs + [
            (unique_positions[u], unique_positions[v], round(float(s), 6))
            for u, v, s in near
        ]
        pairs = [(i, j, s) if i < j else (j, i, s) for i, j, s in pairs]
        pairs.sort(key=lambda p: (p[0], p[1]))

        union = _Union(n)
        for i, j, _ in pairs:
            union.union(i, j)
        buckets: Dict[int, List[int]] = {}
        for index in range(n):
            buckets.setdefault(union.find(index), []).append(index)
        groups = sorted(
            (sorted(members) for members in buckets.values() if len(members) > 1),
            key=lambda group: group[0],
        )

        removed: List[int] = []
        if drop:
            for group in groups:
                keeper = self._choose_keeper(group, items)
                removed.extend(index for index in group if index != keeper)
            removed.sort()
        removed_set = set(removed)
        kept = [index for index in range(n) if index not in removed_set]

        return DedupeResult(
            kept=kept,
            removed=removed,
            groups=groups,
            pairs=pairs,
            texts=[items[index] for index in kept],
            threshold=self.threshold,
            method=method,
            keep=self.keep,
            n_texts=n,
            dropped=drop,
            warnings=warnings,
            all_texts=items,
        )


def dedupe(
    texts: Any,
    *,
    threshold: float = 0.82,
    method: str = "auto",
    keep: str = "longest",
    embed: Optional[EmbedFn] = None,
    **kwargs: Any,
) -> DedupeResult:
    """Drop passages that repeat the meaning of an earlier one.

    Args:
        texts: a list of strings, or a path to a ``.txt`` (one per line),
            ``.csv`` or ``.jsonl`` file.
        threshold: minimum similarity to treat two texts as duplicates.
            ``1.0`` keeps only exact matches.
        method: ``"tfidf"`` (default engine), ``"minhash"`` (fast candidate
            generation for big inputs), ``"embed"`` (use your vectors) or
            ``"auto"`` - embed if given, else minhash above 1000 texts.
        keep: which member of a duplicate group survives.
        embed: ``callable(list[str]) -> ndarray``, one row per text.

    Returns:
        A :class:`DedupeResult`; ``result.texts`` are the survivors.
    """
    return Deduper(
        threshold=threshold, method=method, keep=keep, embed=embed, **kwargs
    ).run(texts, drop=True)


def find_duplicates(texts: Any, **kwargs: Any) -> DedupeResult:
    """Same analysis as :func:`dedupe`, but nothing is removed.

    ``result.groups`` and ``result.pairs`` are filled in exactly as they would
    be, while ``result.kept`` stays the whole input.
    """
    threshold = kwargs.pop("threshold", 0.82)
    method = kwargs.pop("method", "auto")
    keep = kwargs.pop("keep", "longest")
    embed = kwargs.pop("embed", None)
    return Deduper(
        threshold=threshold, method=method, keep=keep, embed=embed, **kwargs
    ).run(texts, drop=False)


def similarity(a: str, b: str, *, method: str = "tfidf") -> float:
    """How alike two texts are, from 0.0 to 1.0.

    ``method="tfidf"`` (the default) is the cosine used by :func:`dedupe`;
    ``method="minhash"`` is the exact Jaccard of the same feature sets, which
    is what the MinHash stage estimates.

    Note that the TF-IDF weighting is relative to the texts being compared, so
    this scores the pair on its own while :func:`dedupe` scores it against the
    whole corpus. The two are close but not identical, and a threshold tuned
    here will not transfer exactly - calibrate on the corpus you will run.
    """
    if not isinstance(a, str) or not isinstance(b, str):
        raise TypeError("similarity() takes two strings")
    if method == "auto":
        method = "tfidf"
    if method == "embed":
        raise ValueError(
            "similarity() has no embed hook; call your embedder and compare the "
            "two vectors, or use dedupe(..., method='embed', embed=...)"
        )
    if method not in ("tfidf", "minhash"):
        raise ValueError(f"method must be 'tfidf' or 'minhash', got {method!r}")
    if normalize(a) == normalize(b):
        return 1.0
    matrix = _vectors.build_tfidf([a, b])
    if method == "minhash":
        return round(_minhash.jaccard(matrix, 0, 1), 6)
    left_idx, left_val = matrix.row(0)
    right_idx, right_val = matrix.row(1)
    if left_idx.size == 0 or right_idx.size == 0:
        return 0.0
    buffer = np.zeros(matrix.n_features, dtype=np.float64)
    buffer[left_idx] = left_val
    score = float((buffer[right_idx] * right_val).sum())
    return round(min(1.0, max(0.0, score)), 6)
