"""Public API: DuplicateFinder, DuplicateResult, find_duplicates, dedupe."""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from . import _images, _records, _text

logger = logging.getLogger(__name__)

KINDS = ("auto", "text", "records", "images")
Pair = Tuple[int, int, float]


# --------------------------------------------------------------------------- #
# Result
# --------------------------------------------------------------------------- #
@dataclass
class DuplicateResult:
    """What ``find_duplicates`` found. All indices are 0-based positions in the input.

    Attributes:
        groups: connected components of near-duplicate indices (size >= 2),
            each ascending, ordered by their smallest member.
        pairs: ``(i, j, score)`` edges that were found, ``i < j``, ordered by
            ``(i, j)``. Exact duplicates score 1.0.
        representatives: the index kept for each group (aligned with ``groups``):
            the longest text, the first row, or the first image.
        keep_indices: every singleton plus one representative per group, ascending.
        n_items: number of input items.
        kind: ``"text"``, ``"records"`` or ``"images"``.
        threshold: the similarity threshold that was used.
    """

    groups: List[List[int]] = field(default_factory=list)
    pairs: List[Pair] = field(default_factory=list)
    representatives: List[int] = field(default_factory=list)
    keep_indices: List[int] = field(default_factory=list)
    n_items: int = 0
    kind: str = "text"
    threshold: float = 0.85

    @property
    def n_groups(self) -> int:
        """Number of duplicate groups."""
        return len(self.groups)

    @property
    def n_duplicates(self) -> int:
        """Number of items that would be removed by ``dedupe`` (group sizes minus one each)."""
        return sum(len(g) - 1 for g in self.groups)

    @property
    def drop_indices(self) -> List[int]:
        """Indices that ``dedupe`` removes, ascending."""
        keep = set(self.keep_indices)
        return [i for i in range(self.n_items) if i not in keep]

    def summary(self) -> str:
        """Human-readable overview."""
        n = self.n_items
        noun = {"text": "texts", "records": "rows", "images": "images"}.get(self.kind, "items")
        lines = [f"near-dupes: {n:,} {noun}, threshold {self.threshold:g}"]
        if n == 0:
            lines.append("  nothing to compare")
            return "\n".join(lines)
        pct = 100.0 * self.n_duplicates / n if n else 0.0
        lines.append(
            f"  {self.n_groups:,} duplicate group{'s' if self.n_groups != 1 else ''}, "
            f"{self.n_duplicates:,} duplicate{'s' if self.n_duplicates != 1 else ''} ({pct:.1f}%), "
            f"{len(self.keep_indices):,} kept"
        )
        if self.groups:
            exact = sum(1 for _, _, s in self.pairs if s >= 1.0)
            largest = max(len(g) for g in self.groups)
            lines.append(f"  exact pairs: {exact:,}, near pairs: {len(self.pairs) - exact:,}, largest group: {largest}")
            shown = self.groups[:10]
            group_of = {i: k for k, g in enumerate(shown) for i in g}
            lowest = [1.0] * len(shown)
            for i, _, s in self.pairs:  # both ends of an edge sit in the same group
                k = group_of.get(i)
                if k is not None and s < lowest[k]:
                    lowest[k] = s
            for k, (g, rep) in enumerate(zip(shown, self.representatives)):
                members = str(g) if len(g) <= 8 else f"[{', '.join(map(str, g[:6]))}, ... {len(g)} items]"
                lines.append(f"  group {k + 1}: {members} -> keep {rep} (min score {lowest[k]:.3f})")
            if self.n_groups > len(shown):
                lines.append(f"  ... {self.n_groups - len(shown):,} more groups")
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """JSON-safe dictionary of the result."""
        return {
            "kind": self.kind,
            "threshold": float(self.threshold),
            "n_items": int(self.n_items),
            "n_groups": int(self.n_groups),
            "n_duplicates": int(self.n_duplicates),
            "groups": [[int(i) for i in g] for g in self.groups],
            "representatives": [int(i) for i in self.representatives],
            "pairs": [[int(i), int(j), float(s)] for i, j, s in self.pairs],
            "keep_indices": [int(i) for i in self.keep_indices],
        }

    def dedupe(self, items: Any) -> Any:
        """Return ``items`` with duplicates removed, keeping ``keep_indices``.

        A DataFrame/Series comes back as the same type with its original index
        labels; lists, tuples and numpy arrays keep their type; a ``.csv`` /
        ``.parquet`` path is loaded and returned as a DataFrame.
        """
        keep = list(self.keep_indices)
        items = _materialize(items)
        if _records.is_table_path(items):
            items = _records.load_table(items)
        if isinstance(items, (pd.DataFrame, pd.Series)):
            self._check_len(len(items))
            return items.iloc[keep]
        if isinstance(items, np.ndarray):
            self._check_len(len(items))
            return items[keep]
        seq = items if isinstance(items, (list, tuple)) else list(items)
        self._check_len(len(seq))
        kept = [seq[i] for i in keep]
        return tuple(kept) if isinstance(items, tuple) else kept

    def _check_len(self, n: int) -> None:
        if n != self.n_items:
            raise ValueError(f"dedupe() got {n} items but the result was computed on {self.n_items}")


# --------------------------------------------------------------------------- #
# Finder
# --------------------------------------------------------------------------- #
class DuplicateFinder:
    """Configurable near-duplicate finder; ``find_duplicates`` is this with defaults.

    Args:
        kind: ``"auto"`` (detect from the input), ``"text"``, ``"records"`` or ``"images"``.
        threshold: minimum similarity in ``(0, 1]``. Jaccard on character
            shingles for text/records, ``1 - hamming/64`` for images. ``1.0``
            means exact duplicates only.
        key: column name or list of names to compare DataFrame rows on
            (default: all columns).
        n_gram: character shingle length for text/records.
        num_perm: MinHash permutations.
        random_state: seed for the MinHash permutations.
        normalize: fold case, NFKC, collapse whitespace (and strip punctuation
            for records) before comparing. ``False`` compares raw strings.
        all_pairs_max: compare every pair instead of using LSH when there are
            at most this many distinct items.
    """

    def __init__(
        self,
        *,
        kind: str = "auto",
        threshold: float = 0.85,
        key: Any = None,
        n_gram: int = 3,
        num_perm: int = 128,
        random_state: int = 0,
        normalize: bool = True,
        all_pairs_max: int = 2000,
    ) -> None:
        if kind not in KINDS:
            raise ValueError(f"kind must be one of {KINDS}, got {kind!r}")
        threshold = float(threshold)
        if not 0.0 < threshold <= 1.0:
            raise ValueError(f"threshold must be in (0, 1], got {threshold}")
        if int(n_gram) < 1:
            raise ValueError(f"n_gram must be >= 1, got {n_gram}")
        if int(num_perm) < 2:
            raise ValueError(f"num_perm must be >= 2, got {num_perm}")
        self.kind = kind
        self.threshold = threshold
        self.key = key
        self.n_gram = int(n_gram)
        self.num_perm = int(num_perm)
        self.random_state = int(random_state)
        self.normalize = bool(normalize)
        self.all_pairs_max = int(all_pairs_max)

    # -- public ------------------------------------------------------------- #
    def find(self, items: Any) -> DuplicateResult:
        """Find near-duplicates in ``items`` (list of str, DataFrame, list of dicts, image paths/PIL images)."""
        data, kind = self._resolve(items)
        if kind == "text":
            return self._find_text(data)
        if kind == "records":
            return self._find_records(data)
        return self._find_images(data)

    def dedupe(self, items: Any) -> Any:
        """``find(items).dedupe(items)`` in one step."""
        items = _materialize(items)
        return self.find(items).dedupe(items)

    # -- input handling ----------------------------------------------------- #
    def _resolve(self, items: Any) -> Tuple[Any, str]:
        kind = self.kind
        if _records.is_table_path(items):
            items = _records.load_table(items)
        if isinstance(items, pd.DataFrame):
            if kind in ("auto", "records"):
                _records.check_columns(items)
                return items, "records"
            raise ValueError(f"kind={kind!r} does not accept a DataFrame; use kind='records'")
        if isinstance(items, pd.Series):
            items = items.tolist()
        elif isinstance(items, np.ndarray):
            items = items.tolist() if items.ndim == 1 else list(items)
        items = _materialize(items)
        if kind == "records":
            if all(isinstance(x, dict) for x in items):
                frame = _records.frame_from_records(items)
                _records.check_columns(frame)
                return frame, "records"
            raise ValueError("kind='records' expects a DataFrame, a .csv/.parquet path or a list of dicts")
        if kind == "auto":
            if items and all(isinstance(x, dict) for x in items):
                frame = _records.frame_from_records(items)
                _records.check_columns(frame)
                return frame, "records"
            if items and all(_images.is_pil_image(x) for x in items):
                return items, "images"
            if items and all(_images.looks_like_image_path(x) for x in items):
                return items, "images"
            if items and all(isinstance(x, os.PathLike) for x in items):
                raise ValueError("Got Path objects that are not image files; pass kind='text' or kind='images'")
            return items, "text"
        return items, kind

    # -- per kind ----------------------------------------------------------- #
    def _find_text(self, items: List[Any]) -> DuplicateResult:
        _reject_opaque(items)
        if self.normalize:
            keys = [_text.normalize_text(x) for x in items]
        else:
            keys = ["" if x is None or (isinstance(x, float) and x != x) else str(x) for x in items]
        blank = [k == "" for k in keys]
        lengths = [len(str(x)) if x is not None else 0 for x in items]
        return self._assemble(
            keys,
            blank,
            "text",
            near=lambda reps: self._text_pairs([keys[i] for i in reps]),
            choose=lambda group: max(group, key=lambda i: (lengths[i], -i)),
        )

    def _find_records(self, df: pd.DataFrame) -> DuplicateResult:
        keys, blank = _records.row_keys(df, self.key, normalize=self.normalize)
        return self._assemble(
            keys,
            blank,
            "records",
            near=lambda reps: self._text_pairs([keys[i] for i in reps]),
            choose=min,
        )

    def _find_images(self, items: List[Any]) -> DuplicateResult:
        hashes = _images.dhash_many(items) if items else np.empty(0, dtype=np.uint64)
        keys = hashes.tolist()
        blank = [False] * len(keys)
        return self._assemble(
            keys,
            blank,
            "images",
            near=lambda reps: _images.find_image_pairs(hashes[reps], threshold=self.threshold),
            choose=min,
        )

    def _text_pairs(self, texts: List[str]) -> List[Pair]:
        return _text.find_text_pairs(
            texts,
            threshold=self.threshold,
            n_gram=self.n_gram,
            num_perm=self.num_perm,
            random_state=self.random_state,
            all_pairs_max=self.all_pairs_max,
        )

    # -- shared assembly ---------------------------------------------------- #
    def _assemble(self, keys: Sequence[Any], blank: Sequence[bool], kind: str, *, near, choose) -> DuplicateResult:
        n = len(keys)
        first_of: Dict[Any, int] = {}
        exact_members: Dict[int, List[int]] = {}
        for i, (k, is_blank) in enumerate(zip(keys, blank)):
            if is_blank:
                continue
            rep = first_of.setdefault(k, i)
            if rep != i:
                exact_members.setdefault(rep, []).append(i)
        reps = list(first_of.values())  # ascending: dict keeps first-seen order

        pairs: List[Pair] = []
        for rep, members in exact_members.items():
            pairs.extend((rep, m, 1.0) for m in members)
        if self.threshold < 1.0 and len(reps) >= 2:
            for p, q, score in near(reps):
                a, b = reps[p], reps[q]
                pairs.append((min(a, b), max(a, b), float(score)))
        pairs.sort(key=lambda t: (t[0], t[1]))
        logger.debug("near-dupes: %d items, %d distinct, %d pairs", n, len(reps), len(pairs))

        groups = _connected_components(n, pairs)
        representatives = [int(choose(g)) for g in groups]
        grouped = {i for g in groups for i in g}
        keep = sorted(set(range(n)) - grouped | set(representatives))
        return DuplicateResult(
            groups=groups,
            pairs=pairs,
            representatives=representatives,
            keep_indices=keep,
            n_items=n,
            kind=kind,
            threshold=self.threshold,
        )


def _reject_opaque(items: Sequence[Any]) -> None:
    """Refuse items whose ``str()`` is the default ``<Foo object at 0x...>`` form.

    Those strings carry a memory address, so comparing them would compare
    addresses: the result would differ from run to run and group unrelated
    objects that happen to share hex digits. Better to say so than to return a
    confident wrong answer.
    """
    for x in items:
        if type(x).__repr__ is object.__repr__:
            raise TypeError(
                f"cannot compare {type(x).__name__} objects as text: str() of them is a memory "
                "address, so the result would not be reproducible. Give the objects a __repr__ "
                "or __str__, pass a list of strings, or pass a DataFrame / list of dicts with "
                "kind='records'."
            )


def _connected_components(n: int, pairs: Sequence[Pair]) -> List[List[int]]:
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i, j, _ in pairs:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[max(ri, rj)] = min(ri, rj)
    touched = sorted({k for i, j, _ in pairs for k in (i, j)})
    members: Dict[int, List[int]] = {}
    for i in touched:  # ascending, so every group comes out ascending
        members.setdefault(find(i), []).append(i)
    groups = [g for g in members.values() if len(g) >= 2]
    groups.sort(key=lambda g: g[0])
    return groups


def _materialize(items: Any) -> Any:
    """Return ``items`` as something indexable; reject inputs that cannot be a collection of items."""
    if isinstance(items, (str, bytes, os.PathLike)):
        if _records.is_table_path(items):
            return items
        raise TypeError(
            "items must be a list of texts, a DataFrame (or a .csv/.parquet path) or a list of images; "
            f"got a single {type(items).__name__} - wrap it in a list"
        )
    if isinstance(items, (list, tuple, pd.DataFrame, pd.Series, np.ndarray)):
        return items
    if isinstance(items, Iterable):
        return list(items)
    raise TypeError(f"items must be a list of texts, a DataFrame or a list of images, got {type(items).__name__}")


# --------------------------------------------------------------------------- #
# Convenience functions
# --------------------------------------------------------------------------- #
def find_duplicates(
    items: Any,
    *,
    kind: str = "auto",
    threshold: float = 0.85,
    key: Any = None,
    n_gram: int = 3,
) -> DuplicateResult:
    """Find near-duplicates in a list of texts, a DataFrame, or a list of images.

    Args:
        items: ``list[str]`` (text); ``pandas.DataFrame`` or ``.csv``/``.parquet``
            path or list of dicts (records); list of image paths / PIL images
            (images, needs ``pip install near-dupes[images]``).
        kind: ``"auto"`` detects the input type; or force ``"text"``, ``"records"``, ``"images"``.
        threshold: minimum similarity in ``(0, 1]``; ``1.0`` means exact duplicates only.
        key: DataFrame column(s) to compare rows on (default: all columns).
        n_gram: character shingle length for text and records.

    Returns:
        A ``DuplicateResult`` with ``groups``, ``pairs``, ``keep_indices``,
        ``summary()``, ``to_dict()`` and ``dedupe(items)``.
    """
    return DuplicateFinder(kind=kind, threshold=threshold, key=key, n_gram=n_gram).find(items)


def dedupe(items: Any, **kw: Any) -> Any:
    """Return ``items`` without near-duplicates; keyword arguments go to ``find_duplicates``."""
    items = _materialize(items)
    return find_duplicates(items, **kw).dedupe(items)
