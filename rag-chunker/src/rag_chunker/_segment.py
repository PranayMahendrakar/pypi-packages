"""Segmentation primitives.

Everything here works with *units*: half-open character spans that tile the
source text end to end with no gaps and no overlaps. Chunk boundaries are only
ever placed on unit boundaries, which is what makes
``"".join(chunk bodies) == source`` true by construction.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, FrozenSet, Iterable, List, Sequence, Set, Tuple

from ._source import Source

CostFn = Callable[[str], int]

# Scripts written without spaces (CJK, kana) count one character as one word,
# otherwise a whole Japanese page would look like a single unsplittable word.
CJK_RANGES = "぀-ヿ㐀-䶿一-鿿豈-﫿ｦ-ﾟ"
WORD_RE = re.compile("[^\\s" + CJK_RANGES + "]+|[" + CJK_RANGES + "]")
_WORD_RE = WORD_RE
_PARA_BREAK = re.compile(r"\n[ \t]*\n\s*")


def count_words(text: str) -> int:
    """Words in ``text``; one CJK or kana character counts as one word."""
    return len(WORD_RE.findall(text))

_SENT_END = re.compile("[.!?…。！？]+[\\)\\]\"'’”」』]*")

# Words that end in a period without ending a sentence.
_ABBREVIATIONS = frozenset(
    """mr mrs ms dr prof sr jr st vs etc eg ie fig figs no nos al approx inc ltd co
    corp dept est min max vol ed eds pp cf viz ca resp ref refs sec secs chap chaps
    univ dept jan feb mar apr jun jul aug sep sept oct nov dec mon tue wed thu fri
    sat sun""".split()
)


@dataclass
class Unit:
    """One indivisible span of the source text."""

    start: int
    end: int
    atomic: bool
    cost: int


# --------------------------------------------------------------------------- #
# cut points
# --------------------------------------------------------------------------- #
def word_spans(text: str, start: int, end: int) -> List[Tuple[int, int]]:
    """Spans tiling ``text[start:end]``, one per word, whitespace attached to the left."""
    if end <= start:
        return []
    starts = [m.start() for m in _WORD_RE.finditer(text, start, end)]
    if not starts:
        return [(start, end)]
    starts[0] = start
    spans: List[Tuple[int, int]] = []
    for i, s in enumerate(starts):
        spans.append((s, starts[i + 1] if i + 1 < len(starts) else end))
    return spans


def word_cuts(text: str) -> Set[int]:
    """Offsets where a word begins (never 0)."""
    return {m.start() for m in _WORD_RE.finditer(text) if m.start() > 0}


def paragraph_cuts(src: Source) -> Set[int]:
    """Offsets after a blank line, plus every heading boundary."""
    text = src.text
    cuts = {m.end() for m in _PARA_BREAK.finditer(text)}
    cuts |= _heading_cuts(src)
    return cuts


def _heading_cuts(src: Source) -> Set[int]:
    text = src.text
    cuts: Set[int] = set()
    for offset in src.heading_offsets:
        cuts.add(offset)
        newline = text.find("\n", offset)
        cuts.add(len(text) if newline < 0 else newline + 1)
    return cuts


def _is_abbreviation(text: str, punct_index: int) -> bool:
    if text[punct_index] != ".":
        return False
    start = punct_index
    while start > 0 and text[start - 1].isalpha():
        start -= 1
    word = text[start:punct_index]
    if not word:
        return False
    if len(word) == 1 and word.isupper():
        return True
    return word.lower() in _ABBREVIATIONS


def sentence_cuts(src: Source) -> Set[int]:
    """Offsets where a new sentence begins.

    Trailing whitespace belongs to the sentence it follows, so the spans tile
    the text exactly.
    """
    text = src.text
    length = len(text)
    cuts = paragraph_cuts(src)
    for match in _SENT_END.finditer(text):
        end = match.end()
        if end < length and not text[end].isspace():
            continue
        if _is_abbreviation(text, match.start()):
            continue
        cursor = end
        while cursor < length and text[cursor].isspace():
            cursor += 1
        if 0 < cursor < length:
            cuts.add(cursor)
    return cuts


# --------------------------------------------------------------------------- #
# units
# --------------------------------------------------------------------------- #
def _inside_atomic(atomic: Sequence[Tuple[int, int]], offset: int) -> bool:
    return any(s < offset < e for s, e in atomic)


def _is_atomic_span(atomic: Sequence[Tuple[int, int]], start: int, end: int) -> bool:
    return any(s <= start and end <= e for s, e in atomic)


def units_from_cuts(
    text: str,
    cuts: Iterable[int],
    atomic: Sequence[Tuple[int, int]],
    cost_fn: CostFn,
) -> List[Unit]:
    """Build units from cut offsets, forcing atomic blocks to stay whole."""
    length = len(text)
    valid = {c for c in cuts if 0 < c < length and not _inside_atomic(atomic, c)}
    for start, end in atomic:
        if 0 < start < length:
            valid.add(start)
        if 0 < end < length:
            valid.add(end)
    bounds = [0] + sorted(valid) + [length]
    units: List[Unit] = []
    for i in range(len(bounds) - 1):
        start, end = bounds[i], bounds[i + 1]
        if end <= start:
            continue
        units.append(
            Unit(start, end, _is_atomic_span(atomic, start, end), cost_fn(text[start:end]))
        )
    if not units:
        units.append(Unit(0, length, False, cost_fn(text)))
    return units


def expand_with_cuts(
    units: Sequence[Unit],
    text: str,
    size: int,
    cuts: Iterable[int],
    atomic: Sequence[Tuple[int, int]],
    cost_fn: CostFn,
) -> List[Unit]:
    """Break any unit larger than ``size`` at the given finer cut points."""
    ordered = sorted(c for c in cuts if not _inside_atomic(atomic, c))
    out: List[Unit] = []
    for unit in units:
        if unit.cost <= size or unit.atomic:
            out.append(unit)
            continue
        inner = [c for c in ordered if unit.start < c < unit.end]
        if not inner:
            out.append(unit)
            continue
        bounds = [unit.start] + inner + [unit.end]
        for i in range(len(bounds) - 1):
            start, end = bounds[i], bounds[i + 1]
            out.append(
                Unit(start, end, _is_atomic_span(atomic, start, end), cost_fn(text[start:end]))
            )
    return out


def expand_to_words(
    units: Sequence[Unit], text: str, size: int, cost_fn: CostFn
) -> List[Unit]:
    """Last resort: split oversized non-atomic units at word boundaries."""
    out: List[Unit] = []
    for unit in units:
        if unit.cost <= size or unit.atomic:
            out.append(unit)
            continue
        spans = word_spans(text, unit.start, unit.end)
        if len(spans) <= 1:
            out.append(unit)
            continue
        for start, end in spans:
            out.append(Unit(start, end, False, cost_fn(text[start:end])))
    return out


# --------------------------------------------------------------------------- #
# packing
# --------------------------------------------------------------------------- #
def pack(
    units: Sequence[Unit],
    size: int,
    forced: FrozenSet[int] = frozenset(),
    soft: FrozenSet[int] = frozenset(),
    min_fill: int = 0,
) -> List[Tuple[int, int]]:
    """Group unit indices into ``(start, end)`` runs that stay within ``size``.

    ``forced`` indices always start a new group; ``soft`` indices start one only
    once the current group holds at least ``min_fill``.
    """
    groups: List[Tuple[int, int]] = []
    start = 0
    total = 0
    for i, unit in enumerate(units):
        if i > start:
            hard = total + unit.cost > size
            if i in forced or hard or (i in soft and total >= min_fill):
                groups.append((start, i))
                start = i
                total = 0
        total += unit.cost
    groups.append((start, len(units)))
    return groups


def overlap_start(
    units: Sequence[Unit],
    group_start: int,
    overlap: int,
    text: str,
    cost_fn: CostFn,
    floor: int = 0,
) -> int:
    """Where a chunk's text should begin so it repeats up to ``overlap`` of the last one.

    ``floor`` is the index of the first unit of the *previous* group. The walk
    back never consumes it, so a chunk can only ever repeat the chunk directly
    before it and can never swallow that chunk whole. Without this clamp, a run
    of sections shorter than ``overlap`` (an API reference, an FAQ, a changelog)
    lets the walk sail past several earlier boundaries at once and re-absorb
    them, producing chunks that are strict prefixes of one another.
    """
    core = units[group_start].start
    if overlap <= 0 or group_start == 0:
        return core
    limit = max(0, min(int(floor), group_start - 1))
    taken = 0
    start = core
    index = group_start - 1
    while index > limit:
        unit = units[index]
        if taken + unit.cost > overlap:
            break
        taken += unit.cost
        start = unit.start
        index -= 1
    if taken == 0:
        previous = units[group_start - 1]
        # The previous group's own first unit may only be repeated in part,
        # otherwise this chunk would contain the whole of the previous one.
        keep_some = group_start - 1 == limit
        if not previous.atomic:
            for word_start, _ in reversed(word_spans(text, previous.start, previous.end)):
                if keep_some and word_start <= previous.start:
                    break
                if cost_fn(text[word_start:previous.end]) > overlap:
                    break
                start = word_start
    return start
