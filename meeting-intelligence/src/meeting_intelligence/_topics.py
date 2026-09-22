"""Split a meeting into topic segments and label each one.

The method is a hashed bag-of-words vector per turn, smoothed over a small
window, cut where the cosine similarity across a boundary dips into a local
valley. Labels come from the words that are distinctive to a segment relative to
the rest of the meeting. ``numpy`` does the arithmetic; nothing is downloaded and
no model is involved.
"""
from __future__ import annotations

import re
import zlib
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ._parse import Turn
from ._text import label_words, title_case, tokenize, truncate

_ACRONYM = re.compile(r"\b[A-Z][A-Z0-9]{1,4}\b")

__all__ = ["Topic", "segment_topics"]

_HASH_DIMENSIONS = 512
_SMOOTHING_WINDOW = 2
_MIN_TURNS_PER_TOPIC = 3
_TARGET_TURNS_PER_TOPIC = 9
_MAX_TOPICS = 8
_KEYWORDS_PER_TOPIC = 3


@dataclass
class Topic:
    """One stretch of the discussion, labelled by its distinctive words."""

    label: str
    keywords: List[str] = field(default_factory=list)
    start_turn: int = 0
    end_turn: int = 0
    start: Optional[float] = None
    end: Optional[float] = None
    n_turns: int = 0
    speakers: List[str] = field(default_factory=list)
    preview: str = ""

    @property
    def duration(self) -> Optional[float]:
        """Seconds spent on this topic, when the transcript was timed."""
        if self.start is None or self.end is None:
            return None
        return max(0.0, self.end - self.start)

    def to_dict(self) -> Dict[str, Any]:
        """JSON-safe view of the topic."""
        return {
            "label": self.label,
            "keywords": list(self.keywords),
            "start_turn": self.start_turn,
            "end_turn": self.end_turn,
            "start": self.start,
            "end": self.end,
            "duration": self.duration,
            "n_turns": self.n_turns,
            "speakers": list(self.speakers),
            "preview": self.preview,
        }


def _hash_token(token: str) -> int:
    """Deterministic bucket for a token, identical on every machine and run."""
    return zlib.crc32(token.encode("utf-8")) % _HASH_DIMENSIONS


def _turn_vectors(token_lists: Sequence[Sequence[str]]) -> np.ndarray:
    """L2-normalised hashed bag-of-words matrix, one row per turn."""
    matrix = np.zeros((len(token_lists), _HASH_DIMENSIONS), dtype=np.float32)
    for row, tokens in enumerate(token_lists):
        for token in tokens:
            matrix[row, _hash_token(token)] += 1.0
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return matrix / norms


def _boundary_similarity(vectors: np.ndarray, window: int) -> np.ndarray:
    """Cosine similarity across each turn boundary, using a window either side."""
    count = vectors.shape[0]
    if count < 2:
        return np.zeros(0, dtype=np.float32)
    cumulative = np.vstack(
        [np.zeros((1, vectors.shape[1]), dtype=np.float32), np.cumsum(vectors, axis=0)]
    )
    similarities = np.zeros(count - 1, dtype=np.float32)
    for boundary in range(count - 1):
        left_from = max(0, boundary + 1 - window)
        left = cumulative[boundary + 1] - cumulative[left_from]
        right_to = min(count, boundary + 1 + window)
        right = cumulative[right_to] - cumulative[boundary + 1]
        left_norm = float(np.linalg.norm(left))
        right_norm = float(np.linalg.norm(right))
        if left_norm == 0.0 or right_norm == 0.0:
            similarities[boundary] = 1.0
        else:
            similarities[boundary] = float(np.dot(left, right) / (left_norm * right_norm))
    return similarities


def _pick_cuts(similarities: np.ndarray, wanted: int) -> List[int]:
    """Choose boundary positions at the deepest local valleys."""
    if wanted <= 0 or similarities.size == 0:
        return []
    valleys = []  # type: List[Tuple[float, int]]
    for position in range(similarities.size):
        value = float(similarities[position])
        left = float(similarities[position - 1]) if position > 0 else float("inf")
        right = float(similarities[position + 1]) if position + 1 < similarities.size else float("inf")
        if value <= left and value <= right:
            valleys.append((value, position))
    if not valleys:
        order = np.argsort(similarities)
        valleys = [(float(similarities[index]), int(index)) for index in order]
    valleys.sort(key=lambda item: (item[0], item[1]))
    chosen = []  # type: List[int]
    for _value, position in valleys:
        if len(chosen) >= wanted:
            break
        # Keep segments from collapsing to one or two turns.
        if any(abs(position - taken) < _MIN_TURNS_PER_TOPIC for taken in chosen):
            continue
        if position + 1 < _MIN_TURNS_PER_TOPIC:
            continue
        if similarities.size - position < _MIN_TURNS_PER_TOPIC:
            continue
        chosen.append(position)
    return sorted(chosen)


def _distinctive_words(
    segment_counts: Counter,
    total_counts: Counter,
    used: set,
    limit: int = _KEYWORDS_PER_TOPIC,
) -> List[str]:
    """Words that are common in this segment relative to the whole meeting."""
    segment_total = sum(segment_counts.values())
    overall_total = sum(total_counts.values())
    if segment_total == 0 or overall_total == 0:
        return []
    scored = []  # type: List[Tuple[float, int, str]]
    for word, count in segment_counts.items():
        if count < 1:
            continue
        in_segment = count / segment_total
        elsewhere = max(total_counts.get(word, 0) - count, 0)
        elsewhere_total = max(overall_total - segment_total, 1)
        background = (elsewhere / elsewhere_total) if elsewhere_total else 0.0
        # Add-one smoothing keeps a word seen only here from scoring infinitely.
        score = in_segment / (background + (1.0 / overall_total))
        penalty = 0.45 if word in used else 1.0
        scored.append((score * penalty * (1.0 + 0.12 * min(count, 6)), count, word))
    scored.sort(key=lambda item: (-item[0], -item[1], item[2]))
    picked = []  # type: List[str]
    for _score, _count, word in scored:
        if word in picked:
            continue
        picked.append(word)
        if len(picked) >= limit:
            break
    return picked


def segment_topics(turns: Sequence[Turn]) -> Tuple[List[Topic], List[str]]:
    """Split ``turns`` into topics; returns ``(topics, warnings)``.

    A meeting too short or too uniform to cut comes back as a single topic with
    a warning saying so, rather than as an empty list.
    """
    warnings = []  # type: List[str]
    if not turns:
        return [], warnings

    # A name is who was talking, not what about; dropping the roster keeps
    # "Alice" out of a topic label that should read "Event Store Migration".
    names = set()  # type: set
    for turn in turns:
        if turn.speaker:
            names.update(tokenize(turn.speaker))
    token_lists = [
        [token for token in label_words(turn.text) if token not in names] for turn in turns
    ]
    casing = _acronym_casing(turns)
    non_empty = sum(1 for tokens in token_lists if tokens)

    if len(turns) < _MIN_TURNS_PER_TOPIC * 2 or non_empty < 2:
        if len(turns) > 1:
            warnings.append(
                "transcript too short to split into topics; reported as a single topic"
            )
        return (
            [_build_topic(turns, 0, len(turns) - 1, token_lists, Counter(), casing=casing)],
            warnings,
        )

    vectors = _turn_vectors(token_lists)
    similarities = _boundary_similarity(vectors, _SMOOTHING_WINDOW)
    wanted = int(np.clip(len(turns) // _TARGET_TURNS_PER_TOPIC, 1, _MAX_TOPICS)) - 1
    cuts = _pick_cuts(similarities, wanted)

    if not cuts:
        warnings.append(
            "no clear topic shift found; the whole discussion is reported as one topic"
        )

    total_counts = Counter()  # type: Counter
    for tokens in token_lists:
        total_counts.update(tokens)

    bounds = []  # type: List[Tuple[int, int]]
    previous = 0
    for cut in cuts:
        bounds.append((previous, cut))
        previous = cut + 1
    bounds.append((previous, len(turns) - 1))

    topics = []  # type: List[Topic]
    used = set()  # type: set
    for first, last in bounds:
        if first > last:
            continue
        topic = _build_topic(turns, first, last, token_lists, total_counts, used, casing)
        used.update(topic.keywords)
        topics.append(topic)
    return topics, warnings


def _acronym_casing(turns: Sequence[Turn]) -> Dict[str, str]:
    """Map a lower-cased token to its all-capitals spelling, when it had one.

    Tokenizing lower-cases everything, so without this a label built from "PR"
    reads "Pr". Only short all-capitals words are collected, which is what an
    acronym looks like in a transcript.
    """
    capitals = {}  # type: Dict[str, str]
    for turn in turns:
        for match in _ACRONYM.finditer(turn.text):
            word = match.group(0)
            capitals.setdefault(word.lower(), word)
    return capitals


def _build_topic(
    turns: Sequence[Turn],
    first: int,
    last: int,
    token_lists: Sequence[Sequence[str]],
    total_counts: Counter,
    used: Optional[set] = None,
    casing: Optional[Dict[str, str]] = None,
) -> Topic:
    """Assemble one :class:`Topic` for the turn range ``first..last``."""
    used = used if used is not None else set()
    segment_counts = Counter()  # type: Counter
    for position in range(first, last + 1):
        segment_counts.update(token_lists[position])
    if not total_counts:
        total_counts = segment_counts
    keywords = _distinctive_words(segment_counts, total_counts, used)
    label = title_case(keywords, casing) if keywords else "General discussion"

    speakers = []  # type: List[str]
    for position in range(first, last + 1):
        name = turns[position].speaker
        if name and name not in speakers:
            speakers.append(name)

    starts = [turns[position].start for position in range(first, last + 1)]
    starts = [value for value in starts if value is not None]
    ends = [turns[position].end for position in range(first, last + 1)]
    ends = [value for value in ends if value is not None]
    tail_start = turns[last].start

    longest = max(
        range(first, last + 1), key=lambda position: len(turns[position].text)
    )
    return Topic(
        label=label,
        keywords=keywords,
        start_turn=first,
        end_turn=last,
        start=min(starts) if starts else None,
        end=max(ends) if ends else (tail_start if tail_start is not None else None),
        n_turns=last - first + 1,
        speakers=speakers,
        preview=truncate(turns[longest].text, 110),
    )
