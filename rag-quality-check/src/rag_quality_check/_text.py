"""Tokenizing, sentence splitting and the two similarity backends.

Every similarity in this package goes through :class:`Similarity`, so there is
exactly one place where "how alike are these two strings" is decided, and the
report can always name which backend produced a number.
"""
from __future__ import annotations

import re
import unicodedata
from typing import Callable, Dict, FrozenSet, Iterable, List, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "STOPWORDS",
    "Similarity",
    "coverage",
    "cosine",
    "looks_like_passage",
    "split_sentences",
    "tokenize",
]

#: Unicode categories of combining marks: non-spacing and spacing-combining.
_MARK_CATEGORIES = ("Mn", "Mc")


def _escape(point: int) -> str:
    """One code point as a regex escape, wide form above the BMP."""
    if point <= 0xFFFF:
        return "\\u{0:04x}".format(point)
    return "\\U{0:08x}".format(point)


def _combining_mark_class() -> str:
    r"""Character-class body matching every Unicode combining mark.

    Python's ``re`` has no ``\p{M}`` and its ``\w`` does not match marks, so a
    plain ``\w+`` cuts a word apart at every vowel sign: ``"पासवर्ड"`` tokenizes
    as ``प``, ``सवर``, ``ड`` rather than as one word, and the same happens to
    Bengali, Tamil, Thai, pointed Hebrew and vocalized Arabic. The ranges are
    derived from :mod:`unicodedata` instead of hardcoded so they stay correct as
    Python's Unicode tables move; the scan runs once at import and costs about
    ten milliseconds.
    """
    ranges: List[Tuple[int, int]] = []
    start = prev = -2
    for point in range(0x0300, 0x1E950):
        if unicodedata.category(chr(point)) in _MARK_CATEGORIES:
            if point == prev + 1:
                prev = point
            else:
                if start >= 0:
                    ranges.append((start, prev))
                start = prev = point
    if start >= 0:
        ranges.append((start, prev))
    parts: List[str] = []
    for low, high in ranges:
        if low == high:
            parts.append(_escape(low))
        else:
            parts.append(_escape(low) + "-" + _escape(high))
    return "".join(parts)


#: A word is a run of word characters together with any marks that belong to
#: them. See :func:`_combining_mark_class` for why the marks have to be spelled
#: out rather than left to ``\w``.
_WORD = re.compile("[\\w" + _combining_mark_class() + "]+", re.UNICODE)

# Sentence enders. Latin ``. ! ?``, the Urdu full stop and the Devanagari danda
# end a sentence only when whitespace follows, so "3.14" and "sec. 2" stay in
# one piece. The CJK enders need no following space, because Japanese and
# Chinese are not written with spaces between sentences - requiring one would
# make a whole Japanese paragraph a single sentence and quietly flatten
# groundedness, which is measured per sentence. Any run of newlines also ends a
# sentence.
_SENTENCE_SPLIT = re.compile(r"(?<=[。！？])\s*|(?<=[.!?۔।])\s+|\n+")

# Scripts written without spaces: one character is one token.
_CJK_RANGES = (
    (0x3040, 0x30FF),   # hiragana, katakana
    (0x3400, 0x4DBF),   # CJK extension A
    (0x4E00, 0x9FFF),   # CJK unified ideographs
    (0xF900, 0xFAFF),   # CJK compatibility ideographs
    (0xAC00, 0xD7AF),   # hangul syllables
)

#: English function words dropped before comparing texts. Short on purpose: a
#: longer list starts removing words that carry meaning in a technical query.
#: Non-English text keeps all of its words, which only makes matching stricter.
STOPWORDS: FrozenSet[str] = frozenset(
    """a about an and are as at be been being but by can could did do does for from
    had has have he her his how i if in into is it its may might must no not of on or
    our should so than that the their them then there these they this those to us was
    we were what when where which who will with would you your""".split()
)


def _is_cjk(char: str) -> bool:
    point = ord(char)
    return any(low <= point <= high for low, high in _CJK_RANGES)


def _graphemes(word: str) -> List[str]:
    """``word`` as base characters, each carrying the marks that follow it.

    Used only for the per-character CJK split, where naively iterating the
    string would tear a combining voiced sound mark (U+3099, as in か + ゙) off
    the kana it belongs to and make it a token of its own.
    """
    out: List[str] = []
    for char in word:
        if out and unicodedata.category(char) in _MARK_CATEGORIES:
            out[-1] += char
        else:
            out.append(char)
    return out


def tokenize(text: str) -> FrozenSet[str]:
    r"""Content words of ``text`` as a lowercased set.

    A word is a run of ``\w`` characters together with any combining marks that
    belong to them, casefolded; the marks matter, because without them every
    Devanagari, Bengali, Tamil, Thai or vocalized Arabic word would be cut into
    fragments at its vowel signs. Characters from scripts written without
    spaces (Han, kana, Hangul) each become their own token, so CJK text is
    compared character by character instead of as one blob. English stopwords
    are dropped - unless that would empty the set, in which case every token is
    kept, because a query of pure function words still has to be comparable to
    something.
    """
    if not text:
        return frozenset()
    tokens: List[str] = []
    for match in _WORD.finditer(text):
        word = match.group(0).casefold()
        if any(_is_cjk(char) for char in word):
            tokens.extend(_graphemes(word))
        else:
            tokens.append(word)
    if not tokens:
        return frozenset()
    content = [token for token in tokens if token not in STOPWORDS]
    return frozenset(content) if content else frozenset(tokens)


def split_sentences(text: str) -> List[str]:
    """Split ``text`` into sentences on ``. ! ?``, their CJK forms, and newlines.

    Always returns at least one sentence for non-blank text, so a one-line
    answer without punctuation is still measurable.
    """
    if not text or not text.strip():
        return []
    pieces = [piece.strip() for piece in _SENTENCE_SPLIT.split(text)]
    sentences = [piece for piece in pieces if piece]
    return sentences or [text.strip()]


#: Scripts written without spaces between words: a run of these is a sentence,
#: not an identifier. CJK and Hangul from :data:`_CJK_RANGES`, plus Thai, Lao,
#: Tibetan, Myanmar, Khmer and the CJK punctuation block.
_SPACELESS_RANGES = _CJK_RANGES + (
    (0x0E00, 0x0EFF),   # Thai, Lao
    (0x0F00, 0x0FFF),   # Tibetan
    (0x1000, 0x109F),   # Myanmar
    (0x1780, 0x17FF),   # Khmer
    (0x3000, 0x303F),   # CJK symbols and punctuation
)

#: A string with no whitespace in it that is longer than this is prose, not an
#: id: no realistic identifier runs this long, and a spaceless script this
#: package does not know by name still gets treated as text.
_ID_MAX_CHARS = 40


def looks_like_passage(text: str) -> bool:
    """True when ``text`` reads as passage prose rather than a bare id.

    ``"p_reset"``, ``"doc-42"``, ``"7f3c9b"`` and a bare UUID are identifiers:
    one unbroken run with no prose in it for an answer to be checked against. A
    string counts as prose when it contains whitespace, or when it is written in
    a script that has no spaces between words (Han, kana, Hangul, Thai, Lao,
    Tibetan, Myanmar, Khmer).

    Length alone is NOT evidence of prose. This used to treat anything over 40
    characters as a passage "which no identifier does" - but a source URL and a
    sha256 chunk id both run well past that, so passing either in ``retrieved``
    scored the answer against a bare id, and every such case was reported as an
    ungrounded answer when nothing was wrong. A long unbroken run is now checked
    for the shapes ids actually take.

    This never changes a number. It decides only whether ``groundedness`` and
    ``citation_coverage`` can be measured at all, because scoring an answer
    against the string ``"p_reset"`` reports a hallucination that did not
    happen.
    """
    flat = text.strip()
    if not flat:
        return False
    if any(char.isspace() for char in flat):
        return True
    for char in flat:
        point = ord(char)
        if any(low <= point <= high for low, high in _SPACELESS_RANGES):
            return True
    lowered = flat.lower()
    if lowered.startswith(("http://", "https://", "s3://", "gs://", "file://", "urn:", "doc:")):
        return False
    if len(lowered) >= 16 and all(char in "0123456789abcdef" for char in lowered):
        return False                                   # a hex digest
    if "/" in flat or "\\" in flat:
        return False                                   # a path or key
    if flat.count("-") + flat.count("_") >= 2 and " " not in flat:
        return False                                   # a separator-joined key
    return len(flat) > _ID_MAX_CHARS


def coverage(query_tokens: FrozenSet[str], passage_tokens: FrozenSet[str]) -> float:
    """Share of ``query_tokens`` that also appear in ``passage_tokens``.

    ``coverage(A, B) = |A intersect B| / |A|``, and ``0.0`` when ``A`` is empty.
    Deliberately asymmetric: it asks how much of the first text the second one
    accounts for, which is the question both "is this passage about the query"
    and "is this sentence supported by this passage" are really asking. A long
    passage is therefore not punished for also containing other material.
    """
    if not query_tokens:
        return 0.0
    return len(query_tokens & passage_tokens) / len(query_tokens)


def cosine(left: np.ndarray, right: np.ndarray) -> float:
    """Cosine similarity of two vectors, negatives clipped to ``0.0``.

    ``cosine(u, v) = max(0, (u . v) / (|u| |v|))``, and ``0.0`` if either vector
    has zero length. Clipping keeps every reported metric inside ``0..1``.
    """
    left_norm = float(np.linalg.norm(left))
    right_norm = float(np.linalg.norm(right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    value = float(np.dot(left, right) / (left_norm * right_norm))
    if value != value:  # NaN guard for degenerate vectors
        return 0.0
    return max(0.0, min(1.0, value))


class Similarity:
    """One similarity function, backed either by ``embed`` or by word overlap.

    ``embed`` is any ``callable(list[str]) -> vectors`` (a sentence-transformer
    ``.encode``, an API client, anything). It is called in batches and every
    vector is cached by its exact text, so a passage shared between cases is
    embedded once per run. With no ``embed``, similarity is :func:`coverage`
    over content words.
    """

    def __init__(self, embed: Optional[Callable[[List[str]], Sequence]] = None) -> None:
        if embed is not None and not callable(embed):
            raise ValueError("embed must be a callable(list[str]) -> vectors, or None")
        self._embed = embed
        self._vectors: Dict[str, np.ndarray] = {}
        self._tokens: Dict[str, FrozenSet[str]] = {}

    @property
    def kind(self) -> str:
        """``"embeddings"`` or ``"lexical overlap"`` - what produced the numbers."""
        return "embeddings" if self._embed is not None else "lexical overlap"

    @property
    def uses_embeddings(self) -> bool:
        """True when an ``embed`` callable is doing the work."""
        return self._embed is not None

    def tokens(self, text: str) -> FrozenSet[str]:
        """Cached :func:`tokenize`."""
        cached = self._tokens.get(text)
        if cached is None:
            cached = tokenize(text)
            self._tokens[text] = cached
        return cached

    def warm(self, texts: Iterable[str]) -> None:
        """Embed every text not already cached, in one call to ``embed``.

        Doing this once per run rather than once per case is what keeps a real
        model usable here: 1000 cases become a single batched call.
        """
        if self._embed is None:
            return
        wanted: List[str] = []
        seen = set()
        for text in texts:
            if text and text not in self._vectors and text not in seen:
                seen.add(text)
                wanted.append(text)
        if not wanted:
            return
        matrix = np.asarray(self._embed(wanted), dtype=float)
        if matrix.ndim == 1 and len(wanted) == 1:
            matrix = matrix.reshape(1, -1)
        if matrix.ndim != 2 or matrix.shape[0] != len(wanted):
            raise ValueError(
                "embed must return one vector per input text: got shape "
                "{0!r} for {1} texts".format(matrix.shape, len(wanted))
            )
        for text, vector in zip(wanted, matrix):
            self._vectors[text] = vector

    def score(self, left: str, right: str) -> float:
        """Similarity of ``left`` to ``right``, in ``0..1``.

        With ``embed`` this is :func:`cosine`; otherwise :func:`coverage`, which
        reads as "share of ``left``'s content words that ``right`` contains".
        """
        if not left or not right:
            return 0.0
        if self._embed is not None:
            self.warm([left, right])
            return cosine(self._vectors[left], self._vectors[right])
        return coverage(self.tokens(left), self.tokens(right))

    def best(self, left: str, candidates: Sequence[str]) -> float:
        """Highest :meth:`score` of ``left`` against any candidate, ``0.0`` if none."""
        if not candidates:
            return 0.0
        return max(self.score(left, candidate) for candidate in candidates)
