"""Turn raw text into the canonical tokens and features everything else compares.

Pipeline: NFKC + casefold + whitespace collapse -> word split -> phrase and
synonym canonicalization -> stopword removal -> light suffix stemming.
Features are word n-grams plus character n-grams *of the canonical string*, so
two differently worded sentences that canonicalize alike also share characters.
"""
from __future__ import annotations

import re
import unicodedata
from typing import Dict, List, Sequence, Tuple

from ._lexicon import ABSORBED_VERBS, PHRASES, STOPWORDS, SYNONYMS

__all__ = ["normalize", "canonical_tokens", "canonical_string", "feature_counts"]

_WORD_RE = re.compile(r"[^\W_]+", re.UNICODE)
_WS_RE = re.compile(r"\s+")

_MAX_PHRASE = max((len(key.split()) for key in PHRASES), default=1)

# Longest suffix first; a rule only fires if it leaves at least 3 characters.
_SUFFIXES: Tuple[Tuple[str, str], ...] = (
    ("ational", "ate"),
    ("ization", "ize"),
    ("isation", "ize"),
    ("iveness", "ive"),
    ("fulness", "ful"),
    ("ousness", "ous"),
    ("ements", ""),
    ("ement", ""),
    ("ments", ""),
    ("ment", ""),
    ("ness", ""),
    ("ingly", ""),
    ("edly", ""),
    ("sses", "ss"),
    ("ies", "y"),
    ("ing", ""),
    ("ed", ""),
    ("ly", ""),
    ("es", ""),
    ("s", ""),
)


def normalize(text: str) -> str:
    """Casefold, NFKC-normalize and collapse whitespace.

    Two strings with the same ``normalize()`` output count as exact duplicates.
    """
    folded = unicodedata.normalize("NFKC", text).casefold()
    return _WS_RE.sub(" ", unicodedata.normalize("NFKC", folded)).strip()


def stem(word: str) -> str:
    """Strip common English inflections. Crude on purpose, and deterministic.

    Only alphabetic tokens are stemmed. Numbers and mixed alphanumeric
    identifiers ("1000", "2022", "sku12") are English-morphology-free and are
    returned untouched, so two passages that differ only by an amount or an id
    keep differing after canonicalization.
    """
    if len(word) <= 3 or not word.isalpha():
        return word
    out = word
    for suffix, replacement in _SUFFIXES:
        if out.endswith(suffix):
            candidate = out[: len(out) - len(suffix)] + replacement
            if len(candidate) >= 3:
                out = candidate
            break
    if len(out) > 3 and out.endswith("e"):
        out = out[:-1]
    if len(out) > 3 and out[-1] == out[-2] and out[-1] not in "lsfz":
        out = out[:-1]
    return out


def _apply_phrases(words: Sequence[str]) -> List[str]:
    """Replace known multi-word phrases with their canonical single token."""
    out: List[str] = []
    i = 0
    n = len(words)
    while i < n:
        matched = False
        for size in range(min(_MAX_PHRASE, n - i), 1, -1):
            phrase = " ".join(words[i : i + size])
            canonical = PHRASES.get(phrase)
            if canonical is not None:
                out.append(canonical)
                i += size
                matched = True
                break
        if not matched:
            out.append(words[i])
            i += 1
    return out


def canonical_tokens(text: str) -> List[str]:
    """Canonical content tokens for ``text`` (may be empty for empty input)."""
    words = _WORD_RE.findall(normalize(text))
    if not words:
        return []
    words = _apply_phrases(words)
    words = [SYNONYMS.get(word, word) for word in words]
    absorbed: set = set()
    present = set(words)
    for canonical, verbs in ABSORBED_VERBS.items():
        if canonical in present:
            absorbed |= verbs
    if absorbed:
        words = [word for word in words if word not in absorbed] or words
    kept = [word for word in words if word not in STOPWORDS]
    if not kept:
        # All stopwords ("the", "of course"). Keep them rather than emit an
        # empty, meaningless vector.
        kept = words
    return [stem(word) for word in kept]


def canonical_string(text: str) -> str:
    """The canonical tokens of ``text`` joined by single spaces."""
    return " ".join(canonical_tokens(text))


#: How much a token carrying a digit is worth relative to a word.
#: An identifier or an amount is the most distinguishing thing in a passage,
#: but its character n-grams overlap heavily with those of a near miss
#: ("1000" literally contains "100"), so the word-level evidence that the two
#: differ gets outvoted by the characters they share. Counting such a token
#: twice, and emitting it once more whole - a feature only an equal token can
#: match - keeps two passages that differ by an amount apart, and pulls two
#: that quote the same reference together.
_NUMERIC_WEIGHT = 2


def _has_digit(token: str) -> bool:
    """True for numbers and mixed identifiers: ``1000``, ``2022``, ``sku12``."""
    return any(char.isdigit() for char in token)


def _char_ngrams(text: str, low: int, high: int) -> List[str]:
    grams: List[str] = []
    length = len(text)
    for size in range(low, high + 1):
        if size <= 0 or length < size:
            continue
        for start in range(length - size + 1):
            grams.append(text[start : start + size])
    return grams


def feature_counts(
    text: str,
    word_ngram: Tuple[int, int] = (1, 2),
    char_ngram: Tuple[int, int] = (3, 4),
) -> Dict[str, int]:
    """Count the features of one text: word n-grams and canonical char n-grams.

    Tokens carrying a digit are additionally emitted whole and weighted by
    :data:`_NUMERIC_WEIGHT`, so a passage is not confused with the same passage
    quoting a different amount or ticket number.

    Returns an empty dict only for text with no word characters at all.
    """
    tokens = canonical_tokens(text)
    counts: Dict[str, int] = {}
    if not tokens:
        return counts
    low, high = word_ngram
    for size in range(max(1, low), max(1, high) + 1):
        if len(tokens) < size:
            continue
        for start in range(len(tokens) - size + 1):
            key = "w:" + " ".join(tokens[start : start + size])
            counts[key] = counts.get(key, 0) + 1
    numeric: Dict[str, int] = {}
    for token in tokens:
        if _has_digit(token):
            numeric[token] = numeric.get(token, 0) + 1
    for token, occurrences in numeric.items():
        key = "w:" + token
        if key in counts:
            counts[key] *= _NUMERIC_WEIGHT
        counts["n:" + token] = _NUMERIC_WEIGHT * occurrences
    joined = " ".join(tokens)
    for gram in _char_ngrams(joined, char_ngram[0], char_ngram[1]):
        key = "c:" + gram
        counts[key] = counts.get(key, 0) + 1
    return counts
