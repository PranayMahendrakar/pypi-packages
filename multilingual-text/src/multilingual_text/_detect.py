"""Language detection from character scripts and common words.

The method, in full, so there are no surprises:

1. Every character that carries a script is counted.  The script with the most
   characters is the *dominant* script and it decides which languages are even
   on the table - Thai text is never going to be Polish.
2. Within that script the candidates are scored on how much of the text is made
   of their function words (a word several languages share counts for less),
   on letters that give a language away (``ñ``, ``ß``, ``ł``, Ukrainian ``і``),
   and against letters the orthography does not use.
3. The scores are turned into probabilities that sum to one, and then scaled
   down for short text, because four letters are not evidence.

No model, no download, no network, no non-standard-library dependency.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Sequence, Tuple

from ._normalize import fold_marks
from ._scripts import (
    COMMON,
    RTL_SCRIPTS,
    UNKNOWN,
    char_script,
    dominant_script,
    is_mixed,
    is_script_char,
    rtl_ratio,
    script_counts,
    script_shares,
)
from ._words import (
    ALLOWED_MARKS,
    CANDIDATES,
    DISTINCTIVE,
    EXCLUDED,
    LANGUAGE_NAMES,
    MARKER_WEIGHTS,
    WORD_WEIGHTS,
)

__all__ = ["Detection", "UNDETERMINED", "detect", "detect_batch", "is_rtl", "script_of"]

#: The code returned when there is nothing to go on.
UNDETERMINED = "und"

# How hard each piece of evidence pushes.  Tuned so that one ordinary sentence
# lands above 0.8 and a bare word stays well below it.
_WORD_WEIGHT = 3.5
_DISTINCTIVE_WEIGHT = 1.0
_PENALTY_WEIGHT = 1.5
_SHARPNESS = 5.0
_EXPONENT_CAP = 30.0

# Short text is scored the same way but believed less.
_FULL_EVIDENCE_AT = 24
_EVIDENCE_FLOOR = 0.35

_MIN_LETTERS_FOR_RELIABLE = 12
_MIN_CONFIDENCE_FOR_RELIABLE = 0.60
_MAX_CONFIDENCE = 0.99

# Folded once at import so a token and a table entry meet in the same shape.
_FOLDED_WORD_WEIGHTS: Dict[str, Dict[str, float]] = {
    lang: {fold_marks(word): weight for word, weight in table.items()}
    for lang, table in WORD_WEIGHTS.items()
}


@dataclass(frozen=True)
class Detection:
    """What :func:`detect` found, and how much of it to believe.

    Attributes:
        language: ISO 639-1 code, or ``"und"`` when there was nothing to go on.
        name: the language's English name, for printing.
        confidence: 0.0 to 1.0.  It already accounts for how short the text is,
            so a correct guess from three letters still scores low.
        script: the dominant writing system, e.g. ``"Latin"``, ``"Devanagari"``.
        alternatives: the runners-up as ``(code, confidence)``, highest first.
            ``detect(text, top=3)`` fills this with two entries; the default
            ``top=1`` leaves it empty.
        reliable: ``True`` only when the text is long enough and the winner is
            far enough ahead to act on without a human looking.  Check this
            before you branch on ``language``.
        mixed: ``True`` when the text really is written in two or more scripts.
            Japanese kanji plus kana does not count as mixed.
        scripts: every script found, as fractions of the text, largest first.
        characters: how many script-carrying characters the text had.
        words: how many word-like tokens were found (0 for Chinese, Japanese,
            Korean and Thai, which are scored on character sequences instead).
    """

    language: str
    name: str
    confidence: float
    script: str
    reliable: bool
    mixed: bool = False
    alternatives: List[Tuple[str, float]] = field(default_factory=list)
    scripts: Dict[str, float] = field(default_factory=dict)
    characters: int = 0
    words: int = 0

    @property
    def ranked(self) -> List[Tuple[str, float]]:
        """The winner followed by the alternatives, as ``(code, confidence)``."""
        return [(self.language, self.confidence)] + list(self.alternatives)

    @property
    def rtl(self) -> bool:
        """True when the dominant script is written right to left."""
        return self.script in RTL_SCRIPTS

    def summary(self) -> str:
        """One line of plain ASCII, safe for any console."""
        if self.language == UNDETERMINED:
            return (
                "multilingual-text: undetermined (confidence 0.00, script %s, "
                "%d characters) - not enough to go on" % (self.script, self.characters)
            )
        parts = [
            "multilingual-text: %s (%s), confidence %.2f, script %s"
            % (self.language, self.name, self.confidence, self.script)
        ]
        parts.append("reliable" if self.reliable else "NOT reliable, treat as a guess")
        if self.mixed:
            shown = ", ".join(
                "%s %.0f%%" % (name, share * 100)
                for name, share in list(self.scripts.items())[:3]
            )
            parts.append("mixed scripts (%s)" % shown)
        if self.alternatives:
            parts.append(
                "next: "
                + ", ".join("%s %.2f" % (code, conf) for code, conf in self.alternatives)
            )
        return "; ".join(parts)

    def to_dict(self) -> Dict[str, object]:
        """JSON-safe view of the result."""
        return {
            "language": self.language,
            "name": self.name,
            "confidence": self.confidence,
            "script": self.script,
            "reliable": self.reliable,
            "mixed": self.mixed,
            "rtl": self.rtl,
            "alternatives": [list(item) for item in self.alternatives],
            "scripts": dict(self.scripts),
            "characters": self.characters,
            "words": self.words,
        }

    def __str__(self) -> str:  # pragma: no cover - convenience
        return self.summary()


def _unknown(script: str, characters: int, scripts: Dict[str, float]) -> Detection:
    return Detection(
        language=UNDETERMINED,
        name=LANGUAGE_NAMES[UNDETERMINED],
        confidence=0.0,
        script=script,
        reliable=False,
        mixed=False,
        alternatives=[],
        scripts=scripts,
        characters=characters,
        words=0,
    )


# Machine addresses, which are not written in any language.  Deliberately only
# the shapes that cannot be mistaken for prose: a scheme, a bare "www." host,
# or an e-mail address.  A bare "example.com" is left alone, because the same
# pattern matches "e.g" and half the abbreviations in ordinary writing.
_MACHINE_ADDRESS = re.compile(
    r"""
      [A-Za-z][A-Za-z0-9+.\-]*://\S+     # scheme://host/path?query
    | \bwww\.\S+                         # www.host/path
    | [^\s@]+@[^\s@]+\.[A-Za-z]{2,}      # someone@example.com
    """,
    re.VERBOSE | re.IGNORECASE,
)


def _without_addresses(text: str) -> str:
    """Blank out URLs and e-mail addresses so they are not read as words.

    A URL is not written in a language, but its pieces look like short words
    and score like them: ``"https://example.com/a/b?c=1"`` used to come back as
    Portuguese at 0.57, because ``com`` and ``a`` are both Portuguese function
    words.  Removing them first leaves nothing to go on, which is the honest
    answer and the same one digits and punctuation already get.
    """
    if "://" not in text and "@" not in text and "www." not in text.lower():
        return text
    return _MACHINE_ADDRESS.sub(" ", text)


def _tokens(text: str) -> List[str]:
    r"""Word-like tokens, casefolded and stripped of generic accents.

    A token is a run of letters and combining marks, so digits, punctuation and
    apostrophes all break a word.  Devanagari vowel signs stay attached to the
    consonant they belong to, which ``\w`` in a regular expression would not do.
    """
    folded = fold_marks(text).casefold()
    out: List[str] = []
    current: List[str] = []
    for ch in folded:
        if is_script_char(ch):
            current.append(ch)
        elif current:
            out.append("".join(current))
            current = []
    if current:
        out.append("".join(current))
    return out


def _word_hit_rate(lang: str, tokens: Sequence[str]) -> float:
    table = _FOLDED_WORD_WEIGHTS.get(lang)
    if not table or not tokens:
        return 0.0
    hit = sum(table.get(token, 0.0) for token in tokens)
    return min(1.0, hit / len(tokens))


def _marker_hit_rate(lang: str, text: str, characters: int) -> float:
    table = MARKER_WEIGHTS.get(lang)
    if not table or not characters:
        return 0.0
    covered = 0.0
    for marker, weight in table.items():
        count = text.count(marker)
        if count:
            covered += count * len(marker) * weight
    return min(1.0, covered / characters)


def _distinctive_bonus(lang: str, letters: str, text: str) -> float:
    """Credit for the characters that give a language away on sight.

    Letters are counted from ``letters``, which is already stripped of
    everything that does not carry a script.  A few DISTINCTIVE entries are
    *not* letters -- Spanish's inverted question and exclamation marks are
    category Po -- so those are counted from the raw ``text`` instead.  Counted
    only from ``letters`` they were silently filtered out before the scorer
    ever saw them, which left "Como estas" and "Como estas" (inverted marks)
    scoring byte-identically.
    """
    marks = DISTINCTIVE.get(lang)
    if not marks:
        return 0.0
    seen = sum(1 for ch in letters if ch in marks)
    seen += sum(1 for ch in text if ch in marks and not is_script_char(ch))
    return 0.2 * min(seen, 5)


def _mark_penalty(lang: str, letters: str, characters: int) -> float:
    """How much of the text is spelled with letters this language never uses."""
    if not characters:
        return 0.0
    bad = 0
    allowed = ALLOWED_MARKS.get(lang)
    if allowed is not None:
        for ch in letters:
            if ch.isascii():
                continue
            if ch not in allowed:
                bad += 1
    excluded = EXCLUDED.get(lang)
    if excluded:
        bad += sum(1 for ch in letters if ch in excluded)
    return min(1.0, 4.0 * bad / characters)


def _evidence_factor(characters: int) -> float:
    """Short text is believed less, on a smooth ramp, never below the floor."""
    if characters <= 0:
        return 0.0
    ramp = characters / _FULL_EVIDENCE_AT
    return min(1.0, _EVIDENCE_FLOOR + (1.0 - _EVIDENCE_FLOOR) * ramp)


def _score(
    candidates: Sequence[str],
    text: str,
    letters: str,
    tokens: Sequence[str],
    characters: int,
    counts: Dict[str, int],
) -> List[Tuple[str, float]]:
    """Raw, unnormalised scores for every candidate, highest first."""
    kana = counts.get("Hiragana", 0) + counts.get("Katakana", 0)
    kana_share = kana / characters if characters else 0.0
    scored: List[Tuple[str, float]] = []
    for order, lang in enumerate(candidates):
        if lang in MARKER_WEIGHTS:
            hit = _marker_hit_rate(lang, text, characters)
        else:
            hit = _word_hit_rate(lang, tokens)
        evidence = (
            _WORD_WEIGHT * hit
            + _DISTINCTIVE_WEIGHT * _distinctive_bonus(lang, letters, text)
            - _PENALTY_WEIGHT * _mark_penalty(lang, letters, characters)
        )
        if lang == "ja":
            evidence += 2.0 * kana_share
        elif lang == "zh":
            evidence -= 2.0 * kana_share
        raw = math.exp(min(_SHARPNESS * evidence, _EXPONENT_CAP))
        scored.append((lang, raw))
    scored.sort(key=lambda item: (-item[1], candidates.index(item[0])))
    return scored


def detect(text: str, *, top: int = 1) -> Detection:
    """Work out what language ``text`` is written in.

    Args:
        text: the text to inspect.  An empty string, or one with no letters in
            it at all, gives back an undetermined result rather than raising.
            URLs and e-mail addresses are removed before scoring, because they
            are not written in any language; a string that is nothing but an
            address is undetermined, and one inside a sentence simply drops
            out.
        top: how many candidates to rank in total.  ``top=1`` (the default)
            fills ``.language`` only; ``top=3`` also puts the two runners-up in
            ``.alternatives``.

    Returns:
        A :class:`Detection`.  Always check ``.reliable`` before branching on
        ``.language``: on very short strings the winner is a guess and says so.

    Raises:
        TypeError: if ``text`` is not a string.
        ValueError: if ``top`` is less than 1.

    Examples:
        ``detect("El perro es muy grande").language`` is ``"es"``;
        ``detect("").language`` is ``"und"`` with confidence ``0.0``;
        ``detect("12345").language`` is ``"und"`` too, because digits are not a
        language, and so is ``detect("https://example.com/a/b").language``,
        because a URL is not one either.
    """
    if not isinstance(text, str):
        raise TypeError("detect() needs a str, got %s" % type(text).__name__)
    if not isinstance(top, int) or isinstance(top, bool) or top < 1:
        raise ValueError("top must be an integer of at least 1, not %r" % (top,))

    content = _without_addresses(text)
    counts = script_counts(content)
    shares = script_shares(counts)
    characters = sum(counts.values())
    if characters == 0:
        return _unknown(COMMON if text else UNKNOWN, 0, shares)

    script = dominant_script(counts)
    candidates = CANDIDATES.get(script)
    if not candidates:
        return _unknown(script, characters, shares)

    letters = "".join(ch for ch in content if is_script_char(ch))
    tokens = _tokens(content)
    scored = _score(candidates, content, letters, tokens, characters, counts)
    total = sum(raw for _, raw in scored) or 1.0
    factor = _evidence_factor(characters)

    ranked: List[Tuple[str, float]] = [
        (lang, round(min(_MAX_CONFIDENCE, raw / total * factor), 4))
        for lang, raw in scored
    ]
    best_code, best_confidence = ranked[0]
    alternatives = ranked[1:top]

    reliable = (
        characters >= _MIN_LETTERS_FOR_RELIABLE
        and best_confidence >= _MIN_CONFIDENCE_FOR_RELIABLE
    )
    return Detection(
        language=best_code,
        name=LANGUAGE_NAMES.get(best_code, best_code),
        confidence=best_confidence,
        script=script,
        reliable=reliable,
        mixed=is_mixed(counts),
        alternatives=alternatives,
        scripts=shares,
        characters=characters,
        words=len(tokens),
    )


def detect_batch(texts: Iterable[str]) -> List[Detection]:
    """Run :func:`detect` over many strings and keep the order.

    Args:
        texts: any iterable of strings.  Passing a single string is a mistake
            common enough to be worth an explicit error, so it raises.

    Returns:
        One :class:`Detection` per input, in the same order.

    Raises:
        TypeError: if ``texts`` is a single string, or holds a non-string.
    """
    if isinstance(texts, str):
        raise TypeError(
            "detect_batch() takes a list of strings, not one string; "
            "use detect() for a single string"
        )
    out: List[Detection] = []
    for index, item in enumerate(texts):
        if not isinstance(item, str):
            raise TypeError(
                "detect_batch() item %d is %s, not a str" % (index, type(item).__name__)
            )
        out.append(detect(item))
    return out


def script_of(text: str) -> str:
    """The dominant writing system of ``text``.

    Returns a Unicode script name such as ``"Latin"``, ``"Devanagari"`` or
    ``"Han"``.  Text made only of digits, punctuation or spaces is
    ``"Common"``; an empty string is ``"Unknown"``, and so is a letter from a
    script this package does not name.

    Examples:
        ``script_of("hello")`` is ``"Latin"``, ``script_of("мир")`` is
        ``"Cyrillic"``, ``script_of("42 + 7")`` is ``"Common"``.
    """
    if not isinstance(text, str):
        raise TypeError("script_of() needs a str, got %s" % type(text).__name__)
    if not text:
        return UNKNOWN
    counts = script_counts(text)
    if not counts:
        return COMMON
    return dominant_script(counts)


def is_rtl(text: str) -> bool:
    """True when the text is mostly written right to left.

    Arabic, Hebrew, Syriac, Thaana and N'Ko count as right to left.  For text
    that mixes directions the majority of the letters wins, so
    ``is_rtl("Hello مرحبا")`` is ``False`` but
    ``is_rtl("مرحبا hi")`` is ``True``.  An empty
    string is ``False``.
    """
    if not isinstance(text, str):
        raise TypeError("is_rtl() needs a str, got %s" % type(text).__name__)
    return rtl_ratio(text) > 0.5


def language_name(code: str) -> str:
    """The English name for an ISO 639-1 code this package can return."""
    return LANGUAGE_NAMES.get(code, code)


def char_script_of(ch: str) -> str:
    """The script of a single character, e.g. ``char_script_of("A") == "Latin"``."""
    return char_script(ch)
