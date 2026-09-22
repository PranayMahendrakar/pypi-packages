"""Which writing system a character belongs to, and how a string is made up.

The Unicode script of a character is looked up in a sorted range table with
:func:`bisect`, so the cost is logarithmic and the answer never depends on the
host platform's locale.  Only the scripts this package can say something useful
about are named individually; anything else falls back to ``"Unknown"``.
"""
from __future__ import annotations

import unicodedata
from bisect import bisect_right
from typing import Dict, List, Tuple

__all__ = [
    "COMMON",
    "LTR_SCRIPTS",
    "RTL_SCRIPTS",
    "UNKNOWN",
    "char_script",
    "dominant_script",
    "families",
    "is_mixed",
    "is_script_char",
    "letters",
    "rtl_ratio",
    "script_block",
    "script_counts",
    "script_names",
    "script_shares",
]

UNKNOWN = "Unknown"
COMMON = "Common"

# (first codepoint, last codepoint, script).  Sorted at import time.
_RAW_RANGES: Tuple[Tuple[int, int, str], ...] = (
    (0x0041, 0x005A, "Latin"),
    (0x0061, 0x007A, "Latin"),
    (0x00AA, 0x00AA, "Latin"),
    (0x00BA, 0x00BA, "Latin"),
    (0x00C0, 0x00FF, "Latin"),
    (0x0100, 0x02AF, "Latin"),
    (0x0370, 0x03FF, "Greek"),
    (0x1F00, 0x1FFF, "Greek"),
    (0x0400, 0x052F, "Cyrillic"),
    (0x1C80, 0x1C8F, "Cyrillic"),
    (0x2DE0, 0x2DFF, "Cyrillic"),
    (0xA640, 0xA69F, "Cyrillic"),
    (0x0530, 0x058F, "Armenian"),
    (0x0590, 0x05FF, "Hebrew"),
    (0xFB1D, 0xFB4F, "Hebrew"),
    (0x0600, 0x06FF, "Arabic"),
    (0x0750, 0x077F, "Arabic"),
    (0x0870, 0x08FF, "Arabic"),
    (0xFB50, 0xFDFF, "Arabic"),
    (0xFE70, 0xFEFF, "Arabic"),
    (0x0700, 0x074F, "Syriac"),
    (0x0780, 0x07BF, "Thaana"),
    (0x07C0, 0x07FF, "Nko"),
    (0x0900, 0x097F, "Devanagari"),
    (0xA8E0, 0xA8FF, "Devanagari"),
    (0x0980, 0x09FF, "Bengali"),
    (0x0A00, 0x0A7F, "Gurmukhi"),
    (0x0A80, 0x0AFF, "Gujarati"),
    (0x0B00, 0x0B7F, "Oriya"),
    (0x0B80, 0x0BFF, "Tamil"),
    (0x0C00, 0x0C7F, "Telugu"),
    (0x0C80, 0x0CFF, "Kannada"),
    (0x0D00, 0x0D7F, "Malayalam"),
    (0x0D80, 0x0DFF, "Sinhala"),
    (0x0E00, 0x0E7F, "Thai"),
    (0x0E80, 0x0EFF, "Lao"),
    (0x0F00, 0x0FFF, "Tibetan"),
    (0x1000, 0x109F, "Myanmar"),
    (0x10A0, 0x10FF, "Georgian"),
    (0x1C90, 0x1CBF, "Georgian"),
    (0x1100, 0x11FF, "Hangul"),
    (0x3130, 0x318F, "Hangul"),
    (0xA960, 0xA97F, "Hangul"),
    (0xAC00, 0xD7AF, "Hangul"),
    (0x1200, 0x137F, "Ethiopic"),
    (0x1780, 0x17FF, "Khmer"),
    (0x1E00, 0x1EFF, "Latin"),
    (0x2C60, 0x2C7F, "Latin"),
    (0xA720, 0xA7FF, "Latin"),
    (0xAB30, 0xAB6F, "Latin"),
    (0xFB00, 0xFB06, "Latin"),
    (0x3040, 0x309F, "Hiragana"),
    (0x30A0, 0x30FF, "Katakana"),
    (0x31F0, 0x31FF, "Katakana"),
    (0x3400, 0x4DBF, "Han"),
    (0x4E00, 0x9FFF, "Han"),
    (0xF900, 0xFAFF, "Han"),
    (0x20000, 0x2A6DF, "Han"),
    (0x2A700, 0x2EBEF, "Han"),
    (0xFF21, 0xFF3A, "Latin"),
    (0xFF41, 0xFF5A, "Latin"),
    (0xFF66, 0xFF9D, "Katakana"),
)

_RANGES = tuple(sorted(_RAW_RANGES))
_STARTS = tuple(item[0] for item in _RANGES)

#: Scripts written right to left.
RTL_SCRIPTS = frozenset({"Arabic", "Hebrew", "Syriac", "Thaana", "Nko"})

#: Every script this package names, minus the right-to-left ones.
LTR_SCRIPTS = frozenset(
    {script for _, _, script in _RAW_RANGES if script not in RTL_SCRIPTS}
)

# The Japanese scripts are treated as one family when deciding whether a string
# really mixes writing systems: ordinary Japanese always uses all three.
_FAMILY: Dict[str, str] = {
    "Han": "Japanese/Chinese",
    "Hiragana": "Japanese/Chinese",
    "Katakana": "Japanese/Chinese",
}

# Prefers the more informative script when two are equally common: kana beats
# Han in Japanese, and a real script always beats "Common".
_TIE_BREAK: Dict[str, int] = {
    "Hiragana": 0,
    "Katakana": 1,
    "Hangul": 2,
    "Han": 3,
}
_DEFAULT_TIE_BREAK = 5
_LAST_TIE_BREAK = 9


def is_script_char(ch: str) -> bool:
    """True for letters and combining marks, i.e. characters that carry a script.

    Marks count because an Indic vowel sign or an Arabic vowel point belongs to
    its script just as much as the consonant it hangs on, and leaving them out
    would make Devanagari text look half as long as it is.
    """
    return unicodedata.category(ch)[0] in ("L", "M")


def letters(text: str) -> str:
    """The script-carrying characters of ``text``, in order."""
    return "".join(ch for ch in text if is_script_char(ch))


def char_script(ch: str) -> str:
    """The script of a single character.

    Non-letters (digits, punctuation, spaces, symbols) are ``"Common"``.
    Letters from a script this package does not name are ``"Unknown"``.
    """
    if not is_script_char(ch):
        return COMMON
    code = ord(ch)
    index = bisect_right(_STARTS, code) - 1
    if index >= 0:
        start, end, script = _RANGES[index]
        if start <= code <= end:
            return script
    return _script_from_name(ch)


def script_block(ch: str) -> str:
    """The script of the Unicode *block* ``ch`` sits in, letter or not.

    :func:`char_script` answers for letters only and calls everything else
    ``"Common"``, which is right when counting how much of a text is in each
    writing system.  Transliteration needs the other answer: a Devanagari danda
    and a Devanagari digit are punctuation and numbers by category, but they
    belong to Devanagari and have to be romanised with it.  Characters outside
    every named block - ASCII digits, ASCII punctuation, spaces, emoji - are
    ``"Common"``.
    """
    code = ord(ch)
    index = bisect_right(_STARTS, code) - 1
    if index >= 0:
        start, end, script = _RANGES[index]
        if start <= code <= end:
            return script
    return COMMON if not is_script_char(ch) else char_script(ch)


def _script_from_name(ch: str) -> str:
    """Fallback: read the script out of the character's Unicode name."""
    try:
        name = unicodedata.name(ch)
    except ValueError:
        return UNKNOWN
    head = name.split(" ", 1)[0]
    if head in ("CJK", "IDEOGRAPHIC"):
        return "Han"
    for script in ("LATIN", "GREEK", "CYRILLIC", "ARABIC", "HEBREW", "DEVANAGARI",
                   "BENGALI", "GURMUKHI", "GUJARATI", "TAMIL", "TELUGU", "KANNADA",
                   "MALAYALAM", "THAI", "HIRAGANA", "KATAKANA", "HANGUL"):
        if head == script:
            return script.capitalize()
    return UNKNOWN


def script_counts(text: str) -> Dict[str, int]:
    """How many script-carrying characters ``text`` has in each script."""
    counts: Dict[str, int] = {}
    for ch in text:
        if not is_script_char(ch):
            continue
        script = char_script(ch)
        counts[script] = counts.get(script, 0) + 1
    return counts


def script_shares(counts: Dict[str, int]) -> Dict[str, float]:
    """Turn raw script counts into fractions that sum to 1, largest first."""
    total = sum(counts.values())
    if not total:
        return {}
    ordered = sorted(counts.items(), key=lambda kv: (-kv[1], _order(kv[0])))
    return {script: round(count / total, 4) for script, count in ordered}


def _order(script: str) -> Tuple[int, str]:
    rank = _TIE_BREAK.get(script, _DEFAULT_TIE_BREAK)
    if script in (UNKNOWN, COMMON):
        rank = _LAST_TIE_BREAK
    return (rank, script)


def families(counts: Dict[str, int]) -> Dict[str, int]:
    """Collapse Han/Hiragana/Katakana into one family before judging mixture."""
    grouped: Dict[str, int] = {}
    for script, count in counts.items():
        key = _FAMILY.get(script, script)
        grouped[key] = grouped.get(key, 0) + count
    return grouped


def dominant_script(counts: Dict[str, int]) -> str:
    """The most common script in ``counts``, ``"Unknown"`` when there is none.

    The winning *family* is found first, so the same grouping that decides
    whether a text is mixed also decides what it is mostly written in.  Without
    that, a Japanese sentence with one English word in it would come back
    "Latin": kanji, hiragana and katakana each lose to the Latin run on their
    own, while together they are two thirds of the text.

    Within the winning family ties are broken deterministically, preferring the
    script that says more: Japanese kana ahead of Han, any real script ahead of
    ``"Common"``.
    """
    if not counts:
        return UNKNOWN
    grouped = families(counts)
    winner = min(grouped.items(), key=lambda kv: (-kv[1], _order(kv[0])))[0]
    members = {
        script: count
        for script, count in counts.items()
        if _FAMILY.get(script, script) == winner
    }
    return min(members.items(), key=lambda kv: (-kv[1], _order(kv[0])))[0]


def is_mixed(counts: Dict[str, int], threshold: float = 0.20) -> bool:
    """True when two or more writing systems each hold at least ``threshold``.

    Japanese is not "mixed" just for using kanji and kana together, so those
    three scripts are counted as one family here.
    """
    grouped = families(counts)
    total = sum(grouped.values())
    if total == 0:
        return False
    significant = [name for name, count in grouped.items() if count / total >= threshold]
    return len(significant) >= 2


def rtl_ratio(text: str) -> float:
    """Fraction of the script-carrying characters that are right-to-left."""
    right = 0
    total = 0
    for ch in text:
        if not is_script_char(ch):
            continue
        total += 1
        if char_script(ch) in RTL_SCRIPTS:
            right += 1
    return right / total if total else 0.0


def script_names() -> List[str]:
    """Every script name this package can report, sorted."""
    return sorted({script for _, _, script in _RAW_RANGES})
