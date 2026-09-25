"""Spoken English numbers: "twenty one", "a hundred and five", "minus three".

The parser works on comparison keys (see :func:`voice_commands_ai._text.fold`)
and understands:

* cardinals up to the trillions, with or without "and"
  ("one thousand two hundred and thirty four", "twelve hundred", "a dozen");
* digits, alone or with a scale word ("21", "2.5", "1,000", "1.5 million");
* signs ("minus three", "negative two", "-3") and decimals ("two point five",
  "point seven five", "three point one four");
* halves and quarters ("two and a half", "a quarter", "three quarters");
* ordinals ("third", "twenty first", "21st");
* digit-by-digit readings ("four five six" is 456, "one oh one" is 101) and
  years read in pairs ("nineteen eighty four" is 1984).

The whole text has to be a number: ``"twenty one degrees"`` is not.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Dict, List, Optional, Sequence, Tuple

from ._text import DISFLUENCIES, raw_tokens

_UNITS: Dict[str, int] = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9,
}
_TEENS: Dict[str, int] = {
    "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
    "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19,
}
_TENS: Dict[str, int] = {
    "twenty": 20, "thirty": 30, "forty": 40, "fourty": 40, "fifty": 50, "sixty": 60,
    "seventy": 70, "eighty": 80, "ninety": 90,
}
_SCALES: Dict[str, int] = {
    "thousand": 10 ** 3, "million": 10 ** 6, "billion": 10 ** 9, "trillion": 10 ** 12,
}
_ORDINALS: Dict[str, str] = {
    "first": "one", "second": "two", "third": "three", "fourth": "four",
    "fifth": "five", "sixth": "six", "seventh": "seven", "eighth": "eight",
    "ninth": "nine", "tenth": "ten", "eleventh": "eleven", "twelfth": "twelve",
    "thirteenth": "thirteen", "fourteenth": "fourteen", "fifteenth": "fifteen",
    "sixteenth": "sixteen", "seventeenth": "seventeen", "eighteenth": "eighteen",
    "nineteenth": "nineteen", "twentieth": "twenty", "thirtieth": "thirty",
    "fortieth": "forty", "fiftieth": "fifty", "sixtieth": "sixty",
    "seventieth": "seventy", "eightieth": "eighty", "ninetieth": "ninety",
    "hundredth": "hundred", "thousandth": "thousand", "millionth": "million",
    "billionth": "billion",
}
_DIGIT_WORDS: Dict[str, int] = dict(_UNITS, oh=0, o=0)
_SIGNS = {"minus": -1, "negative": -1, "plus": 1}

#: Words that can appear inside a spoken number (used to bound slot spans).
NUMBER_WORDS = frozenset(
    set(_UNITS) | set(_TEENS) | set(_TENS) | set(_SCALES) | set(_ORDINALS)
    | {"hundred", "dozen", "and", "a", "an", "point", "half", "quarter", "quarters",
       "halves", "oh", "o"} | set(_SIGNS)
)

_DIGIT_RE = re.compile(r"([-+]?)(\d+(?:[.,]\d+)*)(st|nd|rd|th)?")
_THOUSANDS_RE = re.compile(r"\d{1,3}(?:,\d{3})+(?:\.\d+)?")

Parsed = Tuple[float, bool]  # (value, written as a whole number)


def _digit_value(word: str) -> Optional[Tuple[float, bool]]:
    """Value of a digit token ("21", "-3", "2.5", "1,000", "21st"), with a flag
    saying whether it was written without a decimal part."""
    m = _DIGIT_RE.fullmatch(word)
    if not m:
        return None
    sign, body, suffix = m.groups()
    if suffix and not body.isdigit():
        return None
    if "," in body:
        if _THOUSANDS_RE.fullmatch(body):
            body = body.replace(",", "")
        elif body.count(",") == 1 and "." not in body:
            body = body.replace(",", ".")  # decimal comma: "2,5"
        else:
            return None
    try:
        value = float(body)
    except ValueError:
        return None
    return (-value if sign == "-" else value), ("." not in body)


def _below_hundred(words: Sequence[str], i: int) -> Optional[Tuple[int, int]]:
    """Parse 0-99 in words starting at ``i``: returns ``(value, next index)``."""
    if i >= len(words):
        return None
    w = words[i]
    if w in _TENS:
        value = _TENS[w]
        if i + 1 < len(words) and _UNITS.get(words[i + 1], 0) > 0:
            return value + _UNITS[words[i + 1]], i + 2
        return value, i + 1
    if w in _TEENS:
        return _TEENS[w], i + 1
    if w in _UNITS:
        return _UNITS[w], i + 1
    return None


def _group(words: Sequence[str], i: int) -> Optional[Tuple[float, int]]:
    """One group below a scale word: "three", "twenty five", "a hundred and five",
    "twelve hundred", "a dozen", "21". Returns ``(value, next index)``."""
    n = len(words)
    w = words[i]
    if w in ("a", "an"):
        if i + 1 < n and (words[i + 1] in ("hundred", "dozen") or words[i + 1] in _SCALES):
            head: float = 1
            j = i + 1
        else:
            return None
    elif w in ("hundred", "dozen") or w in _SCALES:
        head, j = 1, i
    else:
        below = _below_hundred(words, i)
        if below is not None:
            head, j = below
        else:
            digits = _digit_value(w)
            if digits is None or digits[0] < 0:
                return None
            head, j = digits[0], i + 1
    if j < n and words[j] == "dozen":
        return head * 12, j + 1
    if j < n and words[j] == "hundred":
        if head != int(head) or not 1 <= head <= 99:
            return None
        value = head * 100
        j += 1
        if j < n and words[j] == "and":
            rest = _below_hundred(words, j + 1)
            if rest is None:
                return None
            return value + rest[0], rest[1]
        rest = _below_hundred(words, j)
        if rest is not None:
            return value + rest[0], rest[1]
        return value, j
    return head, j


def _cardinal(words: Sequence[str]) -> Optional[float]:
    """A whole cardinal such as "two million three thousand and five"."""
    n = len(words)
    if n == 0:
        return None
    i = 0
    total: float = 0
    last_scale = float("inf")
    while i < n:
        parsed = _group(words, i)
        if parsed is None:
            return None
        value, i = parsed
        if i < n and words[i] in _SCALES:
            scale = _SCALES[words[i]]
            if scale >= last_scale:
                return None
            total += value * scale
            last_scale = scale
            i += 1
            if i < n and words[i] == "and":
                i += 1
                if i == n:
                    return None
        else:
            if i != n or value >= last_scale:
                return None
            total += value
    return total


def _digit_sequence(words: Sequence[str]) -> Optional[float]:
    """Digits read one by one: "four five six" -> 456."""
    if len(words) < 2:
        return None
    out: List[str] = []
    for w in words:
        if w in _DIGIT_WORDS:
            out.append(str(_DIGIT_WORDS[w]))
        elif w.isascii() and w.isdigit():
            out.append(w)
        else:
            return None
    return float(int("".join(out)))


def _year(words: Sequence[str]) -> Optional[float]:
    """Years read in pairs: "nineteen eighty four", "twenty twenty six", "nineteen oh five"."""
    first = _below_hundred(words, 0)
    if first is None or not 10 <= first[0] <= 99:
        return None
    rest = list(words[first[1]:])
    if len(rest) == 2 and rest[0] in ("oh", "o", "zero") and _UNITS.get(rest[1], 0) > 0:
        return float(first[0] * 100 + _UNITS[rest[1]])
    second = _below_hundred(words, first[1])
    if second is None or second[1] != len(words) or not 10 <= second[0] <= 99:
        return None
    return float(first[0] * 100 + second[0])


def _fraction_digits(words: Sequence[str]) -> Optional[str]:
    """The digits after "point": "five" -> "5", "one four" -> "14", "twenty five" -> "25"."""
    out: List[str] = []
    i = 0
    while i < len(words):
        w = words[i]
        if w in _DIGIT_WORDS:
            out.append(str(_DIGIT_WORDS[w]))
            i += 1
            continue
        if w.isascii() and w.isdigit():
            out.append(w)
            i += 1
            continue
        below = _below_hundred(words, i)
        if below is None or below[0] < 10:
            return None
        out.append(str(below[0]))
        i = below[1]
    return "".join(out) or None


def _fraction_value(words: Sequence[str]) -> Optional[float]:
    """"a half", "one quarter", "three quarters" on their own."""
    words = list(words)
    if words in (["half"], ["a", "half"], ["one", "half"]):
        return 0.5
    if words in (["quarter"], ["a", "quarter"], ["one", "quarter"]):
        return 0.25
    if len(words) == 2 and words[1] in ("quarters", "halves") and _UNITS.get(words[0], 0) > 0:
        return _UNITS[words[0]] * (0.25 if words[1] == "quarters" else 0.5)
    return None


def parse_keys(keys: Sequence[str]) -> Optional[Parsed]:
    """Parse folded words as one number. Returns ``(value, whole)`` or None.

    ``whole`` is True when the number was said or written without a decimal or
    fractional part, which is what decides between ``int`` and ``float`` slots.
    """
    words = [k for k in keys if k]
    if not words or len(words) > 24:
        return None
    sign = 1
    if words[0] in _SIGNS:
        sign = _SIGNS[words[0]]
        words = words[1:]
        if not words:
            return None
    if len(words) == 1:
        digits = _digit_value(words[0])
        if digits is not None:
            if digits[0] < 0 and sign == -1:
                return None
            return sign * digits[0], digits[1]
    fraction = _fraction_value(words)
    if fraction is not None:
        return sign * fraction, False
    if "point" in words:
        idx = words.index("point")
        head, tail = words[:idx], words[idx + 1:]
        whole_part = _cardinal(head) if head else 0.0
        digits = _fraction_digits(tail)
        if whole_part is None or digits is None or whole_part != int(whole_part):
            return None
        return sign * (whole_part + float("0." + digits)), False
    extra = 0.0
    if len(words) > 3 and words[-3:-1] == ["and", "a"] and words[-1] in ("half", "quarter"):
        extra = 0.5 if words[-1] == "half" else 0.25
        words = words[:-3]
    if words[-1] in _ORDINALS:
        words = words[:-1] + [_ORDINALS[words[-1]]]
    value = _cardinal(words)
    if value is None and not extra:
        value = _digit_sequence(words)
        if value is None:
            value = _year(words)
    if value is None:
        return None
    total = value + extra
    return sign * total, (not extra and float(value).is_integer())


def parse_number(text: str) -> Optional[float]:
    """Read a spoken English number, or return None if ``text`` is not one.

    >>> parse_number("a hundred and five"), parse_number("two point five"), parse_number("minus three")
    (105.0, 2.5, -3.0)

    Case, hyphens, punctuation and hesitations ("twenty, um, one") are ignored.
    The whole text has to be a number, so ``"twenty one degrees"`` returns None.
    Zero comes back as ``0.0``, so test the result with ``is None``.
    """
    if not isinstance(text, str):
        raise TypeError("parse_number() expects a str, got %s" % type(text).__name__)
    keys = [
        t.key for t in raw_tokens(unicodedata.normalize("NFC", text))
        if t.key not in DISFLUENCIES
    ]
    parsed = parse_keys(keys)
    return None if parsed is None else float(parsed[0])


def number_spans(keys: Sequence[str], max_words: int = 12) -> List[List[Tuple[int, float, bool]]]:
    """For every start position, the spans that read as a number.

    Returns ``spans[i] = [(length, value, whole), ...]``.
    """
    n = len(keys)
    spans: List[List[Tuple[int, float, bool]]] = [[] for _ in range(n)]
    for i in range(n):
        first = keys[i]
        if first not in NUMBER_WORDS and _digit_value(first) is None:
            continue
        for k in range(1, min(max_words, n - i) + 1):
            last = keys[i + k - 1]
            if last not in NUMBER_WORDS and _digit_value(last) is None:
                break
            parsed = parse_keys(keys[i:i + k])
            if parsed is not None:
                spans[i].append((k, parsed[0], parsed[1]))
    return spans
