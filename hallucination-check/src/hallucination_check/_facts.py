"""Extraction of the things people actually get wrong: numbers, dates, names, quantities.

Everything here is regex and dictionary work. No model, no download, no network.
"""
from __future__ import annotations

import re
import unicodedata
from decimal import Decimal, InvalidOperation
from typing import Dict, List, Optional, Sequence, Set, Tuple

from ._text import COMMON_WORDS, STOPWORDS, stem, token_spans

__all__ = ["Fact", "extract_facts", "collect_keys", "mask_numbers", "number_tokens"]

# How many content tokens either side of a name count as its context. A name is a fixed
# label, so what tells "developed at Pfizer" from "Separately, Pfizer reported earnings"
# is the company it keeps, not the word immediately next to it.
NAME_WINDOW = 4

_SCALE_WORDS: Dict[str, int] = {
    "hundred": 100,
    "thousand": 1000,
    "lakh": 100000,
    "million": 1000000,
    "crore": 10000000,
    "billion": 1000000000,
    "trillion": 1000000000000,
}
_SCALE_ABBR: Dict[str, int] = {"k": 1000, "m": 1000000, "bn": 1000000000, "tn": 1000000000000}

_UNITS: Dict[str, int] = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
    "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13,
    "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
    "nineteen": 19, "twenty": 20, "thirty": 30, "forty": 40, "fourty": 40, "fifty": 50,
    "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
}
_ORDINALS: Dict[str, int] = {
    "first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5, "sixth": 6, "seventh": 7,
    "eighth": 8, "ninth": 9, "tenth": 10, "eleventh": 11, "twelfth": 12, "thirteenth": 13,
    "fourteenth": 14, "fifteenth": 15, "sixteenth": 16, "seventeenth": 17, "eighteenth": 18,
    "nineteenth": 19, "twentieth": 20, "thirtieth": 30, "fortieth": 40, "fiftieth": 50,
    "sixtieth": 60, "seventieth": 70, "eightieth": 80, "ninetieth": 90, "hundredth": 100,
    "thousandth": 1000, "millionth": 1000000,
}
_NUM_WORDS: Set[str] = set(_UNITS) | set(_SCALE_WORDS)

_MONTHS: Dict[str, int] = {
    "january": 1, "jan": 1, "february": 2, "feb": 2, "march": 3, "mar": 3, "april": 4,
    "apr": 4, "may": 5, "june": 6, "jun": 6, "july": 7, "jul": 7, "august": 8, "aug": 8,
    "september": 9, "sep": 9, "sept": 9, "october": 10, "oct": 10, "november": 11, "nov": 11,
    "december": 12, "dec": 12,
}

_DIGITS_RE = re.compile(
    r"(?<![0-9A-Za-z.])"
    r"([+-]?\d{1,3}(?:,\d{3})+|[+-]?\d+(?:\.\d+)?)"
    r"(?:(k|m|bn|tn)(?![A-Za-z0-9]))?"
    r"(?:\s*(%|percent|per cent))?"
    r"(?:\s+(hundred|thousand|lakh|million|crore|billion|trillion)\b)?",
    re.I,
)
_ISO_RE = re.compile(r"(?<!\d)(\d{4})-(\d{1,2})-(\d{1,2})(?!\d)")
_DMY_RE = re.compile(
    r"(?<!\d)(\d{1,2})(?:st|nd|rd|th)?\s+(?:of\s+)?([A-Za-z]{3,9})\.?,?\s+(\d{4})(?!\d)", re.I
)
_MDY_RE = re.compile(
    r"\b([A-Za-z]{3,9})\.?\s+(\d{1,2})(?:st|nd|rd|th)?,?\s+(\d{4})(?!\d)", re.I
)
_SLASH_RE = re.compile(r"(?<!\d)(\d{1,2})[/.](\d{1,2})[/.](\d{4})(?!\d)")
_MY_RE = re.compile(r"\b([A-Za-z]{3,9})\.?\s+(\d{4})(?!\d)", re.I)
_MONTH_WORD_RE = re.compile(r"\b(%s)\b" % "|".join(sorted(_MONTHS, key=len, reverse=True)), re.I)
_MASK_RE = re.compile(r"[+-]?\d[\d,.]*")


class Fact:
    """One checkable detail lifted out of a piece of text.

    Besides its own keys a fact carries *context* keys that tie it to the words around
    it. Bare membership ("does 1,200 occur anywhere in any passage?") is far too weak:
    it lets "1,200 percent" be backed up by "1,200 participants" in an unrelated
    passage, and lets "11 metres" be backed up by the 11 in "Apollo 11". The context
    keys make a fact count only when it appears in the same role.
    """

    __slots__ = (
        "kind", "text", "keys", "parts", "start", "end", "flaggable",
        "context", "strict", "strict_keys", "before", "after", "value",
    )

    def __init__(
        self,
        kind: str,
        text: str,
        keys: Tuple[str, ...],
        start: int,
        end: int,
        parts: Tuple[str, ...] = (),
        flaggable: bool = True,
        value: Optional[Decimal] = None,
    ) -> None:
        self.kind = kind
        self.text = text
        self.keys = keys
        self.parts = parts
        self.start = start
        self.end = end
        self.flaggable = flaggable
        self.value = value
        # Filled in by _attach_context once every token of the text is known.
        self.context: Tuple[str, ...] = ()
        self.strict = False
        self.strict_keys: Tuple[str, ...] = ()
        self.before = ""
        self.after = ""

    def is_in(self, keys: Set[str]) -> bool:
        """True when this fact is backed by ``keys`` (a source key set).

        ``strict`` facts -- a number with a unit or head noun beside it, and any name --
        are backed up only by one of their ``strict_keys``, so the same digits or the
        same name used elsewhere for something else do not count. Everything else tries
        its context keys first and falls back to plain membership.
        """
        if self.strict:
            return any(k in keys for k in self.strict_keys)
        if self.context and any(c in keys for c in self.context):
            return True
        if any(k in keys for k in self.keys):
            return True
        return bool(self.parts) and all(p in keys for p in self.parts)

    def all_keys(self) -> Tuple[str, ...]:
        """Every key this fact contributes when it comes from a source."""
        return self.keys + self.parts + self.context

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return "Fact(%s, %r)" % (self.kind, self.text)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Fact) and (self.kind, self.keys) == (other.kind, other.keys)

    def __hash__(self) -> int:
        return hash((self.kind, self.keys))


def _canon(value: Decimal) -> str:
    """Canonical text form of a number: 1,200 / 1200 / 1.2 thousand all give '1200'."""
    try:
        text = format(value.normalize(), "f")
    except (InvalidOperation, ValueError):  # pragma: no cover - defensive
        return str(value)
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text or "0"


def _num_key(value: Decimal) -> str:
    return "num:" + _canon(value)


def _overlaps(start: int, end: int, taken: List[Tuple[int, int]]) -> bool:
    return any(start < t_end and t_start < end for t_start, t_end in taken)


def _month_number(word: str) -> int:
    return _MONTHS.get(word.casefold().rstrip("."), 0)


def _date_fact(text: str, start: int, end: int, year: int, month: int, day: int) -> Fact:
    keys = ("date:%04d-%02d-%02d" % (year, month, day),)
    parts = ("num:%d" % year, "month:%d" % month, "num:%d" % day)
    return Fact("date", text, keys, start, end, parts)


def _extract_dates(text: str, taken: List[Tuple[int, int]]) -> List[Fact]:
    facts: List[Fact] = []

    def add(fact: Fact) -> None:
        if not _overlaps(fact.start, fact.end, taken):
            taken.append((fact.start, fact.end))
            facts.append(fact)

    for match in _ISO_RE.finditer(text):
        year, month, day = (int(g) for g in match.groups())
        if 1 <= month <= 12 and 1 <= day <= 31:
            add(_date_fact(match.group(0), match.start(), match.end(), year, month, day))
    for match in _DMY_RE.finditer(text):
        month = _month_number(match.group(2))
        day, year = int(match.group(1)), int(match.group(3))
        if month and 1 <= day <= 31:
            add(_date_fact(match.group(0), match.start(), match.end(), year, month, day))
    for match in _MDY_RE.finditer(text):
        month = _month_number(match.group(1))
        day, year = int(match.group(2)), int(match.group(3))
        if month and 1 <= day <= 31:
            add(_date_fact(match.group(0), match.start(), match.end(), year, month, day))
    for match in _SLASH_RE.finditer(text):
        a, b, year = int(match.group(1)), int(match.group(2)), int(match.group(3))
        keys: List[str] = []
        if 1 <= b <= 12 and 1 <= a <= 31:
            keys.append("date:%04d-%02d-%02d" % (year, b, a))
        if 1 <= a <= 12 and 1 <= b <= 31:
            keys.append("date:%04d-%02d-%02d" % (year, a, b))
        if keys:
            add(
                Fact(
                    "date",
                    match.group(0),
                    tuple(keys),
                    match.start(),
                    match.end(),
                    ("num:%d" % year, "num:%d" % a, "num:%d" % b),
                )
            )
    for match in _MY_RE.finditer(text):
        month = _month_number(match.group(1))
        year = int(match.group(2))
        if month:
            add(
                Fact(
                    "date",
                    match.group(0),
                    ("date:%04d-%02d" % (year, month),),
                    match.start(),
                    match.end(),
                    ("num:%d" % year, "month:%d" % month),
                )
            )
    return facts


def _extract_digit_numbers(text: str, taken: List[Tuple[int, int]]) -> List[Fact]:
    facts: List[Fact] = []
    for match in _DIGITS_RE.finditer(text):
        raw, abbr, pct, scale = match.groups()
        if _overlaps(match.start(), match.end(), taken):
            continue
        try:
            value = Decimal(raw.replace(",", ""))
        except InvalidOperation:  # pragma: no cover - regex keeps this unreachable
            continue
        if abbr:
            value *= _SCALE_ABBR[abbr.casefold()]
        if scale:
            value *= _SCALE_WORDS[scale.casefold()]
        keys = [_num_key(value)]
        if pct:
            keys.append("pct:" + _canon(value))
        taken.append((match.start(), match.end()))
        facts.append(
            Fact(
                "number",
                match.group(0).strip(),
                tuple(keys),
                match.start(),
                match.end(),
                value=value,
            )
        )
    return facts


def _words_to_number(words: Sequence[str]) -> Decimal:
    """Turn ``['twenty', 'three']`` into ``Decimal(23)``."""
    total = Decimal(0)
    current = Decimal(0)
    for word in words:
        if word in _UNITS:
            current += _UNITS[word]
        elif word == "hundred":
            current = (current or Decimal(1)) * 100
        else:
            scale = _SCALE_WORDS[word]
            total += (current or Decimal(1)) * scale
            current = Decimal(0)
    return total + current


def _extract_word_numbers(text: str, taken: List[Tuple[int, int]]) -> List[Fact]:
    facts: List[Fact] = []
    spans = token_spans(text)
    i = 0
    while i < len(spans):
        token = spans[i][0].casefold()
        if token in _ORDINALS:
            start, end = spans[i][1], spans[i][2]
            if not _overlaps(start, end, taken):
                taken.append((start, end))
                facts.append(
                    Fact(
                        "ordinal",
                        spans[i][0],
                        (_num_key(Decimal(_ORDINALS[token])),),
                        start,
                        end,
                        flaggable=False,
                        value=Decimal(_ORDINALS[token]),
                    )
                )
            i += 1
            continue
        if token not in _NUM_WORDS:
            i += 1
            continue
        run: List[str] = []
        j = i
        last = i
        while j < len(spans):
            word = spans[j][0].casefold()
            if word in _NUM_WORDS:
                run.append(word)
                last = j
            elif word == "and" and run and j + 1 < len(spans) and spans[j + 1][0].casefold() in _NUM_WORDS:
                pass
            else:
                break
            j += 1
        start, end = spans[i][1], spans[last][2]
        if not _overlaps(start, end, taken):
            value = _words_to_number(run)
            # A bare "one" is usually the pronoun or the article, not a quantity.
            flaggable = len(run) > 1 or value >= 2
            taken.append((start, end))
            facts.append(
                Fact(
                    "number",
                    text[start:end],
                    (_num_key(value),),
                    start,
                    end,
                    flaggable=flaggable,
                    value=value,
                )
            )
        i = last + 1
    return facts


_SENTENCE_ENDERS = frozenset(".!?…。！？\n\r")


def _starts_a_sentence(text: str, start: int) -> bool:
    index = start - 1
    while index >= 0 and text[index].isspace():
        if text[index] in "\n\r":
            return True
        index -= 1
    return index < 0 or text[index] in _SENTENCE_ENDERS


def _extract_names(text: str, taken: List[Tuple[int, int]]) -> List[Fact]:
    facts: List[Fact] = []
    spans = token_spans(text)
    for position, (token, start, end) in enumerate(spans):
        if _overlaps(start, end, taken):
            continue
        folded = token.casefold()
        if len(token) < 2 or folded in COMMON_WORDS:
            continue
        acronym = token.isupper() and token.isalpha()
        titled = token[:1].isupper() and not token.isupper()
        if not (acronym or titled):
            continue
        if titled and _starts_a_sentence(text, start):
            # A capitalized word at the start of a sentence is usually just a word.
            # Keep it only when the next token is capitalized too, as in "Eiffel Tower".
            nxt = spans[position + 1][0] if position + 1 < len(spans) else ""
            if not (nxt[:1].isupper() and nxt[:1].isalpha()):
                continue
        taken.append((start, end))
        facts.append(Fact("name", token, ("name:" + folded,), start, end))
    return facts


def _content_tokens(norm: str) -> List[Tuple[str, str, int, int]]:
    """``(folded, stemmed, start, end)`` for the tokens that carry meaning."""
    out: List[Tuple[str, str, int, int]] = []
    for token, start, end in token_spans(norm):
        folded = token.casefold()
        if folded in STOPWORDS:
            continue
        out.append((folded, stem(folded), start, end))
    return out


def _number_context(fact: Fact, content: Sequence[Tuple[str, str, int, int]]) -> None:
    """Bind a number to the content word before it and the one after it.

    A number that has a word after it is a measurement or a count ("11 *metres*"), and
    is only ever backed up by a source that binds the same figure to the same word --
    which is what tells it from the 11 in "Apollo 11". A number with nothing after it
    ("completed in 1889.") keeps the looser plain-value match, so a paraphrase that
    moves the surrounding words is not flagged.

    Which of the two neighbours the number is *attached* to is a matter of distance,
    not of side. In "compressor three was replaced" the number belongs to "compressor"
    a space away, not to the verb four words on, and a source writing "the number three
    compressor" says exactly the same thing with the words in the other order. Binding
    strictness to the nearer neighbour and matching it without regard to side is what
    stops a faithful paraphrase from being called a fabrication, while "11 metres" stays
    flagged against "Apollo 11 stood 110 metres tall" because that source binds no 11 to
    "metre".
    """
    if fact.value is None:
        return
    before = after = ""
    before_end = after_start = -1
    for _folded, stemmed, start, end in content:
        if end <= fact.start:
            before, before_end = stemmed, end
        elif start >= fact.end:
            after, after_start = stemmed, start
            break
    fact.before, fact.after = before, after
    canon = _canon(fact.value)
    keys: List[str] = []
    if after:
        keys.append("n>%s|%s" % (canon, after))
    if before:
        keys.append("n<%s|%s" % (canon, before))
    # Order-free companions to the two above. A source contributes all of them, so the
    # side a neighbour sits on never has to agree for the figure to count as backed up.
    keys += ["n:%s|%s" % (canon, token) for token in (after, before) if token]
    fact.context = tuple(dict.fromkeys(keys))
    fact.strict = bool(after)
    if fact.strict:
        gap_after = after_start - fact.end
        gap_before = fact.start - before_end if before else None
        head = after if gap_before is None or gap_after <= gap_before else before
        fact.strict_keys = ("n:%s|%s" % (canon, head),)


def _name_context(fact: Fact, content: Sequence[Tuple[str, str, int, int]]) -> None:
    """Bind a name to the content words near it."""
    position = None
    for index, (_folded, _stemmed, start, _end) in enumerate(content):
        if start == fact.start:
            position = index
            break
    if position is None:  # pragma: no cover - a name is always a content token
        return
    folded = fact.text.casefold()
    low = max(0, position - NAME_WINDOW)
    high = min(len(content), position + NAME_WINDOW + 1)
    fact.context = tuple(
        "c:%s|%s" % (folded, content[j][1]) for j in range(low, high) if j != position
    )
    fact.strict = bool(fact.context)
    fact.strict_keys = fact.context


def _attach_context(norm: str, facts: Sequence[Fact]) -> None:
    """Give every fact the keys that tie it to the words around it."""
    if not facts:
        return
    content = _content_tokens(norm)
    if not content:
        return
    for fact in facts:
        if fact.kind in ("number", "ordinal"):
            _number_context(fact, content)
        elif fact.kind == "name":
            _name_context(fact, content)


def extract_facts(text: str, *, for_source: bool = False) -> List[Fact]:
    """Pull every number, date, quantity and proper noun out of ``text``.

    ``for_source=True`` keeps facts that are never flagged in an answer (ordinals,
    a bare "one") because a source still backs them up.
    """
    norm = unicodedata.normalize("NFKC", text)
    taken: List[Tuple[int, int]] = []
    facts = _extract_dates(norm, taken)
    facts += _extract_digit_numbers(norm, taken)
    facts += _extract_word_numbers(norm, taken)
    facts += _extract_names(norm, taken)
    if not for_source:
        facts = [f for f in facts if f.flaggable]
    facts.sort(key=lambda f: (f.start, f.end))
    _attach_context(norm, facts)
    return facts


def collect_keys(text: str) -> Set[str]:
    """Every key ``text`` can back up: its facts, its context, and every token in it."""
    keys: Set[str] = set()
    for fact in extract_facts(text, for_source=True):
        keys.update(fact.all_keys())
    norm = unicodedata.normalize("NFKC", text)
    for match in _MONTH_WORD_RE.finditer(norm):
        keys.add("month:%d" % _month_number(match.group(1)))
    for token, _start, _end in token_spans(norm):
        folded = token.casefold()
        keys.add("name:" + folded)
        keys.add("name:" + stem(folded))
    # Context keys for every token that could answer a claim's name, not only the ones
    # this text capitalizes: a source may well write a name in lower case.
    content = _content_tokens(norm)
    for index, (folded, stemmed, _start, _end) in enumerate(content):
        if len(folded) < 2 or folded in COMMON_WORDS or folded[0].isdigit():
            continue
        low = max(0, index - NAME_WINDOW)
        high = min(len(content), index + NAME_WINDOW + 1)
        for j in range(low, high):
            if j == index:
                continue
            other = content[j][1]
            keys.add("c:%s|%s" % (folded, other))
            if stemmed != folded:
                keys.add("c:%s|%s" % (stemmed, other))
    return keys


def number_tokens(text: str) -> List[str]:
    """Canonical digit strings for every quantity in ``text``.

    Added to a text's token set so "twenty-three" and "23" count as the same token.
    """
    out: List[str] = []
    for fact in extract_facts(text, for_source=True):
        if fact.kind in ("number", "ordinal"):
            for key in fact.keys:
                if key.startswith("num:"):
                    out.append(key[4:])
    return out


def mask_numbers(text: str) -> str:
    """Replace every digit run with ``#`` so two sentences can be compared apart from
    the numbers they carry."""
    return _MASK_RE.sub("#", unicodedata.normalize("NFKC", text))
