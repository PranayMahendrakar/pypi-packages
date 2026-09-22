"""Romanisation for the three scripts this package covers properly.

Covered: **Devanagari** (IAST), **Cyrillic** (BGN/PCGN, with a Ukrainian table
that is used automatically when the text contains Ukrainian-only letters) and
**Greek** (ISO 843 transliteration, accents folded away first).

Everything else - Arabic, Hebrew, Han, kana, Hangul, Thai, Tamil and the rest -
is returned exactly as it came in.  That is deliberate: a half-right
romanisation of a script is worse than none, because it looks finished.  The
result says which scripts were left alone, so a caller can act on it.
"""
from __future__ import annotations

import unicodedata
from typing import Dict, List, Tuple

from ._normalize import fold_marks
from ._normalize import normalize as _normalize
from ._scripts import COMMON, script_block

__all__ = ["SUPPORTED_SCRIPTS", "TARGETS", "Transliteration", "transliterate"]

#: The scripts :func:`transliterate` can romanise.
SUPPORTED_SCRIPTS: Tuple[str, ...] = ("Devanagari", "Cyrillic", "Greek")

#: The values ``to=`` accepts.
TARGETS: Tuple[str, ...] = ("latin", "ascii")

# --------------------------------------------------------------------------
# Devanagari (IAST)
# --------------------------------------------------------------------------
_DEVA_CONSONANTS: Dict[str, str] = {
    "क": "k", "ख": "kh", "ग": "g", "घ": "gh", "ङ": "ṅ",
    "च": "c", "छ": "ch", "ज": "j", "झ": "jh", "ञ": "ñ",
    "ट": "ṭ", "ठ": "ṭh", "ड": "ḍ", "ढ": "ḍh", "ण": "ṇ",
    "त": "t", "थ": "th", "द": "d", "ध": "dh", "न": "n",
    "प": "p", "फ": "ph", "ब": "b", "भ": "bh", "म": "m",
    "य": "y", "र": "r", "ल": "l", "ळ": "ḷ", "व": "v",
    "श": "ś", "ष": "ṣ", "स": "s", "ह": "h",
    "ऩ": "ṉ", "ऱ": "ṟ", "ऴ": "ḻ",
    # Precomposed nukta letters, in case the input was never NFC-normalised.
    "क़": "q", "ख़": "ḵh", "ग़": "ġ", "ज़": "z",
    "ड़": "ṛ", "ढ़": "ṛh", "फ़": "f", "य़": "y",
}

# consonant + U+093C NUKTA
_DEVA_NUKTA: Dict[str, str] = {
    "क": "q", "ख": "ḵh", "ग": "ġ", "ज": "z",
    "ड": "ṛ", "ढ": "ṛh", "फ": "f", "य": "y",
}

_DEVA_VOWELS: Dict[str, str] = {
    "अ": "a", "आ": "ā", "इ": "i", "ई": "ī",
    "उ": "u", "ऊ": "ū", "ऋ": "ṛ", "ॠ": "ṝ",
    "ऌ": "ḷ", "ॡ": "ḹ",
    "ऍ": "ê", "ऎ": "e", "ए": "e", "ऐ": "ai",
    "ऑ": "ô", "ऒ": "o", "ओ": "o", "औ": "au",
}

_DEVA_MATRAS: Dict[str, str] = {
    "ा": "ā", "ि": "i", "ी": "ī",
    "ु": "u", "ू": "ū", "ृ": "ṛ",
    "ॄ": "ṝ", "ॢ": "ḷ", "ॣ": "ḹ",
    "ॅ": "ê", "ॆ": "e", "े": "e", "ै": "ai",
    "ॉ": "ô", "ॊ": "o", "ो": "o", "ौ": "au",
}

_DEVA_SIGNS: Dict[str, str] = {
    "ं": "ṃ",      # anusvara
    "ँ": "ṁ",      # candrabindu
    "ः": "ḥ",      # visarga
    "ॐ": "oṃ",     # om
    "ऽ": "'",            # avagraha
    "।": ".",            # danda
    "॥": ".",            # double danda
    "॰": ".",            # abbreviation sign
}

_VIRAMA = "्"
_NUKTA = "़"


def _composed(table: Dict[str, str]) -> Dict[str, str]:
    """NFC every value, so the output never comes back half-decomposed.

    The tables are written with letters like ``ā`` and ``ṣ``.  Saved
    from one editor those are single codepoints, from another they are a letter
    plus a combining mark, and the difference would leak straight into the
    result and break string equality for the caller.  Composing here settles it
    once, at import.
    """
    return {key: unicodedata.normalize("NFC", value) for key, value in table.items()}

_DEVA_CONSONANTS = _composed(_DEVA_CONSONANTS)
_DEVA_NUKTA = _composed(_DEVA_NUKTA)
_DEVA_VOWELS = _composed(_DEVA_VOWELS)
_DEVA_MATRAS = _composed(_DEVA_MATRAS)
_DEVA_SIGNS = _composed(_DEVA_SIGNS)

# --------------------------------------------------------------------------
# Cyrillic
# --------------------------------------------------------------------------
_CYRILLIC_RU: Dict[str, str] = {
    "а": "a", "б": "b", "в": "v", "г": "g", "д": "d", "е": "e", "ё": "yo",
    "ж": "zh", "з": "z", "и": "i", "й": "y", "к": "k", "л": "l", "м": "m",
    "н": "n", "о": "o", "п": "p", "р": "r", "с": "s", "т": "t", "у": "u",
    "ф": "f", "х": "kh", "ц": "ts", "ч": "ch", "ш": "sh", "щ": "shch",
    "ъ": "", "ы": "y", "ь": "", "э": "e", "ю": "yu", "я": "ya",
    "і": "i", "ї": "yi", "є": "ye", "ґ": "g", "ў": "w", "ј": "j",
    # Serbian and Macedonian. Without these a Serbian name came back half
    # romanised while the result still reported supported=True, which is worse
    # than refusing outright: the caller had no way to tell it was incomplete.
    "ђ": "dj", "ћ": "c", "љ": "lj", "њ": "nj", "џ": "dz",
    "ѓ": "g", "ќ": "k", "ѕ": "dz",
}

# The Ukrainian national system, which is what road signs and passports use and
# what turns "Київ" into "Kyiv" rather than "Kyyiv".  Four letters depend on
# where in the word they sit.
_CYRILLIC_UK: Dict[str, str] = dict(_CYRILLIC_RU)
_CYRILLIC_UK.update(
    {"г": "h", "ґ": "g", "и": "y", "і": "i", "ї": "i", "є": "ie", "й": "i",
     "ю": "iu", "я": "ia", "щ": "shch", "х": "kh", "ц": "ts"}
)
_CYRILLIC_UK_INITIAL: Dict[str, str] = {
    "є": "ye", "ї": "yi", "й": "y", "ю": "yu", "я": "ya",
}

#: Letters that only Ukrainian uses; seeing one switches the Cyrillic table.
_UKRAINIAN_MARKERS = frozenset("іїєґ")

# --------------------------------------------------------------------------
# Greek, romanised to ISO 843 - the standard the README names. ISO 843 covers
# MODERN Greek, so beta is "v" and phi is "f", the way the letters are said now.
# This table held the classical scheme instead ("b", "ph", macrons on eta and
# omega) while the documentation promised ISO 843. The documentation is what
# users read, so the table was the thing that was wrong.
# (accents are folded away before the lookup)
# --------------------------------------------------------------------------
_GREEK: Dict[str, str] = {
    "α": "a", "β": "v", "γ": "g", "δ": "d", "ε": "e", "ζ": "z", "η": "i", "θ": "th",
    "ι": "i", "κ": "k", "λ": "l", "μ": "m", "ν": "n", "ξ": "x", "ο": "o", "π": "p",
    "ρ": "r", "σ": "s", "ς": "s", "τ": "t", "υ": "y", "φ": "f", "χ": "ch", "ψ": "ps",
    "ω": "o", "ϐ": "v", "ϑ": "th", "ϕ": "f", "ϱ": "r",
}

#: The two punctuation marks that belong to Greek rather than to Common.
#:
#: They are mapped straight off the raw text, before :func:`fold_marks` runs,
#: because both are canonical singletons: U+0387 GREEK ANO TELEIA decomposes to
#: U+00B7 MIDDLE DOT and U+037E GREEK QUESTION MARK to an ASCII semicolon, and
#: NFC never puts a singleton back together.  Kept in ``_GREEK`` alongside the
#: letters they were dead entries -- the fold had already rewritten them out of
#: existence by the time the lookup ran.
_GREEK_PUNCTUATION = {0x0387: ";", 0x037E: "?"}


_CYRILLIC_RU = _composed(_CYRILLIC_RU)
_CYRILLIC_UK = _composed(_CYRILLIC_UK)
_CYRILLIC_UK_INITIAL = _composed(_CYRILLIC_UK_INITIAL)
_GREEK = _composed(_GREEK)


class Transliteration(str):
    """The romanised text, plus what happened to it.

    It *is* a ``str``, so it compares, slices, formats and joins like any other
    string.  The extra attributes are there when a caller needs to know whether
    anything was actually converted::

        out = transliterate("مرحبا")
        out == "مرحبا"        # True: nothing this package can romanise
        out.supported          # False
        out.note               # "Arabic not covered and returned unchanged"
    """

    __slots__ = ("source", "target", "scripts", "converted", "untouched", "note")

    def __new__(
        cls,
        text: str,
        source: str,
        target: str,
        scripts: Tuple[str, ...],
        converted: Tuple[str, ...],
        untouched: Tuple[str, ...],
        note: str,
    ) -> "Transliteration":
        obj = super().__new__(cls, text)
        obj.source = source
        obj.target = target
        obj.scripts = scripts
        obj.converted = converted
        obj.untouched = untouched
        obj.note = note
        return obj

    @property
    def supported(self) -> bool:
        """True when every script in the input could be romanised."""
        return not self.untouched

    @property
    def changed(self) -> bool:
        """True when the output differs from the input."""
        return str(self) != self.source

    def summary(self) -> str:
        """One line of plain ASCII describing the conversion."""
        head = "multilingual-text: transliterate to %s" % self.target
        return "%s -- %s" % (head, self.note)

    def to_dict(self) -> Dict[str, object]:
        """JSON-safe view of the result."""
        return {
            "text": str(self),
            "source": self.source,
            "target": self.target,
            "scripts": list(self.scripts),
            "converted": list(self.converted),
            "untouched": list(self.untouched),
            "supported": self.supported,
            "changed": self.changed,
            "note": self.note,
        }

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return "Transliteration(%s, supported=%s)" % (str.__repr__(self), self.supported)


def _runs(text: str) -> List[Tuple[str, str]]:
    """Split ``text`` into (script, chunk) runs, keeping the original order."""
    out: List[Tuple[str, str]] = []
    for ch in text:
        script = script_block(ch)
        if out and out[-1][0] == script:
            out[-1] = (script, out[-1][1] + ch)
        else:
            out.append((script, ch))
    return out


def _apply_case(word: str, romanised: str) -> str:
    """Carry the word's capitalisation across to the romanised form."""
    cased = [ch for ch in word if ch.isalpha()]
    if not cased or not romanised:
        return romanised
    if len(cased) > 1 and word.isupper():
        return romanised.upper()
    if cased[0].isupper():
        return romanised[:1].upper() + romanised[1:]
    return romanised


def _devanagari(text: str) -> str:
    """Devanagari to IAST, honouring the inherent vowel and the virama."""
    out: List[str] = []
    index = 0
    length = len(text)
    while index < length:
        ch = text[index]
        if ch in _DEVA_CONSONANTS:
            base = _DEVA_CONSONANTS[ch]
            index += 1
            if index < length and text[index] == _NUKTA:
                base = _DEVA_NUKTA.get(ch, base)
                index += 1
            if index < length and text[index] == _VIRAMA:
                out.append(base)
                index += 1
            elif index < length and text[index] in _DEVA_MATRAS:
                out.append(base + _DEVA_MATRAS[text[index]])
                index += 1
            else:
                out.append(base + "a")
            continue
        if ch in _DEVA_VOWELS:
            out.append(_DEVA_VOWELS[ch])
        elif ch in _DEVA_SIGNS:
            out.append(_DEVA_SIGNS[ch])
        elif ch in _DEVA_MATRAS:
            # A stray matra with no consonant in front of it.
            out.append(_DEVA_MATRAS[ch])
        elif ch in (_VIRAMA, _NUKTA):
            pass
        elif "०" <= ch <= "९":
            out.append(chr(ord("0") + ord(ch) - 0x0966))
        else:
            out.append(ch)
        index += 1
    return "".join(out)


def _cyrillic(text: str, table: Dict[str, str], ukrainian: bool = False) -> str:
    """Map a Cyrillic word, honouring the word-initial forms Ukrainian needs."""
    out: List[str] = []
    lowered = text.lower()
    for index, ch in enumerate(lowered):
        if ukrainian and index == 0 and ch in _CYRILLIC_UK_INITIAL:
            out.append(_CYRILLIC_UK_INITIAL[ch])
            continue
        out.append(table.get(ch, ch))
    return "".join(out)


def _greek(text: str) -> str:
    """Romanise a Greek run, punctuation included.

    The punctuation is translated first: :func:`fold_marks` would otherwise
    canonicalise the ano teleia into a middle dot and the Greek question mark
    into an ASCII semicolon, and neither survives to be looked up.
    """
    folded = fold_marks(text.translate(_GREEK_PUNCTUATION)).lower()
    return "".join(_GREEK.get(ch, ch) for ch in folded)


def _to_ascii(text: str) -> str:
    return _normalize(
        text, form="NFC", whitespace=False, quotes=False, diacritics=True
    )


def _note(converted: Tuple[str, ...], untouched: Tuple[str, ...], changed: bool) -> str:
    if untouched and converted:
        return "%s romanised; %s not covered and returned unchanged" % (
            ", ".join(converted),
            ", ".join(untouched),
        )
    if untouched:
        return "%s not covered and returned unchanged" % ", ".join(untouched)
    if converted:
        return "%s romanised" % ", ".join(converted)
    if changed:
        return "only spacing or punctuation changed"
    return "nothing to romanise; the text is already Latin"


def transliterate(text: str, *, to: str = "latin") -> Transliteration:
    """Romanise Devanagari, Cyrillic and Greek; leave every other script alone.

    Args:
        text: the text to romanise.
        to: ``"latin"`` for scholarly Latin with diacritics (IAST for
            Devanagari, macrons for Greek eta and omega), or ``"ascii"`` for
            the same thing with the diacritics folded away.

    Returns:
        A :class:`Transliteration`, which is a ``str`` carrying ``.supported``,
        ``.changed``, ``.converted``, ``.untouched`` and ``.note``.  Text in a
        script this package does not cover comes back byte-for-byte unchanged
        and is named in ``.untouched``; nothing is raised for it.

    Raises:
        TypeError: if ``text`` is not a string.
        ValueError: if ``to`` is not ``"latin"`` or ``"ascii"``.

    Examples:
        ``transliterate("नमस्ते")`` gives ``"namaste"``,
        ``transliterate("Привет, мир!")`` gives ``"Privet, mir!"`` and
        ``transliterate("Ελλάδα")`` gives ``"Ellada"``.  A script that is not
        covered comes straight back::

            out = transliterate("日本")
            str(out) == "日本"   # True
            out.supported        # False
            out.untouched        # ("Han",)
    """
    if not isinstance(text, str):
        raise TypeError("transliterate() needs a str, got %s" % type(text).__name__)
    target = (to or "").lower()
    if target not in TARGETS:
        raise ValueError(
            "to must be one of %s, not %r" % (", ".join(TARGETS), to)
        )
    if not text:
        return Transliteration("", "", target, (), (), (), "empty text")

    cyrillic_table = (
        _CYRILLIC_UK
        if any(ch in _UKRAINIAN_MARKERS for ch in text.lower())
        else _CYRILLIC_RU
    )

    pieces: List[str] = []
    seen: List[str] = []
    converted: List[str] = []
    untouched: List[str] = []
    for script, chunk in _runs(text):
        if script not in seen and script not in (COMMON,):
            seen.append(script)
        if script == "Devanagari":
            piece = _devanagari(chunk)
        elif script == "Cyrillic":
            piece = _apply_case(
                chunk, _cyrillic(chunk, cyrillic_table, cyrillic_table is _CYRILLIC_UK)
            )
        elif script == "Greek":
            piece = _apply_case(chunk, _greek(chunk))
        else:
            piece = chunk
        if script in SUPPORTED_SCRIPTS:
            if script not in converted:
                converted.append(script)
            if target == "ascii":
                piece = _to_ascii(piece)
        elif script == "Latin":
            if target == "ascii":
                piece = _to_ascii(piece)
        elif script != COMMON and script not in untouched:
            untouched.append(script)
        pieces.append(piece)

    out = "".join(pieces)
    note = _note(tuple(converted), tuple(untouched), out != text)
    return Transliteration(
        out, text, target, tuple(seen), tuple(converted), tuple(untouched), note
    )
