"""Text clean-up: one function, seven switches, every one of them idempotent.

``normalize(normalize(x)) == normalize(x)`` holds for every combination of
options, which is what makes it safe to put in front of a cache key, a
deduplication hash, or a search index.
"""
from __future__ import annotations

import re
import unicodedata
from typing import Dict, Optional

__all__ = ["CASE_MODES", "FORMS", "fold_marks", "normalize"]

#: The Unicode normalisation forms ``normalize`` accepts, plus ``None``.
FORMS = ("NFC", "NFD", "NFKC", "NFKD")

#: The values ``case=`` accepts, plus ``None`` for "leave the case alone".
CASE_MODES = ("lower", "upper", "title", "fold")

# Curly quotes, guillemets, dashes, ellipses and primes folded to ASCII.
_PUNCTUATION_FOLD: Dict[int, str] = {
    0x2018: "'", 0x2019: "'", 0x201A: "'", 0x201B: "'",
    0x201C: '"', 0x201D: '"', 0x201E: '"', 0x201F: '"',
    0x2032: "'", 0x2033: '"', 0x2035: "'", 0x2036: '"',
    0x00AB: '"', 0x00BB: '"', 0x2039: "'", 0x203A: "'",
    0x301D: '"', 0x301E: '"', 0x301F: '"',
    0xFF02: '"', 0xFF07: "'",
    0x2010: "-", 0x2011: "-", 0x2012: "-", 0x2013: "-", 0x2014: "-",
    0x2015: "-", 0x2212: "-", 0xFE58: "-", 0xFE63: "-", 0xFF0D: "-",
    0x2026: "...",
    0x00B7: ".", 0x2022: "*", 0x2027: ".",
}

# Anything that should read as a plain space, plus the invisible characters
# that break string comparison for no visible reason.
_SPACE_CHARS = "".join(
    chr(code)
    for code in (
        0x00A0,  # no-break space
        0x1680,  # ogham space mark
        0x2000, 0x2001, 0x2002, 0x2003, 0x2004, 0x2005, 0x2006,
        0x2007,  # figure space
        0x2008, 0x2009, 0x200A,
        0x202F,  # narrow no-break space
        0x205F,  # medium mathematical space
        0x3000,  # ideographic space
    )
)
_ZERO_WIDTH = "".join(
    chr(code)
    for code in (
        0x200B,  # zero width space
        0x200C,  # zero width non-joiner
        0x200D,  # zero width joiner
        0x2060,  # word joiner
        0xFEFF,  # zero width no-break space / byte order mark
        0x00AD,  # soft hyphen
    )
)
_WHITESPACE_FOLD: Dict[int, Optional[str]] = {ord(ch): " " for ch in _SPACE_CHARS}
_WHITESPACE_FOLD.update({ord(ch): None for ch in _ZERO_WIDTH})

# Latin letters with no canonical decomposition, so NFD cannot strip them.
_UNDECOMPOSABLE: Dict[int, str] = {
    0x00D8: "O", 0x00F8: "o", 0x0110: "D", 0x0111: "d",
    0x0126: "H", 0x0127: "h", 0x0141: "L", 0x0142: "l",
    0x00C6: "AE", 0x00E6: "ae", 0x0152: "OE", 0x0153: "oe",
    0x00DE: "Th", 0x00FE: "th", 0x00D0: "D", 0x00F0: "d",
    0x0166: "T", 0x0167: "t", 0x0131: "i", 0x0130: "I",
    0x00DF: "ss", 0x1E9E: "SS",
}

# The generic combining-mark blocks.  Deliberately *not* every Mn character:
# stripping those would destroy Devanagari vowel signs and Arabic vowel points,
# which are not decoration but part of the word.
_COMBINING_BLOCKS = (
    (0x0300, 0x036F),
    (0x1AB0, 0x1AFF),
    (0x1DC0, 0x1DFF),
    (0x20D0, 0x20F0),
    (0xFE20, 0xFE2F),
)

_MULTISPACE = re.compile(r"[ \t\r\f\v]+")
_BLANK_LINES = re.compile(r"\n{3,}")


def _is_generic_mark(ch: str) -> bool:
    code = ord(ch)
    return any(low <= code <= high for low, high in _COMBINING_BLOCKS)


def _fold_quotes(text: str) -> str:
    """Smart quotes, guillemets, dashes, primes and ellipses to ASCII.

    Called once up front and then again after every stage that can *uncover* a
    character this table knows about, because a fold that only runs first is a
    fold the second call finishes -- see the call sites in :func:`normalize`.
    """
    return text.translate(_PUNCTUATION_FOLD)


def fold_marks(text: str) -> str:
    """Drop generic combining accents, leaving Indic and Arabic marks alone.

    ``fold_marks("résumé") == "resume"``, while
    ``fold_marks("नमस्ते") == "नमस्ते"``.
    """
    decomposed = unicodedata.normalize("NFD", text)
    stripped = "".join(ch for ch in decomposed if not _is_generic_mark(ch))
    return unicodedata.normalize("NFC", stripped)


def _strip_diacritics(text: str) -> str:
    """Accents off first, then the letters Unicode refuses to decompose.

    The order matters.  With the table first it runs too early to see what the
    accent strip uncovers: U+01FE LATIN CAPITAL LETTER O WITH STROKE AND ACUTE
    decomposes to U+00D8 plus an acute, so the O-with-stroke only appears once
    :func:`fold_marks` has peeled the accent away -- by which time the table had
    already run and would never run again.  The first call then returned an
    O-with-stroke and a second call turned it into "O", which is exactly the
    idempotence this module promises.  Same story for the six letters built on
    U+00C6 / U+00E6 and U+00D8 / U+00F8.
    """
    return fold_marks(text).translate(_UNDECOMPOSABLE)


def _to_ascii_digits(text: str) -> str:
    """Any Unicode decimal digit becomes its ASCII equivalent."""
    out = []
    for ch in text:
        if "0" <= ch <= "9":
            out.append(ch)
            continue
        value = unicodedata.decimal(ch, None)
        out.append(str(value) if value is not None else ch)
    return "".join(out)


def _drop_punctuation(text: str) -> str:
    return "".join(ch for ch in text if unicodedata.category(ch)[0] != "P")


def _apply_case(text: str, case: str) -> str:
    if case == "lower":
        return text.lower()
    if case == "upper":
        return text.upper()
    if case == "title":
        return text.title()
    if case == "fold":
        return text.casefold()
    raise ValueError(
        "case must be one of %s or None, not %r" % (", ".join(CASE_MODES), case)
    )


def _collapse_whitespace(text: str) -> str:
    """Runs of spaces become one space; blank-line runs become one blank line."""
    lines = [_MULTISPACE.sub(" ", line).strip() for line in text.split("\n")]
    joined = "\n".join(lines)
    return _BLANK_LINES.sub("\n\n", joined).strip()


def normalize(
    text: str,
    *,
    form: Optional[str] = "NFC",
    case: Optional[str] = None,
    whitespace: bool = True,
    punctuation: bool = False,
    digits: bool = False,
    quotes: bool = True,
    diacritics: bool = False,
) -> str:
    """Clean up ``text`` so that two strings that look the same compare equal.

    Applying this twice changes nothing: ``normalize(normalize(x))`` always
    equals ``normalize(x)``, for every combination of the options below.

    Args:
        text: the text to clean up.
        form: Unicode normalisation form, one of ``"NFC"``, ``"NFD"``,
            ``"NFKC"``, ``"NFKD"``, or ``None`` to leave the encoding alone.
            ``normalize("e\\u0301")`` gives ``"e\\u0301"`` NFC-composed to
            ``"\\u00e9"``, so a decomposed and a precomposed accent match.
        case: ``"lower"``, ``"upper"``, ``"title"``, ``"fold"``, or ``None``.
            ``normalize("Straße", case="fold")`` gives ``"strasse"``, while
            ``case="lower"`` gives ``"straße"``.
        whitespace: collapse runs of spaces and tabs to one space, turn
            no-break and other exotic spaces into ordinary ones, delete
            zero-width characters, and strip the ends.
            ``normalize("  a\\u00a0 b  ")`` gives ``"a b"``.
        punctuation: delete punctuation.
            ``normalize("Hello, world!", punctuation=True)`` gives
            ``"Hello world"``.  Note that it joins hyphenated words:
            ``"co-operate"`` becomes ``"cooperate"``.
        digits: convert every Unicode decimal digit to ASCII.
            ``normalize("٤٢ और ४२", digits=True)`` gives ``"42 और 42"``.
        quotes: fold smart quotes, guillemets, dashes and ellipses to ASCII.
            ``normalize("“a” – b…")`` gives ``'"a" - b...'``.
        diacritics: strip accents from Latin, Greek and Cyrillic letters, and
            map the ones Unicode cannot decompose (ø, ł, đ, æ, œ, ß) by hand.
            Indic vowel signs and Arabic vowel points are left alone, because
            they are part of the word rather than decoration.
            ``normalize("Crème Brûlée", diacritics=True)`` gives
            ``"Creme Brulee"``; ``normalize("नमस्ते", diacritics=True)`` is
            unchanged.

    Returns:
        The cleaned text.

    Raises:
        TypeError: if ``text`` is not a string.
        ValueError: if ``form`` or ``case`` is not a recognised value.
    """
    if not isinstance(text, str):
        raise TypeError("normalize() needs a str, got %s" % type(text).__name__)
    if form is not None and form not in FORMS:
        raise ValueError(
            "form must be one of %s or None, not %r" % (", ".join(FORMS), form)
        )
    if not text:
        return ""

    if quotes:
        text = _fold_quotes(text)
    if whitespace:
        text = text.translate(_WHITESPACE_FOLD)
    if form is not None:
        # Resolve the encoding *before* the stages that inspect characters.
        # A compatibility form expands a character into several, and some of
        # those carry accents that were invisible a moment ago: NFKC turns the
        # single character U+01C5 into "D" plus a z-with-caron, and U+3371 into
        # "hPa".  Stripping accents or folding case first would miss them, and
        # the second call would catch them -- which is exactly the idempotence
        # this function promises.  It is applied again at the end, because case
        # mapping and diacritic stripping can both denormalise the text.
        text = unicodedata.normalize(form, text)
        if quotes:
            # The fold has to run again, because the form pass *uncovers*
            # punctuation that was not foldable a moment ago.  U+0387 GREEK
            # ANO TELEIA is canonically U+00B7 MIDDLE DOT, which is itself one
            # of the characters folded to "."; NFKC turns U+013F into "L" plus
            # that same middle dot, U+2034 into three primes, U+FE31 into an em
            # dash.  Folding only before the form pass left all of those
            # standing and let the second call fold them -- exactly the
            # idempotence this function promises.  It runs here, ahead of
            # `punctuation`, so a caller who asked for punctuation to be
            # dropped does not get the newly exposed ASCII back in the result.
            text = _fold_quotes(text)
    if digits:
        text = _to_ascii_digits(text)
    if punctuation:
        text = _drop_punctuation(text)
    if diacritics:
        text = _strip_diacritics(text)
        if quotes:
            # And again, for the same reason, because `diacritics` runs a
            # decomposition of its own: fold_marks() NFD-decomposes U+0387 to
            # U+00B7 and NFC never puts a singleton like that back together,
            # so the ano teleia comes out of this stage foldable even when
            # `form` is None and the form pass above never ran.  The only
            # character this can newly expose is that middle dot, which is
            # ordinary punctuation and has already been dropped above when the
            # caller asked for `punctuation=True`.
            text = _fold_quotes(text)
    if case is not None:
        # Case mapping is defined on composed characters.  Titlecasing a
        # decomposed "e + combining grave" treats the accent as a word break
        # and capitalises the letter after it, so "Crem̀e" would come back
        # "Crem̀E" -- and a second call would do it again somewhere else,
        # which is exactly the idempotence this function promises.  Compose
        # first and let `form` below put the text back into the shape the
        # caller asked for.  A compatibility form composes with NFKC, so the
        # case mapping sees the characters that form is going to expand to
        # anyway (the square "hPa" glyph, roman numerals, ligatures).
        if form in ("NFKC", "NFKD"):
            text = unicodedata.normalize("NFKC", text)
        elif form is not None:
            text = unicodedata.normalize("NFC", text)
        text = _apply_case(text, case)
    if form is not None:
        text = unicodedata.normalize(form, text)
    if diacritics and form in ("NFD", "NFKD"):
        # NFD would put the accents back that NFC had recomposed; strip once
        # more so the result is stable under a second call.
        text = "".join(ch for ch in text if not _is_generic_mark(ch))
    if whitespace:
        text = _collapse_whitespace(text)
    return text
