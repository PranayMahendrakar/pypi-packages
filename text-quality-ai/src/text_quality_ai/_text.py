"""Tokenising, sentence splitting and syllable counting.

Everything here is pure Python and deliberately dependency free. No model is
downloaded and no corpus is needed, so the same numbers come out on every
machine. The syllable counter is a heuristic; :func:`syllables` documents its
rules and the README states its limits.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from functools import lru_cache
from typing import List, Sequence, Tuple

from ._words import ABBREVIATIONS

# --------------------------------------------------------------------------
# character classes
# --------------------------------------------------------------------------

#: Scripts written without spaces between words: Han, kana, Hangul.
CJK_RANGES = (
    r"ᄀ-ᇿ"      # Hangul jamo
    r"぀-ヿ"      # hiragana and katakana
    r"ㇰ-ㇿ"      # katakana phonetic extensions
    r"㐀-䶿"      # CJK unified ideographs extension A
    r"一-鿿"      # CJK unified ideographs
    r"ꥠ-꥿"      # Hangul jamo extended A
    r"가-힯"      # Hangul syllables
    r"豈-﫿"      # CJK compatibility ideographs
)

#: Combining marks. ``\w`` does not match them, but they belong to the word they
#: attach to (Devanagari matras and virama, Arabic and Hebrew points, accents).
MARKS = (
    r"̀-ͯ҃-҉֑-ׇؐ-ًؚ-ٰٟ"
    r"ۖ-ܑۜܰ-݊ަ-ްࠖ-࠭"
    r"ऀ-ःऺ-ॏ॑-ॗॢॣ"
    r"ঁ-ঃ়-্ৗৢৣ"
    r"ਁ-ਃ਼-ੑੰੱઁ-ઃ઼-્"
    r"ଁ-ଃ଼-ୗஂா-்ௗ"
    r"ఀ-ఃా-ౖಁ-ಃ಼-್ೕೖ"
    r"ഀ-ഃ഻-്ൗංඃ්-ෟ"
    r"ัิ-ฺ็-๎ັິ-ຼ່-ໍ"
    r"ཱ-྄྆྇ါ-ှၖ-ၙၞ-ၠ"
    r"឴-៓ᩕ-᩿ᬀ-ᬄ᬴-᭄᳐-᳹"
    r"⃐-⃰︀-️"
)

#: Characters allowed to join two halves of one word.
JOINERS = r"'’‐‑-"

_LETTER = r"[^\W\d_" + CJK_RANGES + r"]"
_INWORD = r"(?:[^\W_" + CJK_RANGES + r"]|[" + MARKS + r"]|[" + JOINERS + r"])"

TOKEN_RE = re.compile(
    r"[" + CJK_RANGES + r"]"                # one space-less character = one token
    r"|" + _LETTER + _INWORD + r"*"         # a word, with hyphens and apostrophes inside
    r"|\d+(?:[.,]\d+)*"                     # a number
)
CJK_RE = re.compile(r"[" + CJK_RANGES + r"]")

_TRIM_RE = re.compile(r"^[" + JOINERS + r"]+|[" + JOINERS + r"]+$")
_PARAGRAPH_RE = re.compile(r"\n[ \t\r\f\v]*\n+")

#: Full stop, bang, question mark, ellipsis, their full-width forms, and danda.
SENTENCE_ENDINGS = ".!?…。！？।॥"
_WIDE_ENDINGS = "。！？।॥"
_CLOSERS = "\"')]}»”’」』"

_VOWEL_RUN_RE = re.compile(r"[aeiouy]+")
_NON_LATIN_VOWELS = set(
    "αεηιουω"                          # Greek
    "аеёиоуыэюя"        # Cyrillic
)


# --------------------------------------------------------------------------
# tokens
# --------------------------------------------------------------------------

def tokenize(text: str) -> List[str]:
    """Split ``text`` into word tokens.

    Words in spaced scripts (Latin, Devanagari, Cyrillic and so on) stay whole,
    including their combining marks, internal hyphens and apostrophes. Scripts
    written without spaces (Han, kana, Hangul) give one token per character, so
    such text never reports a single enormous word.
    """
    out: List[str] = []
    for raw in TOKEN_RE.findall(text):
        token = _TRIM_RE.sub("", raw)
        if token:
            out.append(token)
    return out


def cjk_share(tokens: Sequence[str]) -> float:
    """Share of the word characters in ``tokens`` that are space-less (Han, kana, Hangul).

    Counted per character rather than per token, because a space-less script
    produces one token per character and would otherwise dominate a document
    that is mostly Latin with one short quotation in it.
    """
    total = sum(len(t) for t in tokens)
    if not total:
        return 0.0
    hits = sum(len(t) for t in tokens if CJK_RE.match(t))
    return hits / total


# --------------------------------------------------------------------------
# sentences and paragraphs
# --------------------------------------------------------------------------

def split_paragraphs(text: str) -> List[str]:
    """Blocks of ``text`` separated by a blank line; blank blocks are dropped."""
    return [p.strip() for p in _PARAGRAPH_RE.split(text or "") if p.strip()]


def _next_word_is_capitalised(text: str, index: int) -> bool:
    while index < len(text) and text[index].isspace():
        index += 1
    return index < len(text) and text[index].isupper()


def _looks_like_abbreviation(text: str, dot: int, after: int) -> bool:
    i = dot - 1
    dotted = False
    while i >= 0 and (text[i].isalpha() or text[i] == "."):
        if text[i] == ".":
            dotted = True                    # a dotted form such as e.g. or p.m.
            break
        i -= 1
    word = text[i + 1:dot]
    if dotted:
        # "at 4 p.m. He left" ends a sentence; "e.g. this one" does not.
        return not _next_word_is_capitalised(text, after)
    if not word:
        return False
    if len(word) == 1 and word.isupper():
        return True                          # an initial, as in "J. Smith"
    return word.lower() in ABBREVIATIONS


def split_sentences(text: str) -> List[str]:
    """Split ``text`` into sentences.

    A sentence ends at ``.``, ``!``, ``?``, an ellipsis, their full-width forms
    or a danda, and always at a paragraph break. Text with no sentence-ending
    punctuation at all is one sentence. Common abbreviations and initials do not
    end a sentence.
    """
    sentences: List[str] = []
    for para in split_paragraphs(text):
        n = len(para)
        start = 0
        i = 0
        while i < n:
            ch = para[i]
            if ch in SENTENCE_ENDINGS:
                j = i + 1
                while j < n and para[j] in SENTENCE_ENDINGS:
                    j += 1
                while j < n and para[j] in _CLOSERS:
                    j += 1
                wide = ch in _WIDE_ENDINGS
                if wide or j >= n or para[j].isspace():
                    if ch == "." and _looks_like_abbreviation(para, i, j):
                        i = j
                        continue
                    chunk = para[start:j].strip()
                    if chunk:
                        sentences.append(chunk)
                    start = j
                    i = j
                    continue
            i += 1
        tail = para[start:].strip()
        if tail:
            sentences.append(tail)
    return sentences


# --------------------------------------------------------------------------
# syllables
# --------------------------------------------------------------------------

def _devanagari_syllables(word: str) -> int:
    count = 0
    n = len(word)
    for k, ch in enumerate(word):
        code = ord(ch)
        consonant = (
            0x0915 <= code <= 0x0939
            or 0x0958 <= code <= 0x095F
            or 0x0978 <= code <= 0x097F
        )
        if consonant:
            nxt = ord(word[k + 1]) if k + 1 < n else 0
            if nxt == 0x094D:                                  # virama: no vowel here
                continue
            if 0x093E <= nxt <= 0x094C or nxt in (0x0962, 0x0963):
                continue                                       # the matra counts instead
            count += 1                                         # inherent "a"
        elif 0x0905 <= code <= 0x0914:                         # independent vowel
            count += 1
        elif 0x093E <= code <= 0x094C or code in (0x0962, 0x0963):
            count += 1                                         # vowel sign
    return max(1, count)


#: Words whose neighbouring vowels belong to separate syllables (hiatus).
#: :func:`_latin_syllables` counts vowel *runs*, so "i-de-a" and "sci-ence"
#: would come out one short. There is no cheap rule that splits those without
#: also splitting "team" and "nation", so the common cases are simply listed.
#: The list is short on purpose and the README says the counter is a heuristic.
HIATUS_SYLLABLES = {
    "area": 3, "areas": 3,
    "audio": 3,
    "create": 2, "created": 3, "creates": 2, "creation": 3, "creations": 3,
    "creative": 3,
    "diet": 2, "diets": 2,
    "idea": 3, "ideas": 3, "ideal": 3, "ideals": 3,
    "media": 3, "medium": 3,
    "poem": 2, "poems": 2, "poet": 2, "poetry": 3,
    "quiet": 2, "quieter": 3, "quietly": 3,
    "radio": 3, "radios": 3, "ratio": 3, "ratios": 3,
    "react": 2, "reacts": 2, "reaction": 3, "reactions": 3,
    "science": 2, "sciences": 3,
    "trial": 2, "trials": 2,
    "via": 2, "video": 3, "videos": 3,
}


def _latin_syllables(folded: str) -> int:
    count = len(_VOWEL_RUN_RE.findall(folded))
    if count == 0:
        return 1
    if folded.endswith("es") and len(folded) > 3:
        stem = folded[:-2]
        # "-es" keeps a syllable of its own after a sibilant - wat-ches, di-shes - and
        # after a consonant plus l, where the "-le" carries it: bot-tles, can-dles,
        # ta-bles. Treating those as silent shortened every such plural by one syllable
        # and made Flesch report the text as easier than it reads.
        sibilant = stem.endswith(("ch", "sh"))
        syllabic_l = stem.endswith("l") and len(stem) >= 2 and stem[-2] not in "aeiouy"
        if not sibilant and not syllabic_l and folded[-3] not in "cszxgaeiouy":
            count -= 1
    elif folded.endswith("ed") and len(folded) > 3 and folded[-3] not in "tdaeiouy":
        count -= 1
    elif folded.endswith("e") and not folded.endswith(("le", "ee", "ie", "oe", "ue", "ye")):
        count -= 1
    elif folded.endswith("ing") and len(folded) >= 5 and folded[-4] in "aeiouy":
        count += 1                    # be-ing, do-ing, see-ing, play-ing
    return max(1, count)


@lru_cache(maxsize=200_000)
def syllables(word: str) -> int:
    """Estimated number of syllables in one word.

    The rules are applied in this order and each is a heuristic:

    1. space-less scripts (Han, kana, Hangul) count one syllable per character;
    2. Devanagari counts independent vowels, vowel signs and every consonant
       carrying the inherent vowel, skipping consonants followed by a virama;
    3. anything that folds down to Latin letters counts vowel runs, with the
       usual silent-``e``, ``-ed`` and ``-es`` corrections and a ``-ing``
       correction for "be-ing"; a short list of common hiatus words
       (:data:`HIATUS_SYLLABLES`, such as "idea" and "science") is read off
       instead, because counting vowel runs merges their vowels into one;
    4. anything else counts Greek or Cyrillic vowels, and failing that falls
       back to one syllable per three characters.

    It will be wrong on some words, most often names and loanwords. It never
    raises and never returns less than 1 for a non-empty word.
    """
    if not word:
        return 0
    if CJK_RE.search(word):
        return max(1, len(CJK_RE.findall(word)))
    if any(0x0900 <= ord(ch) <= 0x097F for ch in word):
        return _devanagari_syllables(word)
    decomposed = unicodedata.normalize("NFD", word.lower())
    folded = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    folded = re.sub(r"[^a-z]", "", folded)
    if folded:
        known = HIATUS_SYLLABLES.get(folded)
        if known is not None:
            return known
        return _latin_syllables(folded)
    other = sum(1 for ch in word.lower() if ch in _NON_LATIN_VOWELS)
    if other:
        return other
    return max(1, round(len(word) / 3))


def count_syllables(tokens: Sequence[str]) -> int:
    """Total estimated syllables over ``tokens``."""
    return sum(syllables(t) for t in tokens)


# --------------------------------------------------------------------------
# the parsed document
# --------------------------------------------------------------------------

@dataclass
class Document:
    """One parsed text: its paragraphs, sentences, tokens and counts."""

    text: str
    paragraphs: List[str]
    sentences: List[str]
    sentence_tokens: List[List[str]]
    tokens: List[str]
    lower_tokens: List[str]
    word_mode: str                        # "space" or "character"
    syllable_total: int
    paragraph_word_counts: List[int] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    @property
    def n_words(self) -> int:
        return len(self.tokens)

    @property
    def n_sentences(self) -> int:
        return len(self.sentences)

    @property
    def n_paragraphs(self) -> int:
        return len(self.paragraphs)

    @property
    def sentence_lengths(self) -> List[int]:
        return [len(group) for group in self.sentence_tokens]

    @property
    def is_empty(self) -> bool:
        return self.n_words == 0


def parse(text: str) -> Document:
    """Parse ``text`` once. Every measure then works from the result."""
    if not isinstance(text, str):
        raise TypeError(f"text must be a string, got {type(text).__name__}")
    # Paragraphs and sentences that hold no words at all (a row of dashes, a
    # lone ellipsis) are dropped, so the counts never disagree with each other.
    paragraphs: List[str] = []
    paragraph_words: List[int] = []
    for block in split_paragraphs(text):
        count = len(tokenize(block))
        if count:
            paragraphs.append(block)
            paragraph_words.append(count)
    sentences: List[str] = []
    sentence_tokens: List[List[str]] = []
    for sentence in split_sentences(text):
        group = tokenize(sentence)
        if group:
            sentences.append(sentence)
            sentence_tokens.append(group)
    tokens = [t for group in sentence_tokens for t in group]
    notes: List[str] = []
    mode = "space"
    if cjk_share(tokens) > 0.3:
        mode = "character"
        notes.append(
            "This text is mostly written in a script with no spaces between words "
            "(Han, kana or Hangul), so words are counted as characters and the "
            "English readability formulas are reported for reference only."
        )
    return Document(
        text=text,
        paragraphs=paragraphs,
        sentences=sentences,
        sentence_tokens=sentence_tokens,
        tokens=tokens,
        lower_tokens=[t.lower() for t in tokens],
        word_mode=mode,
        syllable_total=count_syllables(tokens),
        paragraph_word_counts=paragraph_words,
        notes=notes,
    )


def ngrams(tokens: Sequence[str], n: int) -> List[Tuple[str, ...]]:
    """Every ``n``-gram of ``tokens``, in order."""
    if len(tokens) < n:
        return []
    return [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]
