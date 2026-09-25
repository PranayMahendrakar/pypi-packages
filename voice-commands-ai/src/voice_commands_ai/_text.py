"""Text normalisation, tokenisation, filler words and word similarity.

Everything here is a heuristic tuned for English voice commands. Comparison
keys are case-folded, NFKC-normalised and stripped of accents on precomposed
letters, so ``"KÜCHE"``, ``"Küche"`` and ``"kuche"`` compare equal while the
original spelling is kept for slot values.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, FrozenSet, Iterable, List, Optional, Tuple

# A number (optionally signed, with decimal/thousands separators and an ordinal
# suffix) or a run of letters/digits with inner apostrophes ("don't", "rock'n'roll").
_TOKEN_RE = re.compile(
    r"(?<![\w.])[-+\u2212]?\d+(?:[.,]\d+)*(?:st|nd|rd|th)?"
    r"|[^\W_]+(?:['\u2019\u02bc][^\W_]+)*"
)

#: Hesitation sounds. They are dropped from the input entirely, including from
#: the middle of free-text slots.
DISFLUENCIES: FrozenSet[str] = frozenset(
    {
        "um", "umm", "ummm", "uh", "uhh", "uhm", "uhmm", "er", "erm", "err",
        "hmm", "hm", "hmmm", "mm", "mmm", "ah", "ahh", "eh",
    }
)

#: Politeness phrases: free to skip, and trimmed from the edges of text slots
#: ("play despacito please" plays "despacito").
POLITE_PHRASES: Tuple[str, ...] = (
    "please", "pls", "plz", "kindly", "thanks", "thank you", "thank you very much",
    "thanks a lot", "for me", "if you can", "if you could", "if possible",
)

#: Request framing ("could you", "i want to"): free to skip, and trimmed from the
#: start of a text slot that opens the pattern.
FRAME_PHRASES: Tuple[str, ...] = (
    "could you", "can you", "would you", "will you", "could you please",
    "can you please", "would you please", "would you mind", "do you mind",
    "i want to", "i wanna", "i want you to", "i'd like to", "i would like to",
    "i'd like you to", "i need to", "i need you to", "go ahead and", "let's", "let us",
)

#: Discourse words ("just", "so", "okay"): free to skip, but kept inside text
#: slots because they can be real content ("play just like heaven", "say hello").
SOFT_PHRASES: Tuple[str, ...] = (
    "hey", "hi", "hello", "ok", "okay", "alright", "all right", "just", "so", "well",
    "actually", "basically", "like", "oh", "you know", "i mean", "maybe",
)

#: Small grammatical words. Missing or extra ones cost less than content words.
FUNCTION_WORDS: FrozenSet[str] = frozenset(
    {
        "the", "a", "an", "to", "of", "my", "your", "our", "their", "his", "her",
        "its", "this", "that", "these", "those", "some", "for", "at", "in", "into",
        "with", "and", "is", "are", "be", "me", "it", "them", "us", "you", "i",
    }
)

_APOSTROPHES = ("\u2019", "\u02bc", "'")


def fold(text: str) -> str:
    """Comparison key: NFKC, case-folded, accents dropped, apostrophes removed.

    Accents are dropped only from Latin, Greek and Cyrillic letters, where
    transcribers often leave them out ("Kuche" for "Küche"); marks that are part
    of the spelling in other scripts (Devanagari vowel signs, Japanese dakuten)
    are kept.
    """
    text = unicodedata.normalize("NFKD", unicodedata.normalize("NFKC", text).casefold())
    out = []
    base = ""
    for ch in text:
        if unicodedata.combining(ch):
            if base and ord(base) < 0x0500:
                continue
        else:
            base = ch
        out.append(ch)
    folded = unicodedata.normalize("NFC", "".join(out))
    for mark in _APOSTROPHES:
        folded = folded.replace(mark, "")
    return folded.replace("\u2212", "-")


@dataclass(frozen=True)
class Token:
    """One word of the input: its spelling, comparison key and position."""

    surface: str
    key: str
    start: int
    end: int
    kind: str = ""  # "", "polite", "frame" or "soft"

    @property
    def is_filler(self) -> bool:
        """True for politeness, framing and discourse words ("please", "could you", "just")."""
        return bool(self.kind)

    @property
    def weight(self) -> float:
        """Evidence the word carries: 0 for a filler, 0.3 for a function word, else 1."""
        if self.kind:
            return 0.0
        if self.key in FUNCTION_WORDS:
            return 0.3
        return 1.0


def raw_tokens(text: str) -> List[Token]:
    """Split text into tokens without any filler handling."""
    return [
        Token(m.group(0), fold(m.group(0)), m.start(), m.end())
        for m in _TOKEN_RE.finditer(text)
    ]


def phrase_key(text: str) -> Tuple[str, ...]:
    """The folded token keys of a phrase."""
    return tuple(t.key for t in raw_tokens(unicodedata.normalize("NFC", text)))


def _phrase_table(phrases: Iterable[str], kind: str) -> Dict[Tuple[str, ...], str]:
    table: Dict[Tuple[str, ...], str] = {}
    for phrase in phrases:
        keys = phrase_key(phrase)
        if keys:
            table[keys] = kind
    return table


#: The built-in filler table: phrase keys -> "polite", "frame" or "soft".
DEFAULT_FILLERS: Dict[Tuple[str, ...], str] = {
    **_phrase_table(SOFT_PHRASES, "soft"),
    **_phrase_table(FRAME_PHRASES, "frame"),
    **_phrase_table(POLITE_PHRASES, "polite"),
}


def filler_table(extra: Iterable[str] = ()) -> Dict[Tuple[str, ...], str]:
    """The default filler table plus ``extra`` phrases (treated as framing words)."""
    table = dict(DEFAULT_FILLERS)
    for phrase in extra:
        if not isinstance(phrase, str):
            raise TypeError("filler phrases must be strings, got %r" % (phrase,))
        keys = phrase_key(phrase)
        if not keys:
            raise ValueError("filler phrase %r has no words in it" % (phrase,))
        table.setdefault(keys, "soft")
    return table


def tokenize(
    text: str,
    fillers: Optional[Dict[Tuple[str, ...], str]] = None,
) -> Tuple[List[Token], List[Token]]:
    """Tokenise ``text`` (already NFC) and mark filler words.

    Returns ``(tokens, dropped)``; ``dropped`` are the hesitation sounds
    ("um", "uh") removed from the stream. Filler phrases match longest first.
    """
    table = DEFAULT_FILLERS if fillers is None else fillers
    tokens: List[Token] = []
    dropped: List[Token] = []
    for tok in raw_tokens(text):
        if tok.key in DISFLUENCIES:
            dropped.append(tok)
        else:
            tokens.append(tok)
    longest = max((len(k) for k in table), default=1)
    keys = [t.key for t in tokens]
    kinds = [""] * len(tokens)
    i = 0
    while i < len(tokens):
        size = min(longest, len(tokens) - i)
        while size > 0:
            kind = table.get(tuple(keys[i:i + size]))
            if kind:
                for k in range(i, i + size):
                    kinds[k] = kind
                break
            size -= 1
        i += max(size, 1)
    marked = [
        Token(t.surface, t.key, t.start, t.end, kinds[n]) for n, t in enumerate(tokens)
    ]
    return marked, dropped


# ---------------------------------------------------------------- similarity

def _osa_distance(a: str, b: str) -> int:
    """Optimal string alignment distance (Levenshtein plus adjacent transpositions)."""
    if a == b:
        return 0
    la, lb = len(a), len(b)
    if la == 0:
        return lb
    if lb == 0:
        return la
    prev2: List[int] = []
    prev = list(range(lb + 1))
    for i in range(1, la + 1):
        cur = [i] + [0] * lb
        ca = a[i - 1]
        for j in range(1, lb + 1):
            cost = 0 if ca == b[j - 1] else 1
            best = prev[j] + 1
            if cur[j - 1] + 1 < best:
                best = cur[j - 1] + 1
            if prev[j - 1] + cost < best:
                best = prev[j - 1] + cost
            if i > 1 and j > 1 and ca == b[j - 2] and a[i - 2] == b[j - 1]:
                if prev2[j - 2] + 1 < best:
                    best = prev2[j - 2] + 1
            cur[j] = best
        prev2, prev = prev, cur
    return prev[lb]


def _allowed_edits(expected: str) -> int:
    """Typos tolerated for a word of this length: none for 3 letters or fewer."""
    n = len(expected)
    if n <= 3:
        return 0
    if n <= 5:
        return 1
    if n <= 8:
        return 2
    return 3


def _collapse_doubles(word: str) -> str:
    out: List[str] = []
    for ch in word:
        if not out or out[-1] != ch:
            out.append(ch)
    return "".join(out)


_VOWELS = frozenset("aeiouy")


def sound_key(word: str) -> str:
    """A rough English sound-alike key ("whether" and "weather" share one).

    Only defined for plain ASCII letters; anything else returns ``""``.
    """
    if not word.isascii() or not word.isalpha():
        return ""
    w = word
    for prefix, repl in (("wh", "w"), ("wr", "r"), ("kn", "n"), ("ps", "s")):
        if w.startswith(prefix):
            w = repl + w[len(prefix):]
    w = w.replace("ph", "f").replace("ck", "k").replace("tch", "ch").replace("dg", "j")
    w = re.sub(r"gh(?=t|$)", "", w)
    w = re.sub(r"c(?=[eiy])", "s", w)
    w = w.replace("c", "k").replace("q", "k").replace("x", "ks").replace("z", "s")
    if len(w) > 3 and w.endswith("e"):
        w = w[:-1]
    w = _collapse_doubles(w)
    out: List[str] = []
    for ch in w:
        if ch in _VOWELS:
            if not out or out[-1] != "a":
                out.append("a")
        else:
            out.append(ch)
    return "".join(out)


@lru_cache(maxsize=65536)
def similarity(expected: str, heard: str) -> Tuple[float, str]:
    """How well a heard word stands in for an expected one, and why.

    Both arguments are comparison keys (see :func:`fold`). Returns
    ``(score, how)`` with ``score`` in [0, 1], where 0 means "not the same
    word"; ``how`` is ``"exact"``, ``"plural"``, ``"doubled letter"``,
    ``"sounds alike"`` or ``"edit distance N"``.
    """
    if expected == heard:
        return 1.0, "exact"
    if not expected or not heard:
        return 0.0, ""
    for longer, shorter in ((expected, heard), (heard, expected)):
        if len(shorter) >= 3 and longer in (shorter + "s", shorter + "es"):
            return 0.95, "plural"
    if len(expected) >= 3 and _collapse_doubles(expected) == _collapse_doubles(heard):
        return 0.9, "doubled letter"
    best, how = 0.0, ""
    limit = _allowed_edits(expected)
    if limit and abs(len(expected) - len(heard)) <= limit:
        dist = _osa_distance(expected, heard)
        if dist <= limit:
            score = 1.0 - dist / max(len(expected), len(heard))
            if score >= 0.6:
                best, how = score, "edit distance %d" % dist
    if best < 0.8 and len(expected) >= 5 and len(heard) >= 5:
        key = sound_key(expected)
        if key and key == sound_key(heard):
            best, how = 0.8, "sounds alike"
    return best, how
