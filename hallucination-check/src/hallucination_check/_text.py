"""Normalization, unit splitting and lexical similarity primitives.

Pure Python: no tokenizer models, no downloads, no third-party text libraries.
"""
from __future__ import annotations

import re
import unicodedata
from collections import Counter
from math import sqrt
from typing import List, Sequence, Tuple

__all__ = [
    "STOPWORDS",
    "COMMON_WORDS",
    "TextVec",
    "char_ngrams",
    "cosine",
    "coverage",
    "normalize",
    "split_sentences",
    "split_units",
    "stem",
    "strip_possessive",
    "tokenize",
    "truncate",
    "vectorize",
]

_ENGLISH_STOPWORDS = frozenset(
    """a about above after again against all also am an another any are aren as at be because been before
    being below between both but by can cannot could did do does doing don down during each few for from
    further had has have having he her here hers herself him himself his how i if in into is it its itself
    just me more most much my myself no nor not of off on once only or other our ours ourselves out over own
    same she should so some such than that the their theirs them themselves then there these they this those
    through to too under until up very was we were what when where which while who whom why will with would
    you your yours yourself yourselves""".split()
)

# Function words of the scripts that are written without spaces, where one character is
# one token. Without these a Chinese pronoun such as 它 counts as content while its
# English translation "it" does not, so the identical answer scores differently in the
# two languages. Devanagari words are space separated but need the same treatment.
_OTHER_STOPWORDS = frozenset(
    # Chinese particles, pronouns, copulas and prepositions
    "的 了 着 过 是 在 和 与 及 也 都 就 而 或 被 把 对 从 以 其 此 该 这 那 它 他 她 我 你 您 们 之 并 "
    "但 还 又 很 呢 吗 吧 啊 于 为 所 由 向 让 给 跟 者 吾".split()
    # Japanese particles and auxiliaries (each kana is its own token)
    + "は が を に へ と も の で や か ね よ な だ ま す し て た".split()
    # Hindi/Marathi function words
    + """का की के को में से पर और या है हैं था थी थे हो होता होती हुआ हुई यह वह ये वे एक कि जो तो ही भी
    पर लिए साथ द्वारा नहीं""".split()
)

STOPWORDS = _ENGLISH_STOPWORDS | _OTHER_STOPWORDS

# Capitalized words that are almost never proper nouns; this keeps sentence-initial
# words from being mistaken for names.
COMMON_WORDS = STOPWORDS | frozenset(
    """according additionally afterwards although among around available based besides beyond compared
    consequently despite due earlier either elsewhere especially even every finally first following
    furthermore generally given hence however importantly including indeed initially instead lastly later
    likewise meanwhile moreover nevertheless nonetheless nothing notably now often otherwise overall per
    perhaps previously rather recently regarding second several similarly since specifically still
    subsequently therefore third though throughout thus together typically ultimately unless unlike usually
    whereas whether within without yet
    also answer based data every finding findings first however important next note overall please question
    report reported research researchers result results second source sources study studies summary take
    that their them then there these third this those together total use used using""".split()
)

_CJK = "぀-ヿ㐀-䶿一-鿿가-힯豈-﫿"
_TOKEN_RE = re.compile(
    "[" + _CJK + "]"                          # CJK: one character is one token
    r"|[0-9]+(?:[.,][0-9]+)*"                 # digits, with decimal or thousands marks
    r"|[^\W\d_]+(?:['’][^\W\d_]+)*",  # words, keeping internal apostrophes
    re.UNICODE,
)
_WS_RE = re.compile(r"\s+")
# Sentence terminators. The trailing whitespace group is optional on purpose: Chinese,
# Japanese and Devanagari do not put a space after a full stop, so requiring one there
# would collapse a whole answer into a single unit. _is_boundary decides what a
# terminator with no space after it means, which is how "3.5" stays one token.
_SENT_END_RE = re.compile("([.!?…。！？।॥]+)([\"'”’)\\]」』）]*)(\\s*)")
# Terminators that never need a following space.
_WIDE_END_RE = re.compile("[。।॥]")
# Scripts that run on without spaces, so a terminator followed straight by one of these
# really is a sentence boundary.
_NO_SPACE_SCRIPT_RE = re.compile("[" + _CJK + "ऀ-ॿ]")
_POSSESSIVE_RE = re.compile("['’]s$", re.UNICODE)
_TRAILING_WORD_RE = re.compile(r"([^\W\d_.]+\.?)$", re.UNICODE)
_CLAUSE_RE = re.compile(
    r"\s*[;:]\s+|\s*,\s+(?=(?:and|but|which|while|whereas|because|so|although|though)\s)", re.I
)

_ABBREVIATIONS = frozenset(
    """mr mrs ms dr prof rev sr jr st inc ltd co corp dept est fig figs no nos vs vol al etc approx
    cf eg ie ca circa min max avg univ gov sen rep capt sgt lt col gen ft mt""".split()
)


def normalize(text: str) -> str:
    """NFKC-normalize ``text`` and collapse runs of whitespace."""
    return _WS_RE.sub(" ", unicodedata.normalize("NFKC", text)).strip()


def strip_possessive(token: str) -> str:
    """Drop a trailing English possessive clitic: ``France's`` -> ``France``.

    Possessives are everywhere in real prose ("the company's revenue", "NASA's
    budget"), and without this the name in them would never match the same name
    standing on its own in the answer.
    """
    base = _POSSESSIVE_RE.sub("", token)
    return base if len(base) >= 2 else token


def tokenize(text: str, *, fold: bool = True) -> List[str]:
    """Split ``text`` into word, number and CJK-character tokens."""
    tokens = [
        strip_possessive(t) for t in _TOKEN_RE.findall(unicodedata.normalize("NFKC", text))
    ]
    return [t.casefold() for t in tokens] if fold else tokens


def token_spans(text: str) -> List[Tuple[str, int, int]]:
    """Tokens of the NFKC-normalized ``text`` with their ``(start, end)`` offsets.

    The possessive clitic is stripped here as well as in :func:`tokenize`, so a source
    that writes ``France's capital`` still backs up a claim about ``France``. The offsets
    stay those of the whole token, clitic included, so spans never overlap.
    """
    norm = unicodedata.normalize("NFKC", text)
    return [
        (strip_possessive(m.group(0)), m.start(), m.end()) for m in _TOKEN_RE.finditer(norm)
    ]


def stem(token: str) -> str:
    """Very light suffix stripping so plurals and tenses still match."""
    if len(token) > 4 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 4 and token.endswith(("sses", "ches", "shes")):
        return token[:-2]
    if len(token) > 3 and token.endswith("s") and not token.endswith(("ss", "us", "is")):
        return token[:-1]
    if len(token) > 5 and token.endswith("ing"):
        return token[:-3]
    if len(token) > 4 and token.endswith("ed"):
        return token[:-2]
    return token


def char_ngrams(text: str, n: int) -> Counter:
    """Counter of character ``n``-grams of the normalized, casefolded ``text``."""
    s = normalize(text).casefold()
    if not s:
        return Counter()
    if len(s) <= n:
        return Counter([s])
    return Counter(s[i : i + n] for i in range(len(s) - n + 1))


def word_ngrams(tokens: Sequence[str]) -> Counter:
    """Counter of word unigrams and bigrams."""
    counts = Counter(tokens)
    counts.update("%s %s" % pair for pair in zip(tokens, tokens[1:]))
    return counts


def cosine(a: Counter, norm_a: float, b: Counter, norm_b: float) -> float:
    """Cosine similarity of two count vectors, given their precomputed norms."""
    if not a or not b or norm_a <= 0.0 or norm_b <= 0.0:
        return 0.0
    if len(a) > len(b):
        a, b, norm_a, norm_b = b, a, norm_b, norm_a
    dot = 0.0
    for key, value in a.items():
        other = b.get(key)
        if other:
            dot += value * other
    return max(0.0, min(1.0, dot / (norm_a * norm_b)))


def _norm_of(counts: Counter) -> float:
    return sqrt(sum(v * v for v in counts.values()))


class TextVec:
    """Everything the scorer needs about one piece of text, computed once."""

    __slots__ = (
        "text",
        "tokens",
        "token_set",
        "content",
        "match_set",
        "char",
        "char_norm",
        "word",
        "word_norm",
    )

    def __init__(self, text: str, char_n: int = 4, extra_tokens: Sequence[str] = ()) -> None:
        self.text = text
        tokens = tokenize(text)
        if extra_tokens:
            known = set(tokens)
            tokens = tokens + [t for t in extra_tokens if t not in known]
        self.tokens = tokens
        self.token_set = set(tokens)
        self.content = [t for t in sorted(self.token_set) if t not in STOPWORDS]
        self.match_set = self.token_set | {stem(t) for t in self.token_set}
        self.char = char_ngrams(text, char_n)
        self.char_norm = _norm_of(self.char)
        self.word = word_ngrams(tokens)
        self.word_norm = _norm_of(self.word)

    @property
    def keys(self) -> List[str]:
        """Content tokens if there are any, otherwise every token."""
        return self.content if self.content else sorted(self.token_set)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return "TextVec(%r)" % truncate(self.text, 40)


def vectorize(text: str, char_n: int = 4, extra_tokens: Sequence[str] = ()) -> TextVec:
    """Build a :class:`TextVec` for ``text``."""
    return TextVec(text, char_n=char_n, extra_tokens=extra_tokens)


def coverage(claim: TextVec, span: TextVec) -> float:
    """Share of the claim's content tokens that also occur in ``span``."""
    keys = claim.keys
    if not keys:
        return 0.0
    hits = 0
    for token in keys:
        if token in span.match_set or stem(token) in span.match_set:
            hits += 1
    return hits / len(keys)


# How far back the word in front of a full stop may reach. Every abbreviation and
# initial this has to recognize is a handful of characters long, so a bounded look-back
# gives the same answer as scanning the whole line -- and it is what keeps the scan
# linear: slicing ``text[: match.start()]`` once per terminator makes splitting one long
# line quadratic in its length, which is how a 300 KB source came to take a minute.
_LOOKBACK = 48


def _is_boundary(text: str, match) -> bool:
    end = match.end()
    length = len(text)
    punct = match.group(1)
    if not match.group(3) and end < length:
        # Nothing separates the terminator from what follows. That is a decimal point or
        # a URL in Latin text, but it is ordinary punctuation in Chinese, Japanese and
        # Devanagari, which never put a space there.
        if not _WIDE_END_RE.search(punct) and not _NO_SPACE_SCRIPT_RE.match(text[end]):
            return False
    if punct[0] != ".":
        return True
    start = match.start()
    before = _TRAILING_WORD_RE.search(text, max(0, start - _LOOKBACK), start)
    if before is not None:
        word = before.group(1).rstrip(".").casefold()
        if word in _ABBREVIATIONS:
            return False
        if len(word) == 1 and word.isalpha():
            return False
    if end >= length:
        return True
    # Anything that is not a lowercase letter starts a new sentence: capitals, digits,
    # opening quotes, and scripts without case such as Chinese or Devanagari.
    return not text[end].islower()


def split_sentences(text: str) -> List[str]:
    """Split ``text`` into sentences. A line break always ends a unit."""
    out: List[str] = []
    for line in unicodedata.normalize("NFKC", text).splitlines():
        line = line.strip()
        if not line:
            continue
        start = 0
        for match in _SENT_END_RE.finditer(line):
            if not _is_boundary(line, match):
                continue
            piece = line[start : match.end()].strip()
            if piece:
                out.append(piece)
            start = match.end()
        tail = line[start:].strip()
        if tail:
            out.append(tail)
    return out


def split_paragraphs(text: str) -> List[str]:
    """Split ``text`` on blank lines."""
    blocks = re.split(r"\n\s*\n", unicodedata.normalize("NFKC", text))
    out = [_WS_RE.sub(" ", b).strip() for b in blocks]
    return [b for b in out if b]


def split_clauses(text: str) -> List[str]:
    """Split into sentences, then into clauses at ``;``, ``:`` and coordinating commas."""
    out: List[str] = []
    for sentence in split_sentences(text):
        parts = [p.strip() for p in _CLAUSE_RE.split(sentence) if p and p.strip()]
        merged: List[str] = []
        for part in parts:
            if merged and len(tokenize(part)) < 3:
                merged[-1] = "%s %s" % (merged[-1], part)
            else:
                merged.append(part)
        out.extend(merged if merged else [sentence])
    return out


def split_units(text: str, granularity: str) -> List[str]:
    """Split ``text`` into the units that get checked, one per claim."""
    if granularity == "sentence":
        return split_sentences(text)
    if granularity == "clause":
        return split_clauses(text)
    if granularity == "paragraph":
        return split_paragraphs(text)
    raise ValueError(
        "granularity must be 'sentence', 'clause' or 'paragraph', got %r" % (granularity,)
    )


def truncate(text: str, limit: int = 100) -> str:
    """Shorten ``text`` for display. ASCII punctuation only."""
    flat = normalize(text)
    if len(flat) <= limit:
        return flat
    return flat[: max(1, limit - 3)].rstrip() + "..."
