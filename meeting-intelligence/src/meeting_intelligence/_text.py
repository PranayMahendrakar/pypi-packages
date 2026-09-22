"""Sentence splitting, tokenizing and stop words.

Everything here is pure Python and regex. No model, no data file, no download.
"""
from __future__ import annotations

import re
import unicodedata
from typing import Iterable, List, Mapping, Optional, Sequence, Tuple

# Abbreviations that end in a period but do not end a sentence.
_ABBREVIATIONS = {
    "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "mt", "no", "vs", "etc",
    "eg", "ie", "al", "inc", "ltd", "co", "corp", "dept", "est", "fig", "approx",
    "cf", "min", "max", "sec", "am", "pm", "a.m", "p.m", "e.g", "i.e", "u.s",
    "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "sept", "oct", "nov", "dec",
}

# Sentence-ending punctuation, Latin and CJK.
_TERMINATORS = ".!?。！？…"
_CJK_TERMINATORS = "。！？"

_SENTENCE_BREAK = re.compile(
    r"(?<=[.!?…])[\"'”’)\]]*\s+|(?<=[。！？])\s*"
)

_WORD_RE = re.compile(r"[^\W\d_]+(?:['’][^\W\d_]+)*|\d+(?:[.,]\d+)*", re.UNICODE)

# Small, deliberately generic stop list. Used for topic keywords and for the
# content-word overlap that matches an answer to a question.
STOP_WORDS = frozenset("""
a about above after again against all also am an and any are aren't as at
be because been before being below between both but by
can cannot could couldn't
did didn't do does doesn't doing don't down during
each even
few for from further
had hadn't has hasn't have haven't having he he'd he'll he's her here here's hers herself him
himself his how how's however
i i'd i'll i'm i've if in into is isn't it it's its itself
just
let let's like
me more most mustn't my myself
no nor not now
of off ok okay on once only or other ought our ours ourselves out over own
really right
same shan't she she'd she'll she's should shouldn't so some such
than that that's the their theirs them themselves then there there's these they they'd they'll
they're they've this those through to too
under until up
very
was wasn't we we'd we'll we're we've were weren't what what's when when's where where's which
while who who's whom why why's with won't would wouldn't
yeah yes you you'd you'll you're you've your yours yourself yourselves
thing things stuff kind sort lot lots bit going get got gets gonna wanna
think thought know knew say said says saying see seen look looks maybe sure
actually basically literally anyway sorry thanks thank hey hi hello
um uh er ah oh mm mmm hmm yep yup nope nah
one two three four five six seven eight nine ten
""".split())

# Words that are common in meeting scaffolding and make useless topic labels.
# A label is supposed to name the subject, so the cue verbs that decisions and
# action items are found with belong here too: "we decided to move the event
# store" is about the event store, not about deciding.
_LABEL_NOISE = frozenset("""
meeting call today yesterday tomorrow week month quarter time minute minutes hour
people team everyone everybody guys folks question questions answer answers
point points thing things update updates start started starting end ends ending
agenda item items next last first second next
decided decide decides deciding decision decisions agreed agree agrees
plan plans planned planning settled settle chose choose chosen picked pick
assigned assign assigns owns owned action actions todo
need needs needed want wants wanted must should shall
morning afternoon evening night hello goodbye welcome
make makes made take takes taken put puts give gives
come comes went gone done doing able sounds seems looks
older newer earlier later bigger smaller better worse older
new old big small good bad long short easy hard quick fast slow
whole full same other another few many much more less
everything nothing something anything someone somebody anybody nobody
go goes gone went
monday tuesday wednesday thursday friday saturday sunday
january february march april june july august september october november december
""".split())


def split_sentences(text: str) -> List[str]:
    """Split ``text`` into sentences, keeping the terminator on each one."""
    if not text:
        return []
    stripped = text.strip()
    if not stripped:
        return []
    pieces = [piece for piece in _SENTENCE_BREAK.split(stripped) if piece and piece.strip()]
    if not pieces:
        return [stripped]
    merged: List[str] = []
    for piece in pieces:
        piece = piece.strip()
        if not piece:
            continue
        if merged and _ends_with_abbreviation(merged[-1]):
            merged[-1] = merged[-1] + " " + piece
        else:
            merged.append(piece)
    return merged or [stripped]


def _ends_with_abbreviation(sentence: str) -> bool:
    """True when a piece ends in an abbreviation rather than a real full stop."""
    if not sentence.endswith("."):
        return False
    tail = sentence[:-1].split()
    if not tail:
        return False
    last = tail[-1].lower().strip("([{\"'")
    if last in _ABBREVIATIONS:
        return True
    # A single capital letter, as in a middle initial.
    return len(last) == 1 and last.isalpha()


def tokenize(text: str) -> List[str]:
    """Lower-cased word tokens; punctuation and symbols are dropped."""
    if not text:
        return []
    return [match.group(0).lower() for match in _WORD_RE.finditer(text)]


def content_words(text: str) -> List[str]:
    """Tokens with stop words and one-character tokens removed."""
    return [
        token
        for token in tokenize(text)
        if token not in STOP_WORDS and len(token) > 1 and not token.isdigit()
    ]


def label_words(text: str) -> List[str]:
    """Content words that are also plausible topic-label words."""
    return [token for token in content_words(text) if token not in _LABEL_NOISE]


def word_count(text: str) -> int:
    """Number of word tokens in ``text``."""
    return len(tokenize(text))


def normalize(text: str) -> str:
    """Collapse whitespace and normalise unicode for comparison keys."""
    return unicodedata.normalize("NFKC", " ".join(text.split()))


def fold(text: str) -> str:
    """Lower-cased, whitespace-collapsed, with typographic quotes flattened."""
    folded = normalize(text).lower()
    return (
        folded.replace("’", "'")
        .replace("‘", "'")
        .replace("“", '"')
        .replace("”", '"')
        .replace("—", " - ")
        .replace("–", " - ")
    )


def truncate(text: str, width: int = 90) -> str:
    """One-line preview of ``text``, at most ``width`` characters."""
    flat = " ".join(str(text).split())
    if len(flat) <= width:
        return flat
    return flat[: max(1, width - 3)].rstrip() + "..."


def strip_terminator(text: str) -> str:
    """Drop a trailing sentence terminator, for phrases embedded in a line."""
    return text.rstrip().rstrip(_TERMINATORS).rstrip()


def title_case(words: Sequence[str], casing: Optional[Mapping[str, str]] = None) -> str:
    """Join label words into a short human-readable topic label.

    ``casing`` maps a lower-cased token to the spelling it had in the
    transcript, so an acronym comes back as "PR" or "API" instead of the
    "Pr" and "Api" that blind capitalisation would produce.
    """
    parts = []
    for word in words:
        original = (casing or {}).get(word.lower())
        if original and original.isupper() and len(original) <= 5:
            parts.append(original)
        elif word.isupper() and len(word) <= 5:
            parts.append(word)
        else:
            parts.append(word[:1].upper() + word[1:])
    return " ".join(parts)


def overlap_score(left: Iterable[str], right: Iterable[str]) -> Tuple[int, float]:
    """Shared content words between two token sequences: (count, jaccard)."""
    left_set = set(left)
    right_set = set(right)
    if not left_set or not right_set:
        return 0, 0.0
    shared = left_set & right_set
    union = left_set | right_set
    return len(shared), len(shared) / len(union)
