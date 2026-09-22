"""The five measures: readability, repetition, structure, clarity, vocabulary.

Every measure takes a parsed :class:`~text_quality_ai._text.Document` and returns
a plain dict that always carries a ``score`` between 0 and 100 and an
``explanation`` in plain English, plus the raw numbers behind them.
"""
from __future__ import annotations

import math
from collections import Counter
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from ._text import Document, ngrams, syllables
from ._words import (
    AGENT_DETERMINERS,
    BE_FORMS,
    DEGREE_ADVERBS,
    FILLER_PHRASES,
    FILLER_WORDS,
    HEDGE_WORDS,
    IRREGULAR_PARTICIPLES,
    NOMINALISATION_EXCEPTIONS,
    NOMINALISATION_SUFFIXES,
    VERB_DERIVED_ANCE_ENCE,
    NON_AGENT_AFTER_BY,
    PREDICATE_ADJECTIVES,
    STOPWORDS,
    TRANSITION_PHRASES,
    TRANSITION_WORDS,
)

MEASURE_NAMES = ("readability", "repetition", "structure", "clarity", "vocabulary")

#: What "good" looks like for each audience.
#:
#: ``nominalisation_rate`` is the share of words that may be nouns built out of
#: verbs before the clarity score starts paying for it, and
#: ``nominalisation_limit`` is where the report raises an issue about them.
#: Both travel with the audience: a technical or academic paper names concepts
#: for a living, so it is given more room than marketing copy.
TARGETS: Dict[str, Dict[str, Any]] = {
    "general": {"grade": 8.0, "long_word_share": 0.14, "label": "a general audience",
                "nominalisation_rate": 0.020, "nominalisation_limit": 0.035},
    "academic": {"grade": 14.0, "long_word_share": 0.28, "label": "an academic audience",
                 "nominalisation_rate": 0.035, "nominalisation_limit": 0.060},
    "marketing": {"grade": 7.0, "long_word_share": 0.10, "label": "marketing copy",
                  "nominalisation_rate": 0.016, "nominalisation_limit": 0.028},
    "technical": {"grade": 12.0, "long_word_share": 0.22, "label": "a technical audience",
                  "nominalisation_rate": 0.035, "nominalisation_limit": 0.060},
    "simple": {"grade": 5.0, "long_word_share": 0.06, "label": "plain, simple language",
               "nominalisation_rate": 0.014, "nominalisation_limit": 0.025},
}

#: How much each measure counts towards the overall score, per target.
WEIGHTS: Dict[str, Dict[str, float]] = {
    "general": {"readability": 0.30, "repetition": 0.20, "structure": 0.15, "clarity": 0.25, "vocabulary": 0.10},
    "academic": {"readability": 0.15, "repetition": 0.20, "structure": 0.25, "clarity": 0.25, "vocabulary": 0.15},
    "marketing": {"readability": 0.30, "repetition": 0.20, "structure": 0.15, "clarity": 0.30, "vocabulary": 0.05},
    "technical": {"readability": 0.20, "repetition": 0.20, "structure": 0.20, "clarity": 0.30, "vocabulary": 0.10},
    "simple": {"readability": 0.40, "repetition": 0.15, "structure": 0.10, "clarity": 0.30, "vocabulary": 0.05},
}

LONG_SENTENCE_WORDS = 30
LONG_PARAGRAPH_WORDS = 200
WORDS_PER_MINUTE = 238          # Brysbaert (2019), silent reading of English prose
CHARS_PER_MINUTE = 400          # rough equivalent for Han / kana / Hangul text


# --------------------------------------------------------------------------
# small numeric helpers
# --------------------------------------------------------------------------

def clip(value: float) -> float:
    """Keep a score inside 0 to 100, and turn NaN into 0."""
    if value != value or value is None:            # NaN guard
        return 0.0
    return float(min(100.0, max(0.0, value)))


def band(value: float, low: float, high: float, width: float) -> float:
    """100 inside ``[low, high]``, falling linearly to 0 ``width`` beyond it."""
    if low <= value <= high:
        return 100.0
    distance = (low - value) if value < low else (value - high)
    return clip(100.0 - 100.0 * distance / max(width, 1e-9))


def r1(value: float) -> float:
    """Round to one decimal, JSON safe."""
    if value != value or value in (float("inf"), float("-inf")):
        return 0.0
    return round(float(value), 1)


def r3(value: float) -> float:
    """Round to three decimals, JSON safe."""
    if value != value or value in (float("inf"), float("-inf")):
        return 0.0
    return round(float(value), 3)


def _shorten(text: str, limit: int = 90) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[:limit - 3].rstrip() + "..."


def _empty(name: str) -> Dict[str, Any]:
    return {"score": 0.0, "explanation": "There is no text to measure."}


# --------------------------------------------------------------------------
# readability
# --------------------------------------------------------------------------

def readability_measure(doc: Document, target: str = "general") -> Dict[str, Any]:
    """Flesch reading ease, Flesch-Kincaid grade, SMOG, and sentence/word length."""
    if doc.is_empty:
        base = _empty("readability")
        base.update(
            flesch_reading_ease=0.0, flesch_kincaid_grade=0.0, smog_index=0.0,
            avg_sentence_length=0.0, avg_word_length=0.0, avg_syllables_per_word=0.0,
            polysyllable_words=0, target=target, target_grade=float(TARGETS[target]["grade"]),
        )
        return base

    words = doc.n_words
    sentences = max(1, doc.n_sentences)
    asl = words / sentences
    asw = doc.syllable_total / words
    fre = 206.835 - 1.015 * asl - 84.6 * asw
    fkg = 0.39 * asl + 11.8 * asw - 15.59
    poly = sum(1 for t in doc.tokens if syllables(t) >= 3)
    smog = 1.0430 * math.sqrt(poly * (30.0 / sentences)) + 3.1291
    awl = float(np.mean([len(t) for t in doc.tokens]))
    ideal = float(TARGETS[target]["grade"])

    if doc.word_mode == "character":
        score = band(asl, 15.0, 45.0, 40.0)
        explanation = (
            f"Words are counted as characters here, so the English formulas are for reference only "
            f"(reading ease {fre:.1f}, grade {fkg:.1f}). Sentences average {asl:.1f} characters, which "
            f"reads comfortably between 15 and 45."
        )
    else:
        # Being simpler than the target is only a problem where precision is
        # expected, so the band is wide below the ideal and tight above it.
        low_edge = ideal - 1.5 if target in ("academic", "technical") else max(0.0, ideal - 6.0)
        score = band(fkg, low_edge, ideal + 1.5, 14.0)
        explanation = (
            f"Flesch reading ease {fre:.1f} and Flesch-Kincaid grade {fkg:.1f}; "
            f"{TARGETS[target]['label']} reads best around grade {ideal:.0f}. "
            f"Sentences average {asl:.1f} words and words average {awl:.1f} characters "
            f"({asw:.2f} syllables)."
        )
    if doc.n_sentences < 30:
        explanation += " SMOG is designed for 30 sentences or more, so treat it as indicative here."

    return {
        "score": r1(score),
        "explanation": explanation,
        "flesch_reading_ease": r1(fre),
        "flesch_kincaid_grade": r1(fkg),
        "smog_index": r1(smog),
        "avg_sentence_length": r1(asl),
        "avg_word_length": r1(awl),
        "avg_syllables_per_word": r3(asw),
        "polysyllable_words": int(poly),
        "target": target,
        "target_grade": ideal,
    }


# --------------------------------------------------------------------------
# repetition
# --------------------------------------------------------------------------

def _repeated_ngrams(groups: Sequence[Sequence[str]], n: int, minimum: int):
    """Repeated ``n``-grams and how many there were in total.

    N-grams never cross a sentence boundary, so the last words of one sentence
    plus the first of the next are not reported as a repeated phrase.
    """
    counts: Counter = Counter()
    total = 0
    for group in groups:
        grams = ngrams(group, n)
        total += len(grams)
        counts.update(grams)
    out = []
    for gram, count in counts.items():
        if count < minimum:
            continue
        if all(word in STOPWORDS for word in gram):
            continue                                  # "of the" repeating is not a problem
        out.append({"phrase": " ".join(gram), "count": int(count)})
    out.sort(key=lambda d: (-d["count"], d["phrase"]))
    return out[:10], total


def repetition_measure(doc: Document) -> Dict[str, Any]:
    """Repeated words, repeated sentence openers and repeated bigrams and trigrams."""
    if doc.is_empty:
        base = _empty("repetition")
        base.update(repeated_words=[], repeated_openers=[], repeated_bigrams=[],
                    repeated_trigrams=[], word_repetition_rate=0.0, opener_repetition_rate=0.0,
                    phrase_repetition_rate=0.0, content_words=0)
        return base

    lower = doc.lower_tokens
    content = [t for t in lower if t not in STOPWORDS and len(t) > 2]
    n_content = len(content)

    repeated_words: List[Dict[str, Any]] = []
    excess = 0
    if n_content:
        threshold = max(3, math.ceil(0.02 * n_content))
        for word, count in Counter(content).most_common():
            if count < threshold:
                break
            excess += count - threshold + 1
            repeated_words.append({"word": word, "count": int(count),
                                   "share": r3(count / n_content)})
    repeated_words = repeated_words[:10]
    word_rate = excess / n_content if n_content else 0.0

    openers: List[Dict[str, Any]] = []
    opener_rate = 0.0
    if doc.n_sentences >= 4:
        firsts = [group[0].lower() for group in doc.sentence_tokens if group]
        counts = Counter(firsts)
        repeats = sum(c - 1 for c in counts.values() if c > 1)
        opener_rate = repeats / max(1, len(firsts))
        openers = [{"word": w, "count": int(c)} for w, c in counts.most_common() if c > 1][:10]

    lower_groups = [[t.lower() for t in group] for group in doc.sentence_tokens]
    bigrams, n_bi = _repeated_ngrams(lower_groups, 2, 3)
    trigrams, n_tri = _repeated_ngrams(lower_groups, 3, 2)
    n_bi = max(1, n_bi)
    n_tri = max(1, n_tri)
    phrase_rate = (
        sum(d["count"] - 1 for d in bigrams) / n_bi
        + sum(d["count"] - 1 for d in trigrams) / n_tri
    )

    penalty = (
        min(45.0, 220.0 * word_rate)
        + min(25.0, 60.0 * opener_rate)
        + min(30.0, 260.0 * phrase_rate)
    )
    score = clip(100.0 - penalty)

    parts = []
    if repeated_words:
        top = repeated_words[0]
        parts.append(f"{len(repeated_words)} word(s) are used a lot, most of all '{top['word']}' ({top['count']} times)")
    if openers:
        parts.append(f"{len(openers)} sentence opener(s) repeat, most of all '{openers[0]['word']}' ({openers[0]['count']} sentences)")
    if trigrams:
        parts.append(f"the phrase '{trigrams[0]['phrase']}' appears {trigrams[0]['count']} times")
    elif bigrams:
        parts.append(f"the phrase '{bigrams[0]['phrase']}' appears {bigrams[0]['count']} times")
    explanation = ("Little repetition: no word, opener or phrase stands out."
                   if not parts else "Repetition found: " + "; ".join(parts) + ".")

    return {
        "score": r1(score),
        "explanation": explanation,
        "repeated_words": repeated_words,
        "repeated_openers": openers,
        "repeated_bigrams": bigrams,
        "repeated_trigrams": trigrams,
        "word_repetition_rate": r3(word_rate),
        "opener_repetition_rate": r3(opener_rate),
        "phrase_repetition_rate": r3(phrase_rate),
        "content_words": int(n_content),
    }


# --------------------------------------------------------------------------
# structure
# --------------------------------------------------------------------------

def _starts_with_transition(tokens: Sequence[str]) -> bool:
    head = " ".join(tokens[:4]).lower()
    if any(head.startswith(phrase) for phrase in TRANSITION_PHRASES):
        return True
    return bool(tokens) and tokens[0].lower() in TRANSITION_WORDS


def structure_measure(doc: Document) -> Dict[str, Any]:
    """Sentence length variation, paragraph balance and transition-word use."""
    if doc.is_empty:
        base = _empty("structure")
        base.update(sentence_length_variation=0.0, longest_paragraph_words=0,
                    paragraph_share_of_longest=0.0, transition_share=0.0,
                    transition_sentences=0, shortest_sentence_words=0,
                    longest_sentence_words=0)
        return base

    lengths = doc.sentence_lengths or [doc.n_words]
    arr = np.asarray(lengths, dtype=float)
    mean_len = float(arr.mean())
    variation = float(arr.std() / mean_len) if mean_len > 0 else 0.0

    if len(lengths) >= 3:
        sub_variation = band(variation, 0.35, 0.90, 0.50)
    else:
        sub_variation = 100.0

    para_counts = doc.paragraph_word_counts or [doc.n_words]
    longest_para = int(max(para_counts))
    share_longest = longest_para / max(1, sum(para_counts))
    if len(para_counts) >= 2:
        sub_paragraphs = min(band(share_longest, 0.0, 0.60, 0.40),
                             band(float(longest_para), 0.0, float(LONG_PARAGRAPH_WORDS), 250.0))
    else:
        sub_paragraphs = band(float(longest_para), 0.0, float(LONG_PARAGRAPH_WORDS), 250.0)

    transition_hits = sum(
        1 for tokens in doc.sentence_tokens
        if _starts_with_transition(tokens)
    )
    transition_share = transition_hits / max(1, doc.n_sentences)
    sub_transitions = band(transition_share, 0.10, 0.45, 0.30) if doc.n_sentences >= 5 else 100.0

    score = 0.40 * sub_variation + 0.30 * sub_paragraphs + 0.30 * sub_transitions

    explanation = (
        f"Sentences run from {int(arr.min())} to {int(arr.max())} words (variation {variation:.2f}; "
        f"0.35 to 0.90 reads well). The longest of {len(para_counts)} paragraph(s) holds "
        f"{longest_para} words. {transition_hits} of {doc.n_sentences} sentence(s) open with a "
        f"linking word such as 'however' or 'for example'."
    )

    return {
        "score": r1(clip(score)),
        "explanation": explanation,
        "sentence_length_variation": r3(variation),
        "shortest_sentence_words": int(arr.min()),
        "longest_sentence_words": int(arr.max()),
        "mean_sentence_words": r1(mean_len),
        "paragraphs": len(para_counts),
        "longest_paragraph_words": longest_para,
        "paragraph_share_of_longest": r3(share_longest),
        "transition_sentences": int(transition_hits),
        "transition_share": r3(transition_share),
    }


# --------------------------------------------------------------------------
# clarity
# --------------------------------------------------------------------------

def _is_participle(token: str) -> bool:
    if token in IRREGULAR_PARTICIPLES:
        return True
    return len(token) >= 5 and token.endswith("ed")


#: Words allowed to sit between "to be" and the participle without breaking the
#: passive frame: "was not sent", "were never circulated", "was then completed".
_INTERVENING = frozenset(
    "not never also already being been just still then only even always "
    "soon recently finally originally".split()
)


def _participle_after(lowered: Sequence[str], position: int) -> Optional[int]:
    """Index of the past participle governed by the "to be" at ``position``.

    ``None`` when there is none, or when a degree adverb ("was very tired")
    shows the word is being used as an adjective instead.
    """
    for index in range(position + 1, min(len(lowered), position + 4)):
        token = lowered[index]
        if _is_participle(token):
            if index > 0 and lowered[index - 1] in DEGREE_ADVERBS:
                return None
            return index
        if token in _INTERVENING or token.endswith("ly"):
            continue
    return None


def _is_agent_by(lowered: Sequence[str], by_index: int) -> bool:
    """Does the "by" at ``by_index`` introduce the doer of the action?

    "written by the committee" does; "packed by nine" and "sent by email" do
    not, because those are adverbial phrases, not an agent.
    """
    index = by_index + 1
    while index < len(lowered) and lowered[index] in AGENT_DETERMINERS:
        index += 1
    if index >= len(lowered):
        return False
    word = lowered[index]
    if word[0].isdigit():
        return False
    return word not in NON_AGENT_AFTER_BY


def _has_agent(lowered: Sequence[str], participle_index: int) -> bool:
    """Is there an agent "by" phrase close after the participle?"""
    for index in range(participle_index + 1, min(len(lowered), participle_index + 4)):
        if lowered[index] == "by":
            return _is_agent_by(lowered, index)
    return False


def passive_sentence_indices(doc: Document) -> List[int]:
    """Indices of sentences that look like passive voice.

    The frame is a form of "to be" (or "get") followed within three tokens, past
    any adverb or "not", by a past participle. Two checks then keep ordinary
    active prose out of the count, because that frame on its own also covers
    every "he was tired" in English:

    * an agent phrase after the participle ("written **by the committee**")
      settles it as passive, whatever the participle is;
    * otherwise a participle that normally works as an adjective
      (:data:`~text_quality_ai._words.PREDICATE_ADJECTIVES`), or one carrying a
      degree adverb ("was **very** tired"), is read as a description of a state
      and is not counted.

    It is still a heuristic and will miss passives whose participle happens to
    sit in that adjective list. The README says so.
    """
    hits: List[int] = []
    for index, tokens in enumerate(doc.sentence_tokens):
        lowered = [t.lower() for t in tokens]
        for position, token in enumerate(lowered):
            if token not in BE_FORMS:
                continue
            participle = _participle_after(lowered, position)
            if participle is None:
                continue
            if not _has_agent(lowered, participle) and lowered[participle] in PREDICATE_ADJECTIVES:
                continue
            hits.append(index)
            break
    return hits


def _nominalisations(lower_tokens: Sequence[str]) -> Counter:
    """Nouns built out of verbs, counted per occurrence.

    Only suffixes that really do mark a verb-derived noun are matched by suffix alone;
    ``-ance`` and ``-ence`` need the word itself to be on a known list, because
    "distance" and "patience" end that way without a verb anywhere behind them.
    See :data:`VERB_DERIVED_ANCE_ENCE` for the reasoning.
    """
    found: Counter = Counter()
    for token in lower_tokens:
        if len(token) < 7 or token in NOMINALISATION_EXCEPTIONS:
            continue
        if token.endswith(NOMINALISATION_SUFFIXES) or token in VERB_DERIVED_ANCE_ENCE:
            found[token] += 1
    return found


def count_filler_phrases(lowered_text: str) -> Counter:
    """Count each filler phrase once per occurrence in ``lowered_text``.

    Some phrases nest inside others: "due to the fact that" contains "the fact
    that". Matching the longest first and blanking out what it matched means one
    occurrence is reported once, under the longest phrase that covers it, rather
    than counted twice.
    """
    counts: Counter = Counter()
    remaining = lowered_text
    for phrase in sorted(FILLER_PHRASES, key=len, reverse=True):
        occurrences = remaining.count(phrase)
        if occurrences:
            counts[phrase] += occurrences
            # A space, not an empty string, so removing a phrase never welds two
            # words together into a match that was not in the text.
            remaining = remaining.replace(phrase, " ")
    return counts


def clarity_measure(
    doc: Document,
    long_sentence_words: int = LONG_SENTENCE_WORDS,
    target: str = "general",
) -> Dict[str, Any]:
    """Passive voice, filler and hedge words, nominalisations and long sentences."""
    if target not in TARGETS:
        target = "general"
    # Technical and academic writing names things that are genuinely nouns
    # ("the configuration", "the distribution"), so the allowance travels with
    # the audience instead of being one flat number for everyone.
    nominal_allowance = float(TARGETS[target]["nominalisation_rate"])
    if doc.is_empty:
        base = _empty("clarity")
        base.update(passive_sentences=0, passive_share=0.0, filler_words=[], filler_count=0,
                    hedge_words=[], hedge_count=0, nominalisations=[], nominalisation_count=0,
                    long_sentences=0, long_sentence_share=0.0,
                    long_sentence_words=int(long_sentence_words), long_sentence_examples=[],
                    target=target, nominalisation_allowance=nominal_allowance)
        return base

    words = doc.n_words
    lower = doc.lower_tokens

    passive = passive_sentence_indices(doc)
    passive_share = len(passive) / max(1, doc.n_sentences)

    filler_counts = Counter(t for t in lower if t in FILLER_WORDS)
    filler_counts.update(count_filler_phrases(doc.text.lower()))
    filler_total = sum(filler_counts.values())

    hedge_counts = Counter(t for t in lower if t in HEDGE_WORDS)
    hedge_total = sum(hedge_counts.values())

    nominal_counts = _nominalisations(lower)
    nominal_total = sum(nominal_counts.values())

    long_indices = [i for i, n in enumerate(doc.sentence_lengths) if n > long_sentence_words]
    long_share = len(long_indices) / max(1, doc.n_sentences)

    penalty = (
        min(45.0, 140.0 * max(0.0, passive_share - 0.10))
        + min(25.0, 800.0 * max(0.0, filler_total / words - 0.005))
        + min(15.0, 400.0 * max(0.0, hedge_total / words - 0.010))
        + min(15.0, 400.0 * max(0.0, nominal_total / words - nominal_allowance))
        + min(30.0, 120.0 * max(0.0, long_share - 0.05))
    )
    score = clip(100.0 - penalty)

    parts = [f"{len(passive)} of {doc.n_sentences} sentence(s) read as passive"]
    parts.append(f"{filler_total} filler word(s)")
    parts.append(f"{hedge_total} hedge(s)")
    parts.append(f"{nominal_total} noun(s) built from a verb")
    parts.append(f"{len(long_indices)} sentence(s) longer than {long_sentence_words} words")
    explanation = "Clarity: " + ", ".join(parts) + "."

    return {
        "score": r1(score),
        "explanation": explanation,
        "passive_sentences": len(passive),
        "passive_share": r3(passive_share),
        "passive_examples": [_shorten(doc.sentences[i]) for i in passive[:3]],
        "filler_words": [{"word": w, "count": int(c)} for w, c in filler_counts.most_common(10)],
        "filler_count": int(filler_total),
        "hedge_words": [{"word": w, "count": int(c)} for w, c in hedge_counts.most_common(10)],
        "hedge_count": int(hedge_total),
        "nominalisations": [{"word": w, "count": int(c)} for w, c in nominal_counts.most_common(10)],
        "nominalisation_count": int(nominal_total),
        "long_sentences": len(long_indices),
        "long_sentence_share": r3(long_share),
        "long_sentence_words": int(long_sentence_words),
        "long_sentence_examples": [_shorten(doc.sentences[i]) for i in long_indices[:3]],
        "target": target,
        "nominalisation_allowance": nominal_allowance,
    }


# --------------------------------------------------------------------------
# vocabulary
# --------------------------------------------------------------------------

def moving_average_ttr(tokens: Sequence[str], window: int = 100) -> float:
    """Type-token ratio averaged over a sliding window, so length does not skew it."""
    n = len(tokens)
    if n == 0:
        return 0.0
    if n <= window:
        return len(set(tokens)) / n
    counts = Counter(tokens[:window])
    total = float(len(counts))
    seen = 1
    for i in range(window, n):
        old = tokens[i - window]
        counts[old] -= 1
        if counts[old] == 0:
            del counts[old]
        counts[tokens[i]] += 1
        total += len(counts)
        seen += 1
    return (total / seen) / window


def vocabulary_measure(doc: Document, target: str = "general") -> Dict[str, Any]:
    """Type-token ratio and the share of long words."""
    if doc.is_empty:
        base = _empty("vocabulary")
        base.update(type_token_ratio=0.0, moving_average_ttr=0.0, unique_words=0,
                    long_word_share=0.0, long_words=0, target=target,
                    target_long_word_share=float(TARGETS[target]["long_word_share"]))
        return base

    lower = doc.lower_tokens
    unique = len(set(lower))
    ttr = unique / len(lower)
    mattr = moving_average_ttr(lower, 100)

    if doc.word_mode == "character":
        long_words = 0
        long_share = 0.0
        sub_long = 100.0
    else:
        long_words = sum(1 for t in doc.tokens if syllables(t) >= 3)
        long_share = long_words / len(lower)
        ideal = float(TARGETS[target]["long_word_share"])
        sub_long = band(long_share, max(0.0, ideal - 0.06), ideal + 0.08, 0.25)

    sub_ttr = band(mattr, 0.45, 0.85, 0.35)
    score = clip(0.55 * sub_ttr + 0.45 * sub_long)

    explanation = (
        f"{unique} different words out of {len(lower)} (type-token ratio {ttr:.2f}, "
        f"length-adjusted {mattr:.2f}; 0.45 to 0.85 is a healthy range)."
    )
    if doc.word_mode == "character":
        explanation += " Long-word share is not meaningful for a script counted by characters."
    else:
        explanation += (
            f" {long_share * 100:.0f}% of words have three or more syllables; "
            f"{TARGETS[target]['label']} sits near {float(TARGETS[target]['long_word_share']) * 100:.0f}%."
        )

    return {
        "score": r1(score),
        "explanation": explanation,
        "type_token_ratio": r3(ttr),
        "moving_average_ttr": r3(mattr),
        "unique_words": int(unique),
        "long_words": int(long_words),
        "long_word_share": r3(long_share),
        "target": target,
        "target_long_word_share": float(TARGETS[target]["long_word_share"]),
    }


def reading_time_seconds(doc: Document) -> float:
    """Estimated silent reading time in seconds."""
    if doc.is_empty:
        return 0.0
    rate = CHARS_PER_MINUTE if doc.word_mode == "character" else WORDS_PER_MINUTE
    return round(60.0 * doc.n_words / rate, 1)
