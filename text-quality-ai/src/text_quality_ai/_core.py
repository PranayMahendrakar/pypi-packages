"""The public entry points: :func:`score`, :func:`readability`, :func:`repetition`,
:func:`compare` and the :class:`TextScorer` class behind them.
"""
from __future__ import annotations

import logging
from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List, Sequence, Union

from ._measures import (
    LONG_SENTENCE_WORDS,
    MEASURE_NAMES,
    TARGETS,
    WEIGHTS,
    clarity_measure,
    clip,
    r1,
    readability_measure,
    reading_time_seconds,
    repetition_measure,
    structure_measure,
    vocabulary_measure,
)
from ._report import Issue, QualityReport, build_issues, grade_for
from ._text import parse

log = logging.getLogger(__name__)

TextInput = Union[str, Sequence[str]]

SEVERITY_RANK = {"low": 1, "medium": 2, "high": 3}


def available_targets() -> List[str]:
    """The audience profiles :func:`score` accepts."""
    return list(TARGETS)


def _check_target(target: str) -> str:
    if target not in TARGETS:
        raise ValueError(
            f"unknown target {target!r}; choose one of {', '.join(sorted(TARGETS))}"
        )
    return target


def _is_text_list(value: Any) -> bool:
    return isinstance(value, (list, tuple)) and not isinstance(value, (str, bytes))


class TextScorer:
    """Score text for readability, repetition, structure, clarity and vocabulary.

    Args:
        target: which audience to aim at. One of ``"general"``, ``"academic"``,
            ``"marketing"``, ``"technical"`` or ``"simple"``. It shifts the ideal
            reading grade and the expected share of long words.
        long_sentence_words: a sentence longer than this counts as a long one.

    ``TextScorer`` holds no state between calls, so one instance can score any
    number of texts.
    """

    def __init__(self, target: str = "general", *, long_sentence_words: int = LONG_SENTENCE_WORDS) -> None:
        self.target = _check_target(target)
        if int(long_sentence_words) < 5:
            raise ValueError("long_sentence_words must be at least 5")
        self.long_sentence_words = int(long_sentence_words)

    # -- one document -----------------------------------------------------

    def score_one(self, text: str) -> QualityReport:
        """Score a single string."""
        doc = parse(text)
        measures: Dict[str, Dict[str, Any]] = {
            "readability": readability_measure(doc, self.target),
            "repetition": repetition_measure(doc),
            "structure": structure_measure(doc),
            "clarity": clarity_measure(doc, self.long_sentence_words, self.target),
            "vocabulary": vocabulary_measure(doc, self.target),
        }
        components = {name: clip(measures[name]["score"]) for name in MEASURE_NAMES}
        if doc.is_empty:
            log.debug("no words in the text; every measure reports 0")
            overall = 0.0
        else:
            weights = WEIGHTS[self.target]
            overall = clip(sum(components[name] * weights[name] for name in MEASURE_NAMES))
        issues, suggestions = build_issues(doc, measures, self.target)
        stats = {
            "words": doc.n_words,
            "sentences": doc.n_sentences,
            "paragraphs": doc.n_paragraphs,
            "syllables": doc.syllable_total,
            "reading_time_seconds": reading_time_seconds(doc),
            "characters": len(doc.text),
            "unique_words": measures["vocabulary"].get("unique_words", 0),
            "word_counting": doc.word_mode,
        }
        notes = list(doc.notes)
        if doc.is_empty:
            notes.append("The text has no words, so every measure is reported as 0.")
        return QualityReport(
            score=r1(overall),
            grade=grade_for(overall),
            target=self.target,
            components={k: r1(v) for k, v in components.items()},
            issues=issues,
            suggestions=suggestions,
            stats=stats,
            measures=measures,
            notes=notes,
        )

    # -- many documents ---------------------------------------------------

    def score_many(self, texts: Iterable[str]) -> QualityReport:
        """Score each string and return one report covering all of them."""
        items = list(texts)
        for i, item in enumerate(items):
            if not isinstance(item, str):
                raise TypeError(
                    f"every item must be a string; item {i} is a {type(item).__name__}"
                )
        documents = [self.score_one(item) for item in items]
        return self._combine(documents)

    def _combine(self, documents: List[QualityReport]) -> QualityReport:
        total_words = sum(doc.stats["words"] for doc in documents)
        if not documents:
            report = self.score_one("")
            report.documents = []
            report.notes.append("No documents were given.")
            return report

        if total_words:
            def weighted(values: List[float]) -> float:
                pairs = zip(values, (doc.stats["words"] for doc in documents))
                return sum(value * words for value, words in pairs) / total_words
        else:
            def weighted(values: List[float]) -> float:
                return sum(values) / len(values)

        components = {
            name: r1(clip(weighted([doc.components[name] for doc in documents])))
            for name in MEASURE_NAMES
        }
        overall = r1(clip(weighted([doc.score for doc in documents])))
        # An average of the raw numbers behind a measure would be misleading
        # (there is no one Flesch grade for five documents), so the aggregate
        # carries the score and the explanation only. Every number is still on
        # report.documents[i].measures, and the explanation says so.
        measures = {
            name: {
                "score": components[name],
                "explanation": (
                    f"Average of {len(documents)} document(s), weighted by length: "
                    f"{components[name]:.1f} out of 100. The numbers behind this measure are "
                    f"per document; see report.documents[i].measures[{name!r}]."
                ),
                "documents": len(documents),
            }
            for name in MEASURE_NAMES
        }
        stats = {
            "words": total_words,
            "sentences": sum(doc.stats["sentences"] for doc in documents),
            "paragraphs": sum(doc.stats["paragraphs"] for doc in documents),
            "syllables": sum(doc.stats["syllables"] for doc in documents),
            "reading_time_seconds": round(sum(doc.stats["reading_time_seconds"] for doc in documents), 1),
            "characters": sum(doc.stats["characters"] for doc in documents),
            "unique_words": sum(doc.stats["unique_words"] for doc in documents),
            "documents": len(documents),
            "word_counting": "mixed" if len({doc.stats["word_counting"] for doc in documents}) > 1
            else documents[0].stats["word_counting"],
        }
        issues, suggestions = _merge_issues(documents)
        notes = [f"{len(documents)} document(s) scored; each score is the average weighted by length."]
        seen = set()
        for doc in documents:
            for note in doc.notes:
                if note not in seen:
                    seen.add(note)
                    notes.append(note)
        report = QualityReport(
            score=overall,
            grade=grade_for(overall),
            target=self.target,
            components=components,
            issues=issues,
            suggestions=suggestions,
            stats=stats,
            measures=measures,
            notes=notes,
        )
        report.documents = documents
        return report

    def score(self, text: TextInput) -> QualityReport:
        """Score a string, or a list of strings."""
        if _is_text_list(text):
            return self.score_many(text)
        return self.score_one(text)


def _merge_issues(documents: List[QualityReport]):
    """Fold per-document issues into one list, most widespread and worst first."""
    rank: Dict[str, float] = defaultdict(float)
    counts: Counter = Counter()
    docs_hit: Counter = Counter()
    examples: Dict[str, List[str]] = defaultdict(list)
    message: Dict[str, str] = {}
    severity: Dict[str, str] = {}
    suggestion: Dict[str, str] = {}

    for doc in documents:
        total = len(doc.issues)
        for position, issue in enumerate(doc.issues):
            kind = issue.kind
            rank[kind] += total - position
            counts[kind] += issue.count
            docs_hit[kind] += 1
            for example in issue.examples:
                if len(examples[kind]) < 4 and example not in examples[kind]:
                    examples[kind].append(example)
            if kind not in message or SEVERITY_RANK[issue.severity] > SEVERITY_RANK[severity[kind]]:
                message[kind] = issue.message
                severity[kind] = issue.severity
            if position < len(doc.suggestions):
                suggestion.setdefault(kind, doc.suggestions[position])

    order = sorted(rank, key=lambda k: (-rank[k], k))
    issues = [
        Issue(
            kind=kind,
            severity=severity[kind],
            message=(f"{docs_hit[kind]} of {len(documents)} document(s): " + message[kind]),
            examples=examples[kind],
            count=int(counts[kind]),
        )
        for kind in order
    ]
    suggestions = [suggestion[kind] for kind in order if kind in suggestion]
    if not suggestions:
        suggestions = ["Nothing major to fix: these texts score well on all five measures."]
    return issues, suggestions


# --------------------------------------------------------------------------
# module-level convenience functions
# --------------------------------------------------------------------------

def score(text: TextInput, *, target: str = "general") -> QualityReport:
    """Score ``text`` and say what to fix.

    Args:
        text: a string, or a list of strings. With a list, the returned report
            covers all of them and ``report.documents`` holds one report each.
        target: ``"general"``, ``"academic"``, ``"marketing"``, ``"technical"``
            or ``"simple"``. It shifts the ideal reading grade and the expected
            share of long words.

    Returns:
        A :class:`~text_quality_ai.QualityReport`: ``.score`` 0 to 100,
        ``.grade`` ``"A"`` to ``"F"``, ``.components``, ``.issues``,
        ``.suggestions``, ``.stats``, and ``.summary()``.

    Empty or whitespace-only text gives a zero-word report rather than an error.
    """
    return TextScorer(target).score(text)


def readability(text: str, *, target: str = "general") -> Dict[str, Any]:
    """Readability only: Flesch reading ease, Flesch-Kincaid grade, SMOG and lengths."""
    _check_target(target)
    return readability_measure(parse(text), target)


def repetition(text: str) -> Dict[str, Any]:
    """Repetition only: repeated words, sentence openers, bigrams and trigrams."""
    return repetition_measure(parse(text))


def compare(
    a: TextInput,
    b: TextInput,
    *,
    target: str = "general",
    long_sentence_words: int = LONG_SENTENCE_WORDS,
) -> Dict[str, Any]:
    """Score two texts and return how ``b`` differs from ``a``.

    Every number is ``b`` minus ``a``, so a positive ``"score"`` means ``b`` is
    the better text. Both sides are scored with the same settings, so
    ``long_sentence_words`` applies to ``a`` and ``b`` alike.
    """
    scorer = TextScorer(target, long_sentence_words=long_sentence_words)
    report_a = scorer.score(a)
    report_b = scorer.score(b)
    numeric = ("words", "sentences", "paragraphs", "syllables", "reading_time_seconds")
    delta = round(report_b.score - report_a.score, 1)
    if delta > 0:
        better, verdict = "b", f"b scores {delta:.1f} points higher than a."
    elif delta < 0:
        better, verdict = "a", f"a scores {abs(delta):.1f} points higher than b."
    else:
        better, verdict = "tie", "Both texts score the same."
    return {
        "score": delta,
        "better": better,
        "summary": verdict,
        "components": {
            name: round(report_b.components[name] - report_a.components[name], 1)
            for name in MEASURE_NAMES
        },
        "stats": {
            name: round(float(report_b.stats.get(name, 0)) - float(report_a.stats.get(name, 0)), 1)
            for name in numeric
        },
        "a": {"score": report_a.score, "grade": report_a.grade, "components": dict(report_a.components)},
        "b": {"score": report_b.score, "grade": report_b.grade, "components": dict(report_b.components)},
        "target": target,
        "long_sentence_words": scorer.long_sentence_words,
    }
