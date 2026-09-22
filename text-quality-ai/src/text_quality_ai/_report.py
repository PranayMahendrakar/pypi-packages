"""The result objects: :class:`Issue` and :class:`QualityReport`.

The report is meant to explain itself. Every measure carries the numbers behind
it and a sentence of plain English, every issue names what was found and shows
examples from the text, and the suggestions are ordered so the first one is the
change that would raise the score most.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ._measures import MEASURE_NAMES, TARGETS, WEIGHTS
from ._text import Document

GRADE_CUTOFFS = ((90.0, "A"), (80.0, "B"), (70.0, "C"), (60.0, "D"), (0.0, "F"))


def grade_for(score: float) -> str:
    """Letter grade for a 0 to 100 score."""
    for cutoff, letter in GRADE_CUTOFFS:
        if score >= cutoff:
            return letter
    return "F"


@dataclass
class Issue:
    """One thing worth fixing, with examples taken from the text."""

    kind: str
    severity: str                       # "high", "medium" or "low"
    message: str
    examples: List[str] = field(default_factory=list)
    count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """JSON-safe view of the issue."""
        return {
            "kind": self.kind,
            "severity": self.severity,
            "message": self.message,
            "examples": list(self.examples),
            "count": int(self.count),
        }

    def __str__(self) -> str:                       # pragma: no cover - cosmetic
        return f"[{self.severity}] {self.kind}: {self.message}"


def _bar(value: float, width: int = 10) -> str:
    filled = int(round(max(0.0, min(100.0, value)) / 100.0 * width))
    return "[" + "#" * filled + "-" * (width - filled) + "]"


@dataclass
class QualityReport:
    """What :func:`text_quality_ai.score` gives back.

    Attributes:
        score: overall quality, 0 to 100.
        grade: ``"A"`` to ``"F"`` for :attr:`score`.
        components: each measure's 0 to 100 score, keyed by name.
        issues: what was found, most worth fixing first.
        suggestions: plain-language fixes, most helpful first.
        stats: words, sentences, paragraphs, syllables, reading_time_seconds.
        measures: the full numbers and explanation behind each component.
        notes: caveats about how this text was counted.
        documents: one report per input when a list of strings was scored.
    """

    score: float
    grade: str
    target: str
    components: Dict[str, float]
    issues: List[Issue] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)
    stats: Dict[str, Any] = field(default_factory=dict)
    measures: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)
    documents: Optional[List["QualityReport"]] = None

    # -- helpers ----------------------------------------------------------

    def explain(self, measure: str) -> str:
        """The plain-English explanation for one measure."""
        if measure not in self.measures:
            raise KeyError(f"unknown measure {measure!r}; try one of {', '.join(MEASURE_NAMES)}")
        return str(self.measures[measure]["explanation"])

    @property
    def weakest(self) -> str:
        """Name of the lowest-scoring measure."""
        if not self.components:
            return ""
        return min(self.components, key=lambda k: self.components[k])

    def summary(self) -> str:
        """A short human-readable report. Plain ASCII punctuation only."""
        stats = self.stats
        lines: List[str] = []
        head = f"Text quality: {self.score:.1f} / 100 (grade {self.grade}) for {TARGETS[self.target]['label']}"
        lines.append(head)
        if self.documents is not None:
            lines.append(f"  {len(self.documents)} document(s) scored together")
        lines.append(
            "  {words:,} words, {sentences:,} sentences, {paragraphs:,} paragraphs, "
            "{syllables:,} syllables, about {rt:.0f}s to read".format(
                words=stats.get("words", 0),
                sentences=stats.get("sentences", 0),
                paragraphs=stats.get("paragraphs", 0),
                syllables=stats.get("syllables", 0),
                rt=stats.get("reading_time_seconds", 0.0),
            )
        )
        lines.append("")
        for name in MEASURE_NAMES:
            if name not in self.components:
                continue
            value = self.components[name]
            lines.append(f"  {name:<12} {value:5.1f} {_bar(value)}")
            explanation = self.measures.get(name, {}).get("explanation")
            if explanation:
                lines.append(f"               {explanation}")
        if self.notes:
            lines.append("")
            for note in self.notes:
                lines.append(f"  Note: {note}")
        if self.issues:
            lines.append("")
            lines.append("  Issues found:")
            for issue in self.issues:
                lines.append(f"    [{issue.severity:<6}] {issue.kind}: {issue.message}")
                for example in issue.examples[:2]:
                    lines.append(f"               e.g. {example}")
        if self.suggestions:
            lines.append("")
            lines.append("  What to fix first:")
            for i, text in enumerate(self.suggestions, start=1):
                lines.append(f"    {i}. {text}")
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """JSON-safe view of the whole report."""
        out: Dict[str, Any] = {
            "score": round(float(self.score), 1),
            "grade": self.grade,
            "target": self.target,
            "components": {k: round(float(v), 1) for k, v in self.components.items()},
            "issues": [issue.to_dict() for issue in self.issues],
            "suggestions": list(self.suggestions),
            "stats": dict(self.stats),
            "measures": {k: dict(v) for k, v in self.measures.items()},
            "notes": list(self.notes),
        }
        if self.documents is not None:
            out["documents"] = [doc.to_dict() for doc in self.documents]
        return out

    def to_json(self, **kwargs: Any) -> str:
        """The report as a JSON string (non-ASCII characters kept as they are)."""
        kwargs.setdefault("ensure_ascii", False)
        kwargs.setdefault("indent", 2)
        return json.dumps(self.to_dict(), **kwargs)

    def __repr__(self) -> str:                      # pragma: no cover - cosmetic
        return (
            f"QualityReport(score={self.score:.1f}, grade={self.grade!r}, "
            f"words={self.stats.get('words', 0)}, issues={len(self.issues)})"
        )


# --------------------------------------------------------------------------
# turning measures into issues and suggestions
# --------------------------------------------------------------------------

def _severity(ratio: float) -> str:
    if ratio >= 2.0:
        return "high"
    if ratio >= 1.3:
        return "medium"
    return "low"


def _examples_from_counts(entries: List[Dict[str, Any]], key: str, limit: int = 4) -> List[str]:
    return [
        "{0} ({1} {2})".format(entry[key], entry["count"], "time" if entry["count"] == 1 else "times")
        for entry in entries[:limit]
    ]


def build_issues(doc: Document, measures: Dict[str, Dict[str, Any]], target: str):
    """Return ``(issues, suggestions)``, both ordered by how much they matter."""
    weights = WEIGHTS[target]
    found: List[tuple] = []          # (impact, Issue, suggestion)

    if doc.is_empty:
        issue = Issue("empty_text", "high", "There is no text to score.", [], 0)
        return [issue], ["Give the scorer some text: this input has no words in it."]

    words = doc.n_words
    sentences = max(1, doc.n_sentences)
    read = measures["readability"]
    rep = measures["repetition"]
    struct = measures["structure"]
    clar = measures["clarity"]
    vocab = measures["vocabulary"]

    # -- clarity ----------------------------------------------------------
    share = clar["long_sentence_share"]
    if clar["long_sentences"] and share > 0.05:
        impact = weights["clarity"] * min(30.0, 120.0 * (share - 0.05))
        found.append((impact, Issue(
            "long_sentences", _severity(share / 0.15),
            f"{clar['long_sentences']} of {sentences} sentences run longer than "
            f"{clar['long_sentence_words']} words.",
            list(clar["long_sentence_examples"]), clar["long_sentences"],
        ), f"Split the {clar['long_sentences']} sentence(s) longer than "
           f"{clar['long_sentence_words']} words; 15 to 20 words each is easy to follow."))

    share = clar["passive_share"]
    if share > 0.20:
        impact = weights["clarity"] * min(45.0, 140.0 * (share - 0.10))
        found.append((impact, Issue(
            "passive_voice", _severity(share / 0.30),
            f"{clar['passive_sentences']} of {sentences} sentences read as passive "
            f"({share * 100:.0f}%).",
            list(clar["passive_examples"]), clar["passive_sentences"],
        ), f"Rewrite the {clar['passive_sentences']} passive sentence(s) so the doer comes first, "
           "for example 'the team tested the system' rather than 'the system was tested by the team'."))

    rate = clar["filler_count"] / words
    if rate > 0.010:
        impact = weights["clarity"] * min(25.0, 800.0 * (rate - 0.005))
        found.append((impact, Issue(
            "filler_words", _severity(rate / 0.020),
            f"{clar['filler_count']} filler word(s) in {words} words ({rate * 100:.1f}%).",
            _examples_from_counts(clar["filler_words"], "word"), clar["filler_count"],
        ), "Cut filler words such as 'very', 'really' and 'just'; they add length but no meaning."))

    rate = clar["hedge_count"] / words
    if rate > 0.025:
        impact = weights["clarity"] * min(15.0, 400.0 * (rate - 0.010))
        found.append((impact, Issue(
            "hedging", _severity(rate / 0.040),
            f"{clar['hedge_count']} hedging word(s) in {words} words ({rate * 100:.1f}%).",
            _examples_from_counts(clar["hedge_words"], "word"), clar["hedge_count"],
        ), "Trim hedges such as 'maybe', 'possibly' and 'generally' wherever you can state the point plainly."))

    rate = clar["nominalisation_count"] / words
    # The limit travels with the audience: a technical or academic piece names
    # concepts for a living, so it is given more room than marketing copy.
    nominal_limit = float(TARGETS[target]["nominalisation_limit"])
    nominal_allowance = float(clar.get("nominalisation_allowance", TARGETS[target]["nominalisation_rate"]))
    if rate > nominal_limit:
        impact = weights["clarity"] * min(15.0, 400.0 * (rate - nominal_allowance))
        found.append((impact, Issue(
            "nominalisations", _severity(rate / (nominal_limit * 1.57)),
            f"{clar['nominalisation_count']} noun(s) built out of verbs ({rate * 100:.1f}% of words).",
            _examples_from_counts(clar["nominalisations"], "word"), clar["nominalisation_count"],
        ), "Turn nouns back into verbs: 'we decided' instead of 'a decision was made'."))

    # -- readability ------------------------------------------------------
    if doc.word_mode == "space" and words >= 25:
        ideal = float(TARGETS[target]["grade"])
        grade_gap = read["flesch_kincaid_grade"] - ideal
        impact = weights["readability"] * (100.0 - read["score"])
        if grade_gap > 2.5:
            found.append((impact, Issue(
                "hard_to_read", _severity(grade_gap / 4.0),
                f"Reading grade {read['flesch_kincaid_grade']:.1f} is above the grade {ideal:.0f} that "
                f"{TARGETS[target]['label']} expects.",
                list(clar["long_sentence_examples"])[:2], int(round(grade_gap)),
            ), "Shorten sentences and swap long words for everyday ones to bring the reading grade down."))
        elif grade_gap < -2.5 and target in ("academic", "technical"):
            found.append((impact, Issue(
                "too_simple", _severity(abs(grade_gap) / 4.0),
                f"Reading grade {read['flesch_kincaid_grade']:.1f} is below the grade {ideal:.0f} that "
                f"{TARGETS[target]['label']} expects.",
                [], int(round(abs(grade_gap))),
            ), "The writing is simpler than this audience expects; precise terms and fuller sentences would fit better."))

    # -- repetition -------------------------------------------------------
    if rep["repeated_words"] and rep["word_repetition_rate"] > 0.010:
        impact = weights["repetition"] * min(45.0, 220.0 * rep["word_repetition_rate"])
        found.append((impact, Issue(
            "repeated_words", _severity(rep["word_repetition_rate"] / 0.030),
            f"{len(rep['repeated_words'])} word(s) are used far more often than the rest.",
            _examples_from_counts(rep["repeated_words"], "word"),
            sum(entry["count"] for entry in rep["repeated_words"]),
        ), "Vary or cut the words you lean on most: " +
           ", ".join(entry["word"] for entry in rep["repeated_words"][:3]) + "."))

    if rep["repeated_openers"] and rep["opener_repetition_rate"] > 0.25:
        impact = weights["repetition"] * min(25.0, 60.0 * rep["opener_repetition_rate"])
        top = rep["repeated_openers"][0]
        found.append((impact, Issue(
            "repeated_openers", _severity(rep["opener_repetition_rate"] / 0.45),
            f"Sentences keep starting the same way; '{top['word']}' opens {top['count']} of them.",
            _examples_from_counts(rep["repeated_openers"], "word"),
            sum(entry["count"] for entry in rep["repeated_openers"]),
        ), f"Start sentences differently; '{top['word']}' opens {top['count']} of {sentences}."))

    phrases = rep["repeated_trigrams"] or rep["repeated_bigrams"]
    if phrases and rep["phrase_repetition_rate"] > 0.010:
        impact = weights["repetition"] * min(30.0, 260.0 * rep["phrase_repetition_rate"])
        found.append((impact, Issue(
            "repeated_phrases", _severity(rep["phrase_repetition_rate"] / 0.025),
            f"{len(phrases)} phrase(s) come back more than once.",
            _examples_from_counts(phrases, "phrase"),
            sum(entry["count"] for entry in phrases),
        ), "Reword the phrases you repeat, starting with '" + phrases[0]["phrase"] + "'."))

    # -- structure --------------------------------------------------------
    if sentences >= 3 and struct["sentence_length_variation"] < 0.35:
        gap = 0.35 - struct["sentence_length_variation"]
        impact = weights["structure"] * 0.40 * min(100.0, 200.0 * gap)
        found.append((impact, Issue(
            "uniform_sentence_length", _severity(gap / 0.20),
            f"Sentences are all about the same length ({struct['mean_sentence_words']:.0f} words, "
            f"variation {struct['sentence_length_variation']:.2f}).",
            [], sentences,
        ), "Mix short and long sentences; a short one after a long one gives the reader a break."))

    if sentences >= 5 and struct["transition_share"] < 0.10:
        impact = weights["structure"] * 0.30 * min(100.0, 100.0 * (0.10 - struct["transition_share"]) / 0.10)
        found.append((impact, Issue(
            "few_transitions", _severity((0.10 - struct["transition_share"]) / 0.07),
            f"Only {struct['transition_sentences']} of {sentences} sentences open with a linking word.",
            [], struct["transition_sentences"],
        ), "Add linking words such as 'however', 'for example' or 'as a result' so the ideas connect."))

    if struct["longest_paragraph_words"] > 200:
        over = struct["longest_paragraph_words"] - 200
        impact = weights["structure"] * 0.30 * min(100.0, 100.0 * over / 250.0)
        found.append((impact, Issue(
            "long_paragraphs", _severity(over / 150.0),
            f"The longest paragraph holds {struct['longest_paragraph_words']} words.",
            [], struct["longest_paragraph_words"],
        ), "Break the longest paragraph up; around 100 words per paragraph is comfortable."))

    # -- vocabulary -------------------------------------------------------
    if words >= 50 and vocab["moving_average_ttr"] < 0.45:
        gap = 0.45 - vocab["moving_average_ttr"]
        impact = weights["vocabulary"] * 0.55 * min(100.0, 100.0 * gap / 0.35)
        found.append((impact, Issue(
            "narrow_vocabulary", _severity(gap / 0.15),
            f"The same words keep coming back (length-adjusted type-token ratio "
            f"{vocab['moving_average_ttr']:.2f}).",
            _examples_from_counts(rep["repeated_words"], "word"), vocab["unique_words"],
        ), "Widen the vocabulary; short stretches of this text reuse the same few words."))

    ideal_long = float(TARGETS[target]["long_word_share"])
    if doc.word_mode == "space" and words >= 25 and vocab["long_word_share"] > ideal_long + 0.08:
        over = vocab["long_word_share"] - ideal_long - 0.08
        impact = weights["vocabulary"] * 0.45 * min(100.0, 100.0 * over / 0.25)
        found.append((impact, Issue(
            "long_words", _severity(over / 0.08),
            f"{vocab['long_word_share'] * 100:.0f}% of words have three or more syllables; "
            f"{TARGETS[target]['label']} sits near {ideal_long * 100:.0f}%.",
            [], vocab["long_words"],
        ), "Swap some long words for short everyday ones, such as 'use' for 'utilise'."))

    if words < 25:
        found.append((0.05, Issue(
            "very_short_text", "low",
            f"Only {words} word(s): the scores are indicative rather than reliable.",
            [], words,
        ), "Score a longer passage if you need a dependable number; below about 25 words the measures are noisy."))

    found.sort(key=lambda entry: -entry[0])
    issues = [entry[1] for entry in found]
    suggestions = [entry[2] for entry in found]
    if not suggestions:
        suggestions = ["Nothing major to fix: this text scores well on all five measures."]
    return issues, suggestions
