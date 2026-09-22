"""Case evaluation, the two result objects, and the public entry points.

Every number this module reports lives in ``0..1`` and has its exact formula in
the docstring of the function that produces it. The point of the package is that
two people running it on two systems can compare the outputs, which is only true
if there is no room left for an undocumented variant.

Two decisions are made once, here, and carried into every metric:

``threshold``
    The similarity at or above which two texts count as a match - a passage as
    relevant to a query, a passage as support for a sentence. The same number is
    also the score below which a case is listed in ``failures``, with one
    deliberate exception: ``answer_relevance`` is reported but never turned into
    a failure line under lexical overlap, because a correct answer routinely
    answers a question without repeating any of its words.
``embed``
    When given, every similarity is the cosine of two embedding vectors. When
    not, it is word overlap. ``report.similarity`` always says which one ran.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Dict,
    Hashable,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)

from ._metrics import hit_rate, mrr, ndcg_at_k, precision_at_k, recall_at_k
from ._text import Similarity, looks_like_passage, split_sentences

logger = logging.getLogger(__name__)

__all__ = [
    "ANSWER_METRICS",
    "METRIC_HELP",
    "RETRIEVAL_METRICS",
    "CaseResult",
    "RagReport",
    "evaluate",
    "evaluate_case",
]

#: Retrieval metrics, in report order.
RETRIEVAL_METRICS: Tuple[str, ...] = (
    "precision_at_k",
    "recall_at_k",
    "mrr",
    "ndcg_at_k",
    "hit_rate",
)

#: Answer metrics, in report order. ``answer_correctness`` only appears for
#: cases that carry a ``ground_truth``.
ANSWER_METRICS: Tuple[str, ...] = (
    "groundedness",
    "citation_coverage",
    "answer_relevance",
    "answer_correctness",
)

#: One line per metric, used by :meth:`RagReport.summary`. The full formula is
#: in each metric function's docstring; these are the reminders.
METRIC_HELP: Dict[str, str] = {
    "precision_at_k": "relevant items in the top k, divided by k",
    "recall_at_k": "relevant items found, divided by all relevant ones",
    "mrr": "1 / rank of the first relevant item, averaged over cases",
    "ndcg_at_k": "DCG@k / ideal DCG@k, linear gains, log2 discount",
    "hit_rate": "share of queries with a relevant item in the top k",
    "groundedness": "share of the answer's words a retrieved passage supports",
    "citation_coverage": "share of retrieved passages the answer actually uses",
    "answer_relevance": "share of the query's content words the answer repeats",
    "answer_correctness": "share of the ground truth the answer covers",
}

_TEXT_KEYS = ("text", "passage", "content", "chunk", "document")

_UNHASHABLE = (list, dict, set, bytearray)


def _as_item(item: Any) -> Tuple[Hashable, str]:
    """Split one retrieved item into ``(id, text)``.

    A plain string is both its own id and its own text. A mapping supplies
    ``id`` and/or one of ``text``/``passage``/``content``/``chunk``/
    ``document``; whichever is missing falls back to the other. Anything else
    (an int id, a UUID) is the id, and ``str()`` of it is the text.
    """
    if isinstance(item, str):
        return item, item
    if isinstance(item, Mapping):
        text = ""
        for key in _TEXT_KEYS:
            value = item.get(key)
            if isinstance(value, str) and value:
                text = value
                break
        ident = item.get("id", text)
        if isinstance(ident, _UNHASHABLE):
            raise ValueError(
                "retrieved id must be hashable, got " + type(ident).__name__
            )
        return ident, text or str(ident)
    if isinstance(item, _UNHASHABLE):
        raise ValueError(
            "retrieved items must be strings, mappings or ids, got "
            + type(item).__name__
        )
    return item, str(item)


def _as_gains(relevant: Any) -> Dict[Hashable, float]:
    """Normalise ``relevant`` into ``{id: gain}`` with gains as floats.

    A list of ids becomes gain ``1.0`` each (binary relevance, the usual case).
    A mapping of ``{id: gain}`` is taken as graded relevance and used as given;
    gains are clipped at ``0`` because a negative gain has no meaning in DCG.
    """
    if relevant is None:
        return {}
    if isinstance(relevant, Mapping):
        pairs: Iterable[Tuple[Any, Any]] = list(relevant.items())
    elif isinstance(relevant, (str, bytes)):
        raise ValueError(
            "'relevant' must be a list of ids or a {id: gain} mapping, "
            "not a single string"
        )
    elif isinstance(relevant, (Sequence, set, frozenset)):
        pairs = [(entry, 1.0) for entry in relevant]
    else:
        raise ValueError(
            "'relevant' must be a list of ids or a {id: gain} mapping, got "
            + type(relevant).__name__
        )
    gains: Dict[Hashable, float] = {}
    for key, gain in pairs:
        ident, _ = _as_item(key)
        try:
            value = float(gain)
        except (TypeError, ValueError):
            raise ValueError(
                "relevance gain for {0!r} is not a number: {1!r}".format(ident, gain)
            ) from None
        if value != value:  # NaN
            value = 0.0
        gains[ident] = max(0.0, value)
    return gains


def _shorten(text: Any, width: int = 60) -> str:
    """One-line preview of ``text`` for messages, never longer than ``width``."""
    flat = " ".join(str(text).split())
    if len(flat) <= width:
        return flat
    return flat[: width - 3] + "..."


@dataclass
class CaseResult:
    """One query's scores, plus everything needed to explain them.

    ``metrics`` holds only the metrics that could be computed; a metric that was
    excluded (recall for a query with no relevant items, the answer metrics for
    a case with no answer) is absent from ``metrics`` and named in ``excluded``,
    never faked as ``0.0``. ``notes`` records anything that changed how a number
    was produced - de-duplication, a short result list, estimated relevance.
    """

    query: str
    index: int = 0
    k: int = 5
    threshold: float = 0.5
    n_retrieved: int = 0
    n_unique: int = 0
    n_relevant: int = 0
    estimated: bool = False
    metrics: Dict[str, float] = field(default_factory=dict)
    excluded: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    failures: List[str] = field(default_factory=list)

    @property
    def score(self) -> float:
        """Mean of this case's available metrics, in ``0..1``.

        ``score = (sum of metrics) / (number of metrics)``, and ``0.0`` when no
        metric could be computed at all. This is the ordering used by
        :meth:`RagReport.weakest`; it is a triage handle, not a headline number,
        which is why it is not in ``metrics``.
        """
        if not self.metrics:
            return 0.0
        return sum(self.metrics.values()) / len(self.metrics)

    @property
    def ok(self) -> bool:
        """True when this case produced no failure lines."""
        return not self.failures

    def summary(self) -> str:
        """Human-readable block describing this one case."""
        head = "[{0}] {1!r}  score {2:.2f}".format(
            self.index, _shorten(self.query, 64), self.score
        )
        if self.estimated:
            head += "  (estimated relevance)"
        lines = [head]
        lines.append(
            "  retrieved {0} unique of {1}, top-{2}, {3} relevant id(s) known".format(
                self.n_unique, self.n_retrieved, self.k, self.n_relevant
            )
        )
        for name in RETRIEVAL_METRICS + ANSWER_METRICS:
            if name in self.metrics:
                lines.append("  {0:<18} {1:.3f}".format(name, self.metrics[name]))
        for note in self.notes:
            lines.append("  note: " + note)
        for failure in self.failures:
            lines.append("  FAIL: " + failure)
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """JSON-safe dict of this case."""
        return {
            "index": self.index,
            "query": self.query,
            "k": self.k,
            "threshold": self.threshold,
            "n_retrieved": self.n_retrieved,
            "n_unique": self.n_unique,
            "n_relevant": self.n_relevant,
            "estimated": self.estimated,
            "score": round(self.score, 6),
            "metrics": {name: round(value, 6) for name, value in self.metrics.items()},
            "excluded": list(self.excluded),
            "notes": list(self.notes),
            "failures": list(self.failures),
        }


@dataclass
class RagReport:
    """The result of :func:`evaluate`: means, per-case detail and plain warnings.

    ``metrics`` averages each metric over the cases where it was defined, so a
    metric is never dragged down by cases that could not produce it. How many
    cases went into each mean is in ``support`` and is printed by
    :meth:`summary`.
    """

    metrics: Dict[str, float] = field(default_factory=dict)
    per_case: List[CaseResult] = field(default_factory=list)
    failures: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    support: Dict[str, int] = field(default_factory=dict)
    k: int = 5
    threshold: float = 0.5
    similarity: str = "lexical overlap"
    n_cases: int = 0
    n_estimated: int = 0

    @property
    def estimated(self) -> bool:
        """True when any case's relevance was estimated rather than measured."""
        return self.n_estimated > 0

    @property
    def fully_estimated(self) -> bool:
        """True when *every* case's relevance was estimated."""
        return self.n_cases > 0 and self.n_estimated == self.n_cases

    @property
    def ok(self) -> bool:
        """True when no case produced a failure line."""
        return not self.failures

    def weakest(self, n: int = 5) -> List[CaseResult]:
        """The ``n`` worst cases, weakest first, ordered by :attr:`CaseResult.score`.

        Ties break on the original case order, so repeated runs list the same
        cases in the same order. ``n <= 0`` returns an empty list.

        A case where nothing could be measured at all - a query labelled as
        having no relevant document and carrying no answer - is left out: its
        score is ``0.0`` for want of any metric, and listing it as the weakest
        case would put the queries a system is *meant* to fail at the top of
        the triage list.
        """
        if n <= 0:
            return []
        scored = [case for case in self.per_case if case.metrics]
        ordered = sorted(scored, key=lambda case: (case.score, case.index))
        return ordered[:n]

    def summary(self) -> str:
        """Plain-text report: what ran, the numbers, what to distrust, what broke."""
        lines: List[str] = []
        lines.append(
            "rag-quality-check: {0} case(s), top-{1}, threshold {2:g}, "
            "similarity by {3}.".format(
                self.n_cases, self.k, self.threshold, self.similarity
            )
        )
        if self.estimated:
            scope = (
                "every case"
                if self.fully_estimated
                else "{0} of {1} cases".format(self.n_estimated, self.n_cases)
            )
            lines.append(
                "  ESTIMATED: {0} had no 'relevant' ids, so relevance was guessed "
                "from".format(scope)
            )
            lines.append(
                "  {0} against the query. Those numbers are estimated, not "
                "measured;".format(self.similarity)
            )
            lines.append("  label them as such before comparing two systems.")
        if not self.metrics:
            lines.append("  no metric could be computed; see the notes below.")
        for group, names in (
            ("retrieval", RETRIEVAL_METRICS),
            ("answer", ANSWER_METRICS),
        ):
            present = [name for name in names if name in self.metrics]
            if not present:
                continue
            lines.append(group + ":")
            for name in present:
                lines.append(
                    "  {0:<18} {1:.3f}   {2} ({3} case(s))".format(
                        name,
                        self.metrics[name],
                        METRIC_HELP[name],
                        self.support.get(name, 0),
                    )
                )
        if self.notes:
            lines.append("notes ({0}):".format(len(self.notes)))
            for note in self.notes[:10]:
                lines.append("  - " + note)
            if len(self.notes) > 10:
                lines.append("  - ... {0} more note(s)".format(len(self.notes) - 10))
        if self.failures:
            lines.append("failures ({0}):".format(len(self.failures)))
            for failure in self.failures[:10]:
                lines.append("  - " + failure)
            if len(self.failures) > 10:
                lines.append(
                    "  - ... {0} more failure(s)".format(len(self.failures) - 10)
                )
        else:
            lines.append("failures: none.")
        weakest = self.weakest(3)
        if weakest and self.n_cases > 1:
            lines.append("weakest cases:")
            for case in weakest:
                lines.append(
                    "  [{0}] {1:.2f}  {2!r}".format(
                        case.index, case.score, _shorten(case.query, 58)
                    )
                )
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """JSON-safe dict of the whole report, per-case detail included."""
        return {
            "k": self.k,
            "threshold": self.threshold,
            "similarity": self.similarity,
            "n_cases": self.n_cases,
            "n_estimated": self.n_estimated,
            "estimated": self.estimated,
            "metrics": {name: round(value, 6) for name, value in self.metrics.items()},
            "support": dict(self.support),
            "notes": list(self.notes),
            "failures": list(self.failures),
            "per_case": [case.to_dict() for case in self.per_case],
        }

    def to_frame(self):
        """One row per case as a ``pandas.DataFrame``, metrics as columns.

        Needs pandas, which is an optional extra:
        ``pip install rag-quality-check[frame]``. A metric a case did not
        produce is ``NaN`` rather than ``0``, so ``frame.mean()`` agrees with
        :attr:`metrics` instead of quietly disagreeing with it.
        """
        try:
            import pandas as pd
        except ImportError:  # pragma: no cover - only without pandas installed
            raise ImportError(
                "to_frame() needs pandas: pip install rag-quality-check[frame]"
            ) from None
        columns = [
            name
            for name in RETRIEVAL_METRICS + ANSWER_METRICS
            if any(name in case.metrics for case in self.per_case)
        ]
        base = [
            "index",
            "query",
            "n_retrieved",
            "n_unique",
            "n_relevant",
            "estimated",
            "score",
        ]
        rows: List[Dict[str, Any]] = []
        for case in self.per_case:
            row: Dict[str, Any] = {
                "index": case.index,
                "query": case.query,
                "n_retrieved": case.n_retrieved,
                "n_unique": case.n_unique,
                "n_relevant": case.n_relevant,
                "estimated": case.estimated,
                "score": case.score,
            }
            for name in columns:
                row[name] = case.metrics.get(name, float("nan"))
            row["failures"] = len(case.failures)
            rows.append(row)
        if not rows:
            return pd.DataFrame(columns=base + columns + ["failures"])
        return pd.DataFrame(rows, columns=base + columns + ["failures"])

    def __str__(self) -> str:  # pragma: no cover - convenience only
        return self.summary()



def _groundedness(
    answer: str,
    passages: Sequence[str],
    sim: Similarity,
    threshold: float,
) -> Tuple[float, List[int]]:
    """``groundedness = (weight of the supported sentences) / (weight of every sentence)``.

    A sentence's weight is the number of distinct content words in it, the same
    tokens :func:`~._text.coverage` compares.

    A sentence is supported when its similarity to at least one retrieved
    passage reaches ``threshold``. Without ``embed`` that similarity is the
    share of the sentence's content words the passage contains, so "supported"
    reads as "this sentence is made of words that are in the context".

    Sentences are weighted by how many content words they hold, not counted one
    each: a leading ``"Yes."`` is one word and must not cancel out a paragraph
    copied verbatim from the context, which is exactly what equal weights did.
    When no sentence has a content word at all (punctuation, digits) every
    sentence weighs ``1`` instead, so the formula still divides by something.

    Also returns the positions of the passages that supported something, which
    is what :func:`_citation_coverage` counts. Range ``0..1``; an answer with no
    sentences, or no passages to check against, scores ``0.0``.
    """
    sentences = split_sentences(answer)
    if not sentences or not passages:
        return 0.0, []
    weights = [float(len(sim.tokens(sentence))) for sentence in sentences]
    total = sum(weights)
    if total <= 0.0:
        weights = [1.0] * len(sentences)
        total = float(len(sentences))
    supported = 0.0
    used: List[int] = []
    seen = set()
    for position_in_answer, sentence in enumerate(sentences):
        best_index = -1
        best_score = 0.0
        for position, passage in enumerate(passages):
            score = sim.score(sentence, passage)
            if score > best_score:
                best_score = score
                best_index = position
        if best_score >= threshold and best_index >= 0:
            supported += weights[position_in_answer]
            if best_index not in seen:
                seen.add(best_index)
                used.append(best_index)
    return min(1.0, supported / total), used


def _citation_coverage(used: Sequence[int], n_passages: int) -> float:
    """``citation_coverage = (passages that supported a sentence) / (passages retrieved)``.

    Read it as "how much of the context the answer actually drew on". A low
    number next to a high ``groundedness`` means most of what was retrieved went
    unused, which is a retriever returning more than the answer needs. There are
    no citation markers to parse here, so *supporting a sentence* is what counts
    as being cited, at the same ``threshold``. Range ``0..1``; ``0.0`` when
    nothing was retrieved.
    """
    if n_passages <= 0:
        return 0.0
    return min(1.0, len(set(used)) / n_passages)


def _answer_relevance(query: str, answer: str, sim: Similarity) -> float:
    """``answer_relevance = similarity(query, answer)``, in ``0..1``.

    Under the default lexical backend that is
    ``|query words in the answer| / |query words|`` - the share of the
    question's own content words the answer repeats. Read it as an echo
    measure, not as a verdict: "It is 29 dollars per seat per month." answers
    "how much does the pro plan cost" perfectly and scores ``0.00`` because it
    repeats none of the question's words. That is why a low value is never
    turned into a failure line unless ``embed`` is given, where the number is a
    cosine between two meanings rather than a word count.
    """
    return sim.score(query, answer)


def _answer_correctness(ground_truth: str, answer: str, sim: Similarity) -> float:
    """``answer_correctness = similarity(ground_truth, answer)``, in ``0..1``.

    Under the default lexical backend that is
    ``|ground truth words in the answer| / |ground truth words|`` - the share of
    the reference answer the generated one covers. Asymmetric on purpose: an
    answer that says everything the reference says plus more is still correct.
    """
    return sim.score(ground_truth, answer)


def _score_answer(
    case: Mapping[str, Any],
    result: "CaseResult",
    index: int,
    texts: Sequence[str],
    checkable: Sequence[str],
    sim: Similarity,
    threshold: float,
    tally: Dict[str, int],
) -> None:
    """Add the answer metrics to ``result``, or exclude them and say why.

    ``checkable`` is the subset of ``texts`` that carries passage prose.
    ``groundedness`` and ``citation_coverage`` are measured against it and only
    it: an item that is just an id has no words to support an answer sentence,
    and scoring against it would report a hallucination that did not happen.
    When ids came back but no text, both are excluded and noted rather than
    reported as ``0.0``; when *nothing* came back they stay ``0.0``, because an
    answer with no context behind it really is ungrounded.
    ``answer_relevance`` and ``answer_correctness`` never touch the passages, so
    they are reported either way.
    """
    answer = case.get("answer")
    if answer is None or not str(answer).strip():
        result.excluded.extend(ANSWER_METRICS)
        if answer is not None:
            result.notes.append(
                "case {0}: 'answer' is blank, so the answer metrics are "
                "excluded".format(index)
            )
        return

    answer = str(answer)
    query = result.query
    n_dropped = len(texts) - len(checkable)

    if checkable or not texts:
        # Nothing retrieved at all is a real ``0``, not an unknown: an answer
        # built on no context is ungrounded, and the case already says the
        # retriever came back empty.
        grounded, used = _groundedness(answer, checkable, sim, threshold)
        result.metrics["groundedness"] = grounded
        result.metrics["citation_coverage"] = _citation_coverage(used, len(checkable))
        if n_dropped:
            result.notes.append(
                "case {0}: {1} of {2} retrieved item(s) carry an id but no passage "
                "text; groundedness and citation_coverage cover the other "
                "{3}".format(index, n_dropped, len(texts), len(checkable))
            )
        if grounded < threshold:
            result.failures.append(
                "case {0} {1!r}: answer not grounded - only {2:.0%} of it is "
                "supported by the retrieved passages".format(
                    index, _shorten(query), grounded
                )
            )
    else:
        result.excluded.extend(("groundedness", "citation_coverage"))
        tally["answers_without_text"] = tally.get("answers_without_text", 0) + 1
        result.notes.append(
            "case {0}: 'retrieved' holds ids, not passage text, so groundedness "
            "and citation_coverage are excluded - an id carries no words for the "
            "answer to be supported by, and scoring against one would report a "
            "hallucination that did not happen; pass the passage text, or "
            "{{'id': ..., 'text': ...}}, to measure them".format(index)
        )

    relevance = _answer_relevance(query, answer, sim)
    result.metrics["answer_relevance"] = relevance
    if sim.uses_embeddings:
        # Cosine between the query and the answer does mean "does this answer
        # address this question", so a low one is a real failure. Word overlap
        # does not, so below it is noted instead of accused.
        if relevance < threshold:
            result.failures.append(
                "case {0} {1!r}: answer does not address the query "
                "(answer_relevance {2:.2f})".format(index, _shorten(query), relevance)
            )
    elif relevance < threshold:
        result.notes.append(
            "case {0}: answer_relevance is {1:.2f} because the answer repeats few "
            "of the query's own words, which a correct answer often does not do; "
            "it is reported, never counted as a failure - pass embed= for a "
            "semantic answer_relevance".format(index, relevance)
        )

    truth = case.get("ground_truth")
    if truth is not None and str(truth).strip():
        correctness = _answer_correctness(str(truth), answer, sim)
        result.metrics["answer_correctness"] = correctness
        if correctness < threshold:
            result.failures.append(
                "case {0} {1!r}: answer differs from ground_truth "
                "(answer_correctness {2:.2f})".format(
                    index, _shorten(query), correctness
                )
            )


def _evaluate_one(
    case: Mapping[str, Any],
    index: int,
    k: int,
    threshold: float,
    sim: Similarity,
    tally: Optional[Dict[str, int]] = None,
) -> CaseResult:
    """Score a single case. Every edge case in the README is decided here.

    ``tally`` collects the few counts that only mean something across the whole
    run - how many labelled cases matched an id, how many answers had no
    passage text to check against - so :func:`evaluate` can add one run-level
    note instead of repeating itself per case.
    """
    if tally is None:
        tally = {}
    query = str(case.get("query", "") or "")
    raw_retrieved = case.get("retrieved")
    if raw_retrieved is None:
        raw_retrieved = []
    if isinstance(raw_retrieved, (str, bytes)):
        raise ValueError(
            "case {0}: 'retrieved' must be a list of passages or ids, not a single "
            "string".format(index)
        )
    if not isinstance(raw_retrieved, Sequence):
        raise ValueError(
            "case {0}: 'retrieved' must be a list, got {1}".format(
                index, type(raw_retrieved).__name__
            )
        )

    result = CaseResult(query=query, index=index, k=k, threshold=threshold)
    result.n_retrieved = len(raw_retrieved)

    # De-duplicate by id, first occurrence wins: a retriever that returns the
    # same passage twice must not be paid twice for it.
    ids: List[Hashable] = []
    texts: List[str] = []
    seen = set()
    duplicates = 0
    for item in raw_retrieved:
        ident, text = _as_item(item)
        if ident in seen:
            duplicates += 1
            continue
        seen.add(ident)
        ids.append(ident)
        texts.append(text)
    result.n_unique = len(ids)
    # Which retrieved items actually carry prose an answer can be checked
    # against. A bare id ("p_reset") carries none, and scoring an answer against
    # it reports a hallucination that never happened. With ``embed`` the
    # caller's own model decides what a string means - it may well map ids to
    # the vectors of their passages - so nothing is filtered there.
    if sim.uses_embeddings:
        checkable = list(texts)
    else:
        checkable = [text for text in texts if looks_like_passage(text)]
    if duplicates:
        result.notes.append(
            "case {0}: {1} duplicate passage(s) in 'retrieved' counted once".format(
                index, duplicates
            )
        )

    if not query.strip():
        result.notes.append(
            "case {0}: the query is empty, so query-based metrics will be 0".format(
                index
            )
        )

    has_relevant = "relevant" in case and case.get("relevant") is not None
    gain_by_id = _as_gains(case.get("relevant")) if has_relevant else {}
    result.n_relevant = sum(1 for gain in gain_by_id.values() if gain > 0)
    unanswerable = has_relevant and result.n_relevant == 0

    if result.n_unique == 0:
        if unanswerable:
            result.notes.append(
                "case {0}: nothing came back, which for a query labelled as "
                "having no right answer is the wanted outcome, not a "
                "failure".format(index)
            )
        else:
            result.failures.append(
                "case {0} {1!r}: retrieved nothing".format(index, _shorten(query))
            )
    elif result.n_unique < k:
        result.notes.append(
            "case {0}: asked for top-{1} but only {2} passage(s) came back; "
            "precision_at_k still divides by {1}".format(index, k, result.n_unique)
        )

    if unanswerable:
        # The caller labelled this query as having no relevant document at all -
        # an unanswerable query, kept in the set on purpose to test refusals.
        # Every retrieval metric can then only be 0, which is not a measurement:
        # it would drag the means down and fill ``failures`` with the expected
        # outcome. Exclude them all and say so, the same way recall was already
        # excluded here.
        result.excluded.extend(RETRIEVAL_METRICS)
        result.notes.append(
            "case {0}: 'relevant' lists no relevant ids at all, so this query is "
            "treated as having no right answer; every retrieval metric is "
            "excluded (retrieving nothing relevant is the wanted result here, "
            "not a failure)".format(index)
        )
        _score_answer(case, result, index, texts, checkable, sim, threshold, tally)
        return result

    if has_relevant:
        gains = [float(gain_by_id.get(ident, 0.0)) for ident in ids]
        ideal = sorted((gain for gain in gain_by_id.values() if gain > 0), reverse=True)
        # Run-level bookkeeping: a labelled case whose relevant ids never appear
        # in the retrieved ids is usually a harness wiring mistake, not a broken
        # retriever, but only the whole run can tell the two apart.
        tally["labelled"] = tally.get("labelled", 0) + 1
        if any(gain > 0 for gain in gains):
            tally["id_matched"] = tally.get("id_matched", 0) + 1
    else:
        # No labels: estimate relevance from similarity to the query. Binary, so
        # every metric keeps the meaning it has with real labels.
        result.estimated = True
        gains = [1.0 if sim.score(query, text) >= threshold else 0.0 for text in texts]
        # Nothing is known about relevant passages that were never retrieved, so
        # the ideal ranking can only be the retrieved set, sorted.
        ideal = sorted((gain for gain in gains if gain > 0), reverse=True)
        result.n_relevant = len(ideal)
        result.notes.append(
            "case {0}: no 'relevant' ids, so relevance was estimated from {1} "
            "against the query at threshold {2:g}".format(index, sim.kind, threshold)
        )
        if result.n_unique and not checkable:
            result.notes.append(
                "case {0}: 'retrieved' holds ids, not passage text, so there is "
                "nothing for the query to overlap with and the estimate can only "
                "come out 0; pass 'relevant' ids, or the passage text, to measure "
                "this case".format(index)
            )

    result.metrics["precision_at_k"] = precision_at_k(gains, k)
    result.metrics["mrr"] = mrr(gains, k)
    result.metrics["hit_rate"] = hit_rate(gains, k)

    if result.estimated:
        result.excluded.append("recall_at_k")
        result.notes.append(
            "case {0}: recall_at_k excluded - without 'relevant' ids there is no "
            "way to know what was missed".format(index)
        )
    else:
        # ``ideal`` always holds at least one positive gain by this point: a
        # query labelled with none at all returned above, where every retrieval
        # metric is excluded rather than divided by nothing.
        recall = recall_at_k(gains, ideal, k)
        if recall is None:  # pragma: no cover - unreachable, see the comment
            result.excluded.append("recall_at_k")
        else:
            result.metrics["recall_at_k"] = recall

    ndcg = ndcg_at_k(gains, ideal, k)
    if ndcg is None:
        result.excluded.append("ndcg_at_k")
        result.notes.append(
            "case {0}: nothing relevant is known for this query, so ndcg_at_k is "
            "excluded".format(index)
        )
    else:
        result.metrics["ndcg_at_k"] = ndcg

    if result.n_unique and result.metrics["hit_rate"] == 0.0:
        tail = " (estimated)" if result.estimated else ""
        result.failures.append(
            "case {0} {1!r}: nothing relevant in the top {2}{3}".format(
                index, _shorten(query), k, tail
            )
        )

    _score_answer(case, result, index, texts, checkable, sim, threshold, tally)

    return result


def _texts_of(case: Mapping[str, Any]) -> List[str]:
    """Every string in a case a similarity might later be asked about."""
    out: List[str] = []
    query = case.get("query")
    if isinstance(query, str) and query:
        out.append(query)
    retrieved = case.get("retrieved") or []
    if isinstance(retrieved, Sequence) and not isinstance(retrieved, (str, bytes)):
        for item in retrieved:
            try:
                _, text = _as_item(item)
            except ValueError:
                continue
            if text:
                out.append(text)
    for key in ("answer", "ground_truth"):
        value = case.get(key)
        if isinstance(value, str) and value.strip():
            out.append(value)
            out.extend(split_sentences(value))
    return out


def evaluate(
    cases: Sequence[Mapping[str, Any]],
    *,
    k: int = 5,
    threshold: float = 0.5,
    embed: Optional[Callable[[List[str]], Sequence]] = None,
) -> RagReport:
    """Score a list of retrieval cases and return a report that explains itself.

    Each case is a dict:

    ``query``
        The question that was asked. A blank one is scored and noted rather than
        raising, because one bad row should not stop an evaluation run.
    ``retrieved``
        What came back, best first: passage strings, ids, or dicts carrying
        ``id`` and/or ``text``. Duplicates are counted once and noted. Ids alone
        are all the retrieval metrics need; ``groundedness`` and
        ``citation_coverage`` need the passage text, and are excluded with a
        note when only ids came in rather than scored ``0``.
    ``relevant`` *(optional)*
        The ids that should have come back - a list (binary relevance) or a
        ``{id: gain}`` mapping (graded). Without it, relevance is **estimated**
        from similarity to the query, and the report says so everywhere a number
        is shown. An empty list means "this query has no right answer": every
        retrieval metric is excluded for that case, because 0 would be the score
        of a system doing exactly what was asked of it.
    ``answer`` *(optional)*
        The generated answer. Turns on ``groundedness``, ``citation_coverage``
        and ``answer_relevance``.
    ``ground_truth`` *(optional)*
        The reference answer. Adds ``answer_correctness``.

    Parameters
    ----------
    k:
        Cut-off for the retrieval metrics, at least 1. It is always the
        denominator of ``precision_at_k``, even when fewer than ``k`` passages
        came back - that case is noted rather than hidden by shrinking ``k``.
    threshold:
        Similarity in ``0..1`` at which two texts count as a match, and the
        score below which a case is reported as a failure. ``answer_relevance``
        is the one metric it never fails a case on under lexical overlap: a
        correct answer routinely answers the question without repeating any of
        its words, so a low value there is noted, not accused. With ``embed`` it
        is a cosine between meanings and does gate a failure.
    embed:
        ``callable(list[str]) -> vectors``. When given, every similarity is the
        cosine of those vectors instead of word overlap; texts are batched into
        as few calls as possible and cached, so a real model stays usable.

    Returns
    -------
    RagReport
        ``.metrics`` (means), ``.per_case``, ``.weakest()``, ``.failures``,
        ``.notes``, ``.summary()``, ``.to_dict()``, ``.to_frame()``.

    Examples
    --------
    >>> report = evaluate([
    ...     {"query": "capital of France",
    ...      "retrieved": ["p1", "p2"],
    ...      "relevant": ["p1"]},
    ... ], k=2)
    >>> report.metrics["hit_rate"]
    1.0
    """
    if isinstance(k, bool) or not isinstance(k, int):
        raise ValueError("k must be an int, got " + type(k).__name__)
    if k < 1:
        raise ValueError("k must be at least 1, got {0}".format(k))
    try:
        threshold = float(threshold)
    except (TypeError, ValueError):
        raise ValueError(
            "threshold must be a number in 0..1, got {0!r}".format(threshold)
        ) from None
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be in 0..1, got {0}".format(threshold))
    if isinstance(cases, Mapping):
        raise ValueError(
            "cases must be a list of case dicts, not a single dict - "
            "wrap it: evaluate([case])"
        )
    if isinstance(cases, (str, bytes)) or not isinstance(cases, Sequence):
        raise ValueError(
            "cases must be a list of case dicts, got " + type(cases).__name__
        )
    for position, case in enumerate(cases):
        if not isinstance(case, Mapping):
            raise ValueError(
                "case {0} must be a dict with 'query' and 'retrieved', got {1}".format(
                    position, type(case).__name__
                )
            )

    sim = Similarity(embed)
    if sim.uses_embeddings:
        # One batched call for the whole run instead of one per comparison.
        everything: List[str] = []
        for case in cases:
            everything.extend(_texts_of(case))
        sim.warm(everything)

    report = RagReport(
        k=k, threshold=threshold, similarity=sim.kind, n_cases=len(cases)
    )
    if not cases:
        report.notes.append("no cases were given, so there is nothing to measure")
        logger.debug("evaluate() called with an empty case list")
        return report

    totals: Dict[str, float] = {}
    counts: Dict[str, int] = {}
    tally: Dict[str, int] = {}
    for index, case in enumerate(cases):
        result = _evaluate_one(case, index, k, threshold, sim, tally)
        report.per_case.append(result)
        if result.estimated:
            report.n_estimated += 1
        report.notes.extend(result.notes)
        report.failures.extend(result.failures)
        for name, value in result.metrics.items():
            totals[name] = totals.get(name, 0.0) + value
            counts[name] = counts.get(name, 0) + 1

    for name in RETRIEVAL_METRICS + ANSWER_METRICS:
        if counts.get(name):
            report.metrics[name] = totals[name] / counts[name]
            report.support[name] = counts[name]

    excluded_recall = sum(
        1 for case in report.per_case if "recall_at_k" in case.excluded
    )
    if excluded_recall:
        report.notes.append(
            "recall_at_k is excluded for {0} of {1} case(s); the reported mean "
            "covers the other {2}".format(
                excluded_recall, len(cases), report.support.get("recall_at_k", 0)
            )
        )
    if report.estimated:
        report.notes.append(
            "{0} of {1} case(s) had no 'relevant' ids: those retrieval numbers are "
            "estimated from similarity, not measured".format(
                report.n_estimated, len(cases)
            )
        )
    labelled = tally.get("labelled", 0)
    if labelled > 1 and not tally.get("id_matched", 0):
        # Every labelled case scored 0 because no id ever lined up. A retriever
        # that bad is possible; ids compared in two different forms is far more
        # likely, and the numbers look identical either way.
        report.notes.append(
            "no 'relevant' id matched any retrieved id in any of the {0} labelled "
            "case(s); check that both sides use the same id form (upper/lower "
            "case, prefixes, str against int) - if they do not, every retrieval "
            "number above is 0 for that reason, not because the retriever found "
            "nothing".format(labelled)
        )
    without_text = tally.get("answers_without_text", 0)
    if without_text:
        report.notes.append(
            "{0} of {1} case(s) with an 'answer' had no passage text in "
            "'retrieved', only ids: groundedness and citation_coverage are "
            "excluded there rather than reported as 0".format(without_text, len(cases))
        )
    return report


def evaluate_case(
    query: str,
    retrieved: Sequence[Any],
    *,
    relevant: Any = None,
    answer: Optional[str] = None,
    ground_truth: Optional[str] = None,
    k: int = 5,
    threshold: float = 0.5,
    embed: Optional[Callable[[List[str]], Sequence]] = None,
) -> CaseResult:
    """Score one query on its own and return its :class:`CaseResult`.

    Same rules and the same numbers as :func:`evaluate` on a one-case list -
    this is the shortcut for a notebook or a unit test, not a second code path.

    Examples
    --------
    >>> case = evaluate_case("solar power", ["solar panels make power"], k=1)
    >>> case.metrics["hit_rate"]
    1.0
    """
    case: Dict[str, Any] = {"query": query, "retrieved": list(retrieved or [])}
    if relevant is not None:
        case["relevant"] = relevant
    if answer is not None:
        case["answer"] = answer
    if ground_truth is not None:
        case["ground_truth"] = ground_truth
    report = evaluate([case], k=k, threshold=threshold, embed=embed)
    return report.per_case[0]
