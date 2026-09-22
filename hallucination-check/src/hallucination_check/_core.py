"""The public API: :func:`check`, :func:`check_batch`, :class:`GroundingChecker`."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from ._facts import Fact, collect_keys, extract_facts, mask_numbers, number_tokens
from ._match import Span, SourceIndex, lexical_score
from ._text import TextVec, split_units, truncate, vectorize

logger = logging.getLogger(__name__)

__all__ = ["Claim", "GroundingReport", "GroundingChecker", "check", "check_batch"]

GRANULARITIES = ("sentence", "clause", "paragraph")
EmbedFn = Callable[[Sequence[str]], Any]


@dataclass
class Claim:
    """One unit of the answer and the verdict on it.

    Attributes:
        text: the claim exactly as it appears in the answer.
        supported: True when the sources back this claim up.
        confidence: 0-1 support score, how well the best source span covers the claim.
        best_source: id of the source the best span came from, or None.
        best_span: the source text that matched best, or "".
        reason: one plain sentence saying why the verdict came out this way.
    """

    text: str
    supported: bool
    confidence: float
    best_source: Optional[str]
    best_span: str
    reason: str

    def to_dict(self) -> Dict[str, Any]:
        """JSON-safe dict of this claim."""
        return {
            "text": self.text,
            "supported": self.supported,
            "confidence": self.confidence,
            "best_source": self.best_source,
            "best_span": self.best_span,
            "reason": self.reason,
        }


class GroundingReport:
    """What the sources do and do not support, plus why."""

    def __init__(
        self,
        claims: List[Claim],
        *,
        threshold: float,
        granularity: str,
        source_ids: Sequence[str],
        contradiction_indices: Sequence[int] = (),
        warnings: Sequence[str] = (),
    ) -> None:
        self.claims: List[Claim] = list(claims)
        self.threshold = float(threshold)
        self.granularity = granularity
        self.source_ids: List[str] = list(source_ids)
        self.warnings: List[str] = list(warnings)
        self._contradiction_indices: List[int] = sorted(set(int(i) for i in contradiction_indices))

    # -- headline numbers -------------------------------------------------
    @property
    def n_claims(self) -> int:
        """How many units the answer was split into."""
        return len(self.claims)

    @property
    def n_supported(self) -> int:
        """How many of those the sources back up."""
        return sum(1 for c in self.claims if c.supported)

    @property
    def score(self) -> float:
        """0-100: the share of the answer's claims the sources support."""
        if not self.claims:
            return 100.0
        return round(100.0 * self.n_supported / len(self.claims), 1)

    @property
    def unsupported(self) -> List[Claim]:
        """The claims the sources do not back up."""
        return [c for c in self.claims if not c.supported]

    @property
    def contradictions(self) -> List[Claim]:
        """Unsupported claims where a source states a different number or date."""
        return [self.claims[i] for i in self._contradiction_indices]

    @property
    def citations(self) -> Dict[int, str]:
        """Claim index -> id of the source that supports it."""
        return {
            i: c.best_source
            for i, c in enumerate(self.claims)
            if c.supported and c.best_source is not None
        }

    # -- output -----------------------------------------------------------
    def summary(self) -> str:
        """Human-readable report. ASCII punctuation only, safe on any console."""
        lines: List[str] = []
        lines.append(
            "hallucination-check: %.1f / 100 grounded (%d of %d claims supported)"
            % (self.score, self.n_supported, self.n_claims)
        )
        lines.append(
            "  sources: %d   threshold: %.2f   granularity: %s"
            % (len(self.source_ids), self.threshold, self.granularity)
        )
        unsupported = [(i, c) for i, c in enumerate(self.claims) if not c.supported]
        contradicted = set(self._contradiction_indices)
        if not self.claims:
            lines.append("  the answer is empty, so there is nothing to check.")
        elif not unsupported:
            lines.append("  every claim is backed by a source.")
        else:
            lines.append("  not supported (%d):" % len(unsupported))
            for i, claim in unsupported:
                mark = " [contradiction]" if i in contradicted else ""
                lines.append("    [%d] score %.2f%s  %s" % (i, claim.confidence, mark, truncate(claim.text)))
                lines.append("         %s" % claim.reason)
        if self.warnings:
            lines.append("  warnings:")
            for warning in self.warnings:
                lines.append("    - %s" % warning)
        lines.append(
            "  note: lexical grounding is a signal, not proof. Pass embed= to add semantic matching."
        )
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """JSON-safe dict of the whole report."""
        return {
            "score": self.score,
            "n_claims": self.n_claims,
            "n_supported": self.n_supported,
            "n_unsupported": self.n_claims - self.n_supported,
            "n_contradictions": len(self._contradiction_indices),
            "threshold": self.threshold,
            "granularity": self.granularity,
            "source_ids": list(self.source_ids),
            "warnings": list(self.warnings),
            "claims": [c.to_dict() for c in self.claims],
            "unsupported": [i for i, c in enumerate(self.claims) if not c.supported],
            "contradictions": list(self._contradiction_indices),
            "citations": {str(k): v for k, v in self.citations.items()},
        }

    def __repr__(self) -> str:
        return "GroundingReport(score=%.1f, claims=%d, unsupported=%d)" % (
            self.score,
            self.n_claims,
            self.n_claims - self.n_supported,
        )


def _prepare_sources(sources: Any) -> Tuple[List[Tuple[str, str]], List[str]]:
    """Turn whatever the caller passed into ``[(id, text), ...]`` plus any warnings."""
    warnings: List[str] = []
    if sources is None:
        return [], warnings
    items: List[Any]
    if isinstance(sources, (str, bytes)):
        items = [sources]
    elif isinstance(sources, dict):
        items = [sources]
    elif isinstance(sources, Iterable):
        items = list(sources)
    else:
        raise TypeError(
            "sources must be a string, a list of strings, or a list of dicts with a "
            "'text' key, got %s" % type(sources).__name__
        )

    prepared: List[Tuple[str, str]] = []
    empty = 0
    for position, item in enumerate(items):
        if isinstance(item, bytes):
            raise TypeError("source %d is bytes; decode it to str first" % position)
        if isinstance(item, str):
            sid, text = "s%d" % (position + 1), item
        elif isinstance(item, dict):
            if "text" not in item:
                raise ValueError("source %d is a dict without a 'text' key" % position)
            raw = item["text"]
            if not isinstance(raw, str):
                raise TypeError("source %d has a non-string 'text'" % position)
            sid = str(item.get("id") or "s%d" % (position + 1))
            text = raw
        else:
            raise TypeError(
                "source %d must be a str or a dict with a 'text' key, got %s"
                % (position, type(item).__name__)
            )
        if not text.strip():
            empty += 1
            continue
        prepared.append((sid, text))
    if empty:
        warnings.append("%d source(s) held no text and were skipped" % empty)
    return prepared, warnings


def _rival_rank(target: Fact, rival: Fact) -> Tuple[int, int, float, int]:
    """How good a stand-in ``rival`` is for the flagged fact ``target``.

    Lower sorts better. A span often holds several numbers -- "Apollo 11 stood 110
    metres tall" holds two -- and naming the first one by document position tells the
    reader the source says something it does not. The one that sits in the same place
    in the sentence, and failing that the one nearest in magnitude, is the figure the
    answer actually contradicts.
    """
    same_kind = 0 if rival.kind == target.kind else 1
    matches = 0
    if target.after and rival.after == target.after:
        matches += 1
    if target.before and rival.before == target.before:
        matches += 1
    distance = 0.0
    if target.value is not None and rival.value is not None:
        try:
            distance = abs(float(rival.value) - float(target.value))
        except (ArithmeticError, ValueError, OverflowError):  # pragma: no cover
            distance = 0.0
    return (same_kind, -matches, distance, rival.start)


def _best_rival(wrong: Sequence[Fact], rivals: Sequence[Fact]) -> Tuple[Fact, Fact]:
    """The flagged fact and the source figure that best rivals it."""
    best: Optional[Tuple[Tuple[Any, ...], Fact, Fact]] = None
    for target in wrong:
        for rival in rivals:
            rank = _rival_rank(target, rival) + (target.start,)
            if best is None or rank < best[0]:
                best = (rank, target, rival)
    assert best is not None  # wrong and rivals are both non-empty
    return best[1], best[2]


def _span_score(index: SourceIndex, claim: TextVec, span: Span) -> float:
    """Lexical score of ``claim`` against ``span``, length difference allowed for.

    A long span is also compared with the window of itself that best fits the claim, and
    the better of the two scores wins, so a faithful paraphrase of one clause is not
    marked down for the rest of the sentence around it.
    """
    score = lexical_score(claim, index.span_vec(span))
    window = index.window_vec(span, claim)
    if window is not None:
        score = max(score, lexical_score(claim, window))
    return score


def _as_matrix(vectors: Any, expected: int) -> Optional[np.ndarray]:
    """Validate what an ``embed`` callable returned."""
    array = np.asarray(vectors, dtype=np.float64)
    if array.ndim != 2 or array.shape[0] != expected or array.shape[1] == 0:
        return None
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return array / norms


class GroundingChecker:
    """Check answers against their sources. The knobs :func:`check` does not expose.

    Args:
        threshold: support score in 0-1 at or above which a claim counts as supported.
        granularity: "sentence", "clause" or "paragraph".
        embed: optional ``embed(texts) -> (n, d) array`` used for semantic matching.
        embed_weight: how much of the final score comes from ``embed`` (0-1).
        flag_missing_facts: flag a claim whose number, date or name is absent from
            the sources even when its overall similarity is high.
        char_n: character n-gram length.
        window: how many consecutive source sentences may form one span.
        top_k: how many source sentences are shortlisted per claim.
        embed_top: how many extra spans per claim the embedding ranking contributes.
        embed_max_sentences: above this many source sentences, ``embed`` only sees the
            lexically shortlisted ones instead of all of them.
    """

    def __init__(
        self,
        *,
        threshold: float = 0.55,
        granularity: str = "sentence",
        embed: Optional[EmbedFn] = None,
        embed_weight: float = 0.5,
        flag_missing_facts: bool = True,
        char_n: int = 4,
        window: int = 3,
        top_k: int = 25,
        embed_top: int = 5,
        embed_max_sentences: int = 2000,
    ) -> None:
        if not isinstance(threshold, (int, float)) or isinstance(threshold, bool):
            raise TypeError("threshold must be a number between 0 and 1")
        if not 0.0 <= float(threshold) <= 1.0:
            raise ValueError("threshold must be between 0 and 1, got %r" % (threshold,))
        if granularity not in GRANULARITIES:
            raise ValueError(
                "granularity must be one of %s, got %r" % (", ".join(GRANULARITIES), granularity)
            )
        if embed is not None and not callable(embed):
            raise TypeError("embed must be a callable taking a list of strings")
        if not 0.0 <= float(embed_weight) <= 1.0:
            raise ValueError("embed_weight must be between 0 and 1, got %r" % (embed_weight,))
        self.threshold = float(threshold)
        self.granularity = granularity
        self.embed = embed
        self.embed_weight = float(embed_weight)
        self.flag_missing_facts = bool(flag_missing_facts)
        self.char_n = int(char_n)
        self.window = int(window)
        self.top_k = int(top_k)
        self.embed_top = max(1, int(embed_top))
        self.embed_max_sentences = max(1, int(embed_max_sentences))

    # -- helpers ----------------------------------------------------------
    def _units(self, answer: str, warnings: List[str]) -> List[str]:
        units = split_units(answer, self.granularity)
        if not units and answer.strip():
            warnings.append(
                "no %s boundary was found; the whole answer was treated as one claim"
                % self.granularity
            )
            units = [" ".join(answer.split())]
        return units

    def _embed_matrix(
        self, texts: Sequence[str], warnings: List[str]
    ) -> Optional[np.ndarray]:
        if self.embed is None or not texts:
            return None
        try:
            raw = self.embed(list(texts))
        except Exception as exc:  # noqa: BLE001 - never let a user hook break the check
            logger.warning("embed callable raised %s; falling back to lexical only", exc)
            warnings.append("the embed callable raised %s; semantic matching was skipped" % exc)
            return None
        try:
            matrix = _as_matrix(raw, len(texts))
        except (TypeError, ValueError) as exc:
            logger.warning("embed callable returned something unusable: %s", exc)
            matrix = None
        if matrix is None:
            warnings.append(
                "the embed callable did not return an (n, d) array of %d vectors; "
                "semantic matching was skipped" % len(texts)
            )
        return matrix

    def _semantic(
        self,
        index: SourceIndex,
        units: Sequence[str],
        shortlists: Sequence[Sequence[Span]],
        warnings: List[str],
    ) -> Tuple[Optional[np.ndarray], Dict[int, int]]:
        """Embed the claims and the source sentences and return the similarity matrix.

        Every source sentence is embedded when there are not too many of them, so the
        hook can find spans that share no words at all with the claim. Past the cap only
        the lexically shortlisted sentences are embedded.
        """
        if self.embed is None or index.n_sentences == 0:
            return None, {}
        if index.n_sentences <= self.embed_max_sentences:
            selected = list(range(index.n_sentences))
        else:
            selected = sorted(
                {sid for shortlist in shortlists for span in shortlist
                 for sid in range(span.start, span.end + 1)}
            )
            warnings.append(
                "the sources hold %d sentences, over the embed_max_sentences limit of %d; "
                "only the %d lexically shortlisted sentences were embedded"
                % (index.n_sentences, self.embed_max_sentences, len(selected))
            )
        if not selected:
            return None, {}
        texts = list(units) + [index.sentences[sid] for sid in selected]
        matrix = self._embed_matrix(texts, warnings)
        if matrix is None:
            return None, {}
        sims = matrix[: len(units)] @ matrix[len(units) :].T
        return np.clip(sims, 0.0, 1.0), {sid: column for column, sid in enumerate(selected)}

    def _contradiction(
        self,
        claim_text: str,
        missing: Sequence[Fact],
        span_text: str,
        window_text: Optional[str] = None,
    ) -> Optional[str]:
        """Does the best span state a different number or date?"""
        wrong = [f for f in missing if f.kind in ("number", "date")]
        if not wrong or not span_text:
            return None
        span_facts = [
            f
            for f in extract_facts(span_text, for_source=True)
            if f.kind in ("number", "date") and f.flaggable
        ]
        # Only a fact's own identity keys disqualify it as a rival. Its decomposed parts
        # must not: a date contributes num:<year>, month:<m> and num:<day>, so any date
        # sharing a year -- extremely common in passages about the same period -- would
        # otherwise look like it was already in the claim and no contradiction would be
        # reported.
        claim_keys = set()
        for fact in extract_facts(claim_text, for_source=True):
            claim_keys.update(fact.keys)
        rivals = [f for f in span_facts if not any(k in claim_keys for k in f.keys)]
        if not rivals:
            return None
        masked_claim = vectorize(mask_numbers(claim_text), char_n=self.char_n)
        masked_span = vectorize(mask_numbers(span_text), char_n=self.char_n)
        similarity = lexical_score(masked_claim, masked_span)
        if window_text:
            masked_window = vectorize(mask_numbers(window_text), char_n=self.char_n)
            similarity = max(similarity, lexical_score(masked_claim, masked_window))
        if similarity < max(0.5, self.threshold):
            return None
        target, rival = _best_rival(wrong, rivals)
        return "the sources say %s where the answer says %s" % (
            truncate(rival.text, 40),
            truncate(target.text, 40),
        )

    # -- the work ---------------------------------------------------------
    def check(self, answer: Any, sources: Any) -> GroundingReport:
        """Check ``answer`` against ``sources`` and return a :class:`GroundingReport`."""
        if answer is None:
            answer = ""
        if not isinstance(answer, str):
            raise TypeError("answer must be a string, got %s" % type(answer).__name__)

        prepared, warnings = _prepare_sources(sources)
        units = self._units(answer, warnings)
        source_ids = [sid for sid, _t in prepared]

        if not units:
            return GroundingReport(
                [],
                threshold=self.threshold,
                granularity=self.granularity,
                source_ids=source_ids,
                warnings=warnings,
            )

        if not prepared:
            warnings.append("no sources were given, so nothing in the answer can be supported")
            claims = [
                Claim(text=unit, supported=False, confidence=0.0, best_source=None,
                      best_span="", reason="no sources were given")
                for unit in units
            ]
            return GroundingReport(
                claims,
                threshold=self.threshold,
                granularity=self.granularity,
                source_ids=source_ids,
                warnings=warnings,
            )

        index = SourceIndex(prepared, char_n=self.char_n, top_k=self.top_k, window=self.window)
        source_keys: set = set()
        for _sid, text in prepared:
            source_keys |= collect_keys(text)

        claim_vecs: List[TextVec] = []
        claim_facts: List[List[Fact]] = []
        shortlists: List[List[Span]] = []
        for unit in units:
            claim_facts.append(extract_facts(unit))
            vec = vectorize(unit, char_n=self.char_n, extra_tokens=number_tokens(unit))
            claim_vecs.append(vec)
            scored = [(span, _span_score(index, vec, span)) for span in index.candidates(vec)]
            scored.sort(key=lambda pair: (-pair[1], pair[0].start))
            shortlists.append([span for span, _s in scored[: max(self.embed_top, 1)]])

        sem, sem_column = self._semantic(index, units, shortlists, warnings)

        if not any(claim_vecs[i].keys or claim_facts[i] for i in range(len(units))):
            warnings.append(
                "no unit of the answer carries a word, number or name that could be "
                "checked against a source"
            )

        claims: List[Claim] = []
        contradictions: List[int] = []
        for position, unit in enumerate(units):
            vec = claim_vecs[position]
            facts = claim_facts[position]
            candidates = list(shortlists[position])
            if sem is not None:
                seeds = [
                    sid
                    for sid in sorted(sem_column, key=lambda s: -sem[position, sem_column[s]])[
                        : self.embed_top
                    ]
                ]
                known = {span.key for span in candidates}
                candidates += [s for s in index.windows(seeds) if s.key not in known]
            best_span: Optional[Span] = None
            best_score = 0.0
            for span in candidates:
                score = _span_score(index, vec, span)
                if sem is not None:
                    columns = [
                        sem_column[sid]
                        for sid in range(span.start, span.end + 1)
                        if sid in sem_column
                    ]
                    if columns:
                        semantic = max(float(sem[position, c]) for c in columns)
                        score = (1.0 - self.embed_weight) * score + self.embed_weight * semantic
                if best_span is None or score > best_score:
                    best_span, best_score = span, score
            best_score = float(max(0.0, min(1.0, best_score)))
            span_text = index.span_text(best_span) if best_span is not None else ""
            source_id = index.source_id(best_span) if best_span is not None else None

            window_text = index.window_text(best_span, vec) if best_span is not None else None
            missing = [f for f in facts if not f.is_in(source_keys)]
            clash = (
                self._contradiction(unit, missing, span_text, window_text) if missing else None
            )

            if missing and self.flag_missing_facts:
                supported = False
                if clash:
                    reason = clash
                    contradictions.append(position)
                else:
                    reason = "these appear nowhere in the sources: %s" % ", ".join(
                        dict.fromkeys(truncate(f.text, 40) for f in missing)
                    )
            elif not vec.keys and not facts:
                supported = True
                reason = "no factual content to check"
            elif best_score >= self.threshold:
                supported = True
                reason = "supported by %s (score %.2f)" % (source_id, best_score)
            else:
                supported = False
                if best_span is None:
                    reason = "no source mentions any of this"
                else:
                    reason = "the closest source span only scores %.2f, below the %.2f threshold" % (
                        best_score,
                        self.threshold,
                    )
            claims.append(
                Claim(
                    text=unit,
                    supported=supported,
                    confidence=round(best_score, 3),
                    best_source=source_id,
                    best_span=span_text,
                    reason=reason,
                )
            )

        return GroundingReport(
            claims,
            threshold=self.threshold,
            granularity=self.granularity,
            source_ids=source_ids,
            contradiction_indices=contradictions,
            warnings=warnings,
        )

    def check_batch(self, pairs: Iterable[Any]) -> List[GroundingReport]:
        """Run :meth:`check` over ``(answer, sources)`` pairs."""
        reports: List[GroundingReport] = []
        for position, pair in enumerate(pairs):
            if isinstance(pair, dict):
                if "answer" not in pair or "sources" not in pair:
                    raise ValueError(
                        "pair %d must be a dict with 'answer' and 'sources' keys" % position
                    )
                answer, sources = pair["answer"], pair["sources"]
            elif isinstance(pair, (tuple, list)) and len(pair) == 2:
                answer, sources = pair[0], pair[1]
            else:
                raise ValueError(
                    "pair %d must be an (answer, sources) tuple or a dict with those keys"
                    % position
                )
            reports.append(self.check(answer, sources))
        return reports


def check(
    answer: str,
    sources: Any,
    *,
    threshold: float = 0.55,
    granularity: str = "sentence",
    embed: Optional[EmbedFn] = None,
) -> GroundingReport:
    """Check an answer against the sources it claims to use.

    Args:
        answer: the generated text.
        sources: a string, a list of strings, or a list of dicts with a "text" key and
            an optional "id", so retrieved passages can be passed straight through.
        threshold: support score in 0-1 at or above which a claim counts as supported.
        granularity: "sentence" (default), "clause" or "paragraph".
        embed: optional ``embed(texts) -> (n, d) array``; when given, semantic
            similarity is blended into the score at equal weight.

    Returns:
        A :class:`GroundingReport`. ``report.score`` is 0-100, ``report.unsupported``
        lists the claims the sources do not back up.
    """
    return GroundingChecker(
        threshold=threshold, granularity=granularity, embed=embed
    ).check(answer, sources)


def check_batch(pairs: Iterable[Any], **kwargs: Any) -> List[GroundingReport]:
    """Check many ``(answer, sources)`` pairs with one set of options.

    Args:
        pairs: an iterable of ``(answer, sources)`` tuples, or of dicts with
            "answer" and "sources" keys.
        **kwargs: passed straight to :func:`check`.

    Returns:
        One :class:`GroundingReport` per pair, in order.
    """
    return GroundingChecker(**kwargs).check_batch(pairs)
