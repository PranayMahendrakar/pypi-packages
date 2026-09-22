"""The :class:`Labeler`, its :class:`LabelResult`, and the one-line :func:`label` helper."""

from __future__ import annotations

import logging
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Union

import numpy as np
import pandas as pd

from auto_label import models
from auto_label._data import TABULAR, TEXT, Prepared, json_safe, prepare
from auto_label.rules import Rule, make_rule, match_rule

logger = logging.getLogger(__name__)

LLM = Callable[[str, List[str]], Optional[str]]
SOURCES = ("rule", "model", "llm")


@dataclass
class LabelResult:
    """Labels for every item, with a confidence and where each label came from.

    ``labels[i]`` is ``None`` for an unlabeled item; then ``confidence[i] == 0.0`` and
    ``source[i] is None``. ``source`` is one of ``"rule"``, ``"model"``, ``"llm"``.
    """

    items: List[Any]
    labels: List[Optional[str]]
    confidence: List[float]
    source: List[Optional[str]]
    index: List[Any] = field(default_factory=list)
    candidates: List[str] = field(default_factory=list)
    mode: str = TEXT
    model_trained: bool = False
    notes: List[str] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.labels)

    @property
    def coverage(self) -> float:
        """Fraction of items that received a label (0.0 for an empty input)."""
        if not self.labels:
            return 0.0
        return sum(lab is not None for lab in self.labels) / len(self.labels)

    @property
    def counts(self) -> Dict[str, int]:
        """Number of items per assigned label, most common first."""
        return dict(Counter(lab for lab in self.labels if lab is not None).most_common())

    @property
    def source_counts(self) -> Dict[str, int]:
        """Number of items per source, plus ``"unlabeled"``."""
        c = Counter(self.source)
        out = {s: c.get(s, 0) for s in SOURCES}
        out["unlabeled"] = c.get(None, 0)
        return out

    def to_frame(self) -> pd.DataFrame:
        """DataFrame with columns ``item, label, confidence, source``, on the index of the input."""
        index = pd.Index(self.index) if len(self.index) == len(self.labels) else None
        return pd.DataFrame(
            {
                "item": pd.Series(self.items, index=index, dtype=object),
                "label": pd.Series(self.labels, index=index, dtype=object),
                "confidence": pd.Series(self.confidence, index=index, dtype="float64"),
                "source": pd.Series(self.source, index=index, dtype=object),
            }
        )

    def to_dict(self) -> Dict[str, Any]:
        """JSON-safe dict: totals, per-label and per-source counts, notes, and every record."""
        index = self.index if len(self.index) == len(self.labels) else list(range(len(self.labels)))
        records = [
            {
                "index": json_safe(idx),
                "item": json_safe(item),
                "label": lab,
                "confidence": round(float(conf), 6),
                "source": src,
            }
            for idx, item, lab, conf, src in zip(index, self.items, self.labels, self.confidence, self.source)
        ]
        return {
            "n_items": len(self.labels),
            "n_labeled": sum(lab is not None for lab in self.labels),
            "coverage": round(self.coverage, 6),
            "mode": self.mode,
            "candidates": list(self.candidates),
            "model_trained": self.model_trained,
            "labels": self.counts,
            "sources": self.source_counts,
            "notes": list(self.notes),
            "records": records,
        }

    def summary(self) -> str:
        """A few human-readable lines: coverage, sources, labels, and what the model did."""
        n = len(self.labels)
        labeled = sum(lab is not None for lab in self.labels)
        lines = [f"auto-label: {labeled}/{n} items labeled ({self.coverage * 100:.1f}%)"]
        if n:
            lines.append("  by source: " + ", ".join(f"{k} {v}" for k, v in self.source_counts.items()))
            if self.counts:
                lines.append("  by label:  " + ", ".join(f"{k} {v}" for k, v in self.counts.items()))
        for note in self.notes:
            lines.append(f"  {note}")
        return "\n".join(lines)


class Labeler:
    """Label items from rules, then a small model on the rest, then an optional LLM.

    Parameters
    ----------
    labels:
        Optional closed set of allowed labels. Rules must use one of them and the LLM
        hook is held to the set. ``None`` means "whatever the rules (and LLM) say".
    min_confidence:
        Items whose label is below this confidence are offered to the LLM hook; model
        predictions below it are left unlabeled when no hook is given.
    random_state:
        Seed passed to the model so results are reproducible.
    """

    def __init__(
        self,
        labels: Optional[Iterable[str]] = None,
        min_confidence: float = 0.6,
        random_state: int = 0,
    ) -> None:
        if isinstance(labels, str):
            labels = [labels]
        self.labels: Optional[List[str]] = None
        if labels is not None:
            cleaned = [str(lab) for lab in labels if str(lab).strip()]
            if not cleaned:
                raise ValueError("labels must contain at least one non-empty label")
            self.labels = list(dict.fromkeys(cleaned))
        if isinstance(min_confidence, bool) or not isinstance(min_confidence, (int, float)):
            raise ValueError("min_confidence must be a number between 0 and 1")
        if not 0.0 <= float(min_confidence) <= 1.0:
            raise ValueError(f"min_confidence must be between 0 and 1, got {min_confidence}")
        self.min_confidence = float(min_confidence)
        if isinstance(random_state, bool) or not isinstance(random_state, (int, np.integer)):
            raise ValueError(f"random_state must be an integer, got {random_state!r}")
        self.random_state = int(random_state)
        self.rules: List[Rule] = []
        self.model_: Any = None

    def __repr__(self) -> str:
        return (
            f"Labeler(labels={self.labels!r}, min_confidence={self.min_confidence}, "
            f"random_state={self.random_state}, rules={len(self.rules)})"
        )

    # rules -----------------------------------------------------------------

    def add_rule(
        self,
        label: str,
        *,
        keywords: Union[None, str, Iterable[str]] = None,
        regex: Optional[Union[str, "re.Pattern[str]"]] = None,
        query: Optional[str] = None,
        func: Optional[Callable[[Any], bool]] = None,
        weight: float = 1.0,
    ) -> "Labeler":
        """Add one rule for ``label``; returns ``self`` so calls can be chained.

        A rule fires when every condition given here matches. ``keywords`` are
        case-insensitive whole words / phrases; ``regex`` is searched as written;
        ``query`` is a pandas query on a DataFrame; ``func(item)`` gets the text or the
        row as a Series, named with that row's label in the caller's index. Rules that
        fire on the same item are combined by summed ``weight``.
        """
        rule = make_rule(label, keywords=keywords, regex=regex, query=query, func=func, weight=weight)
        if self.labels is not None and rule.label not in self.labels:
            raise ValueError(f"rule label {rule.label!r} is not in labels={self.labels!r}")
        self.rules.append(rule)
        return self

    @property
    def candidates(self) -> List[str]:
        """Labels the LLM hook may choose from: ``labels`` if given, else those the rules use."""
        if self.labels is not None:
            return list(self.labels)
        return list(dict.fromkeys(rule.label for rule in self.rules))

    # labelling -------------------------------------------------------------

    def label(self, data: Any, *, model: bool = True, llm: Optional[LLM] = None) -> LabelResult:
        """Label ``data``: rules first, then the model on the rest, then the LLM hook.

        ``data`` is a list of strings or a Series (text), or a DataFrame, a dict of
        columns, or a ``.csv`` / ``.parquet`` path (tabular). ``llm(text, candidate_labels)``
        should return a label or ``None``; if it raises, the error is logged and the item
        is treated as unanswered.
        """
        if llm is not None and not callable(llm):
            raise TypeError(
                "llm must be a callable llm(text, candidate_labels) -> label | None, got "
                f"{type(llm).__name__}"
            )
        prepared = prepare(data)
        n = len(prepared)
        candidates = self.candidates
        labels: List[Optional[str]] = [None] * n
        confidence: List[float] = [0.0] * n
        source: List[Optional[str]] = [None] * n
        notes: List[str] = []
        self.model_ = None

        if n == 0:
            return LabelResult(
                [], [], [], [], index=[], candidates=candidates, mode=prepared.mode, notes=["empty input"]
            )
        if not self.rules:
            logger.warning("Labeler.label called with no rules; nothing can be labeled without them")
        if prepared.text_fallback and any(r.keywords or r.regex is not None for r in self.rules):
            message = (
                "no text-like column found; keyword and regex rules were matched against "
                f"every column rendered as text ({', '.join(prepared.text_columns)})"
            )
            logger.warning("%s", message)
            notes.append(f"rules: {message}")

        self._apply_rules(prepared, labels, confidence, source)
        n_rule = sum(src == "rule" for src in source)
        notes.append(f"rules: {len(self.rules)} rule(s) labeled {n_rule} item(s)")

        trained = False
        if model:
            trained = self._apply_model(prepared, labels, confidence, source, notes)
        else:
            notes.append("model: disabled (model=False)")

        if llm is not None:
            self._apply_llm(llm, prepared, candidates, labels, confidence, source, notes)

        return LabelResult(
            items=list(prepared.items),
            labels=labels,
            confidence=confidence,
            source=source,
            index=list(prepared.index),
            candidates=candidates,
            mode=prepared.mode,
            model_trained=trained,
            notes=notes,
        )

    def _apply_rules(
        self,
        prepared: Prepared,
        labels: List[Optional[str]],
        confidence: List[float],
        source: List[Optional[str]],
    ) -> None:
        if not self.rules:
            return
        masks = [match_rule(rule, prepared) for rule in self.rules]
        hit_any = np.zeros(len(prepared), dtype=bool)
        for m in masks:
            hit_any |= m
        for i in np.flatnonzero(hit_any):
            totals: Dict[str, float] = {}
            first_seen: Dict[str, int] = {}
            for r, rule in enumerate(self.rules):
                if masks[r][i]:
                    totals[rule.label] = totals.get(rule.label, 0.0) + rule.weight
                    first_seen.setdefault(rule.label, r)
            positive = {lab: w for lab, w in totals.items() if w > 0}
            if not positive:
                continue
            best = max(positive.values())
            # Summed weights accumulate rounding (0.1 + 0.2 != 0.3), so an exact ==
            # comparison would hand the win to whichever label picked up a few parts in
            # 1e16 more noise and silently skip the documented "ties go to the rule added
            # first". Compare the totals as the user wrote them instead.
            tied = [
                lab for lab, w in positive.items()
                if math.isclose(w, best, rel_tol=1e-9, abs_tol=1e-12)
            ]
            winner = min(tied, key=first_seen.__getitem__)
            labels[i] = winner
            confidence[i] = best / sum(positive.values())
            source[i] = "rule"

    def _apply_model(
        self,
        prepared: Prepared,
        labels: List[Optional[str]],
        confidence: List[float],
        source: List[Optional[str]],
        notes: List[str],
    ) -> bool:
        train_idx = [i for i, lab in enumerate(labels) if lab is not None]
        predict_idx = [i for i, lab in enumerate(labels) if lab is None]
        n_classes = len(set(labels[i] for i in train_idx))
        if not predict_idx:
            notes.append("model: skipped, every item was labeled by rules")
            return False
        if len(train_idx) < 5:
            notes.append(f"model: skipped, needs at least 5 rule-labeled examples (have {len(train_idx)})")
            return False
        if n_classes < 2:
            notes.append(f"model: skipped, needs at least 2 labels among rule-labeled examples (have {n_classes})")
            return False
        try:
            classes, proba, pipeline, model_notes = models.fit_predict(
                prepared, train_idx, [str(labels[i]) for i in train_idx], predict_idx, self.random_state
            )
        except Exception as exc:  # noqa: BLE001 - degrade gracefully, keep the rule labels
            logger.warning("model training failed, keeping rule labels only: %s", exc)
            notes.append(f"model: skipped, training failed ({exc})")
            return False
        self.model_ = pipeline
        kept = 0
        for row, i in enumerate(predict_idx):
            k = int(np.argmax(proba[row]))
            p = float(proba[row, k])
            if p >= self.min_confidence:
                labels[i] = classes[k]
                confidence[i] = p
                source[i] = "model"
                kept += 1
        notes.append(
            f"model: trained on {len(train_idx)} rule-labeled examples, predicted {len(predict_idx)}, "
            f"kept {kept} at confidence >= {self.min_confidence:g}"
        )
        notes.extend(model_notes)
        if kept == 0:
            # The model generalised to none of the unlabeled items. Say so rather than
            # leaving the user with a bare "kept 0" (CONVENTIONS: an auto heuristic that
            # degenerates must name the problem). Both modes need this: the tabular advice
            # points at --column, and --column is the text branch.
            if prepared.mode == TABULAR:
                message = (
                    "model: no prediction reached the confidence threshold from this table's "
                    "columns; if one column holds the text you are labelling, pass it directly "
                    '(df["text"], or --column on the CLI) so the text model is used'
                )
            else:
                message = (
                    f"model: no prediction reached min_confidence={self.min_confidence:g}, so "
                    "the model added nothing; add more rule-labeled examples, or lower the "
                    "threshold (min_confidence, --min-confidence on the CLI)"
                )
            logger.warning("%s", message)
            notes.append(message)
        return True

    def _apply_llm(
        self,
        llm: LLM,
        prepared: Prepared,
        candidates: List[str],
        labels: List[Optional[str]],
        confidence: List[float],
        source: List[Optional[str]],
        notes: List[str],
    ) -> None:
        asked = answered = failed = 0
        lowered = {c.lower(): c for c in candidates}
        seen_errors: set = set()  # warn once per distinct failure, not once per item
        for i in range(len(prepared)):
            if labels[i] is not None and confidence[i] >= self.min_confidence:
                continue
            asked += 1
            try:
                answer = llm(prepared.llm_texts[i], list(candidates))
            except Exception as exc:  # noqa: BLE001 - a flaky hook must not sink the run
                failed += 1
                key = (type(exc).__name__, str(exc))
                if key not in seen_errors:
                    seen_errors.add(key)
                    logger.warning(
                        "llm hook raised for item %s; treating as no answer (further identical "
                        "failures are counted, not logged): %s: %s",
                        i, key[0], key[1],
                    )
                continue
            chosen = self._accept_llm_answer(answer, lowered)
            if chosen is None:
                continue
            labels[i] = chosen
            confidence[i] = 1.0
            source[i] = "llm"
            answered += 1
        notes.append(f"llm: asked {asked}, labeled {answered}, failed {failed}")

    def _accept_llm_answer(self, answer: Any, lowered: Dict[str, str]) -> Optional[str]:
        if answer is None:
            return None
        if not isinstance(answer, str):
            logger.debug("llm hook returned %r (not a string); ignoring", answer)
            return None
        text = answer.strip()
        if not text:
            return None
        canonical = lowered.get(text.lower())
        if canonical is not None:
            return canonical
        if self.labels is not None:
            logger.debug("llm hook returned %r, not in labels %r; ignoring", text, self.labels)
            return None
        return text


RuleSpec = Union[str, Callable[[Any], bool], Sequence[str], Mapping[str, Any], Sequence[Mapping[str, Any]]]


def _rule_kwargs(label: str, spec: RuleSpec) -> List[Dict[str, Any]]:
    """Expand one ``rules`` dict value into ``add_rule`` keyword-argument dicts."""
    if isinstance(spec, str):
        return [{"regex": spec}]
    if callable(spec):
        return [{"func": spec}]
    if isinstance(spec, Mapping):
        return [dict(spec)]
    if isinstance(spec, (list, tuple, set, frozenset)):
        items = list(spec)
        if items and all(isinstance(it, Mapping) for it in items):
            return [dict(it) for it in items]
        return [{"keywords": items}]
    raise TypeError(
        f"rules[{label!r}] must be a list of keywords, a regex string, a callable, "
        f"or a dict of add_rule keyword arguments; got {type(spec).__name__}"
    )


def label(
    data: Any,
    rules: Mapping[str, RuleSpec],
    *,
    labels: Optional[Iterable[str]] = None,
    min_confidence: float = 0.6,
    random_state: int = 0,
    model: bool = True,
    llm: Optional[LLM] = None,
) -> LabelResult:
    """One-line labelling: ``label(data, {"spam": ["free", "prize"], "work": r"(?i)meeting"})``.

    ``rules`` maps a label to a list of keywords, a regex string, a callable, a dict of
    :meth:`Labeler.add_rule` keyword arguments, or a list of such dicts. The other
    arguments are forwarded to :class:`Labeler` and :meth:`Labeler.label`.
    """
    if not isinstance(rules, Mapping):
        raise TypeError("rules must be a dict mapping label -> keywords | regex | dict | callable")
    labeler = Labeler(labels=labels, min_confidence=min_confidence, random_state=random_state)
    for name, spec in rules.items():
        for kwargs in _rule_kwargs(name, spec):
            labeler.add_rule(name, **kwargs)
    return labeler.label(data, model=model, llm=llm)
