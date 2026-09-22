"""Result objects: :class:`Finding` and :class:`FeatureReport`."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Hashable, Iterator, List, Optional, Sequence, Tuple

import pandas as pd

from ._io import FrameLike, load_frame

log = logging.getLogger(__name__)

#: severities from worst to mildest
SEVERITIES: Tuple[str, ...] = ("high", "medium", "low")
_RANK = {s: i for i, s in enumerate(SEVERITIES)}
#: findings at these severities put a column on ``drop_recommended``
DROP_SEVERITIES = frozenset({"high", "medium"})


@dataclass
class Finding:
    """One problem found on one column.

    ``kind`` is one of :data:`ml_feature_check.KINDS`, ``severity`` is ``"high"``,
    ``"medium"`` or ``"low"``, ``message`` is human text and ``detail`` holds the
    JSON-safe numbers behind it.
    """

    kind: str
    severity: str
    message: str
    detail: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """JSON-safe dictionary."""
        return {
            "kind": self.kind,
            "severity": self.severity,
            "message": self.message,
            "detail": dict(self.detail),
        }


def sort_findings(findings: List[Finding]) -> List[Finding]:
    """Most severe first, otherwise keep the order the checks ran in."""
    return sorted(findings, key=lambda f: _RANK.get(f.severity, len(SEVERITIES)))


def rank_columns(
    features: Dict[Hashable, List[Finding]], order: Sequence[Hashable]
) -> Tuple[List[Hashable], List[Hashable]]:
    """Split columns into ``(drop_recommended, keep)``; drop is ordered by severity."""
    pos = {c: i for i, c in enumerate(order)}

    def worst(col: Hashable) -> int:
        return min(_RANK[f.severity] for f in features[col] if f.severity in DROP_SEVERITIES)

    drop = [c for c in order if any(f.severity in DROP_SEVERITIES for f in features.get(c, []))]
    drop.sort(key=lambda c: (worst(c), pos[c]))
    dropped = set(drop)
    keep = [c for c in order if c not in dropped]
    return drop, keep


@dataclass
class FeatureReport:
    """What :func:`ml_feature_check.check` found.

    ``features`` maps every feature column (never the target) to its findings,
    an empty list meaning the column is clean. ``drop_recommended`` lists the
    columns with a high or medium finding, most severe first; ``keep`` is the
    rest in the original column order. ``notes`` says what was skipped and why.
    """

    features: Dict[Hashable, List[Finding]]
    drop_recommended: List[Hashable]
    keep: List[Hashable]
    target: Optional[Hashable] = None
    notes: List[str] = field(default_factory=list)
    n_rows: int = 0
    n_rows_checked: int = 0
    params: Dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------ access
    @property
    def n_features(self) -> int:
        """Number of feature columns examined."""
        return len(self.features)

    @property
    def flagged(self) -> List[Hashable]:
        """Columns with at least one finding of any severity, in column order."""
        return [c for c, fs in self.features.items() if fs]

    @property
    def review(self) -> List[Hashable]:
        """Kept columns that still have a (low severity) finding worth a look."""
        dropped = set(self.drop_recommended)
        return [c for c, fs in self.features.items() if fs and c not in dropped]

    def columns_with(self, kind: str) -> List[Hashable]:
        """Columns that have a finding of this ``kind``."""
        return [c for c, fs in self.features.items() if any(f.kind == kind for f in fs)]

    def iter_findings(self) -> Iterator[Tuple[Hashable, Finding]]:
        """All ``(column, finding)`` pairs, most severe first."""
        pairs = [(c, f) for c, fs in self.features.items() for f in fs]
        pos = {c: i for i, c in enumerate(self.features)}
        pairs.sort(key=lambda cf: (_RANK.get(cf[1].severity, 9), pos[cf[0]]))
        return iter(pairs)

    # ------------------------------------------------------------------ actions
    def apply(self, df: FrameLike) -> pd.DataFrame:
        """Return a copy of ``df`` without the ``drop_recommended`` columns.

        ``df`` is meant to be the frame that was checked. When it is not - a
        rename, a reordered load, a different split - the mismatch is reported
        through ``logging.getLogger("ml_feature_check")`` rather than passed over
        in silence, because an unfiltered frame looks exactly like a clean one.
        """
        if isinstance(df, pd.DataFrame):
            frame = df
        else:
            try:
                frame = load_frame(df)
            except (TypeError, ValueError) as exc:
                # load_frame speaks about files; apply() is documented as taking a
                # frame, so a caller who passed something else should be told that,
                # and without load_frame's own "expected a ..." repeated after it
                detail = str(exc)
                extra = "" if detail.startswith("expected a pandas DataFrame") else f": {detail}"
                raise TypeError(
                    "apply() expects a pandas DataFrame or a path to a .csv/.tsv/.parquet "
                    f"file, got {type(df).__name__}{extra}"
                ) from exc
        columns = set(frame.columns)
        present = [c for c in self.drop_recommended if c in columns]
        self._warn_mismatch(columns, present)
        return frame.drop(columns=present)

    def _warn_mismatch(self, columns: set, present: Sequence[Hashable]) -> None:
        """Log when the frame handed to ``apply()`` is not the frame we checked."""
        if self.features and not any(c in columns for c in self.features):
            log.warning(
                "apply() was given a frame that shares no column with the checked one, "
                "so nothing was dropped; the report describes %d column(s): %s",
                self.n_features,
                _join(list(self.features)),
            )
            return
        absent = [c for c in self.drop_recommended if c not in columns]
        if absent:
            log.warning(
                "apply() could not drop %d of the %d recommended column(s), which are "
                "not in the frame: %s",
                len(absent),
                len(self.drop_recommended),
                _join(absent),
            )

    # ------------------------------------------------------------------ output
    def summary(self) -> str:
        """Human-readable multi-line summary."""
        lines = [self._headline()]
        review = self.review
        width = self._name_width(list(self.drop_recommended) + list(review))
        if self.drop_recommended:
            lines.append(f"Drop recommended ({len(self.drop_recommended)}):")
            lines.extend(self._block(self.drop_recommended, width))
        else:
            lines.append("Drop recommended (0): none")
        if review:
            lines.append(f"Worth a look ({len(review)}):")
            lines.extend(self._block(review, width))
        if self.keep:
            lines.append(f"Keep ({len(self.keep)}): " + _join(self.keep))
        else:
            lines.append("Keep (0): none")
        if self.notes:
            lines.append("Notes:")
            lines.extend(f"  - {note}" for note in self.notes)
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """JSON-safe dictionary with every finding."""
        return {
            "target": None if self.target is None else str(self.target),
            "n_rows": int(self.n_rows),
            "n_rows_checked": int(self.n_rows_checked),
            "n_features": self.n_features,
            "drop_recommended": [str(c) for c in self.drop_recommended],
            "keep": [str(c) for c in self.keep],
            "notes": list(self.notes),
            "params": dict(self.params),
            "features": {str(c): [f.to_dict() for f in fs] for c, fs in self.features.items()},
        }

    def to_markdown(self) -> str:
        """Markdown report with one table row per feature."""
        dropped = set(self.drop_recommended)
        checked = f" (checked: {self.n_rows_checked:,})" if self.n_rows_checked != self.n_rows else ""
        target = f"`{_md(str(self.target))}`" if self.target is not None else "none"
        lines = [
            "# ml-feature-check report",
            "",
            f"- rows: {self.n_rows:,}{checked}",
            f"- features: {self.n_features}, target: {target}",
            f"- drop recommended: {len(self.drop_recommended)}, keep: {len(self.keep)}",
            "",
            "| Column | Verdict | Findings |",
            "| --- | --- | --- |",
        ]
        order = list(self.drop_recommended) + [c for c in self.features if c not in dropped]
        for col in order:
            findings = self.features[col]
            verdict = "drop" if col in dropped else ("review" if findings else "keep")
            text = "<br>".join(f"**{f.kind}** ({f.severity}): {_md(f.message)}" for f in findings) or "-"
            lines.append(f"| `{_md(str(col))}` | {verdict} | {text} |")
        if self.notes:
            lines.append("")
            lines.extend(f"- {_md(note)}" for note in self.notes)
        return "\n".join(lines) + "\n"

    # ------------------------------------------------------------------ dunder
    def __repr__(self) -> str:
        return (
            f"FeatureReport(features={self.n_features}, drop_recommended={len(self.drop_recommended)}, "
            f"keep={len(self.keep)}, target={self.target!r})"
        )

    def __str__(self) -> str:
        return self.summary()

    # ------------------------------------------------------------------ helpers
    def _headline(self) -> str:
        head = f"ml-feature-check: {self.n_features} feature(s), {self.n_rows:,} rows"
        if self.n_rows_checked != self.n_rows:
            head += f" ({self.n_rows_checked:,} checked)"
        head += f", target {str(self.target)!r}" if self.target is not None else ", no target"
        return head

    @staticmethod
    def _name_width(columns: Sequence[Hashable]) -> int:
        """One shared width for the column names, so every block lines up."""
        return min(max((len(_label(c)) for c in columns), default=4), 28)

    def _block(self, columns: Sequence[Hashable], width: int) -> List[str]:
        out: List[str] = []
        for col in columns:
            for i, f in enumerate(self.features[col]):
                name = _label(col) if i == 0 else ""
                tag = f"{f.kind} [{f.severity}]"
                out.append(f"  {name:<{width}}  {tag:<32} {f.message}")
        return out


def _label(col: Hashable) -> str:
    """Column name as it should appear in text.

    A name that is empty or only whitespace would render as blank padding and
    its finding would read as a second finding of the previous column, so such
    a name is quoted instead.
    """
    text = str(col)
    return text if text.strip() else repr(text)


def _join(columns: Sequence[Hashable]) -> str:
    """Comma-separated column names, every one of them visible."""
    return ", ".join(_label(c) for c in columns) or "none"


def _md(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ")
