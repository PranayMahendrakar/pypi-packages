"""PIIReport: which columns (or which spans of text) carry personal data, and how risky that is."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

from ._detectors import CRITICAL_TYPES, HIGH_TYPES, LOW_TYPES, SENSITIVE_TYPES

RISK_LEVELS = ("none", "low", "medium", "high")


@dataclass
class Finding:
    """One hit inside free text: the type, a masked preview and the ``(start, end)`` span.

    ``doc`` is the index of the text in the input list (0 for a single string).
    """

    type: str
    value_masked: str
    span: Tuple[int, int]
    doc: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.type,
            "value_masked": self.value_masked,
            "span": [int(self.span[0]), int(self.span[1])],
            "doc": int(self.doc),
        }


@dataclass
class ColumnFinding:
    """What a column carries: ``types`` maps each PII type to the share of non-null values hit."""

    column: str
    types: Dict[str, float]
    samples: List[str]
    confidence: float
    primary: str = ""
    dtype: str = ""
    rows_checked: int = 0
    hinted: bool = False

    @property
    def category(self) -> str:
        """The family ``primary`` belongs to: ``"sensitive_category"``, or ``primary``.

        Lets a caller ask "which columns are sensitive categories" without hard-coding
        gender, religion, caste and the rest.
        """
        return "sensitive_category" if self.primary in SENSITIVE_TYPES else self.primary

    def to_dict(self) -> Dict[str, Any]:
        return {
            "column": str(self.column),
            "types": {k: float(round(v, 4)) for k, v in self.types.items()},
            "samples": list(self.samples),
            "confidence": float(round(self.confidence, 2)),
            "primary": self.primary,
            "category": self.category,
            "dtype": self.dtype,
            "rows_checked": int(self.rows_checked),
            "hinted": bool(self.hinted),
        }


def risk_level(types: List[str]) -> str:
    """Combine the PII types found into ``"none" | "low" | "medium" | "high"``."""
    found = set(types)
    if not found:
        return "none"
    high = found & HIGH_TYPES
    if found & CRITICAL_TYPES or len(high) >= 2:
        return "high"
    if {"date_of_birth", "postal_code", "gender"} <= found:  # the classic re-identification triple
        return "high"
    if high:
        return "medium"
    low = found & LOW_TYPES
    if len(low) >= 2:
        return "medium"
    return "low" if low else "medium"


@dataclass
class PIIReport:
    """Result of :func:`privacy_scan_ml.scan`.

    ``columns`` is filled for tabular input (column -> :class:`ColumnFinding`), ``findings``
    for free text (one :class:`Finding` per hit). Values are never stored unmasked.
    """

    columns: Dict[str, ColumnFinding] = field(default_factory=dict)
    findings: List[Finding] = field(default_factory=list)
    mode: str = "table"
    rows: int = 0
    rows_scanned: int = 0
    documents: int = 0
    n_columns: int = 0
    min_share: float = 0.2
    warnings: List[str] = field(default_factory=list)

    @property
    def has_pii(self) -> bool:
        """True when at least one column or one span was flagged."""
        return bool(self.columns) or bool(self.findings)

    @property
    def types(self) -> List[str]:
        """Sorted list of every PII type found anywhere."""
        found = set()
        for cf in self.columns.values():
            found.update(cf.types)
        for f in self.findings:
            found.add(f.type)
        return sorted(found)

    @property
    def risk(self) -> str:
        """``"none"``, ``"low"``, ``"medium"`` or ``"high"``."""
        return risk_level(self.types)

    @property
    def pii_columns(self) -> List[str]:
        """Names of the flagged columns, in frame order."""
        return list(self.columns)

    def to_dict(self) -> Dict[str, Any]:
        """JSON-safe dictionary with every field plus ``has_pii`` and ``risk``."""
        return {
            "has_pii": self.has_pii,
            "risk": self.risk,
            "mode": self.mode,
            "rows": int(self.rows),
            "rows_scanned": int(self.rows_scanned),
            "documents": int(self.documents),
            "n_columns": int(self.n_columns),
            "min_share": float(self.min_share),
            "types": self.types,
            "columns": {str(k): v.to_dict() for k, v in self.columns.items()},
            "findings": [f.to_dict() for f in self.findings],
            "warnings": list(self.warnings),
        }


    # -- markdown ----------------------------------------------------------
    def to_markdown(self) -> str:
        """The report as a Markdown document. Every value in it is masked."""
        if self.mode == "text":
            return self._markdown_text()
        return self._markdown_table()

    def _markdown_table(self) -> str:
        scanned = (
            f"{self.rows_scanned:,} of {self.rows:,} rows scanned"
            if self.rows_scanned != self.rows
            else f"{self.rows:,} rows scanned"
        )
        lines = [
            "# privacy-scan-ml report",
            "",
            f"**Risk: {self.risk}** - {len(self.columns)} of {self.n_columns} columns "
            f"carry personal data, {scanned}.",
            "",
        ]
        if not self.columns:
            lines.append("No personal data found.")
        else:
            lines.append("| column | types | confidence | masked examples |")
            lines.append("| --- | --- | --- | --- |")
            for name, cf in self.columns.items():
                kinds = ", ".join(f"{t} ({share * 100:.0f}%)" for t, share in cf.types.items())
                samples = ", ".join(_code(s) for s in cf.samples) if cf.samples else "-"
                lines.append(
                    f"| {_cell(name)} | {_cell(kinds)} | {cf.confidence:.2f} | {samples} |"
                )
        lines.extend(_markdown_warnings(self.warnings))
        return "\n".join(lines) + "\n"

    def _markdown_text(self) -> str:
        n = len(self.findings)
        docs = f"{self.documents} document" + ("s" if self.documents != 1 else "")
        lines = [
            "# privacy-scan-ml report",
            "",
            f"**Risk: {self.risk}** - {n} finding{'s' if n != 1 else ''} in {docs}.",
            "",
        ]
        if not self.findings:
            lines.append("No personal data found.")
        else:
            lines.append("| document | type | masked value | span |")
            lines.append("| --- | --- | --- | --- |")
            for f in self.findings[:200]:
                span = f"{f.span[0]}-{f.span[1]}"
                lines.append(
                    f"| {f.doc} | {_cell(f.type)} | {_code(f.value_masked)} | {span} |"
                )
            if n > 200:
                lines.extend(["", f"... {n - 200} more findings."])
        lines.extend(_markdown_warnings(self.warnings))
        return "\n".join(lines) + "\n"

    def summary(self) -> str:
        """Human-readable multi-line summary."""
        if self.mode == "text":
            return self._summary_text()
        return self._summary_table()

    def _summary_table(self) -> str:
        n_flagged = len(self.columns)
        scanned = f"{self.rows_scanned:,} of {self.rows:,} rows scanned" if self.rows_scanned != self.rows else f"{self.rows:,} rows scanned"
        head = (
            f"privacy-scan-ml: {n_flagged} of {self.n_columns} columns carry personal data "
            f"(risk: {self.risk}), {scanned}"
        )
        lines = [head]
        if not self.columns:
            lines.append("  no personal data found")
        width = max((len(str(c)) for c in self.columns), default=0)
        width = min(max(width, 6), 28)
        for name, cf in self.columns.items():
            kinds = ", ".join(f"{t} ({share * 100:.0f}%)" for t, share in cf.types.items())
            samples = ", ".join(cf.samples) if cf.samples else "-"
            lines.append(
                f"  {str(name)[:width]:<{width}}  {kinds:<34}  confidence {cf.confidence:.2f}  e.g. {samples}"
            )
        for w in self.warnings:
            lines.append(f"  warning: {w}")
        return "\n".join(lines)

    def _summary_text(self) -> str:
        n = len(self.findings)
        docs = f"{self.documents} document" + ("s" if self.documents != 1 else "")
        lines = [f"privacy-scan-ml: {n} finding{'s' if n != 1 else ''} in {docs} (risk: {self.risk})"]
        if not self.findings:
            lines.append("  no personal data found")
            return "\n".join(lines)
        counts: Dict[str, int] = {}
        for f in self.findings:
            counts[f.type] = counts.get(f.type, 0) + 1
        lines.append("  " + ", ".join(f"{t} x{c}" for t, c in sorted(counts.items())))
        for f in self.findings[:50]:
            where = f"[{f.doc}] " if self.documents > 1 else ""
            lines.append(f"  {where}{f.type:<14} {f.value_masked:<28} at {f.span[0]}-{f.span[1]}")
        if n > 50:
            lines.append(f"  ... {n - 50} more")
        for w in self.warnings:
            lines.append(f"  warning: {w}")
        return "\n".join(lines)


def _cell(text: Any) -> str:
    """Text safe inside a Markdown table cell (a pipe would end the cell)."""
    return str(text).replace("|", "\\|").replace("\n", " ").strip()


def _code(text: Any) -> str:
    """A masked value as inline code, so asterisks stay literal."""
    return "`" + _cell(text).replace("`", "'") + "`"


def _markdown_warnings(warnings: List[str]) -> List[str]:
    if not warnings:
        return []
    out = ["", "## Warnings", ""]
    out.extend(f"- {_cell(w)}" for w in warnings)
    return out
