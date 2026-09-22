"""Result objects: one Action per change, one CleanResult per run."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Tuple

import pandas as pd

__all__ = ["Action", "CleanResult"]


@dataclass(frozen=True)
class Action:
    """One change made to the table.

    ``column`` is None for table-wide changes (dropping duplicate rows, dropping
    rows with missing values). ``rows_affected`` counts the cells changed in
    that column, or the rows dropped for row-level changes; it is 0 for a
    column rename.
    """

    column: Optional[str]
    kind: str
    detail: str
    rows_affected: int

    def to_dict(self) -> Dict[str, Any]:
        """JSON-safe representation."""
        return {
            "column": self.column,
            "kind": self.kind,
            "detail": self.detail,
            "rows_affected": int(self.rows_affected),
        }

    def __str__(self) -> str:
        target = "<table>" if self.column is None else self.column
        return f"[{target}] {self.kind}: {self.detail} ({int(self.rows_affected)} rows)"


@dataclass
class CleanResult:
    """What :func:`smartclean_df.clean` returns.

    Iterating the result yields ``(df, actions)`` so ``df, actions = clean(df)``
    works too. ``warnings`` is empty on a healthy run; it names anything that
    came out degenerate, such as a table that lost every row.
    """

    df: pd.DataFrame
    actions: List[Action] = field(default_factory=list)
    input_shape: Tuple[int, int] = (0, 0)
    output_shape: Tuple[int, int] = (0, 0)
    dry_run: bool = False
    warnings: List[str] = field(default_factory=list)
    #: dtypes of the table the run produced. On a dry run ``df`` is the
    #: untouched input, so this is the schema the run *would* have produced.
    output_dtypes: Optional[Dict[str, str]] = None

    def summary(self) -> str:
        """Human-readable account of every change, in order."""
        in_rows, in_cols = self.input_shape
        out_rows, out_cols = self.output_shape
        header = (
            f"smartclean-df: {in_rows} rows x {in_cols} columns"
            f" -> {out_rows} rows x {out_cols} columns"
        )
        if self.dry_run:
            header += "  (dry run: input returned unchanged)"
        lines = [header]
        if self.actions:
            lines.append(f"  {len(self.actions)} action(s):")
            for number, action in enumerate(self.actions, start=1):
                lines.append(f"    {number}. {action}")
        else:
            lines.append("  no changes needed")
        for message in self.warnings:
            lines.append(f"  warning: {message}")
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """JSON-safe representation (shapes, actions and output dtypes).

        ``dtypes`` always describes the cleaned table, including on a dry run
        where ``df`` itself is the unchanged input.
        """
        dtypes = self.output_dtypes
        if dtypes is None:
            dtypes = {str(col): str(dtype) for col, dtype in self.df.dtypes.items()}
        return {
            "input_shape": [int(self.input_shape[0]), int(self.input_shape[1])],
            "output_shape": [int(self.output_shape[0]), int(self.output_shape[1])],
            "dry_run": bool(self.dry_run),
            "n_actions": len(self.actions),
            "actions": [action.to_dict() for action in self.actions],
            "warnings": list(self.warnings),
            "dtypes": dict(dtypes),
        }

    def __iter__(self) -> Iterator[Any]:
        yield self.df
        yield self.actions
