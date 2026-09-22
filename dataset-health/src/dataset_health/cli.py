"""Command line interface: ``dataset-health data.csv [--target COL] [options]``."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Sequence

from . import __version__
from ._core import diagnose


def build_parser() -> argparse.ArgumentParser:
    """The argparse parser for the ``dataset-health`` command."""
    parser = argparse.ArgumentParser(
        prog="dataset-health",
        description=(
            "One-call health report for a CSV or Parquet dataset: missingness, duplicates, "
            "leakage, imbalance, outliers, correlations and more."
        ),
    )
    parser.add_argument("data", help="dataset to check (.csv, .tsv or .parquet)")
    parser.add_argument(
        "--target", metavar="COL", help="label column; enables the class-imbalance and target-leakage checks"
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=200_000,
        metavar="N",
        help="analyze at most N random rows; 0 uses every row (default: 200000)",
    )
    parser.add_argument(
        "--random-state", type=int, default=0, metavar="SEED", help="seed for the sampling (default: 0)"
    )
    output = parser.add_mutually_exclusive_group()
    output.add_argument("--json", action="store_true", help="print the full report as JSON instead of the summary")
    output.add_argument("--markdown", action="store_true", help="print the report as Markdown instead of the summary")
    parser.add_argument(
        "--output",
        metavar="PATH",
        help="also write the report to PATH: Markdown when it ends with .md, otherwise JSON",
    )
    parser.add_argument(
        "--fail-below",
        type=int,
        metavar="SCORE",
        help="exit with status 1 when the score is below SCORE (handy in CI)",
    )
    parser.add_argument(
        "--fail-on-critical", action="store_true", help="exit with status 1 when any critical issue is found"
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return parser


def _configure_streams() -> None:
    """Make stdout/stderr UTF-8 tolerant so non-Latin column names never raise."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):  # pragma: no cover - exotic stream
                pass


def _emit(text: str) -> None:
    try:
        print(text)
    except UnicodeEncodeError:  # pragma: no cover - only when reconfigure was unavailable
        encoding = getattr(sys.stdout, "encoding", None) or "ascii"
        print(text.encode(encoding, errors="replace").decode(encoding))


def _as_json(report) -> str:
    return json.dumps(report.to_dict(), indent=2, ensure_ascii=False)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Entry point. Returns the process exit status."""
    _configure_streams()
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        report = diagnose(
            args.data,
            target=args.target,
            sample=args.sample or None,
            random_state=args.random_state,
        )
    except (ValueError, TypeError, ImportError, OSError) as exc:
        parser.error(str(exc))
    if args.json:
        _emit(_as_json(report))
    elif args.markdown:
        _emit(report.to_markdown())
    else:
        _emit(report.summary())
    if args.output:
        path = Path(args.output)
        if path.suffix.lower() == ".md":
            path.write_text(report.to_markdown(), encoding="utf-8")
        else:
            path.write_text(_as_json(report), encoding="utf-8")
    failed = (args.fail_below is not None and report.score < args.fail_below) or (
        args.fail_on_critical and bool(report.critical)
    )
    return 1 if failed else 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
