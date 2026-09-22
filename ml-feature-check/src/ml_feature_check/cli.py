"""Command line interface: ``ml-feature-check INPUT [options]``."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Sequence

from . import __version__
from ._io import load_frame, write_frame
from .checker import FeatureChecker

_DESCRIPTION = """Catch useless, redundant, leaking and suspicious features before you train on them.

INPUT is a .csv, .tsv or .parquet file. The summary lists what to drop and why."""

_EPILOG = """examples:
  ml-feature-check train.csv
  ml-feature-check train.csv --target churn
  ml-feature-check train.csv --target churn --json
  ml-feature-check train.csv --target churn --output report.md --apply clean.csv"""


def build_parser() -> argparse.ArgumentParser:
    """The argparse parser, exposed so the options can be tested."""
    parser = argparse.ArgumentParser(
        prog="ml-feature-check",
        description=_DESCRIPTION,
        epilog=_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("input", metavar="INPUT", help="a .csv, .tsv or .parquet file to check")
    parser.add_argument(
        "--target", metavar="COL", default=None,
        help="the label column; unlocks the leakage and importance checks",
    )
    parser.add_argument(
        "--corr-threshold", type=float, default=0.95, dest="corr_threshold",
        help="|corr| or Cramer's V above which two columns are redundant (default: 0.95)",
    )
    parser.add_argument(
        "--missing-threshold", type=float, default=0.6, dest="missing_threshold",
        help="missing fraction above which a column is flagged (default: 0.6)",
    )
    parser.add_argument(
        "--cardinality-threshold", type=float, default=0.98, dest="cardinality_threshold",
        help="distinct-value ratio above which a column looks like an id (default: 0.98)",
    )
    parser.add_argument(
        "--sample", type=int, default=200_000,
        help="check at most this many rows, sampled at random (default: 200000)",
    )
    parser.add_argument(
        "--random-state", type=int, default=0, dest="random_state",
        help="seed for sampling and the models (default: 0)",
    )
    parser.add_argument("--json", action="store_true", help="print the report as JSON instead of the summary")
    parser.add_argument("--markdown", action="store_true", help="print the report as a markdown table")
    parser.add_argument(
        "--output", metavar="PATH",
        help="write the report here (.json, .md or .txt picked from the suffix)",
    )
    parser.add_argument(
        "--apply", metavar="PATH", dest="apply_to",
        help="write INPUT without the drop_recommended columns here (.csv/.tsv/.parquet)",
    )
    parser.add_argument("--version", action="version", version=f"ml-feature-check {__version__}")
    return parser


def _make_console_safe() -> None:
    """Never crash on a console that cannot encode a value from the data."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):  # pragma: no cover - closed or odd streams
                pass


def _render(report, as_json: bool, as_markdown: bool) -> str:
    if as_json:
        return json.dumps(report.to_dict(), indent=2, ensure_ascii=False)
    if as_markdown:
        return report.to_markdown()
    return report.summary()


def write_report(report, path: str) -> None:
    """Write the report to ``path``; the suffix picks JSON, markdown or plain text."""
    suffix = Path(path).suffix.lower()
    if suffix == ".json":
        text = json.dumps(report.to_dict(), indent=2, ensure_ascii=False)
    elif suffix in (".md", ".markdown"):
        text = report.to_markdown()
    else:
        text = report.summary() + "\n"
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Entry point; returns the process exit code."""
    _make_console_safe()
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        frame = load_frame(args.input)
        checker = FeatureChecker(
            corr_threshold=args.corr_threshold,
            missing_threshold=args.missing_threshold,
            cardinality_threshold=args.cardinality_threshold,
            sample=args.sample,
            random_state=args.random_state,
        )
        report = checker.check(frame, args.target)
        print(_render(report, args.json, args.markdown))
        if args.output:
            write_report(report, args.output)
            print(f"wrote the report to {args.output}", file=sys.stderr)
        if args.apply_to:
            cleaned = report.apply(frame)
            write_frame(cleaned, args.apply_to)
            print(
                f"wrote {len(cleaned.columns)} column(s) to {args.apply_to}, "
                f"{len(report.drop_recommended)} dropped",
                file=sys.stderr,
            )
    except (ValueError, KeyError, TypeError, OSError, ImportError) as exc:
        print(f"ml-feature-check: error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
