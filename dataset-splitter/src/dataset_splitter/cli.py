"""Command line entry point: ``dataset-splitter data.csv --target y --group customer_id``."""
from __future__ import annotations

import argparse
import json
import sys
from typing import List, Optional, Union

from . import __version__
from .splitter import Splitter


def _size(text: str) -> Union[int, float]:
    """'0.2' -> fraction, '100' -> absolute row count."""
    if "." in text or "e" in text.lower():
        return float(text)
    return int(text)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dataset-splitter",
        description=(
            "Leakage-safe train/validation/test splits: stratified, grouped, time-aware, "
            "and checked. Prints the split report; exit status 1 when the report is not ok."
        ),
    )
    parser.add_argument("path", help="input table (.csv, .tsv or .parquet)")
    parser.add_argument(
        "--target", metavar="COL", help="column to stratify on and report class balance for"
    )
    parser.add_argument(
        "--group",
        metavar="COL",
        nargs="+",
        help="column(s) whose rows must stay on one side, or 'auto' to detect id-like columns",
    )
    parser.add_argument(
        "--time", metavar="COL", help="chronological split on this column (oldest train, newest test)"
    )
    parser.add_argument(
        "--test-size", type=_size, default=0.2, help="fraction or row count (default 0.2)"
    )
    parser.add_argument(
        "--val-size", type=_size, default=0.1, help="fraction or row count (default 0.1)"
    )
    parser.add_argument("--random-state", type=int, default=0, help="seed for shuffling (default 0)")
    parser.add_argument(
        "--no-dedupe", action="store_true", help="do not keep exact duplicate rows on the same side"
    )
    parser.add_argument("--json", action="store_true", help="print the report as JSON")
    parser.add_argument(
        "--output", metavar="DIR", help="write train/val/test files and report.json here"
    )
    parser.add_argument(
        "--format", choices=("csv", "parquet"), default="csv", help="output file format"
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    """Parse arguments, split the table, print the report. Returns the process exit status."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):  # a stream that cannot be reconfigured
                pass
    args = build_parser().parse_args(argv)
    group = None
    if args.group:
        if args.group == ["auto"]:
            group = "auto"
        elif len(args.group) == 1:
            group = args.group[0]
        else:
            group = list(args.group)
    try:
        splitter = Splitter(
            target=args.target,
            group=group,
            time=args.time,
            test_size=args.test_size,
            val_size=args.val_size,
            random_state=args.random_state,
            dedupe=not args.no_dedupe,
        )
        result = splitter.split(args.path)
        report = result.report()
        if args.output:
            result.save(args.output, format=args.format)
    except (ValueError, TypeError, ImportError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(report.to_dict(), indent=2, ensure_ascii=False))
    else:
        print(report.summary())
    return 0 if report.ok else 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
