"""Command line entry point: ``smartclean-df data.csv --output clean.csv``."""
from __future__ import annotations

import argparse
import json
import sys
from typing import List, Optional

from . import __version__
from ._io import write_frame
from .core import MISSING_MODES, OUTLIER_MODES, clean


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="smartclean-df",
        description=(
            "Detect and fix missing values, duplicate rows, outliers, dirty column names and "
            "numbers/dates/booleans stored as text in a table, and report every change."
        ),
    )
    parser.add_argument("path", help="input table (.csv, .tsv or .parquet)")
    parser.add_argument(
        "--missing",
        choices=MISSING_MODES,
        default="auto",
        help="auto: fill with median / mode / forward-fill; drop: drop rows with missing values; "
        "none: leave them (default auto)",
    )
    parser.add_argument(
        "--outliers",
        choices=OUTLIER_MODES,
        default="clip",
        help="clip: winsorize to the IQR bounds; flag: only report; drop: drop the rows; "
        "none: skip (default clip)",
    )
    parser.add_argument(
        "--iqr-factor",
        type=float,
        default=3.0,
        metavar="K",
        help="outlier bounds are Q1 - K*IQR and Q3 + K*IQR (default 3.0)",
    )
    parser.add_argument("--keep-duplicates", action="store_true", help="do not drop exact duplicate rows")
    parser.add_argument("--keep-column-names", action="store_true", help="do not normalize column names")
    parser.add_argument("--no-parse-numbers", action="store_true", help="leave numbers stored as text alone")
    parser.add_argument("--no-parse-dates", action="store_true", help="leave dates stored as text alone")
    parser.add_argument("--no-parse-booleans", action="store_true", help="leave booleans stored as text alone")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="report what would change without changing anything (cannot be combined with --output)",
    )
    parser.add_argument("--json", action="store_true", help="print the result as JSON instead of the text summary")
    parser.add_argument("--output", metavar="FILE", help="write the cleaned table here (.csv, .tsv or .parquet)")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    """Run the command line; returns the process exit status."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.dry_run and args.output:
        parser.error("--dry-run cannot be combined with --output")
    try:
        result = clean(
            args.path,
            missing=args.missing,
            outliers=args.outliers,
            iqr_factor=args.iqr_factor,
            duplicates=not args.keep_duplicates,
            normalize_columns=not args.keep_column_names,
            parse_numbers=not args.no_parse_numbers,
            parse_dates=not args.no_parse_dates,
            parse_booleans=not args.no_parse_booleans,
            dry_run=args.dry_run,
        )
        written = write_frame(result.df, args.output) if args.output else None
    except (ValueError, TypeError, ImportError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if args.json:
        payload = result.to_dict()
        if written is not None:
            payload["output"] = str(written)
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    else:
        print(result.summary())
        if written is not None:
            print(f"cleaned table written to {written}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
