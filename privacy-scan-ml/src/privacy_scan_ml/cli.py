"""Command line front end: scan a file or a string, print or write the result."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional, Sequence

from ._mask import STRATEGIES
from .report import PIIReport
from .scanner import DEFAULT_MIN_SHARE, DEFAULT_SAMPLE, PIIScanner


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="privacy-scan-ml",
        description=(
            "Find personal data in datasets before it leaks into models: emails, phones, "
            "Aadhaar, PAN, cards, IPs, addresses and more."
        ),
        epilog=(
            "examples:\n"
            "  privacy-scan-ml customers.csv\n"
            "  privacy-scan-ml customers.csv --json -o report.json\n"
            "  privacy-scan-ml customers.csv --mask clean.csv --strategy hash --salt s3cret\n"
            '  privacy-scan-ml --text "write to asha@example.com"\n'
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("path", nargs="?", help="a .csv, .tsv or .parquet file to scan")
    parser.add_argument("--text", metavar="STRING", help="scan this string instead of a file")
    parser.add_argument("--json", action="store_true", help="print the report as JSON")
    parser.add_argument("-o", "--output", metavar="PATH", help="write the report JSON to PATH")
    parser.add_argument(
        "--sample",
        type=int,
        default=DEFAULT_SAMPLE,
        help=f"rows to read from a big table (default: {DEFAULT_SAMPLE})",
    )
    parser.add_argument(
        "--min-share",
        type=float,
        default=DEFAULT_MIN_SHARE,
        help=f"share of a column's values that must validate (default: {DEFAULT_MIN_SHARE})",
    )
    parser.add_argument("--mask", metavar="PATH", help="write a masked copy of the table to PATH")
    parser.add_argument(
        "--strategy",
        choices=STRATEGIES,
        default="redact",
        help="how to mask: redact, hash or partial (default: redact)",
    )
    parser.add_argument("--salt", help="salt for the hash strategy")
    parser.add_argument(
        "--columns",
        help="comma separated columns to mask instead of the ones the scan flags",
    )
    parser.add_argument(
        "--fail-on-pii",
        action="store_true",
        help="exit with status 1 when personal data is found (useful in CI)",
    )
    parser.add_argument("--version", action="version", version=_version_string())
    return parser


def _version_string() -> str:
    from . import __version__

    return f"privacy-scan-ml {__version__}"


def _write_table(frame, path: Path) -> None:
    suffix = path.suffix.lower()
    if suffix in (".parquet", ".pq"):
        frame.to_parquet(path, index=False)
    elif suffix == ".tsv":
        frame.to_csv(path, sep="\t", index=False, encoding="utf-8")
    else:
        frame.to_csv(path, index=False, encoding="utf-8")


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Entry point for the ``privacy-scan-ml`` command. Returns the process exit code."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    parser = _build_parser()
    args = parser.parse_args(argv)
    mask_notes: List[str] = []

    # `is None`, not truthiness: `--text ""` is a string that WAS given, and telling
    # the caller they gave both or neither would be plainly wrong.
    if (args.path is None) == (args.text is None):
        parser.error("give either a file path or --text, not both")

    try:
        scanner = PIIScanner(sample=args.sample, min_share=args.min_share)
        data = args.text if args.text is not None else args.path
        report: PIIReport = scanner.scan(data)

        if args.mask:
            if args.text is not None:
                parser.error("--mask writes a table; use it with a file path, not --text")
            columns: Optional[List[str]] = None
            if args.columns:
                columns = [c.strip() for c in args.columns.split(",") if c.strip()]
            masked = scanner.mask(
                args.path, strategy=args.strategy, columns=columns, salt=args.salt
            )
            _write_table(masked, Path(args.mask))
            mask_notes = list(scanner.last_warnings)
            if args.strategy == "hash" and not args.salt:
                # The generated salt lives only on the in-process frame; a CSV cannot
                # carry it. Print it or the mapping can never be reproduced.
                mask_notes.append(
                    "hash salt generated for this run (keep it to reproduce the mapping, "
                    f"pass it back with --salt): {scanner.last_salt}"
                )
    except (OSError, ValueError, TypeError, KeyError, ImportError) as exc:
        print(f"privacy-scan-ml: {exc}", file=sys.stderr)
        return 2

    payload = json.dumps(report.to_dict(), indent=2, ensure_ascii=False)
    if args.output:
        Path(args.output).write_text(payload + "\n", encoding="utf-8")
    if args.json:
        print(payload)
        for note in mask_notes:
            # stderr, so stdout stays a single parseable JSON document
            print(f"privacy-scan-ml: {note}", file=sys.stderr)
    else:
        print(report.summary())
        if args.mask:
            print(f"  masked copy written to {args.mask}")
            for note in mask_notes:
                print(f"  {note}")
        if args.output:
            print(f"  report written to {args.output}")

    return 1 if (args.fail_on_pii and report.has_pii) else 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
