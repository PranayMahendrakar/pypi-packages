"""Command line interface: ``schema-guard infer|validate|enforce``."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional, Sequence

from . import __version__
from ._io import read_tabular, write_tabular
from ._types import json_dumps
from .result import SchemaError
from .schema import Schema

COMMANDS = ("infer", "validate", "enforce")

#: Suffixes that say a path is data, not a schema - the giveaway for swapped arguments.
_DATA_SUFFIXES = (".csv", ".tsv", ".parquet", ".pq")


def build_parser() -> argparse.ArgumentParser:
    """The argparse parser for the ``schema-guard`` command."""
    parser = argparse.ArgumentParser(
        prog="schema-guard",
        description="Infer a DataFrame schema once, then validate or enforce it on every new batch of data.",
        epilog="A bare data file runs 'infer', so 'schema-guard data.csv' prints the inferred schema.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subparsers = parser.add_subparsers(dest="command", metavar="command")

    infer = subparsers.add_parser("infer", help="infer a schema from a data file and print it (the default)")
    infer.add_argument("data", help="path to a .csv, .tsv or .parquet file")
    infer.add_argument("--json", action="store_true", help="print the schema as JSON instead of a summary table")
    infer.add_argument("--output", "-o", metavar="PATH", help="also write the schema JSON to PATH")
    infer.add_argument(
        "--max-categories",
        type=int,
        default=50,
        metavar="N",
        help="record allowed categories for text columns with at most N distinct values (default 50, 0 disables)",
    )
    infer.add_argument("--no-ranges", action="store_true", help="do not record min/max for numeric columns")
    infer.add_argument(
        "--nullable",
        choices=("observed", "always", "never"),
        default="observed",
        help="how to decide whether a column may hold nulls (default: observed)",
    )

    validate = subparsers.add_parser("validate", help="check a data file against a saved schema (exit 1 on failure)")
    validate.add_argument("schema", help="schema JSON written by 'infer --output' or Schema.save()")
    validate.add_argument("data", help="path to a .csv, .tsv or .parquet file")
    validate.add_argument("--json", action="store_true", help="print the result as JSON instead of a summary")
    validate.add_argument("--output", "-o", metavar="PATH", help="also write the result JSON to PATH")
    validate.add_argument("--strict-ranges", action="store_true", help="treat out-of-range values as errors")

    enforce = subparsers.add_parser("enforce", help="coerce a data file to a saved schema and write the result")
    enforce.add_argument("schema", help="schema JSON written by 'infer --output' or Schema.save()")
    enforce.add_argument("data", help="path to a .csv, .tsv or .parquet file")
    enforce.add_argument("--output", "-o", metavar="PATH", required=True, help="where to write the enforced data")
    enforce.add_argument("--strict", action="store_true", help="fail instead of coercing when the data does not match")
    enforce.add_argument("--extra", choices=("drop", "keep", "raise"), default="drop", help="what to do with extra columns")
    enforce.add_argument("--missing", choices=("fill", "raise"), default="fill", help="what to do with missing columns")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Entry point for the ``schema-guard`` command; returns the process exit code."""
    _make_console_safe()
    args_list: List[str] = list(sys.argv[1:] if argv is None else argv)
    if args_list and args_list[0] not in COMMANDS and not args_list[0].startswith("-"):
        args_list.insert(0, "infer")
    parser = build_parser()
    if not args_list:
        parser.print_help()
        return 2
    args = parser.parse_args(args_list)
    try:
        if args.command == "infer":
            return _run_infer(args)
        if args.command == "validate":
            return _run_validate(args)
        return _run_enforce(args)
    except SchemaError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    except (OSError, ValueError, TypeError, ImportError) as exc:
        print(f"schema-guard: error: {exc}", file=sys.stderr)
        return 1


def _run_infer(args: argparse.Namespace) -> int:
    schema = Schema.infer(
        read_tabular(args.data),
        categorical_max_unique=args.max_categories,
        numeric_ranges=not args.no_ranges,
        nullable=args.nullable,
    )
    if args.output:
        schema.save(args.output)
        print(f"schema written to {args.output}", file=sys.stderr)
    if args.json:
        print(schema.to_json().rstrip("\n"))
    else:
        print(schema.summary())
    return 0


def _load_schema(path: str) -> Schema:
    """Load the schema argument, catching the common slip of passing the two files swapped."""
    try:
        return Schema.load(path)
    except ValueError as exc:
        if Path(path).suffix.lower() in _DATA_SUFFIXES:
            raise ValueError(
                f"{exc} (validate and enforce take the schema first, then the data file)"
            ) from exc
        raise


def _run_validate(args: argparse.Namespace) -> int:
    schema = _load_schema(args.schema)
    result = schema.validate(read_tabular(args.data), strict_ranges=args.strict_ranges)
    payload = result.to_dict()
    if args.output:
        # json_dumps, not json.dump: a batch holding inf would otherwise write a bare
        # Infinity token that only Python can read back. The parent folder is created the
        # same way 'infer --output' and 'enforce --output' create theirs.
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as handle:
            handle.write(json_dumps(payload) + "\n")
        print(f"result written to {args.output}", file=sys.stderr)
    if args.json:
        print(json_dumps(payload))
    else:
        print(result.summary())
    return 0 if result.ok else 1


def _run_enforce(args: argparse.Namespace) -> int:
    schema = _load_schema(args.schema)
    fixed = schema.enforce(
        read_tabular(args.data),
        mode="strict" if args.strict else "coerce",
        extra=args.extra,
        missing=args.missing,
    )
    write_tabular(fixed, args.output)
    print(f"wrote {len(fixed)} rows x {len(fixed.columns)} columns to {args.output}")
    return 0


def _make_console_safe() -> None:
    """Never crash on a console that cannot encode a value from the data."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):  # pragma: no cover - closed or odd streams
                pass


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
