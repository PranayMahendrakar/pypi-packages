"""Command line interface: ``ml-pipeline-kit DATA [checks]``.

The checks a pipeline makes - required columns, dtypes, ranges, no missing
values - are configuration, so they can be run against a file without writing
any Python. Steps cannot: they are your code, and live in your code.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd

from . import __version__
from ._errors import PipelineError
from ._io import read_table
from ._pipeline import Pipeline

_DESCRIPTION = """Check a data file against the columns, dtypes and ranges it should have.

DATA is a .csv, .tsv or .parquet file. Give at least one check; every check that
fails is named in the report, and the exit code is 1 when any of them do - unless
--warn recorded them as warnings instead, which always exits 0."""

_EPILOG = """examples:
  ml-pipeline-kit sales.csv --expect-schema "id:int,city:str"
  ml-pipeline-kit sales.csv --expect-range "price:0:1000" --not-null id
  ml-pipeline-kit sales.csv --expect-schema age:int --warn
  ml-pipeline-kit sales.csv --expect-range "age:18:" --json > report.json
  ml-pipeline-kit --describe scoring.json"""


def build_parser() -> argparse.ArgumentParser:
    """The argument parser, exposed so tests and docs can read it."""
    parser = argparse.ArgumentParser(
        prog="ml-pipeline-kit",
        description=_DESCRIPTION,
        epilog=_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "data", metavar="DATA", nargs="?", help="path to a .csv, .tsv or .parquet file"
    )
    parser.add_argument(
        "--expect-schema",
        metavar="SPEC",
        action="append",
        default=[],
        help='columns and dtypes, e.g. "age:int,city:str"; a bare name checks presence (repeatable)',
    )
    parser.add_argument(
        "--expect-range",
        metavar="SPEC",
        action="append",
        default=[],
        help='a numeric range, e.g. "age:18:100"; leave a side empty to leave it open (repeatable)',
    )
    parser.add_argument(
        "--not-null",
        metavar="COL",
        action="append",
        default=[],
        help="a column that must have no missing values (repeatable)",
    )
    parser.add_argument(
        "--warn",
        action="store_true",
        help="record failed checks as warnings and keep going; a warn-only run exits 0",
    )
    parser.add_argument(
        "--name", default="checks", help="name for the pipeline in the report (default: checks)"
    )
    parser.add_argument(
        "--describe",
        metavar="FILE",
        help="print the steps of a pipeline saved with Pipeline.save and exit",
    )
    parser.add_argument(
        "--json", action="store_true", help="print the full result as JSON instead of the summary"
    )
    parser.add_argument("--output", metavar="PATH", help="write the JSON result to PATH")
    parser.add_argument("--version", action="version", version="ml-pipeline-kit {0}".format(__version__))
    return parser


def _make_console_safe() -> None:
    """Never crash on a console that cannot encode a value from the data."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except (ValueError, OSError):  # pragma: no cover - closed or unusual streams
            try:
                reconfigure(errors="replace")
            except (ValueError, OSError):
                pass


def parse_schema(spec: str) -> Dict[str, Optional[str]]:
    """``"age:int,city:str"`` becomes ``{"age": "int", "city": "str"}``."""
    schema: Dict[str, Optional[str]] = {}
    for piece in spec.split(","):
        piece = piece.strip()
        if not piece:
            continue
        column, sep, dtype = piece.partition(":")
        column = column.strip()
        if not column:
            raise ValueError("--expect-schema {0!r}: a column name is missing".format(spec))
        schema[column] = dtype.strip() if sep and dtype.strip() else None
    if not schema:
        raise ValueError("--expect-schema {0!r}: nothing to check".format(spec))
    return schema


def parse_range(spec: str) -> Tuple[str, Optional[float], Optional[float]]:
    """``"age:18:100"`` becomes ``("age", 18.0, 100.0)``; a side may be empty."""
    parts = spec.split(":")
    if len(parts) != 3 or not parts[0].strip():
        raise ValueError(
            '--expect-range {0!r}: write it as "column:low:high", leaving a side '
            "empty to leave it open".format(spec)
        )
    column = parts[0].strip()
    bounds: List[Optional[float]] = []
    for text, side in ((parts[1], "low"), (parts[2], "high")):
        text = text.strip()
        if not text:
            bounds.append(None)
            continue
        try:
            bounds.append(float(text))
        except ValueError:
            raise ValueError(
                "--expect-range {0!r}: {1} bound {2!r} is not a number".format(spec, side, text)
            ) from None
    if bounds[0] is None and bounds[1] is None:
        raise ValueError("--expect-range {0!r}: give at least one bound".format(spec))
    return column, bounds[0], bounds[1]


def _not_null_check(column: str):
    """A check that `column` has no missing values."""

    def check(frame: Any) -> Tuple[bool, Optional[str]]:
        if not isinstance(frame, pd.DataFrame):
            return False, "expected a table, got {0}".format(type(frame).__name__)
        if column not in frame.columns:
            return False, "column '{0}' is missing".format(column)
        missing = int(frame[column].isna().sum())
        if missing:
            return False, "column '{0}' has {1} of {2} values missing".format(
                column, missing, len(frame)
            )
        return True, None

    return check


def build_pipeline(args: argparse.Namespace) -> Pipeline:
    """Turn the command line options into a checks-only pipeline."""
    severity = "warn" if args.warn else "error"
    pipe = Pipeline(args.name)
    for spec in args.expect_schema:
        pipe.expect_schema(parse_schema(spec), severity=severity)
    for spec in args.expect_range:
        column, low, high = parse_range(spec)
        pipe.expect_range(column, low, high, severity=severity)
    for column in args.not_null:
        pipe.validate(
            _not_null_check(column), name="not_null[{0}]".format(column), severity=severity
        )
    return pipe


def write_report(path: str, payload: Dict[str, Any]) -> None:
    """Write the JSON report to `path`, creating the folder as ``save()`` does.

    ``Pipeline.save()`` makes the parent folders it needs, so ``--output`` does
    too. Anything that still goes wrong is reported as a sentence naming the
    folder, the way every other message here does, not as a bare errno.
    """
    target = Path(path)
    parent = target.parent
    if str(parent) and not parent.exists():
        try:
            parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise ValueError(
                "--output {0}: the folder '{1}' does not exist and could not be created "
                "({2}); create it first or write somewhere else".format(path, parent, exc)
            ) from None
    try:
        with open(target, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
    except OSError as exc:
        raise ValueError(
            "--output {0}: the report could not be written ({1})".format(path, exc)
        ) from None


def data_line(path: str, frame: pd.DataFrame, limit: int = 8) -> str:
    """One line naming the file, its shape and its columns."""
    names = [str(col) for col in frame.columns]
    shown = ", ".join(names[:limit])
    if len(names) > limit:
        shown += ", and {0} more".format(len(names) - limit)
    return "data: {0} ({1:,} rows, {2} columns: {3})".format(
        path, len(frame), len(names), shown or "none"
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Entry point; returns the process exit code (0 ok, 1 checks failed, 2 error)."""
    _make_console_safe()
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.describe:
            print(Pipeline.load(args.describe).describe())
            return 0
        if not args.data:
            parser.error("give a DATA file to check, or --describe FILE")
        frame, notes = read_table(args.data)
        pipe = build_pipeline(args)
        result = pipe.run(frame)
        for note in reversed(notes):
            result.warnings.insert(0, note)
        if not len(pipe):
            result.warnings.append(
                "give --expect-schema, --expect-range or --not-null to check something"
            )
        payload = result.to_dict()
        payload["data"] = str(args.data)
        if args.output:
            write_report(args.output, payload)
        if args.json:
            print(json.dumps(payload, indent=2, ensure_ascii=False))
        else:
            print(data_line(str(args.data), frame))
            print(result.summary())
            if args.output:
                print("  wrote     : {0}".format(args.output))
        return 0 if result.ok else 1
    except (ValueError, KeyError, TypeError, OSError, ImportError, PipelineError) as exc:
        print("ml-pipeline-kit: error: {0}".format(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
