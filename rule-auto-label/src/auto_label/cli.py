"""Command line entry point: ``auto-label INPUT --rule LABEL=kw1,kw2 ...``."""

from __future__ import annotations

import argparse
import io
import json
import logging
import os
import sys
import tempfile
from typing import Any, Callable, Dict, List, Optional

import pandas as pd

from auto_label import __version__
from auto_label._data import TABULAR, decode_text, read_table
from auto_label.labeler import Labeler, LabelResult

logger = logging.getLogger(__name__)

TABLE_SUFFIXES = (".csv", ".parquet")
OUTPUT_SUFFIXES = (".csv", ".parquet", ".json")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="auto-label",
        description=(
            "Label text or tabular data from rules, then extend the labels with a small "
            "model. Rules are given inline (--rule/--regex/--query) or in a JSON file."
        ),
        epilog=(
            "examples:\n"
            "  auto-label tickets.csv --column text --rule billing=invoice,refund --rule bug=crash,error\n"
            "  auto-label lines.txt --regex urgent='(?i)asap|urgent' --json\n"
            "  auto-label orders.csv --query big='amount > 100' --output labeled.csv\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("input", help="a .csv/.parquet file, a text file with one item per line, or - for stdin")
    parser.add_argument(
        "--rule", action="append", default=[], metavar="LABEL=KW1,KW2",
        help="keyword rule: comma-separated case-insensitive words/phrases (repeatable)",
    )
    parser.add_argument(
        "--regex", action="append", default=[], metavar="LABEL=PATTERN",
        help="regex rule searched in each text (repeatable)",
    )
    parser.add_argument(
        "--query", action="append", default=[], metavar="LABEL=EXPR",
        help="pandas query rule for tabular input (repeatable)",
    )
    parser.add_argument(
        "--rules", metavar="FILE.json",
        help='JSON file mapping label -> keyword list | regex string | {"keywords"/"regex"/"query"/"weight"}',
    )
    parser.add_argument(
        "--column", metavar="COL",
        help="for .csv/.parquet input: label this column as text instead of the whole table",
    )
    parser.add_argument(
        "--encoding", metavar="NAME",
        help=(
            "encoding of the input file or stdin (default: UTF-8, with undecodable bytes "
            "replaced and a warning)"
        ),
    )
    parser.add_argument("--labels", metavar="A,B,C", help="closed set of allowed labels")
    parser.add_argument("--min-confidence", type=float, default=0.6, help="default 0.6")
    parser.add_argument("--random-state", type=int, default=0, help="seed for the model (default 0)")
    parser.add_argument("--no-model", action="store_true", help="use rules only, skip the model")
    parser.add_argument("--json", action="store_true", help="print the full result as JSON instead of the summary")
    parser.add_argument(
        "--output", metavar="PATH",
        help="write the labeled table to PATH (.csv, .parquet, or .json for the full result)",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return parser


def _split_spec(spec: str, flag: str) -> "tuple[str, str]":
    if "=" not in spec:
        raise ValueError(f"{flag} expects LABEL=VALUE, got {spec!r}")
    label, value = spec.split("=", 1)
    label, value = label.strip(), value.strip()
    if not label or not value:
        raise ValueError(f"{flag} expects LABEL=VALUE, got {spec!r}")
    return label, value


def _rules_from_file(path: str) -> Dict[str, Any]:
    name = os.path.basename(path) or path
    if os.path.isdir(path):
        raise ValueError(f"--rules file {name!r} is a directory, not a .json file")
    if not os.path.exists(path):
        raise FileNotFoundError(f"--rules file {name!r} does not exist")
    with open(path, "r", encoding="utf-8") as fh:
        try:
            loaded = json.load(fh)
        except json.JSONDecodeError as exc:
            raise ValueError(f"--rules file {name!r} is not valid JSON: {exc}") from None
    if not isinstance(loaded, dict):
        raise ValueError(
            f"--rules file {name!r} must contain a JSON object mapping label -> rule, "
            f"got {type(loaded).__name__}"
        )
    return loaded


def _add_file_rules(labeler: Labeler, rules: Dict[str, Any]) -> None:
    from auto_label.labeler import _rule_kwargs

    for name, spec in rules.items():
        for kwargs in _rule_kwargs(str(name), spec):
            if "func" in kwargs:
                raise ValueError(f"rules file: label {name!r} uses 'func', which cannot be loaded from JSON")
            labeler.add_rule(str(name), **kwargs)


def read_lines(lines: Any, origin: str) -> "tuple[pd.Series, List[str]]":
    """One item per non-blank line, indexed by its 0-based line number in ``origin``.

    Blank lines carry nothing to label, so they are skipped; keeping the real line
    number as the index means every record still points back at its source line, and
    the count of skipped lines is returned so the caller can report it.
    """
    values: List[str] = []
    index: List[int] = []
    skipped = 0
    for lineno, line in enumerate(lines):
        text = line.rstrip("\r\n")
        if text.strip():
            values.append(text)
            index.append(lineno)
        else:
            skipped += 1
    notes: List[str] = []
    if skipped:
        message = (
            f"input: skipped {skipped} blank line(s) in {origin}; the index of each record "
            "is its 0-based line number in the input"
        )
        logger.warning("%s", message)
        notes.append(message)
    return pd.Series(values, index=index, dtype=object), notes


def read_stdin_text(encoding: Optional[str] = None) -> "tuple[str, List[str]]":
    """Everything on stdin, decoded as UTF-8 (or ``encoding``), plus any notes.

    ``sys.stdin`` is opened with the console's locale encoding - cp1252 with
    ``errors="surrogateescape"`` on Windows - so reading it as text turns piped UTF-8
    into mojibake before any rule ever sees it, silently and with a zero exit code. The
    bytes underneath are what the user actually sent, so take those and decode them the
    same way a file is decoded. A stdin replaced by a text object (a test double, or an
    embedder) has no ``.buffer``; that text is already decoded, so it is used as it is.
    """
    stream = sys.stdin
    if stream is None:
        raise ValueError("there is no stdin to read; pass a file path instead of -")
    buffer = getattr(stream, "buffer", None)
    data = buffer.read() if buffer is not None else stream.read()
    if not isinstance(data, (bytes, bytearray)):
        return str(data), []
    return decode_text(bytes(data), encoding, "stdin")


def split_lines(text: str, origin: str) -> "tuple[pd.Series, List[str]]":
    """One item per non-blank line of ``text``.

    ``io.StringIO`` iterates exactly like a text file opened in universal-newline mode,
    so a trailing newline does not turn into an extra blank line in the skipped count.
    """
    return read_lines(io.StringIO(text), origin)


def load_input(
    path: str, column: Optional[str], encoding: Optional[str] = None
) -> "tuple[Any, List[str]]":
    """Turn the CLI input argument into what :meth:`Labeler.label` accepts, plus any notes."""
    if path == "-":
        if column is not None:
            raise ValueError("--column only applies to .csv/.parquet input")
        text, notes = read_stdin_text(encoding)
        series, line_notes = split_lines(text, "stdin")
        return series, notes + line_notes
    if os.path.isdir(path):
        raise ValueError(f"{path!r} is a directory, not a .csv/.parquet/text file")
    suffix = os.path.splitext(path)[1].lower()
    if suffix in TABLE_SUFFIXES:
        frame = read_table(path, encoding)
        if column is None:
            return frame, []
        if column not in frame.columns:
            raise ValueError(f"--column {column!r} not found; columns are {list(frame.columns)}")
        return frame[column], []
    if column is not None:
        raise ValueError("--column only applies to .csv/.parquet input")
    if not os.path.exists(path):
        raise FileNotFoundError(f"no such file: {path}")
    name = os.path.basename(path)
    with open(path, "rb") as fh:
        raw = fh.read()
    text, notes = decode_text(raw, encoding, name)
    series, line_notes = split_lines(text, name)
    return series, notes + line_notes


def _free_name(base: str, taken: set) -> str:
    """``base``, or ``base_2`` / ``base_3``... when the input already uses that column name."""
    name = base
    n = 2
    while name in taken:
        name = f"{base}_{n}"
        n += 1
    taken.add(name)
    return name


def output_frame(result: LabelResult) -> pd.DataFrame:
    """The table ``--output`` writes.

    For tabular input this is the original columns plus ``label, confidence, source``, so
    the file can be joined back to the input. ``result.to_frame()`` would put a Python
    dict repr in the ``item`` column instead, which is neither JSON nor joinable. Text
    input is written as ``item, label, confidence, source``.

    The index carries a name from the same free-name pool as the added columns, because
    the written index becomes a column of its own in the file. An input that already has
    a column called ``index`` would otherwise produce two headers with that name - a
    shape pandas silently renames on the way back in, and one this package's own
    :func:`~auto_label._data.check_columns` refuses to read at all.
    """
    items = list(result.items)
    index = pd.Index(result.index) if len(result.index) == len(result.labels) else None
    if result.mode != TABULAR or not items or not isinstance(items[0], dict):
        frame = result.to_frame()
        frame.index.name = _free_name("index", {str(c) for c in frame.columns})
        return frame
    frame = pd.DataFrame(items, index=index)
    taken = {str(c) for c in frame.columns}
    for base, values in (
        ("label", result.labels),
        ("confidence", result.confidence),
        ("source", result.source),
    ):
        frame[_free_name(base, taken)] = list(values)
    frame.index.name = _free_name("index", taken)
    return frame


def write_atomically(path: str, write: "Callable[[str], None]") -> None:
    """Run ``write(temporary_path)``, then move the finished file onto ``path``.

    Writing straight to the destination truncates whatever is already there before the
    first byte is written, so a failure part way through - an encode error, a full disk,
    a permission revoked on the stream - leaves the user with a header-only CSV or a
    JSON file that no longer parses, and their original data gone. Building the file
    beside the destination and renaming it means ``path`` only ever holds the old file
    or the complete new one, never a half-written mixture.
    """
    directory = os.path.dirname(os.path.abspath(path))
    if not os.path.isdir(directory):
        raise ValueError(f"--output {path!r}: the folder {directory!r} does not exist")
    suffix = os.path.splitext(path)[1]
    fd, tmp = tempfile.mkstemp(prefix=".auto-label-", suffix=suffix, dir=directory)
    os.close(fd)
    try:
        write(tmp)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.remove(tmp)
        except OSError:  # pragma: no cover - the temp file may already be gone
            pass
        raise


def write_output(result: LabelResult, path: str) -> None:
    """Write the result to ``path``; the suffix picks the format.

    The destination is replaced only once the whole file has been written successfully
    (see :func:`write_atomically`).
    """
    suffix = os.path.splitext(path)[1].lower()
    if suffix not in OUTPUT_SUFFIXES:
        raise ValueError(
            f"--output {path!r}: unsupported extension {suffix or '(none)'}; use "
            + ", ".join(OUTPUT_SUFFIXES)
        )

    if suffix == ".json":
        def write(target: str) -> None:
            with open(target, "w", encoding="utf-8") as fh:
                json.dump(result.to_dict(), fh, indent=2, ensure_ascii=False)
    else:
        frame = output_frame(result)
        if suffix == ".parquet":
            def write(target: str) -> None:
                try:
                    frame.to_parquet(target, index=True)
                except ImportError as exc:
                    raise ImportError(
                        "writing .parquet needs pyarrow: pip install 'auto-label[parquet]'"
                    ) from exc
        else:
            def write(target: str) -> None:
                frame.to_csv(target, index=True, index_label=frame.index.name, encoding="utf-8")

    try:
        write_atomically(path, write)
    except UnicodeEncodeError as exc:
        # Lone surrogates from bytes that were never valid UTF-8; the raw codec message
        # ("can't encode character '\\udc8f'") tells the user nothing they can act on.
        raise ValueError(
            f"--output {path!r} could not be written as UTF-8: the labeled text contains "
            f"bytes that are not valid UTF-8 ({exc.reason}). Re-run with --encoding set to "
            "the input's real encoding (for example --encoding cp1252), or pipe UTF-8. "
            f"{path!r} was left unchanged."
        ) from None


def main(argv: Optional[List[str]] = None) -> int:
    """Parse ``argv``, label the input, print the summary or JSON; returns the exit code."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):  # pragma: no cover - already-detached stream
                pass
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        labels = [s.strip() for s in args.labels.split(",") if s.strip()] if args.labels else None
        labeler = Labeler(labels=labels, min_confidence=args.min_confidence, random_state=args.random_state)
        if args.rules:
            _add_file_rules(labeler, _rules_from_file(args.rules))
        for spec in args.rule:
            name, value = _split_spec(spec, "--rule")
            labeler.add_rule(name, keywords=[kw.strip() for kw in value.split(",") if kw.strip()])
        for spec in args.regex:
            name, value = _split_spec(spec, "--regex")
            labeler.add_rule(name, regex=value)
        for spec in args.query:
            name, value = _split_spec(spec, "--query")
            labeler.add_rule(name, query=value)
        if not labeler.rules:
            parser.error("give at least one rule with --rule, --regex, --query or --rules")

        data, input_notes = load_input(args.input, args.column, args.encoding)
        result = labeler.label(data, model=not args.no_model)
        result.notes = list(input_notes) + result.notes

        if args.output:
            write_output(result, args.output)
        if args.json:
            print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))
        else:
            print(result.summary())
            if args.output:
                print(f"  wrote {args.output}")
        return 0
    except (ValueError, TypeError, FileNotFoundError, ImportError, OSError, json.JSONDecodeError) as exc:
        # library errors already say "auto_label: "; don't print the prefix twice
        message = str(exc)
        if message.startswith("auto_label: "):
            message = message[len("auto_label: "):]
        print(f"auto-label: error: {message}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
