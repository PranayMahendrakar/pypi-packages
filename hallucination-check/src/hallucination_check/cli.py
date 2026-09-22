"""Command line interface: ``hallucination-check ANSWER [-s SOURCE ...] [options]``."""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, List, Optional, Sequence

from . import __version__
from ._core import GRANULARITIES, GroundingChecker

_DESCRIPTION = """Check an answer against the sources it claims to use and flag every
unsupported sentence.

ANSWER and each SOURCE may be a file path or the text itself; use "-" to read the
answer from stdin. A .jsonl source file is read one JSON object per line, each with a
"text" key and an optional "id", so retrieved passages can be piped straight in."""

_EPILOG = """examples:
  hallucination-check answer.txt -s passage1.txt -s passage2.txt
  hallucination-check "Paris is the capital of France." -s "France's capital is Paris."
  cat answer.txt | hallucination-check - -s retrieved.jsonl --json
  hallucination-check answer.txt -s docs.txt --fail-under 80"""


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser."""
    parser = argparse.ArgumentParser(
        prog="hallucination-check",
        description=_DESCRIPTION,
        epilog=_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("answer", metavar="ANSWER", help='answer file, literal text, or "-" for stdin')
    parser.add_argument(
        "-s", "--source", dest="sources", action="append", metavar="SOURCE", default=[],
        help="source file (.txt/.jsonl) or literal text; repeat for several sources",
    )
    parser.add_argument(
        "--threshold", type=float, default=0.55,
        help="support score in 0-1 at or above which a claim counts as supported (default: 0.55)",
    )
    parser.add_argument(
        "--granularity", choices=list(GRANULARITIES), default="sentence",
        help="what one claim is (default: sentence)",
    )
    parser.add_argument(
        "--no-fact-flags", dest="flag_missing_facts", action="store_false",
        help="do not flag a claim purely because a number, date or name is missing",
    )
    parser.add_argument("--json", action="store_true", help="print the full report as JSON")
    parser.add_argument("--output", metavar="PATH", help="write the JSON report to PATH")
    parser.add_argument(
        "--fail-under", type=float, metavar="SCORE", default=None,
        help="exit with status 2 when the score is below SCORE",
    )
    parser.add_argument("--version", action="version", version="hallucination-check %s" % __version__)
    return parser


def _read_text(path: str) -> str:
    with open(path, encoding="utf-8", errors="replace") as handle:
        return handle.read()


# Suffixes that make a value a file path and nothing else. A bare word with one of these
# on the end is never prose.
_PATH_SUFFIXES = (".txt", ".text", ".md", ".json", ".jsonl", ".ndjson")


def _check_not_a_typo(value: str, what: str) -> None:
    """Refuse a value that plainly means a file but names one that is not there.

    Both ANSWER and SOURCE accept either a path or the text itself, so a mistyped path
    would otherwise be checked as if it were the answer or the whole corpus: a confident
    wrong report, exit code 0, and under --fail-under a build that fails for a reason
    that has nothing to do with the answer.
    """
    if value.split() == [value] and value.lower().endswith(_PATH_SUFFIXES):
        raise ValueError(
            "%s %r looks like a file path but there is no such file; pass an existing "
            "file, or quote the text if you really meant it literally" % (what, value)
        )


def load_answer(value: str) -> str:
    """Read the answer from stdin, a file, or take the value as the text itself."""
    if value == "-":
        return sys.stdin.read()
    if os.path.isfile(value):
        return _read_text(value)
    _check_not_a_typo(value, "the answer")
    return value


def load_sources(values: Sequence[str]) -> List[Any]:
    """Turn each ``--source`` value into a source string or dict."""
    out: List[Any] = []
    for value in values:
        if value == "-":
            out.append(sys.stdin.read())
            continue
        if not os.path.isfile(value):
            _check_not_a_typo(value, "the source")
            out.append(value)
            continue
        text = _read_text(value)
        if value.lower().endswith((".jsonl", ".ndjson")):
            for number, line in enumerate(text.splitlines(), start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except ValueError as exc:
                    raise ValueError("%s line %d is not valid JSON: %s" % (value, number, exc))
                if isinstance(record, str):
                    out.append(record)
                elif isinstance(record, dict) and "text" in record:
                    out.append({"text": record["text"], "id": record.get("id")})
                else:
                    raise ValueError(
                        "%s line %d must be a string or an object with a 'text' key" % (value, number)
                    )
            continue
        out.append({"text": text, "id": os.path.basename(value)})
    return out


def _make_console_safe() -> None:
    """Make stdin, stdout and stderr UTF-8 tolerant.

    Under a legacy console codepage (cp1252 on Windows, ascii in a bare container)
    writing a non-Latin character raises UnicodeEncodeError and reading piped UTF-8
    raises UnicodeDecodeError. Both are the environment's fault, never the data's, so
    force UTF-8 and replace anything that still will not convert.
    """
    for stream in (sys.stdin, sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except (ValueError, OSError, LookupError):  # pragma: no cover - odd or closed streams
            try:
                reconfigure(errors="replace")
            except (ValueError, OSError):
                pass


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Entry point; returns the process exit code."""
    _make_console_safe()
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        answer = load_answer(args.answer)
        sources = load_sources(args.sources)
        checker = GroundingChecker(
            threshold=args.threshold,
            granularity=args.granularity,
            flag_missing_facts=args.flag_missing_facts,
        )
        report = checker.check(answer, sources)
        if args.output:
            with open(args.output, "w", encoding="utf-8") as handle:
                json.dump(report.to_dict(), handle, indent=2, ensure_ascii=False)
        if args.json:
            print(json.dumps(report.to_dict(), indent=2, ensure_ascii=False))
        else:
            print(report.summary())
            if args.output:
                print("  wrote the JSON report to %s" % args.output)
    except (ValueError, TypeError, OSError) as exc:
        print("hallucination-check: error: %s" % exc, file=sys.stderr)
        return 1
    if args.fail_under is not None and report.score < args.fail_under:
        print(
            "hallucination-check: score %.1f is below --fail-under %.1f"
            % (report.score, args.fail_under),
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
