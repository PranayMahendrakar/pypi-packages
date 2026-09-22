"""Command line interface: ``rag-quality-check CASES [options]``."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from . import __version__
from ._core import METRIC_HELP, RagReport, evaluate

_DESCRIPTION = """Measure whether a retrieval system is actually retrieving the right things.

CASES is a .json file holding a list of case objects, a .jsonl file holding one
case object per line, or "-" to read either form from standard input. Each case
is {"query": ..., "retrieved": [...]} plus the optional "relevant", "answer" and
"ground_truth" keys."""

_EPILOG = "metrics:\n" + "\n".join(
    "  {0:<19} {1}".format(name, help_text) for name, help_text in METRIC_HELP.items()
) + """

examples:
  rag-quality-check cases.json
  rag-quality-check cases.jsonl --k 3 --threshold 0.4
  rag-quality-check cases.json --json > report.json
  cat cases.json | rag-quality-check - --weakest 10"""


def build_parser() -> argparse.ArgumentParser:
    """The argparse parser, exposed so tests and docs can read it."""
    parser = argparse.ArgumentParser(
        prog="rag-quality-check",
        description=_DESCRIPTION,
        epilog=_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "input",
        metavar="CASES",
        help='.json or .jsonl file of evaluation cases, or "-" for stdin',
    )
    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="cut-off for the retrieval metrics (default: 5)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="similarity at which two texts count as a match (default: 0.5)",
    )
    parser.add_argument(
        "--weakest",
        type=int,
        default=3,
        metavar="N",
        help="show the N worst cases in full under the summary (default: 3, 0 for none)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="as_json",
        help="print the full report as JSON instead of the summary",
    )
    parser.add_argument(
        "--output",
        metavar="PATH",
        help="write the report here (.json for to_dict(), otherwise the summary text)",
    )
    parser.add_argument(
        "--version",
        action="version",
        version="rag-quality-check {0}".format(__version__),
    )
    return parser


def parse_cases(raw: str, origin: str) -> List[Dict[str, Any]]:
    """Read cases from ``raw``: a JSON list, a JSON object, or one object per line.

    ``origin`` only names the source in error messages. A file whose whole body
    parses as JSON is used as-is; otherwise every non-blank line must be its own
    JSON object, which is the shape most evaluation harnesses log.
    """
    text = raw.strip()
    if not text:
        raise ValueError("{0} is empty; expected JSON cases".format(origin))
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        value = None
    if value is None:
        cases: List[Dict[str, Any]] = []
        for number, line in enumerate(text.splitlines(), start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                entry = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    "{0} line {1} is not valid JSON ({2})".format(origin, number, exc)
                ) from None
            if not isinstance(entry, dict):
                raise ValueError(
                    "{0} line {1} is not a JSON object".format(origin, number)
                )
            cases.append(entry)
        if not cases:
            raise ValueError("{0} held no cases".format(origin))
        return cases
    if isinstance(value, dict):
        return [value]
    if not isinstance(value, list):
        raise ValueError(
            "{0} must hold a list of case objects, got {1}".format(
                origin, type(value).__name__
            )
        )
    for position, entry in enumerate(value):
        if not isinstance(entry, dict):
            raise ValueError(
                "{0}: case {1} is not a JSON object".format(origin, position)
            )
    return value


def read_input(target: str) -> List[Dict[str, Any]]:
    """Load the cases named by ``target``; ``-`` reads standard input."""
    if target == "-":
        raw = sys.stdin.buffer.read().decode("utf-8", errors="replace")
        return parse_cases(raw, "standard input")
    path = Path(target)
    if not path.is_file():
        raise FileNotFoundError("{0!r} does not exist".format(target))
    return parse_cases(path.read_text(encoding="utf-8", errors="replace"), target)


def render_summary(report: RagReport, weakest: int) -> str:
    """The report summary plus the ``weakest`` worst cases in full."""
    lines = [report.summary()]
    shown = report.weakest(weakest) if weakest > 0 else []
    if shown:
        lines.append("")
        lines.append("worst {0} case(s) in detail:".format(len(shown)))
        for case in shown:
            lines.append(case.summary())
    return "\n".join(lines)


def write_output(path: str, report: RagReport, weakest: int) -> None:
    """Write the report to ``path``; a ``.json`` suffix picks ``to_dict()``."""
    if path.lower().endswith(".json"):
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(report.to_dict(), handle, indent=2, ensure_ascii=False)
            handle.write("\n")
        return
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(render_summary(report, weakest))
        handle.write("\n")


def _make_console_safe() -> None:
    """Never crash on a console or pipe that cannot encode a character from the data."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except (ValueError, OSError, LookupError):  # pragma: no cover - odd streams
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
        cases = read_input(args.input)
        report = evaluate(cases, k=args.k, threshold=args.threshold)
        if args.output:
            write_output(args.output, report, args.weakest)
        if args.as_json:
            print(json.dumps(report.to_dict(), indent=2, ensure_ascii=False))
        else:
            print(render_summary(report, args.weakest))
            if args.output:
                print("  wrote the report to {0}".format(args.output))
    except (ValueError, KeyError, TypeError, OSError, json.JSONDecodeError) as exc:
        print("rag-quality-check: error: {0}".format(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
