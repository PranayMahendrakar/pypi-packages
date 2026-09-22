"""Command line interface: ``meeting-intelligence TRANSCRIPT [options]``.

The input is a transcript file -- ``.txt``, ``.vtt``, ``.srt`` or ``.md`` -- or
``-`` to read the transcript from standard input. No audio is read here or
anywhere else in this package.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, List, Optional, Sequence

from . import __version__
from ._analyse import analyse
from ._report import MeetingReport

__all__ = ["build_parser", "main", "render"]

_DESCRIPTION = """Turn a meeting transcript into decisions, action items and a summary.

TRANSCRIPT is a .txt, .vtt, .srt or .md file, or "-" to read the transcript
from standard input. This reads text only: it never opens or decodes audio."""

_EPILOG = """examples:
  meeting-intelligence standup.vtt
  meeting-intelligence notes.txt --speakers "Alice,Bob Chen"
  meeting-intelligence notes.txt --json > report.json
  meeting-intelligence notes.txt --markdown --output minutes.md
  cat notes.txt | meeting-intelligence - --actions"""


def build_parser() -> argparse.ArgumentParser:
    """The argparse parser, exposed so tests and docs can read it."""
    parser = argparse.ArgumentParser(
        prog="meeting-intelligence",
        description=_DESCRIPTION,
        epilog=_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "transcript",
        metavar="TRANSCRIPT",
        help='transcript file (.txt/.vtt/.srt/.md), or "-" for standard input',
    )
    parser.add_argument(
        "--speakers",
        metavar="NAMES",
        help="comma-separated roster used to resolve action owners, "
        'e.g. --speakers "Alice,Bob Chen"',
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="as_json",
        help="print the whole report as JSON instead of the text summary",
    )
    parser.add_argument(
        "--markdown",
        action="store_true",
        dest="as_markdown",
        help="print the report as Markdown minutes",
    )
    parser.add_argument(
        "--actions",
        action="store_true",
        help="print only the action items, one per line",
    )
    parser.add_argument(
        "--decisions",
        action="store_true",
        help="print only the decisions, one per line",
    )
    parser.add_argument(
        "--output",
        metavar="PATH",
        help="also write the report here (.json, .md, otherwise the text summary)",
    )
    parser.add_argument(
        "--version",
        action="version",
        version="meeting-intelligence %s" % __version__,
    )
    return parser


def parse_speakers(raw: Optional[str]) -> List[str]:
    """Split ``--speakers`` on commas into a clean roster."""
    if not raw:
        return []
    names = []  # type: List[str]
    for piece in raw.split(","):
        name = piece.strip()
        if name and name not in names:
            names.append(name)
    return names


def read_input(target: str) -> Any:
    """Return the transcript text (for ``-``) or the path to read."""
    if target == "-":
        data = sys.stdin.buffer.read()
        return data.decode("utf-8-sig", errors="replace")
    path = Path(target)
    if not path.is_file():
        raise FileNotFoundError("transcript file not found: %r" % target)
    return path


def render_actions(report: MeetingReport) -> str:
    """One line per action item, or a plain note when there are none."""
    if not report.action_items:
        return "no action items found"
    return "\n".join(
        "[%.2f] %s" % (action.confidence, action.describe())
        for action in report.action_items
    )


def render_decisions(report: MeetingReport) -> str:
    """One line per decision, or a plain note when there are none."""
    if not report.decisions:
        return "no decisions found"
    lines = []  # type: List[str]
    for decision in report.decisions:
        who = " -- %s" % decision.speaker if decision.speaker else ""
        lines.append("[%.2f] %s%s" % (decision.confidence, decision.text, who))
    return "\n".join(lines)


def render(report: MeetingReport, args: argparse.Namespace) -> str:
    """Pick the rendering the flags asked for."""
    if args.as_json:
        return json.dumps(report.to_dict(), indent=2, ensure_ascii=False)
    if args.as_markdown:
        return report.to_markdown().rstrip("\n")
    if args.actions:
        return render_actions(report)
    if args.decisions:
        return render_decisions(report)
    return report.summary()


def write_output(path: str, report: MeetingReport) -> None:
    """Write the report to ``path``; the suffix picks the format."""
    lowered = path.lower()
    if lowered.endswith(".json"):
        body = json.dumps(report.to_dict(), indent=2, ensure_ascii=False)
    elif lowered.endswith((".md", ".markdown")):
        body = report.to_markdown().rstrip("\n")
    else:
        body = report.summary()
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(body)
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
        report = analyse(read_input(args.transcript), speakers=parse_speakers(args.speakers))
        if args.output:
            write_output(args.output, report)
        print(render(report, args))
        if args.output and not (args.as_json or args.as_markdown):
            print("  wrote the report to %s" % args.output)
    except (ValueError, TypeError, OSError) as error:
        print("meeting-intelligence: error: %s" % error, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
