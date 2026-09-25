"""Command line front end: ``call-ai-metrics call.wav`` or ``call-ai-metrics segments.csv``.

Prints the same report the library returns. Exits 0 when there was speech to
measure, 1 when the call was read but held no speech, and 2 when the input could
not be read, so it drops straight into a shell loop over a folder of recordings.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Optional, Sequence

from . import __version__
from .core import CallAnalyzer

MEASURED = 0
"""Exit code when the call held speech and was measured."""
NO_SPEECH = 1
"""Exit code when the call was read but nobody spoke."""
BAD_INPUT = 2
"""Exit code when the input could not be read at all."""


def _make_utf8_safe() -> None:
    """Stop non-ASCII names and transcripts from crashing the output.

    Windows consoles, pipes and CI log capture often hand Python a cp1252 or
    ASCII stdout. Party names and file names go into the report, so a call
    between people named in Cyrillic or Chinese would raise UnicodeEncodeError
    the moment the output was piped - exactly when nobody is watching.
    """
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):  # pragma: no cover - exotic streams
                pass


def build_parser() -> argparse.ArgumentParser:
    """The argument parser, also used to render ``--help``."""
    parser = argparse.ArgumentParser(
        prog="call-ai-metrics",
        description=(
            "Measure talk time, interruptions, silence and pace in a two-party "
            "call. Give it a stereo .wav with one party per channel, or diarized "
            "segments as .csv/.tsv (start, end, speaker[, text]) or .json from any "
            "tool. Every figure is only as good as the speech segmentation behind "
            "it; on audio that is a per-channel energy heuristic."
        ),
        epilog=(
            "exit codes: 0 measured, 1 no speech in the call, 2 the input could not "
            "be read. examples: call-ai-metrics call.wav --speakers agent customer "
            "-- call-ai-metrics segments.csv --json --output report.json"
        ),
    )
    parser.add_argument(
        "source",
        nargs="?",
        help="a stereo .wav (one party per channel) or a .csv/.tsv/.json of segments",
    )
    parser.add_argument(
        "--speakers",
        nargs="+",
        metavar="NAME",
        help="names for the parties: one per channel for audio, or the expected parties for segments",
    )
    parser.add_argument(
        "--grace",
        type=float,
        default=0.5,
        metavar="S",
        help="overlap in seconds allowed before talking over someone is an interruption (default: 0.5)",
    )
    parser.add_argument(
        "--max-turn-pause",
        type=float,
        default=3.0,
        metavar="S",
        help="a pause longer than this ends a turn (default: 3.0)",
    )
    parser.add_argument("--json", action="store_true", help="print the full report as JSON")
    parser.add_argument(
        "--explain", action="store_true", help="also print how every figure was measured"
    )
    parser.add_argument(
        "--output",
        metavar="PATH",
        help="write the report to this file (UTF-8) as well as printing it",
    )
    parser.add_argument("--quiet", action="store_true", help="print nothing; only set the exit code")
    parser.add_argument(
        "--version", action="version", version="call-ai-metrics {}".format(__version__)
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run the command line tool.

    Args:
        argv: Arguments to parse; ``None`` reads ``sys.argv``.

    Returns:
        0 when the call was measured, 1 when it held no speech, 2 when the input
        could not be read.
    """
    _make_utf8_safe()
    parser = build_parser()
    args = parser.parse_args(argv)

    if not args.source:
        parser.print_usage(sys.stderr)
        print("call-ai-metrics: give one .wav, .csv, .tsv or .json file", file=sys.stderr)
        return BAD_INPUT

    try:
        analyzer = CallAnalyzer(grace_s=args.grace, max_turn_pause_s=args.max_turn_pause)
        report = analyzer.analyze(args.source, speakers=args.speakers)
    except (ValueError, OSError) as exc:
        print("call-ai-metrics: {}".format(exc), file=sys.stderr)
        return BAD_INPUT

    if args.json:
        text = json.dumps(report.to_dict(), indent=2, ensure_ascii=False)
    else:
        text = report.summary()
        if args.explain:
            text += "\n\n" + report.explain()

    if args.output:
        try:
            with open(args.output, "w", encoding="utf-8") as handle:
                handle.write(text + "\n")
        except OSError as exc:
            print("call-ai-metrics: cannot write {}: {}".format(args.output, exc), file=sys.stderr)
            return BAD_INPUT

    if not args.quiet:
        print(text)
    return MEASURED if report.talk_time_s > 0 else NO_SPEECH


if __name__ == "__main__":  # pragma: no cover - module entry point
    raise SystemExit(main())
