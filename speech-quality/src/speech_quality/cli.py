"""Command line front end: ``speech-quality interview.wav``.

Prints the same report the library returns and sets an exit code, so a whole
folder of recordings can be sorted into usable and not from a shell loop without
anything reading the text.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Dict, List, Optional, Sequence

from . import __version__
from .core import assess, assess_batch
from .report import BatchReport

__all__ = ["BAD_USAGE", "NOT_USABLE", "USABLE", "build_parser", "main"]

USABLE = 0
"""Exit code when every recording given is good enough to use."""
NOT_USABLE = 1
"""Exit code when at least one recording is not good enough."""
BAD_USAGE = 2
"""Exit code when nothing could be read, or the arguments made no sense."""

_OVERRIDES = (
    ("target_dbfs", "target_rms_dbfs", float, "the RMS level a healthy recording sits at"),
    ("min_snr", "min_snr_db", float, "signal-to-noise under this is called out"),
    ("min_bandwidth", "min_bandwidth_hz", float, "bandwidth in Hz under this is called out"),
    ("usable_score", "usable_score", float, "overall score at or above which a file passes"),
)
"""``--flag`` names mapped to the threshold each one moves."""


def _make_utf8_safe() -> None:
    """Stop non-ASCII text from raising UnicodeEncodeError on any console.

    Windows consoles, pipes and CI log capture often hand Python a cp1252 or
    ASCII stdout. File names go into the report, so a recording named in
    Japanese or Cyrillic would crash the tool rather than report on it - and the
    crash would only show up once the output was piped, which is exactly when
    nobody is watching.
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):  # pragma: no cover - exotic streams
                pass


def build_parser() -> argparse.ArgumentParser:
    """The argument parser, also used to render ``--help``."""
    parser = argparse.ArgumentParser(
        prog="speech-quality",
        description=(
            "Measure whether a voice recording is clean enough to transcribe or "
            "publish. Reports level, clipping, noise, silence, speech share, "
            "dynamics and bandwidth, each with a raw number, a 0-100 score and a "
            "sentence you can act on. These are signal measurements, not a "
            "perceptual model."
        ),
        epilog=(
            "exit codes: 0 every recording is usable, 1 at least one is not, 2 "
            "nothing could be read. "
            "examples: speech-quality interview.wav -- "
            "speech-quality takes/*.wav --worst 3 -- "
            "speech-quality interview.wav --json --output report.json -- "
            "speech-quality interview.wav --min-snr 20 --usable-score 70"
        ),
    )
    parser.add_argument(
        "recordings",
        nargs="*",
        metavar="RECORDING",
        help="one or more .wav files (PCM 8, 16, 24 or 32 bit, any sample rate)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="print the full report as JSON instead of text",
    )
    parser.add_argument(
        "--output",
        metavar="PATH",
        help="write the report to this file (UTF-8) as well as printing it",
    )
    parser.add_argument(
        "--worst",
        type=int,
        default=5,
        metavar="N",
        help="with several recordings, how many low scorers to list (default: 5)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="print nothing; only set the exit code",
    )
    for flag, name, kind, help_text in _OVERRIDES:
        parser.add_argument(
            "--{}".format(flag.replace("_", "-")),
            dest=name,
            type=kind,
            default=None,
            metavar="N",
            help=help_text,
        )
    parser.add_argument(
        "--version",
        action="version",
        version="speech-quality {}".format(__version__),
    )
    return parser


def _overrides_from(args: argparse.Namespace) -> Dict[str, Any]:
    """The thresholds the caller moved on the command line."""
    chosen: Dict[str, Any] = {}
    for _flag, name, _kind, _help_text in _OVERRIDES:
        value = getattr(args, name, None)
        if value is not None:
            chosen[name] = value
    return chosen


def _render(result: Any, as_json: bool, worst: int) -> str:
    """The text to print for one report or a batch of them."""
    if as_json:
        return json.dumps(result.to_dict(), indent=2, ensure_ascii=False)
    if isinstance(result, BatchReport):
        parts: List[str] = [result.summary(worst=worst)]
        for report in result.results:
            parts.append("")
            parts.append(report.summary())
        return "\n".join(parts)
    return result.summary()


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run the command line tool.

    Args:
        argv: arguments to parse; ``None`` reads ``sys.argv``.

    Returns:
        0 when every recording is usable, 1 when at least one is not, 2 when
        nothing could be read or the arguments made no sense.
    """
    _make_utf8_safe()
    parser = build_parser()
    args = parser.parse_args(argv)

    if not args.recordings:
        parser.print_usage(sys.stderr)
        print("speech-quality: give at least one .wav file to assess", file=sys.stderr)
        return BAD_USAGE

    overrides = _overrides_from(args)
    result: Any
    if len(args.recordings) == 1:
        try:
            result = assess(args.recordings[0], thresholds=overrides or None)
        except (FileNotFoundError, ValueError, TypeError, OSError) as exc:
            print("speech-quality: {}".format(exc), file=sys.stderr)
            return BAD_USAGE
        everything_usable = bool(result.usable)
        read_anything = True
    else:
        try:
            result = assess_batch(args.recordings, thresholds=overrides or None)
        except (ValueError, TypeError) as exc:
            print("speech-quality: {}".format(exc), file=sys.stderr)
            return BAD_USAGE
        read_anything = bool(result.results)
        everything_usable = read_anything and not result.unusable and not result.failures

    text = _render(result, args.json, args.worst)

    if args.output:
        try:
            with open(args.output, "w", encoding="utf-8") as handle:
                handle.write(text + "\n")
        except OSError as exc:
            print(
                "speech-quality: cannot write {}: {}".format(args.output, exc),
                file=sys.stderr,
            )
            return BAD_USAGE

    if not args.quiet:
        print(text)

    if not read_anything:
        return BAD_USAGE
    return USABLE if everything_usable else NOT_USABLE


if __name__ == "__main__":  # pragma: no cover - module entry point
    raise SystemExit(main())
