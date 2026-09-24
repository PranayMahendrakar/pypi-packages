"""Command line front end: ``voice-activity-ai recording.wav``.

Prints the same report the library returns, and exits non-zero when there is no
speech in the file, so it drops straight into a shell loop that is sorting real
recordings from ones that are nothing but room tone.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Optional, Sequence

from . import __version__
from ._audio import write_wav
from .core import detect

SPEECH_FOUND = 0
"""Exit code when at least one speech segment was found."""
NO_SPEECH = 1
"""Exit code when the recording was read but held no speech."""
BAD_USAGE = 2
"""Exit code when the input could not be read at all."""


def _make_utf8_safe() -> None:
    """Stop non-ASCII text from raising UnicodeEncodeError on any console.

    Windows consoles, pipes and CI log capture often hand Python a cp1252 or
    ASCII stdout. The file name goes into the report, so a recording called
    something in Japanese or Cyrillic would crash the tool rather than report on
    it - and the crash would only appear once the output was piped, which is
    exactly when nobody is watching.
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
        prog="voice-activity-ai",
        description=(
            "Find where someone is actually speaking in a recording, and where "
            "it is just noise. Prints the speech segments with timestamps, how "
            "much of the file is speech, and can write the trimmed audio back "
            "out. An energy and spectral heuristic, not a neural detector."
        ),
        epilog=(
            "exit codes: 0 speech was found, 1 the file holds no speech, 2 the "
            "input could not be read. "
            "examples: voice-activity-ai meeting.wav -- "
            "voice-activity-ai meeting.wav --json --output segments.json -- "
            "voice-activity-ai meeting.wav --trim clean.wav --sensitivity 0.7"
        ),
    )
    parser.add_argument(
        "recording",
        nargs="?",
        help="path to a .wav file (PCM or IEEE float, any sample rate, mono or multi-channel)",
    )
    parser.add_argument(
        "--frame-ms",
        type=float,
        default=30.0,
        metavar="MS",
        help="analysis frame length, and so the resolution of every boundary (default: 30)",
    )
    parser.add_argument(
        "--sensitivity",
        type=float,
        default=0.5,
        metavar="N",
        help="0 to 1; higher calls more of the recording speech (default: 0.5)",
    )
    parser.add_argument(
        "--min-speech-ms",
        type=float,
        default=200.0,
        metavar="MS",
        help="speech runs shorter than this are dropped (default: 200)",
    )
    parser.add_argument(
        "--min-silence-ms",
        type=float,
        default=300.0,
        metavar="MS",
        help="gaps inside speech shorter than this are bridged (default: 300)",
    )
    parser.add_argument(
        "--trim",
        metavar="PATH",
        help="write the audio with leading and trailing silence removed to this .wav",
    )
    parser.add_argument(
        "--json", action="store_true", help="print the full result as JSON"
    )
    parser.add_argument(
        "--output",
        metavar="PATH",
        help="write the report to this file (UTF-8) as well as printing it",
    )
    parser.add_argument(
        "--quiet", action="store_true", help="print nothing; only set the exit code"
    )
    parser.add_argument(
        "--version",
        action="version",
        version="voice-activity-ai {}".format(__version__),
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run the command line tool.

    Args:
        argv: Arguments to parse; ``None`` reads ``sys.argv``.

    Returns:
        0 when speech was found, 1 when the recording holds none, 2 when the
        input could not be read.
    """
    _make_utf8_safe()
    parser = build_parser()
    args = parser.parse_args(argv)

    if not args.recording:
        parser.print_usage(sys.stderr)
        print(
            "voice-activity-ai: give one .wav file to analyse",
            file=sys.stderr,
        )
        return BAD_USAGE

    try:
        activity = detect(
            args.recording,
            frame_ms=args.frame_ms,
            sensitivity=args.sensitivity,
            min_speech_ms=args.min_speech_ms,
            min_silence_ms=args.min_silence_ms,
        )
    except (ValueError, FileNotFoundError, OSError) as exc:
        print("voice-activity-ai: {}".format(exc), file=sys.stderr)
        return BAD_USAGE

    if args.json:
        text = json.dumps(activity.to_dict(), indent=2, ensure_ascii=False)
    else:
        text = activity.summary()

    if args.output:
        try:
            with open(args.output, "w", encoding="utf-8") as handle:
                handle.write(text + "\n")
        except OSError as exc:
            print(
                "voice-activity-ai: cannot write {}: {}".format(args.output, exc),
                file=sys.stderr,
            )
            return BAD_USAGE

    if args.trim:
        trimmed = activity.trim()
        if trimmed.size == 0:
            print(
                "voice-activity-ai: no speech found, so {} was not written".format(
                    args.trim
                ),
                file=sys.stderr,
            )
        else:
            try:
                write_wav(args.trim, trimmed, activity.sample_rate)
            except OSError as exc:
                print(
                    "voice-activity-ai: cannot write {}: {}".format(args.trim, exc),
                    file=sys.stderr,
                )
                return BAD_USAGE

    if not args.quiet:
        print(text)
    return SPEECH_FOUND if activity.n_segments else NO_SPEECH


if __name__ == "__main__":  # pragma: no cover - module entry point
    raise SystemExit(main())
