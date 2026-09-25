"""Command line interface: ``speaker-diarize-lite RECORDING.wav [options]``."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Optional, Sequence

from . import __version__
from ._diarize import Diarizer

_DESCRIPTION = """Work out who spoke when in a recording.

RECORDING is a .wav file (8/16/24/32-bit PCM or 32/64-bit float; stereo and
multichannel audio is mixed to mono). The default method is a lightweight
heuristic: it separates clearly different voices and can merge similar ones."""

_EPILOG = """examples:
  speaker-diarize-lite meeting.wav
  speaker-diarize-lite meeting.wav --speakers 2
  speaker-diarize-lite meeting.wav --rttm > meeting.rttm
  speaker-diarize-lite meeting.wav --json
  speaker-diarize-lite meeting.wav --output meeting.json
  speaker-diarize-lite meeting.wav --output meeting.rttm

exit status: 0 on success, 2 when the recording could not be read or the
results could not be written."""


def build_parser() -> argparse.ArgumentParser:
    """The argument parser, exposed so tests and docs can read it."""
    parser = argparse.ArgumentParser(
        prog="speaker-diarize-lite",
        description=_DESCRIPTION,
        epilog=_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("recording", metavar="RECORDING", help="path to a .wav file")
    parser.add_argument(
        "--speakers",
        type=int,
        default=None,
        metavar="N",
        help="how many speakers to find (default: estimate from the recording)",
    )
    parser.add_argument(
        "--max-speakers",
        type=int,
        default=8,
        dest="max_speakers",
        metavar="N",
        help="most speakers to consider when estimating (default: 8)",
    )
    parser.add_argument(
        "--min-segment",
        type=float,
        default=0.5,
        dest="min_segment_s",
        metavar="SECONDS",
        help="shortest turn reported (default: 0.5)",
    )
    parser.add_argument(
        "--file-id",
        default=None,
        dest="file_id",
        metavar="NAME",
        help="recording name used in RTTM output (default: the file name without extension)",
    )
    output = parser.add_mutually_exclusive_group()
    output.add_argument("--json", action="store_true", help="print the full result as JSON")
    output.add_argument("--rttm", action="store_true", help="print the segments as RTTM")
    parser.add_argument(
        "--output",
        metavar="PATH",
        help="also write the result to PATH: RTTM if it ends in .rttm, JSON otherwise",
    )
    parser.add_argument("--version", action="version", version="%(prog)s " + __version__)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run the command line tool; returns the exit status."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        diarizer = Diarizer(
            num_speakers=args.speakers,
            min_segment_s=args.min_segment_s,
            max_speakers=args.max_speakers,
        )
    except ValueError as exc:
        parser.error(str(exc))
        return 2
    try:
        result = diarizer.diarize(args.recording)
    except (OSError, ValueError) as exc:
        print("speaker-diarize-lite: error: {}".format(exc), file=sys.stderr)
        return 2
    if args.file_id is not None:
        result.file_id = "_".join(args.file_id.split()) or "audio"

    if args.json:
        print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))
    elif args.rttm:
        sys.stdout.write(result.to_rttm())
    else:
        print(result.summary())

    if args.output:
        try:
            with open(args.output, "w", encoding="utf-8", newline="\n") as handle:
                if args.output.lower().endswith(".rttm"):
                    handle.write(result.to_rttm())
                else:
                    json.dump(result.to_dict(), handle, indent=2, ensure_ascii=False)
                    handle.write("\n")
        except OSError as exc:
            print("speaker-diarize-lite: error: cannot write {}: {}".format(args.output, exc), file=sys.stderr)
            return 2
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
