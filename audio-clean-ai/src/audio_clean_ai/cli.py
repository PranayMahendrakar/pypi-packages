"""Command line front end: ``audio-clean-ai noisy.wav --output clean.wav``.

Prints the same report the library returns and writes the cleaned audio when
``--output`` is given. Without ``--output`` it is a dry run: the report says
what would be removed, and nothing is written.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Optional, Sequence

from . import __version__
from ._audio import WRITE_BITS
from .core import SpectralGate

OK = 0
"""Exit code when the recording was read (and, with --output, written)."""
BAD_INPUT = 2
"""Exit code when the input could not be read or the output not written."""


def _make_utf8_safe() -> None:
    """Stop non-ASCII text from raising UnicodeEncodeError on any console.

    Windows consoles, pipes and CI log capture often hand Python a cp1252 or
    ASCII stdout. File names go into the report, so a recording called
    something in Japanese or Cyrillic would crash the tool rather than clean it,
    and only once the output was piped, which is exactly when nobody is watching.
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
        prog="audio-clean-ai",
        description=(
            "Remove steady background noise (hum, hiss, fans, air conditioning) "
            "from a speech recording, without a model. Spectral gating: the noise "
            "is learned from the quietest stretch of the recording, or from a "
            "noise-only clip, and turned down wherever the recording sits close to "
            "it. A heuristic: it does little for music, babble or a second speaker."
        ),
        epilog=(
            "exit codes: 0 done, 2 the input could not be read or the output not "
            "written. examples: audio-clean-ai noisy.wav -- "
            "audio-clean-ai noisy.wav --output clean.wav -- "
            "audio-clean-ai noisy.wav --output clean.wav --strength 0.6 --json -- "
            "audio-clean-ai noisy.wav --noise-clip room.wav --output clean.wav"
        ),
    )
    parser.add_argument(
        "recording",
        nargs="?",
        help="path to a .wav file (PCM or IEEE float, 8 to 48 kHz, mono or multi-channel)",
    )
    parser.add_argument(
        "--output",
        "-o",
        metavar="PATH",
        help="write the cleaned audio to this .wav file",
    )
    parser.add_argument(
        "--strength",
        type=float,
        default=0.8,
        metavar="N",
        help="0 to 1, how hard to cut; 0 leaves the audio untouched (default: 0.8)",
    )
    parser.add_argument(
        "--no-preserve-speech",
        action="store_true",
        help="let the 300-3400 Hz speech band be cut as deep as everything else",
    )
    parser.add_argument(
        "--noise-clip",
        metavar="PATH",
        help="a .wav of the noise alone (same sample rate) to learn the profile from",
    )
    parser.add_argument(
        "--bits",
        type=int,
        choices=WRITE_BITS,
        metavar="N",
        help="PCM depth of --output: 8, 16, 24 or 32 (default: the input's, else 16)",
    )
    parser.add_argument(
        "--json", action="store_true", help="print the full report as JSON"
    )
    parser.add_argument(
        "--report",
        metavar="PATH",
        help="also write the report (text, or JSON with --json) to this file, UTF-8",
    )
    parser.add_argument(
        "--quiet", action="store_true", help="print nothing; only set the exit code"
    )
    parser.add_argument(
        "--version",
        action="version",
        version="audio-clean-ai {}".format(__version__),
    )
    return parser


def _fail(message: str) -> int:
    """Print an error the same way every time and return the failure code."""
    print("audio-clean-ai: {}".format(message), file=sys.stderr)
    return BAD_INPUT


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run the command line tool.

    Args:
        argv: Arguments to parse; ``None`` reads ``sys.argv``.

    Returns:
        0 on success, 2 when the input could not be read or the output written.
    """
    _make_utf8_safe()
    parser = build_parser()
    args = parser.parse_args(argv)

    if not args.recording:
        parser.print_usage(sys.stderr)
        return _fail("give one .wav file to clean")

    try:
        gate = SpectralGate(
            strength=args.strength, preserve_speech=not args.no_preserve_speech
        )
        profile = None
        if args.noise_clip:
            profile = gate.learn_profile(args.noise_clip)
        result = gate.clean(args.recording, noise_profile=profile)
    except (ValueError, FileNotFoundError, OSError) as exc:
        return _fail(str(exc))

    if args.output:
        try:
            result.save(args.output, bits=args.bits)
        except (ValueError, OSError) as exc:
            return _fail("cannot write {}: {}".format(args.output, exc))

    if args.json:
        payload = result.to_dict()
        payload["output"] = args.output
        text = json.dumps(payload, indent=2, ensure_ascii=False)
    else:
        text = result.summary()
        if args.output:
            text += "\n  written to {}".format(args.output)

    if args.report:
        try:
            with open(args.report, "w", encoding="utf-8") as handle:
                handle.write(text + "\n")
        except OSError as exc:
            return _fail("cannot write {}: {}".format(args.report, exc))

    if not args.quiet:
        print(text)
        if not args.output:
            print(
                "audio-clean-ai: dry run, nothing written; add --output clean.wav "
                "to save the cleaned audio",
                file=sys.stderr,
            )
    return OK


if __name__ == "__main__":  # pragma: no cover - module entry point
    raise SystemExit(main())
