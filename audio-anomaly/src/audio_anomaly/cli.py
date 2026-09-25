"""Command line interface: ``audio-anomaly RECORDING.wav [options]``."""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Optional, Sequence

import numpy as np

from . import __version__
from ._detect import detect, spectral_profile

_DESCRIPTION = """Find unusual sounds in a machine or environment recording.

RECORDING is a .wav file (8/16/24/32-bit PCM or 32/64-bit float, any number of
channels). Without --reference it is judged against its own typical sound; with
one, against a known-good recording or a saved profile."""

_EPILOG = """examples:
  audio-anomaly pump.wav
  audio-anomaly pump_today.wav --reference pump_healthy.wav
  audio-anomaly pump_healthy.wav --save-profile pump.npy
  audio-anomaly pump_today.wav --reference pump.npy --sensitivity 4
  audio-anomaly pump.wav --json > findings.json
  audio-anomaly pump.wav --output findings.json --fail-on-anomaly

exit status: 0 on success, 1 with --fail-on-anomaly when anything was found,
2 when the input could not be read or the results could not be written."""

_PROFILE_HEADER = "frequency_hz,level_db,spread_db"


def build_parser() -> argparse.ArgumentParser:
    """The argument parser, exposed so tests and docs can read it."""
    parser = argparse.ArgumentParser(
        prog="audio-anomaly",
        description=_DESCRIPTION,
        epilog=_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("recording", metavar="RECORDING", help="path to a .wav file")
    parser.add_argument(
        "--reference",
        metavar="PATH",
        help="known-good .wav, or a profile saved with --save-profile (.npy or .csv)",
    )
    parser.add_argument(
        "--sensitivity",
        type=float,
        default=3.0,
        help="score a frame must reach to be anomalous, in robust sigmas (each also "
        "needs 2 dB of real change); higher flags less (default: 3.0)",
    )
    parser.add_argument(
        "--frame-ms",
        type=float,
        default=50.0,
        dest="frame_ms",
        metavar="MS",
        help="analysis frame length in milliseconds (default: 50)",
    )
    parser.add_argument("--json", action="store_true", help="print the full report as JSON")
    parser.add_argument(
        "--output",
        metavar="PATH",
        help="also write the full report to PATH as JSON",
    )
    parser.add_argument(
        "--save-profile",
        metavar="PATH",
        dest="save_profile",
        help="instead of checking, save RECORDING's typical spectrum to PATH "
        "(.npy, or .csv) for use as --reference later",
    )
    parser.add_argument(
        "--fail-on-anomaly",
        action="store_true",
        dest="fail_on_anomaly",
        help="exit with status 1 when any anomaly is found (for scripts and cron)",
    )
    parser.add_argument("--version", action="version", version="%(prog)s " + __version__)
    return parser


def _load_reference(path: str) -> Any:
    """A saved profile for .npy/.csv paths, otherwise the path itself (a .wav)."""
    lower = path.lower()
    if lower.endswith(".npy"):
        return np.load(path, allow_pickle=False)
    if lower.endswith(".csv"):
        return np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2, encoding="utf-8")
    return path


def _save_profile(profile: np.ndarray, path: str) -> None:
    """Write a profile as .npy, or as UTF-8 CSV with a header for anything else."""
    if path.lower().endswith(".npy"):
        np.save(path, profile, allow_pickle=False)
        return
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(_PROFILE_HEADER + "\n")
        for row in profile:
            handle.write("%.3f,%.4f,%.4f\n" % (row[0], row[1], row[2]))


def _fail(message: str) -> int:
    print("audio-anomaly: error: %s" % message, file=sys.stderr)
    return 2


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Entry point for the ``audio-anomaly`` command.

    Every failure exits with status 2, never 1: status 1 means "anomalies found"
    under ``--fail-on-anomaly``, and a script must never mistake a crash for that.
    """
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        if args.save_profile:
            profile = spectral_profile(args.recording, frame_ms=args.frame_ms)
            _save_profile(profile, args.save_profile)
            print(
                "Saved the typical spectrum of %s (%d bands, %.0f-%.0f Hz) to %s"
                % (args.recording, profile.shape[0], profile[0, 0], profile[-1, 0], args.save_profile)
            )
            return 0
        reference = _load_reference(args.reference) if args.reference else None
        report = detect(
            args.recording,
            reference=reference,
            sensitivity=args.sensitivity,
            frame_ms=args.frame_ms,
        )
    except (OSError, ValueError, TypeError) as exc:
        return _fail(str(exc))
    except Exception as exc:  # anything else must still not look like exit status 1
        return _fail("could not analyse %s (%s: %s)" % (args.recording, type(exc).__name__, exc))

    data = report.to_dict()
    if args.reference:
        data["reference"] = os.fspath(args.reference)
    if args.output:
        try:
            with open(args.output, "w", encoding="utf-8") as handle:
                json.dump(data, handle, ensure_ascii=False, indent=2)
                handle.write("\n")
        except OSError as exc:
            return _fail("could not write the report to %s: %s" % (args.output, exc))
    if args.json:
        print(json.dumps(data, ensure_ascii=False, indent=2))
    else:
        print(report.summary())
        if args.output:
            print("\nFull report written to %s" % args.output)
    if args.fail_on_anomaly and report.has_anomalies:
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
