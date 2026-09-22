"""Command line interface: ``ml-inference-profiler [report.json] [--demo]``.

Profiling itself happens in your own code, so the command line has two jobs: show a
worked example with ``--demo``, and read back a report saved with
``ProfileReport.save(path)`` so it can be printed, re-rendered as a tree, or converted to
JSON in a shell pipeline.
"""

from __future__ import annotations

import argparse
import sys
from typing import List, Optional, Sequence

from . import __version__
from .profiler import Profiler
from .report import ProfileReport, load_report


def build_parser() -> argparse.ArgumentParser:
    """The argparse parser for the ``ml-inference-profiler`` command."""
    parser = argparse.ArgumentParser(
        prog="ml-inference-profiler",
        description=(
            "Find the slow step in an ML inference pipeline. Print a report saved with"
            " ProfileReport.save(path), or run --demo to see what a report looks like."
        ),
        epilog=(
            "Examples: ml-inference-profiler --demo | less  |  "
            "ml-inference-profiler report.json --json > report-flat.json"
        ),
    )
    parser.add_argument(
        "report",
        nargs="?",
        metavar="REPORT",
        help="path to a JSON report written by ProfileReport.save()",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="profile a small built-in pipeline and print its report",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="print the report as JSON (to_dict()) instead of the summary",
    )
    parser.add_argument(
        "--tree",
        action="store_true",
        help="print only the stage tree",
    )
    parser.add_argument(
        "--output",
        "-o",
        metavar="PATH",
        help="also write the report JSON to PATH (UTF-8)",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=3,
        metavar="N",
        help=(
            "timed passes for --demo (default 3); ignored when a REPORT file is given,"
            " because that report was already timed"
        ),
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Entry point for the ``ml-inference-profiler`` command; returns the exit code."""
    _make_console_safe()
    args_list: List[str] = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()
    if not args_list:
        parser.print_help()
        return 2
    args = parser.parse_args(args_list)
    if args.report and args.demo:
        print(
            "ml-inference-profiler: error: give a report file or --demo, not both",
            file=sys.stderr,
        )
        return 2
    if not args.report and not args.demo:
        parser.print_help()
        return 2
    try:
        report = demo_report(repeats=args.repeats) if args.demo else load_report(args.report)
        if args.output:
            report.save(args.output)
            print(f"report written to {args.output}", file=sys.stderr)
        if args.json:
            print(report.to_json())
        elif args.tree:
            print(report.tree())
        else:
            print(report.summary())
        return 0
    except (OSError, ValueError, TypeError) as exc:
        print(f"ml-inference-profiler: error: {exc}", file=sys.stderr)
        return 1


def demo_report(repeats: int = 3) -> ProfileReport:
    """Profile a small built-in pipeline: a shape a real image classifier has.

    The stage labels are deliberately not ASCII, and the model stage builds its weights on
    the first call, so the demo shows off nested stages, a cold cache and the suggestions
    that follow from them.
    """
    import numpy as np

    if repeats < 1:
        raise ValueError(f"repeats must be >= 1, got {repeats}")

    rng = np.random.default_rng(0)
    batch = rng.random((384, 64))
    weights: dict = {}
    profiler = Profiler(name="démo: classifieur d'images")

    def preprocess(x):
        with profiler.stage("mise à l'échelle"):
            x = (x - x.mean()) / (x.std() + 1e-9)
        norms = []
        for row in x:  # per-item work: exactly what the batching advice is about
            with profiler.stage("normalisation par élément"):
                norms.append(float(np.linalg.norm(row)))
        return x / (np.asarray(norms).reshape(-1, 1) + 1e-9)

    def model(x):
        if "w" not in weights:  # cold cache: the first call pays for the weights
            with profiler.stage("chargement des poids (cache froid)"):
                weights["w"] = rng.random((64, 512))
                _ = x @ weights["w"]
        return x @ weights["w"]

    def postprocess(x):
        shifted = np.exp(x - x.max(axis=1, keepdims=True))
        return (shifted / shifted.sum(axis=1, keepdims=True)).argmax(axis=1)

    steps = [
        ("prétraitement", preprocess),
        ("modèle", model),
        ("post-traitement", postprocess),
    ]
    return profiler.run(steps, batch, repeats=repeats, warmup=0)


def _make_console_safe() -> None:
    """Never crash on a console that cannot encode a stage label."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):  # pragma: no cover - closed or odd streams
                pass


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
