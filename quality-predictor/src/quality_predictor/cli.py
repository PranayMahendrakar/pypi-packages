"""Command line interface: ``quality-predictor data.csv --target quality``."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd

from . import __version__
from ._api import predict_quality
from ._io import load_frame, write_frame


def _make_console_safe() -> None:
    """Never crash on a console that cannot encode a value from the data."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):  # pragma: no cover - closed or odd streams
                pass


def _parse_features(text: Optional[str]) -> Optional[List[str]]:
    """Turn ``--features a,b,c`` into a list of column names."""
    if text is None:
        return None
    picked = [part.strip() for part in text.split(",") if part.strip()]
    if not picked:
        raise ValueError("--features was given but named no columns")
    return picked


def build_parser() -> argparse.ArgumentParser:
    """The argparse parser for the ``quality-predictor`` command."""
    parser = argparse.ArgumentParser(
        prog="quality-predictor",
        description=(
            "Predict product quality from manufacturing parameters before final "
            "inspection, and see which settings drive it."
        ),
        epilog=(
            "Example: quality-predictor runs.csv --target quality --predict today.csv"
        ),
    )
    parser.add_argument("data", help="path to a .csv, .tsv or .parquet file of past runs")
    parser.add_argument(
        "--target",
        "-t",
        metavar="COLUMN",
        help="the outcome column (the last column is used when omitted)",
    )
    parser.add_argument(
        "--features",
        metavar="A,B,C",
        help="comma-separated parameter columns (every other column when omitted)",
    )
    parser.add_argument(
        "--task",
        choices=("auto", "classification", "regression"),
        default="auto",
        help="force pass/fail or numeric prediction instead of detecting it",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        metavar="FRACTION",
        help="fraction of rows held out for honest scores (default 0.2)",
    )
    parser.add_argument(
        "--random-state", type=int, default=0, metavar="N", help="seed, for repeatable runs"
    )
    parser.add_argument(
        "--predict",
        metavar="PATH",
        help="a second file of new rows to predict (the training rows otherwise)",
    )
    parser.add_argument(
        "--explain",
        type=int,
        metavar="ROW",
        help="also explain this row position of the predicted data",
    )
    parser.add_argument(
        "--top", type=int, default=8, metavar="N", help="how many parameters to list (default 8)"
    )
    parser.add_argument(
        "--predictions",
        metavar="PATH",
        help="write the predicted rows with their prediction to a .csv/.tsv/.parquet file",
    )
    parser.add_argument("--json", action="store_true", help="print to_dict() as JSON")
    parser.add_argument("--output", "-o", metavar="PATH", help="also write the result JSON to PATH")
    parser.add_argument("--version", action="version", version="%(prog)s {0}".format(__version__))
    return parser


def _write_json(payload: Dict[str, Any], path: str) -> None:
    """Write the result as UTF-8 JSON, creating the parent folder if needed."""
    target = Path(path)
    if str(target.parent) not in ("", "."):
        target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def _pick_target(frame: pd.DataFrame) -> str:
    """The last column, used when ``--target`` was not given."""
    if not len(frame.columns):
        raise ValueError("the data has no columns")
    return str(frame.columns[-1])


def _run(args: argparse.Namespace) -> int:
    """Do the work for one parsed command line."""
    frame = load_frame(args.data)
    notes: List[str] = []
    target = args.target
    if target is None:
        target = _pick_target(frame)
        notes.append(
            "no --target was given, so the last column ({0}) was used as the "
            "outcome".format(target)
        )
    if target not in frame.columns:
        known = ", ".join(str(c) for c in list(frame.columns)[:12])
        raise ValueError(
            "target column {0!r} is not in {1}; columns are: {2}".format(
                target, args.data, known
            )
        )

    new_data = load_frame(args.predict) if args.predict else None
    result = predict_quality(
        frame,
        target,
        new_data,
        features=_parse_features(args.features),
        task=args.task,
        test_size=args.test_size,
        random_state=args.random_state,
    )
    result.notes = notes + list(result.notes)

    predicted = new_data if new_data is not None else frame
    if args.predictions and result.predictions is not None:
        out = predicted.copy()
        out["predicted_{0}".format(target)] = result.predictions
        write_frame(out, args.predictions)
        print("predictions written to {0}".format(args.predictions), file=sys.stderr)

    payload = result.to_dict()
    explanation = None
    if args.explain is not None:
        if args.explain < 0 or args.explain >= len(predicted):
            raise ValueError(
                "--explain {0} is out of range: the predicted data has {1} row(s)".format(
                    args.explain, len(predicted)
                )
            )
        explanation = result.model.explain(predicted.iloc[[args.explain]])
        payload["explanation"] = explanation.to_dict()

    if args.output:
        _write_json(payload, args.output)
        print("result written to {0}".format(args.output), file=sys.stderr)

    if args.json:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    else:
        print(result.summary(top=args.top))
        if explanation is not None:
            print("")
            print(explanation.summary())
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Entry point for the ``quality-predictor`` command; returns the exit code."""
    _make_console_safe()
    args_list: List[str] = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()
    if not args_list:
        parser.print_help()
        return 2
    args = parser.parse_args(args_list)
    try:
        return _run(args)
    except (OSError, ValueError, TypeError, ImportError, KeyError) as exc:
        print("quality-predictor: error: {0}".format(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
