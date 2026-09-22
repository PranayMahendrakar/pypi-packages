"""Command line interface: ``model-benchmark MODEL [MODEL ...] [options]``.

A model on the command line is written ``name=module:attribute``; the current
directory is on the import path, so ``model-benchmark fast=mymodels:fast`` works
straight from a project folder. ``--demo`` runs a self-contained example that
needs no files at all.
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

from . import __version__
from ._core import DEFAULT_REPEATS, DEFAULT_WARMUP, benchmark
from ._metrics import METRICS
from ._result import BenchmarkReport

_DESCRIPTION = """Benchmark several models on the same task and compare latency,
memory and accuracy side by side.

Each MODEL is name=module:attribute, for example fast=mymodels:predict_fast.
The attribute is either a callable or an object with a .predict method."""

_EPILOG = """examples:
  model-benchmark --demo
  model-benchmark mymodels:fast mymodels:slow --data mymodels:SAMPLE
  model-benchmark linear=m:lin forest=m:rf --data rows.csv --target label --metric accuracy
  model-benchmark --demo --batch-sizes 1,16,64 --repeats 25
  model-benchmark --demo --json > benchmark.json
  model-benchmark --demo --output table.csv"""


# --------------------------------------------------------------------- demo
def _demo_always_zero(rows: Any) -> Any:
    import numpy as np

    return np.zeros(len(rows), dtype=int)


def _demo_threshold(rows: Any) -> Any:
    import numpy as np

    return (np.asarray(rows)[:, 0] > 0).astype(int)


def _demo_two_features(rows: Any) -> Any:
    import numpy as np

    array = np.asarray(rows)
    return (array[:, 0] + array[:, 1] > 0).astype(int)


def _demo_broken(rows: Any) -> Any:
    raise RuntimeError("this model is deliberately broken")


def demo_case() -> Tuple[Dict[str, Any], Tuple[Any, Any], str]:
    """The models, data and metric behind ``--demo``. Deterministic."""
    import numpy as np

    rng = np.random.default_rng(0)
    features = rng.normal(size=(200, 4))
    labels = (features[:, 0] + features[:, 1] > 0).astype(int)
    models: Dict[str, Any] = {
        "always_zero": _demo_always_zero,
        "threshold": _demo_threshold,
        "two_features": _demo_two_features,
        "broken": _demo_broken,
    }
    return models, (features, labels), "accuracy"


# ------------------------------------------------------------------ loading
def _import_object(spec: str) -> Any:
    """Import ``module:attribute`` and return the attribute."""
    if ":" not in spec:
        raise ValueError(
            f"{spec!r} is not module:attribute (for example mymodels:predict)"
        )
    module_name, _, attribute = spec.partition(":")
    module_name = module_name.strip()
    attribute = attribute.strip()
    if not module_name or not attribute:
        raise ValueError(f"{spec!r} is not module:attribute (for example mymodels:predict)")
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        raise ValueError(f"cannot import module {module_name!r}: {exc}") from exc
    target: Any = module
    for part in attribute.split("."):
        if not hasattr(target, part):
            raise ValueError(f"module {module_name!r} has no attribute {attribute!r}")
        target = getattr(target, part)
    return target


def parse_model_specs(specs: Sequence[str]) -> Dict[str, Any]:
    """Turn ``['name=module:attr', ...]`` into ``{name: object}``."""
    models: Dict[str, Any] = {}
    for spec in specs:
        colon = spec.find(":")
        equals = spec.find("=")
        if equals != -1 and (colon == -1 or equals < colon):
            name, _, reference = spec.partition("=")
            name = name.strip()
        else:
            name, reference = "", spec
        reference = reference.strip()
        obj = _import_object(reference)
        if not name:
            name = reference.partition(":")[2] or reference
        if name in models:
            raise ValueError(f"duplicate model name {name!r}; give each model its own name=")
        models[name] = obj
    if not models:
        raise ValueError("no models given; pass at least one name=module:attribute or --demo")
    return models


def _read_table(path: str) -> Any:
    import pandas as pd

    if path.lower().endswith((".parquet", ".pq")):
        return pd.read_parquet(path)
    return pd.read_csv(path)


def load_data(spec: Optional[str], target: Optional[str]) -> Tuple[Any, bool]:
    """Load ``--data``; returns (data, is_tuple_with_target)."""
    if spec is None:
        return None, False
    looks_like_a_file = spec.lower().endswith((".csv", ".parquet", ".pq"))
    if ":" in spec and not looks_like_a_file and not os.path.exists(spec):
        return _import_object(spec), False
    if not os.path.exists(spec):
        raise ValueError(f"no such data file: {spec}")
    frame = _read_table(spec)
    if target:
        if target not in frame.columns:
            columns = ", ".join(str(column) for column in frame.columns)
            raise ValueError(f"no column {target!r} in {spec}; it has {columns}")
        y_true = frame[target]
        x_data = frame.drop(columns=[target])
        return (x_data, y_true), True
    return frame, False


def parse_batch_sizes(text: Optional[str]) -> Any:
    """Parse ``--batch-sizes 1,16,all`` into a tuple."""
    if not text:
        return (1,)
    sizes: List[Any] = []
    for chunk in text.split(","):
        piece = chunk.strip()
        if not piece:
            continue
        if piece.lower() == "all":
            sizes.append("all")
            continue
        try:
            sizes.append(int(piece))
        except ValueError:
            raise ValueError(f"batch size {piece!r} is not a whole number or 'all'") from None
    if not sizes:
        raise ValueError("--batch-sizes is empty")
    return tuple(sizes)


# ------------------------------------------------------------------- parser
def build_parser() -> argparse.ArgumentParser:
    """The argument parser, exposed so tests and docs can read it."""
    parser = argparse.ArgumentParser(
        prog="model-benchmark",
        description=_DESCRIPTION,
        epilog=_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "models",
        metavar="MODEL",
        nargs="*",
        help="name=module:attribute for each model to benchmark",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="benchmark a built-in example instead, no files needed",
    )
    parser.add_argument(
        "--data",
        metavar="SPEC",
        help="module:attribute holding the input, or a .csv / .parquet file",
    )
    parser.add_argument(
        "--target",
        metavar="COL",
        help="column of --data holding the true labels, making the data (X, y_true)",
    )
    parser.add_argument(
        "--metric",
        metavar="NAME",
        help="score the predictions: " + ", ".join(sorted(METRICS)) + ", or module:attribute",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=DEFAULT_REPEATS,
        help=f"timed calls per batch size (default: {DEFAULT_REPEATS})",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=DEFAULT_WARMUP,
        help=f"untimed calls before the measured ones (default: {DEFAULT_WARMUP})",
    )
    parser.add_argument(
        "--batch-sizes",
        metavar="LIST",
        default="1",
        help="comma separated rows per timed call, or 'all' (default: 1)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        metavar="SECONDS",
        help="give up on a model after this long and record it as timed out",
    )
    parser.add_argument(
        "--best-by",
        choices=("latency", "score", "memory"),
        help="print only the winning model name, nothing else",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="how many models to list in the summary table (default: all)",
    )
    parser.add_argument(
        "--json", action="store_true", help="print the full report as JSON instead of the summary"
    )
    parser.add_argument(
        "--output",
        metavar="PATH",
        help="write the comparison table (.csv or .parquet) with one row per model",
    )
    parser.add_argument("--version", action="version", version=f"model-benchmark {__version__}")
    return parser


def _make_console_safe() -> None:
    """Never crash on a console that cannot encode a model name or a message."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except (ValueError, OSError):  # pragma: no cover - closed or unusual streams
            try:
                reconfigure(errors="replace")
            except (ValueError, OSError):
                pass


def write_output(path: str, report: BenchmarkReport) -> int:
    """Write the comparison table to `path`; returns the number of rows written."""
    frame = report.to_frame()
    if path.lower().endswith((".parquet", ".pq")):
        frame.to_parquet(path, index=False)
    else:
        frame.to_csv(path, index=False, encoding="utf-8")
    return int(len(frame))


def _resolve_metric_argument(metric: Optional[str]) -> Any:
    if metric is None:
        return None
    if metric.strip().lower() in METRICS:
        return metric.strip().lower()
    if ":" in metric:
        return _import_object(metric)
    known = ", ".join(sorted(METRICS))
    raise ValueError(f"unknown metric {metric!r}; use one of {known}, or module:attribute")


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Entry point; returns the process exit code."""
    _make_console_safe()
    parser = build_parser()
    args = parser.parse_args(argv)

    cwd = os.getcwd()
    if cwd not in sys.path:
        sys.path.insert(0, cwd)

    try:
        if args.demo:
            # --demo brings its own models, data and labels, so anything pointing at
            # data of your own would be silently dropped. Say so instead.
            clashing = [
                flag
                for flag, value in (("--data", args.data), ("--target", args.target))
                if value
            ]
            if clashing:
                parser.error(
                    f"--demo runs a built-in example and ignores {' and '.join(clashing)}; "
                    "drop --demo to benchmark your own models and data"
                )
            if args.models:
                parser.error(
                    "--demo runs a built-in example, so it cannot be combined with a "
                    "MODEL argument; drop one of the two"
                )
            models, data, metric = demo_case()
            if args.metric:
                metric = _resolve_metric_argument(args.metric)
        else:
            if not args.models:
                parser.error("give at least one MODEL as name=module:attribute, or use --demo")
            models = parse_model_specs(args.models)
            data, has_target = load_data(args.data, args.target)
            metric = _resolve_metric_argument(args.metric)
            if metric is not None and not has_target:
                if not (isinstance(data, tuple) and len(data) == 2):
                    parser.error(
                        "--metric needs true labels: add --target COL, or point --data at a "
                        "module:attribute holding the tuple (X, y_true)"
                    )

        report = benchmark(
            models,
            data,
            metric=metric,
            warmup=args.warmup,
            repeats=args.repeats,
            batch_sizes=parse_batch_sizes(args.batch_sizes),
            timeout=args.timeout,
        )

        rows = write_output(args.output, report) if args.output else 0
        if args.best_by:
            print(report.best(args.best_by))
        elif args.json:
            print(json.dumps(report.to_dict(), indent=2, ensure_ascii=False))
        else:
            print(report.summary(limit=args.limit))
            if args.output:
                print(f"  wrote     : {rows:,} rows to {args.output}")
        # A model that failed is part of the report, not a CLI failure. Only a run
        # where nothing at all could be measured is an error.
        return 0 if report.ok_results else 1
    except (ValueError, KeyError, TypeError, OSError, ImportError) as exc:
        print(f"model-benchmark: error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
