"""The objects a benchmark hands back: Timing, ModelResult, BenchmarkReport.

Everything here is a plain dataclass. ``summary()`` is human text in plain ASCII,
``to_dict()`` is JSON safe (no NaN, no infinities), and ``to_frame()`` is a
pandas DataFrame with one row per model.
"""
from __future__ import annotations

import math
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

__all__ = ["BenchmarkReport", "ModelResult", "Timing"]

_NAN = float("nan")
_BY_CHOICES = ("latency", "score", "memory")


def _clean(value: Any) -> Any:
    """JSON safe: NaN and infinities become None, numpy scalars become Python ones."""
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return int(value)
    try:
        number = float(value)
    except (TypeError, ValueError):
        return value
    if math.isnan(number) or math.isinf(number):
        return None
    return number


def _missing(value: Any) -> bool:
    if value is None:
        return True
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return math.isnan(number) or math.isinf(number)


def _cell_width(character: str) -> int:
    """How many terminal columns one character occupies: 0, 1 or 2."""
    if unicodedata.combining(character):
        return 0
    return 2 if unicodedata.east_asian_width(character) in ("W", "F") else 1


def _display_width(text: str) -> int:
    """Width of `text` in terminal columns, not in code points.

    A CJK name is half as many characters as it is columns wide, so padding with
    ``str.ljust`` would shift that row's numbers out of line with the header.
    """
    return sum(_cell_width(character) for character in text)


def _pad(text: str, width: int) -> str:
    """Left-align `text` in `width` terminal columns."""
    return text + " " * max(0, width - _display_width(text))


def _shorten(text: str, width: int) -> str:
    """`text` trimmed to `width` terminal columns, ending in '.' when trimmed."""
    if _display_width(text) <= width:
        return text
    kept: List[str] = []
    used = 0
    for character in text:
        size = _cell_width(character)
        if used + size > width - 1:
            break
        kept.append(character)
        used += size
    return "".join(kept) + "."


def _places(value: float) -> int:
    """Enough decimals to keep a small number readable, without silly precision."""
    size = abs(float(value))
    if size == 0.0:
        return 3
    if size < 0.001:
        return 6
    if size < 0.01:
        return 5
    if size < 1.0:
        return 4
    if size < 1000.0:
        return 3
    return 1


def _text(value: Optional[float], places: Optional[int] = None) -> str:
    """A number for prose, or '-' when there is nothing to show."""
    if _missing(value):
        return "-"
    number = float(value)
    return f"{number:,.{_places(number) if places is None else places}f}"


def _fmt(value: Optional[float], places: Optional[int] = 3, width: int = 9) -> str:
    """A fixed-width cell, or '-' when there is nothing to show.

    `places` of None picks the number of decimals from the size of the value, so
    a microsecond model does not report itself as 0.000 ms.
    """
    if _missing(value):
        return "-".rjust(width)
    number = float(value)
    chosen = _places(number) if places is None else places
    return f"{number:{width},.{chosen}f}"


@dataclass
class Timing:
    """Latency statistics for one model at one batch size."""

    batch_size: int
    rows_per_call: Optional[int]
    repeats: int
    latency_mean_ms: float
    latency_p50_ms: float
    latency_p95_ms: float
    latency_min_ms: float
    latency_max_ms: float
    latency_std_ms: float
    throughput_per_s: float

    def to_dict(self) -> Dict[str, Any]:
        """JSON safe dict of this timing."""
        return {
            "batch_size": int(self.batch_size),
            "rows_per_call": None if self.rows_per_call is None else int(self.rows_per_call),
            "repeats": int(self.repeats),
            "latency_mean_ms": _clean(self.latency_mean_ms),
            "latency_p50_ms": _clean(self.latency_p50_ms),
            "latency_p95_ms": _clean(self.latency_p95_ms),
            "latency_min_ms": _clean(self.latency_min_ms),
            "latency_max_ms": _clean(self.latency_max_ms),
            "latency_std_ms": _clean(self.latency_std_ms),
            "throughput_per_s": _clean(self.throughput_per_s),
        }


@dataclass
class ModelResult:
    """What was measured for one model. Present even when the model failed."""

    name: str
    ok: bool = True
    error: Optional[str] = None
    timed_out: bool = False
    latency_mean_ms: float = _NAN
    latency_p50_ms: float = _NAN
    latency_p95_ms: float = _NAN
    latency_min_ms: float = _NAN
    latency_max_ms: float = _NAN
    latency_std_ms: float = _NAN
    throughput_per_s: float = _NAN
    peak_memory_kb: float = _NAN
    rss_delta_kb: Optional[float] = None
    score: Optional[float] = None
    metric_name: Optional[str] = None
    score_error: Optional[str] = None
    n_predictions: Optional[int] = None
    repeats: int = 0
    warmup: int = 0
    timings: Dict[int, Timing] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)

    @property
    def latency_ms(self) -> float:
        """Alias for `latency_mean_ms`, the number people usually mean."""
        return self.latency_mean_ms

    def to_dict(self) -> Dict[str, Any]:
        """JSON safe dict of everything measured for this model."""
        return {
            "name": self.name,
            "ok": bool(self.ok),
            "error": self.error,
            "timed_out": bool(self.timed_out),
            "latency_mean_ms": _clean(self.latency_mean_ms),
            "latency_p50_ms": _clean(self.latency_p50_ms),
            "latency_p95_ms": _clean(self.latency_p95_ms),
            "latency_min_ms": _clean(self.latency_min_ms),
            "latency_max_ms": _clean(self.latency_max_ms),
            "latency_std_ms": _clean(self.latency_std_ms),
            "throughput_per_s": _clean(self.throughput_per_s),
            "peak_memory_kb": _clean(self.peak_memory_kb),
            "rss_delta_kb": _clean(self.rss_delta_kb),
            "score": _clean(self.score),
            "metric_name": self.metric_name,
            "score_error": self.score_error,
            "n_predictions": self.n_predictions,
            "repeats": int(self.repeats),
            "warmup": int(self.warmup),
            "timings": {str(size): timing.to_dict() for size, timing in self.timings.items()},
            "warnings": list(self.warnings),
        }

    def summary(self) -> str:
        """One plain-text line describing this model."""
        if not self.ok:
            return f"{self.name}: failed ({self.error})"
        parts = [
            f"{self.name}: {_text(self.latency_mean_ms)} ms mean",
            f"p95 {_text(self.latency_p95_ms)} ms",
            f"{_text(self.throughput_per_s, 1)}/s",
            f"{_text(self.peak_memory_kb)} KB peak",
        ]
        if self.score is not None:
            parts.append(f"{self.metric_name} {self.score:.4f}")
        return ", ".join(parts)

    def __str__(self) -> str:
        return self.summary()


@dataclass
class BenchmarkReport:
    """Every model's numbers, side by side.

    ``results`` holds one :class:`ModelResult` per model, in the order the models
    were given. ``failures`` holds just the broken ones, as readable strings, so
    one model blowing up never costs you the rest of the run.
    """

    results: Dict[str, ModelResult] = field(default_factory=dict)
    failures: Dict[str, str] = field(default_factory=dict)
    metric_name: Optional[str] = None
    higher_is_better: bool = True
    repeats: int = 0
    warmup: int = 0
    batch_sizes: Tuple[int, ...] = ()
    n_samples: Optional[int] = None
    timeout: Optional[float] = None
    elapsed_s: float = 0.0
    warnings: List[str] = field(default_factory=list)

    # ---------------------------------------------------------------- basics
    @property
    def names(self) -> List[str]:
        """Every model name, in the order they were given."""
        return list(self.results)

    @property
    def ok_results(self) -> Dict[str, ModelResult]:
        """Only the models that produced measurements."""
        return {name: result for name, result in self.results.items() if result.ok}

    @property
    def primary_batch_size(self) -> Optional[int]:
        """The batch size the headline latency numbers come from."""
        return self.batch_sizes[0] if self.batch_sizes else None

    def __len__(self) -> int:
        return len(self.results)

    def __getitem__(self, name: str) -> ModelResult:
        return self.results[name]

    def __iter__(self):
        return iter(self.results.values())

    def __contains__(self, name: object) -> bool:
        return name in self.results

    # ----------------------------------------------------------------- picks
    def best(self, by: str = "latency") -> str:
        """Name of the winning model: `by` is 'latency', 'score' or 'memory'.

        Failed models are never picked. Ties go to whichever model was given first.
        """
        key = str(by).strip().lower()
        if key not in _BY_CHOICES:
            raise ValueError(f"by must be one of {', '.join(_BY_CHOICES)}, got {by!r}")
        ranked = self.rank(key)
        if ranked:
            return ranked[0]
        if not self.ok_results:
            raise ValueError("no model produced measurements; see report.failures")
        if key == "score":
            if self.metric_name is None:
                raise ValueError(
                    "no model has a score; pass metric= to benchmark() and data as (X, y_true)"
                )
            raise ValueError(
                f"no model produced a usable {self.metric_name} score, so there is nothing "
                "to rank; see score_error on each result in report.results"
            )
        raise ValueError(f"no model produced a {key} measurement")

    def rank(self, by: str = "latency") -> List[str]:
        """Every measured model, best first, using the same rules as `best`."""
        key = str(by).strip().lower()
        if key not in _BY_CHOICES:
            raise ValueError(f"by must be one of {', '.join(_BY_CHOICES)}, got {by!r}")
        candidates = list(self.ok_results.values())
        if key == "latency":
            usable = [r for r in candidates if not _missing(r.latency_mean_ms)]
            return [r.name for r in sorted(usable, key=lambda r: r.latency_mean_ms)]
        if key == "memory":
            usable = [r for r in candidates if not _missing(r.peak_memory_kb)]
            return [r.name for r in sorted(usable, key=lambda r: r.peak_memory_kb)]
        usable = [r for r in candidates if not _missing(r.score)]
        return [
            r.name
            for r in sorted(usable, key=lambda r: float(r.score), reverse=bool(self.higher_is_better))
        ]

    def _require(self, name: str) -> ModelResult:
        if name not in self.results:
            known = ", ".join(self.results) or "nothing"
            raise ValueError(f"no model named {name!r}; this report has {known}")
        result = self.results[name]
        if not result.ok:
            raise ValueError(f"model {name!r} failed and has nothing to compare: {result.error}")
        return result

    def compare(self, a: str, b: str) -> Dict[str, Any]:
        """How model `a` stacks up against model `b`, as ratios.

        A ``latency_ratio`` below 1 means `a` is that much faster than `b`, and
        ``speedup`` says the same thing the other way up.
        """
        left = self._require(a)
        right = self._require(b)

        def ratio(top: Any, bottom: Any) -> Optional[float]:
            """top / bottom, or None when either side is missing or zero."""
            if _missing(top) or _missing(bottom) or float(bottom) == 0.0:
                return None
            return float(top) / float(bottom)

        latency_ratio = ratio(left.latency_mean_ms, right.latency_mean_ms)
        out: Dict[str, Any] = {
            "a": a,
            "b": b,
            "latency_ratio": latency_ratio,
            "speedup": None if not latency_ratio else 1.0 / latency_ratio,
            "throughput_ratio": ratio(left.throughput_per_s, right.throughput_per_s),
            "memory_ratio": ratio(left.peak_memory_kb, right.peak_memory_kb),
            "score_ratio": ratio(left.score, right.score),
            "score_delta": (
                None
                if _missing(left.score) or _missing(right.score)
                else float(left.score) - float(right.score)
            ),
            "metric_name": self.metric_name,
            "faster": None,
            "leaner": None,
            "better_score": None,
        }
        if latency_ratio is not None and latency_ratio != 1.0:
            out["faster"] = a if latency_ratio < 1 else b
        memory_ratio = out["memory_ratio"]
        if memory_ratio is not None and memory_ratio != 1.0:
            out["leaner"] = a if memory_ratio < 1 else b
        if not _missing(left.score) and not _missing(right.score):
            if float(left.score) != float(right.score):
                left_wins = float(left.score) > float(right.score)
                if not self.higher_is_better:
                    left_wins = not left_wins
                out["better_score"] = a if left_wins else b
        out["summary"] = self._compare_summary(a, b, out)
        return out

    @staticmethod
    def _compare_summary(a: str, b: str, out: Dict[str, Any]) -> str:
        speedup = out.get("speedup")
        if speedup is None:
            line = f"{a} vs {b}: latency not comparable"
        elif speedup >= 1:
            line = f"{a} is {speedup:.2f}x faster than {b}"
        else:
            line = f"{a} is {1.0 / speedup:.2f}x slower than {b}"
        memory_ratio = out.get("memory_ratio")
        if memory_ratio:
            line += f", uses {memory_ratio:.2f}x the peak memory"
        delta = out.get("score_delta")
        if delta is not None:
            line += f", {out.get('metric_name') or 'score'} {delta:+.4f}"
        return line

    # ----------------------------------------------------------------- views
    def to_frame(self, batch_size: Optional[int] = None) -> pd.DataFrame:
        """One row per model, in the order the models were given.

        `batch_size` picks which batch size the latency columns come from; the
        first one benchmarked is used by default. Asking for a batch size that
        was never benchmarked is a :class:`ValueError`, not a silent fallback to
        the primary one.
        """
        if batch_size is None:
            size = self.primary_batch_size
        else:
            size = int(batch_size)
            if size not in self.batch_sizes:
                known = ", ".join(str(item) for item in self.batch_sizes) or "none"
                raise ValueError(
                    f"batch size {size} was not benchmarked; this report has {known}"
                )
        rows: List[Dict[str, Any]] = []
        for name, result in self.results.items():
            timing = result.timings.get(size) if size is not None else None
            rows.append(
                {
                    "model": name,
                    "ok": bool(result.ok),
                    "latency_ms": timing.latency_mean_ms if timing else result.latency_mean_ms,
                    "p50_ms": timing.latency_p50_ms if timing else result.latency_p50_ms,
                    "p95_ms": timing.latency_p95_ms if timing else result.latency_p95_ms,
                    "min_ms": timing.latency_min_ms if timing else result.latency_min_ms,
                    "max_ms": timing.latency_max_ms if timing else result.latency_max_ms,
                    "throughput_per_s": (
                        timing.throughput_per_s if timing else result.throughput_per_s
                    ),
                    "peak_memory_kb": result.peak_memory_kb,
                    "rss_delta_kb": result.rss_delta_kb,
                    "score": result.score,
                    "error": result.error,
                }
            )
        return pd.DataFrame(
            rows,
            columns=[
                "model",
                "ok",
                "latency_ms",
                "p50_ms",
                "p95_ms",
                "min_ms",
                "max_ms",
                "throughput_per_s",
                "peak_memory_kb",
                "rss_delta_kb",
                "score",
                "error",
            ],
        )

    def batch_frame(self) -> pd.DataFrame:
        """One row per model and batch size, for runs with several `batch_sizes`."""
        rows: List[Dict[str, Any]] = []
        for name, result in self.results.items():
            for size in self.batch_sizes:
                timing = result.timings.get(size)
                rows.append(
                    {
                        "model": name,
                        "batch_size": size,
                        "rows_per_call": timing.rows_per_call if timing else None,
                        "latency_ms": timing.latency_mean_ms if timing else _NAN,
                        "p95_ms": timing.latency_p95_ms if timing else _NAN,
                        "throughput_per_s": timing.throughput_per_s if timing else _NAN,
                    }
                )
        return pd.DataFrame(
            rows,
            columns=[
                "model",
                "batch_size",
                "rows_per_call",
                "latency_ms",
                "p95_ms",
                "throughput_per_s",
            ],
        )

    def to_dict(self) -> Dict[str, Any]:
        """JSON safe dict of the whole report."""
        return {
            "n_models": len(self.results),
            "n_ok": len(self.ok_results),
            "n_failed": len(self.failures),
            "metric": self.metric_name,
            "higher_is_better": bool(self.higher_is_better),
            "repeats": int(self.repeats),
            "warmup": int(self.warmup),
            "batch_sizes": [int(size) for size in self.batch_sizes],
            "n_samples": self.n_samples,
            "timeout": _clean(self.timeout),
            "elapsed_s": _clean(self.elapsed_s),
            "best": self._best_or_none(),
            "results": {name: result.to_dict() for name, result in self.results.items()},
            "failures": dict(self.failures),
            "warnings": list(self.warnings),
        }

    def _best_or_none(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for key in _BY_CHOICES:
            try:
                out[key] = self.best(key)
            except ValueError:
                out[key] = None
        return out

    # --------------------------------------------------------------- summary
    def summary(self, limit: Optional[int] = None) -> str:
        """The plain-text report. ASCII only, safe on any console."""
        lines: List[str] = []
        total = len(self.results)
        n_ok = len(self.ok_results)
        n_failed = len(self.failures)
        head = f"model-benchmark: {total} model{'' if total == 1 else 's'} in {self.elapsed_s:.2f} s"
        head += f", {n_ok} measured"
        if n_failed:
            head += f", {n_failed} failed"
        lines.append(head)

        if self.n_samples is None:
            lines.append("  data      : one input, not a collection of rows")
        else:
            lines.append(f"  data      : {self.n_samples:,} rows")
        lines.append(f"  calls     : {self.warmup} warmup + {self.repeats} timed, per batch size")
        lines.append(f"  batch     : {self._batch_line()}")
        if self.metric_name:
            direction = "higher is better" if self.higher_is_better else "lower is better"
            lines.append(f"  metric    : {self.metric_name} ({direction})")
        if self.timeout is not None:
            lines.append(f"  timeout   : {self.timeout:g} s per model")

        if n_ok:
            fastest = self.best("latency")
            lines.append(
                f"  fastest   : {fastest} ({_text(self.results[fastest].latency_mean_ms)} ms)"
            )
            leanest = self.best("memory")
            peak = _text(self.results[leanest].peak_memory_kb)
            primary = self.primary_batch_size
            where = "" if primary is None else f", one call at batch size {primary}"
            lines.append(f"  leanest   : {leanest} ({peak} KB peak{where})")
            scored = self.rank("score")
            if scored:
                winner = self.results[scored[0]]
                lines.append(
                    f"  best score: {winner.name} ({float(winner.score):.4f} {self.metric_name})"
                )

        lines.append("")
        lines.extend(self._table(limit))

        if self.failures:
            lines.append("")
            lines.append("  failures:")
            for name, message in self.failures.items():
                lines.append(f"    {name}: {message}")

        notes = list(self.warnings)
        for result in self.results.values():
            for note in result.warnings:
                notes.append(f"{result.name}: {note}")
        if notes:
            lines.append("")
            lines.append("  warnings:")
            for note in notes:
                lines.append(f"    - {note}")
        return "\n".join(lines)

    def _batch_line(self) -> str:
        if not self.batch_sizes:
            return "-"
        sizes = ", ".join(str(size) for size in self.batch_sizes)
        if self.n_samples is None:
            return f"{sizes} (the input is passed whole)"
        return f"{sizes} of {self.n_samples:,} rows per timed call"

    def _table(self, limit: Optional[int]) -> List[str]:
        names = list(self.results)
        if limit is not None and limit >= 0:
            names = names[:limit]
        width = min(max([_display_width(name) for name in names] + [5]), 32)
        header = (
            f"  {_pad('model', width)}  {'latency_ms':>11}  {'p50_ms':>10}  {'p95_ms':>10}"
            f"  {'per_s':>13}  {'peak_KB':>10}"
        )
        if self.metric_name:
            header += f"  {'score':>9}"
        rows = [header]
        for name in names:
            result = self.results[name]
            label = _shorten(name, width)
            row = (
                f"  {_pad(label, width)}  {_fmt(result.latency_mean_ms, None, 11)}"
                f"  {_fmt(result.latency_p50_ms, None, 10)}  {_fmt(result.latency_p95_ms, None, 10)}"
                f"  {_fmt(result.throughput_per_s, 1, 13)}  {_fmt(result.peak_memory_kb, None, 10)}"
            )
            if self.metric_name:
                row += f"  {_fmt(result.score, 4, 9)}"
            rows.append(row)
        if limit is not None and 0 <= limit < len(self.results):
            rows.append(f"  ... and {len(self.results) - limit} more")
        return rows

    def __str__(self) -> str:
        return self.summary()

    def __repr__(self) -> str:
        return (
            f"BenchmarkReport({len(self.results)} models, {len(self.failures)} failed, "
            f"metric={self.metric_name!r})"
        )
