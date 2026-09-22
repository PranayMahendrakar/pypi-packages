"""The measurement engine.

Timing is always :func:`time.perf_counter`, never ``time.time``. Latency is
measured with allocation tracing switched off, because tracing makes every call
several times slower; the memory numbers come from a separate pass afterwards.
No GPU is assumed, queried or required, and nothing here imports torch or
tensorflow: a torch model is just a callable like any other.
"""
from __future__ import annotations

import logging
import threading
from time import perf_counter
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from ._metrics import resolve_metric
from ._resources import measure_allocations, rss_bytes
from ._result import BenchmarkReport, ModelResult, Timing

__all__ = ["Benchmark", "benchmark", "compare"]

log = logging.getLogger(__name__)

DEFAULT_WARMUP = 2
DEFAULT_REPEATS = 10
DEFAULT_BATCH_SIZES: Tuple[Union[int, str], ...] = (1,)
_MIN_SECONDS = 1e-9
_NAN = float("nan")


class _ModelTimeout(Exception):
    """Raised internally when a model outruns its timeout."""


class _Abandoned(Exception):
    """Raised inside the worker once its deadline has passed, to stop the schedule."""


# ---------------------------------------------------------------- model input
def _as_callable(name: str, model: Any) -> Callable[..., Any]:
    """The thing to call for this model: its `.predict`, or the model itself."""
    predict = getattr(model, "predict", None)
    if callable(predict):
        return predict
    if callable(model):
        return model
    raise TypeError(
        f"model {name!r} is a {type(model).__name__}, which is neither callable "
        "nor an object with a .predict method"
    )


def _normalise_models(models: Any) -> "Dict[str, Any]":
    """Accept a dict, a list of (name, model) pairs, or a plain list of models."""
    if models is None:
        raise ValueError("models must be a dict of name -> model, not None")
    if isinstance(models, Mapping):
        items: List[Tuple[Any, Any]] = list(models.items())
    elif isinstance(models, (list, tuple)):
        items = []
        for index, entry in enumerate(models):
            if isinstance(entry, tuple) and len(entry) == 2 and isinstance(entry[0], str):
                items.append((entry[0], entry[1]))
            else:
                label = getattr(entry, "__name__", None) or f"model_{index}"
                items.append((str(label), entry))
    else:
        raise TypeError(
            "models must be a dict of name -> model (or a list of models), got "
            f"{type(models).__name__}"
        )
    if not items:
        raise ValueError("models is empty; give at least one model to benchmark")

    out: "Dict[str, Any]" = {}
    duplicates: List[str] = []
    for raw_name, model in items:
        name = str(raw_name)
        if name in out:
            duplicates.append(name)
            continue
        out[name] = model
    if duplicates:
        listed = ", ".join(sorted(set(duplicates)))
        raise ValueError(f"duplicate model names: {listed}; every model needs its own name")
    return out


# ----------------------------------------------------------------- data shape
def _row_count(data: Any) -> Optional[int]:
    """How many rows `data` holds, or None when it is a single opaque input."""
    if data is None:
        return None
    if isinstance(data, (str, bytes, bytearray, dict, set)):
        return None
    shape = getattr(data, "shape", None)
    if isinstance(shape, tuple) and shape:
        try:
            return int(shape[0])
        except (TypeError, ValueError):
            return None
    if isinstance(data, (list, tuple)):
        return len(data)
    return None


def _check_columns(data: Any) -> None:
    """Refuse a DataFrame with duplicate column names, naming the duplicates.

    pandas lets a frame carry the same column name twice, and every model would
    then silently see something different from what the caller thinks it sent.
    Catching it here beats an AttributeError from deep inside pandas later.
    """
    if not isinstance(data, pd.DataFrame):
        return
    columns = pd.Index(data.columns)
    duplicated = columns[columns.duplicated()]
    if len(duplicated) == 0:
        return
    listed = ", ".join(sorted({str(column) for column in duplicated}))
    raise ValueError(
        f"X has duplicate column names: {listed}. Rename them so every model "
        "sees the same, unambiguous input."
    )


def _slice_rows(data: Any, n: int) -> Any:
    """The first `n` rows of `data`."""
    iloc = getattr(data, "iloc", None)
    if iloc is not None:
        return iloc[:n]
    return data[:n]


def _normalise_batch_sizes(
    batch_sizes: Any, n_samples: Optional[int], warnings: List[str]
) -> Tuple[int, ...]:
    """Turn the `batch_sizes` argument into a deduplicated tuple of positive ints."""
    if batch_sizes is None:
        raw: List[Any] = [None]
    elif isinstance(batch_sizes, int) and not isinstance(batch_sizes, bool):
        raw = [batch_sizes]
    elif isinstance(batch_sizes, str):
        raw = [batch_sizes]
    elif isinstance(batch_sizes, Iterable):
        raw = list(batch_sizes)
    else:
        raise TypeError(
            f"batch_sizes must be an int or a sequence of ints, got {type(batch_sizes).__name__}"
        )
    if not raw:
        raise ValueError("batch_sizes is empty; use (1,) for one row per call")

    sizes: List[int] = []
    for entry in raw:
        if entry is None or (isinstance(entry, str) and entry.strip().lower() == "all"):
            sizes.append(int(n_samples) if n_samples else 1)
            continue
        if isinstance(entry, bool) or not isinstance(entry, (int, np.integer)):
            raise TypeError(f"batch size {entry!r} is not an int (use 'all' for every row)")
        value = int(entry)
        if value < 1:
            raise ValueError(f"batch size {value} is not positive; batch sizes start at 1")
        sizes.append(value)

    if n_samples is None:
        if any(size != 1 for size in sizes):
            warnings.append(
                "data is a single input rather than a collection of rows, so batch sizes "
                "were ignored and the input was passed whole"
            )
        return (1,)

    deduped: List[int] = []
    for size in sizes:
        if size not in deduped:
            deduped.append(size)
    if n_samples == 0:
        # Every batch size is "too large" for an empty input, which reads backwards.
        # Say the useful thing once instead of once per size.
        warnings.append("data has 0 rows, so there is nothing to time")
        return tuple(deduped)
    for size in deduped:
        if size > n_samples:
            warnings.append(
                f"batch size {size} is larger than the {n_samples} rows of data, "
                f"so {n_samples} rows were used for it"
            )
    return tuple(deduped)


def _build_batches(
    x_data: Any, batch_sizes: Tuple[int, ...], n_samples: Optional[int]
) -> "Dict[int, Tuple[Any, Optional[int]]]":
    """Map each batch size to (the input for one call, how many rows that is)."""
    batches: "Dict[int, Tuple[Any, Optional[int]]]" = {}
    for size in batch_sizes:
        if x_data is None or n_samples is None:
            batches[size] = (x_data, None)
        elif size >= n_samples:
            batches[size] = (x_data, n_samples)
        else:
            batches[size] = (_slice_rows(x_data, size), size)
    return batches


# ------------------------------------------------------------------ predictions
def _prediction_length(values: Any) -> Optional[int]:
    if values is None:
        return None
    shape = getattr(values, "shape", None)
    if isinstance(shape, tuple) and shape:
        try:
            return int(shape[0])
        except (TypeError, ValueError):
            return None
    try:
        return len(values)
    except TypeError:
        return None


def _shape_problem(y_true: Any, y_pred: Any) -> Optional[str]:
    """A readable complaint when predictions cannot line up with the truth."""
    n_true = _prediction_length(y_true)
    n_pred = _prediction_length(y_pred)
    if n_pred is None:
        return (
            f"returned {type(y_pred).__name__}, which has no length; the metric needs "
            "one prediction per input"
        )
    if n_true is not None and n_pred != n_true:
        return f"returned {n_pred} predictions for {n_true} inputs"
    try:
        true_dim = int(np.asarray(y_true).ndim) if y_true is not None else 1
    except Exception:
        true_dim = 1
    pred_dim = getattr(y_pred, "ndim", None)
    if pred_dim is None:
        return None
    if int(pred_dim) > 1 and int(true_dim) == 1:
        shape = tuple(getattr(y_pred, "shape", ()))
        trailing = [int(dim) for dim in shape[1:]]
        if any(dim != 1 for dim in trailing):
            return (
                f"returned an array of shape {shape}; the metric needs one prediction "
                "per input, so reduce it first (for example with argmax)"
            )
    return None


# ----------------------------------------------------------------- measurement
def _latency_stats(durations: Sequence[float], rows_per_call: Optional[int]) -> Dict[str, float]:
    """Latency statistics in milliseconds; safe for a single repeat."""
    seconds = np.asarray(list(durations), dtype=float)
    millis = seconds * 1000.0
    mean_seconds = float(seconds.mean())
    items = rows_per_call if rows_per_call and rows_per_call > 0 else 1
    return {
        "latency_mean_ms": float(millis.mean()),
        "latency_p50_ms": float(np.percentile(millis, 50)),
        "latency_p95_ms": float(np.percentile(millis, 95)),
        "latency_min_ms": float(millis.min()),
        "latency_max_ms": float(millis.max()),
        "latency_std_ms": float(millis.std(ddof=0)),
        "throughput_per_s": float(items) / max(mean_seconds, _MIN_SECONDS),
    }


def _call(function: Callable[..., Any], payload: Any, pass_nothing: bool) -> Any:
    return function() if pass_nothing else function(payload)


def _check_deadline(deadline: Optional[float]) -> None:
    """Give up on this model when its timeout has already passed.

    Checked between calls, never inside a timed window. Without it an abandoned
    model would go on to run its whole remaining schedule in the background.
    """
    if deadline is not None and perf_counter() >= deadline:
        raise _Abandoned()


def _measure_model(
    name: str,
    function: Callable[..., Any],
    batches: "Dict[int, Tuple[Any, Optional[int]]]",
    batch_sizes: Tuple[int, ...],
    x_data: Any,
    y_true: Any,
    metric_function: Optional[Callable[[Any, Any], float]],
    metric_name: Optional[str],
    warmup: int,
    repeats: int,
    pass_nothing: bool,
    deadline: Optional[float] = None,
) -> ModelResult:
    """Everything measured for one model. Raises if the model itself raises.

    `deadline` is a :func:`time.perf_counter` reading after which this model is
    abandoned: it is checked between calls so a model that has already outrun
    its timeout stops there instead of working through the rest of the schedule.
    """
    result = ModelResult(name=name, metric_name=metric_name, repeats=repeats, warmup=warmup)
    primary = batch_sizes[0]
    primary_payload = batches[primary][0]

    # 1 + 2. warm up at each batch size, so every timed run starts warm, then time it
    #        with allocation tracing off so the numbers are honest.
    for size in batch_sizes:
        payload, rows = batches[size]
        for _ in range(warmup):
            _check_deadline(deadline)
            _call(function, payload, pass_nothing)
        durations: List[float] = []
        for _ in range(repeats):
            _check_deadline(deadline)
            started = perf_counter()
            _call(function, payload, pass_nothing)
            durations.append(perf_counter() - started)
        stats = _latency_stats(durations, rows)
        result.timings[size] = Timing(
            batch_size=size,
            rows_per_call=rows,
            repeats=repeats,
            **stats,
        )
        if size == primary:
            for key, value in stats.items():
                setattr(result, key, value)

    # 3. memory, in its own pass; tracemalloc starts and stops around this call only.
    #    This is one call at the primary (first) batch size, which the README says.
    _check_deadline(deadline)
    rss_before = rss_bytes()
    with measure_allocations() as reading:
        _call(function, primary_payload, pass_nothing)
    rss_after = rss_bytes()
    result.peak_memory_kb = reading.peak_bytes / 1024.0
    if rss_before is not None and rss_after is not None:
        result.rss_delta_kb = max(0.0, float(rss_after - rss_before)) / 1024.0

    # 4. the metric, scored once on the whole of X.
    if metric_function is not None:
        _check_deadline(deadline)
        predictions = _call(function, x_data, pass_nothing)
        result.n_predictions = _prediction_length(predictions)
        problem = _shape_problem(y_true, predictions)
        if problem is not None:
            result.score_error = problem
            result.warnings.append(f"{problem}; no {metric_name} score was computed")
        else:
            try:
                result.score = float(metric_function(y_true, predictions))
            except Exception as exc:  # the metric is user code too
                result.score_error = f"{type(exc).__name__}: {exc}"
                result.warnings.append(
                    f"metric {metric_name!r} could not score the predictions "
                    f"({result.score_error})"
                )
    return result


def _run_with_timeout(work: Callable[[], Any], timeout: Optional[float]) -> Any:
    """Run `work` on a daemon thread, giving up after `timeout` seconds.

    Python cannot force a running call to stop, so on a timeout the in-flight
    call is abandoned rather than killed. Two things keep that from costing the
    caller unbounded time anyway:

    * the worker is a plain daemon thread, so nothing joins it at interpreter
      exit and the process still exits on schedule (a pool would register an
      ``atexit`` hook that waits for the worker to finish);
    * `work` checks its deadline between calls, so the abandoned model stops
      after the call it is already in rather than running the rest of its
      schedule in the background.
    """
    if timeout is None:
        return work()
    box: "Dict[str, Any]" = {}

    def runner() -> None:
        """Run the measurement, keeping whatever came back for the caller."""
        try:
            box["value"] = work()
        except BaseException as exc:  # handed back to the calling thread below
            box["error"] = exc

    thread = threading.Thread(target=runner, name="model-benchmark-worker", daemon=True)
    thread.start()
    thread.join(timeout)
    if thread.is_alive():
        raise _ModelTimeout(timeout)
    error = box.get("error")
    if error is not None:
        # The worker can notice the deadline a hair before join() gives up.
        if isinstance(error, _Abandoned):
            raise _ModelTimeout(timeout)
        raise error
    return box["value"]


# ------------------------------------------------------------------- public API
class Benchmark:
    """The benchmark settings, for people who run the same comparison repeatedly.

    ``benchmark()`` is this class with one call::

        bench = Benchmark(metric="accuracy", repeats=50, batch_sizes=(1, 32))
        report = bench.run(models, (X, y))
    """

    def __init__(
        self,
        *,
        metric: Any = None,
        warmup: int = DEFAULT_WARMUP,
        repeats: int = DEFAULT_REPEATS,
        batch_sizes: Any = DEFAULT_BATCH_SIZES,
        timeout: Optional[float] = None,
    ) -> None:
        self.metric = metric
        self.warmup = int(warmup)
        self.repeats = int(repeats)
        self.batch_sizes = batch_sizes
        self.timeout = None if timeout is None else float(timeout)
        if self.repeats < 1:
            raise ValueError(f"repeats must be at least 1, got {repeats}")
        if self.warmup < 0:
            raise ValueError(f"warmup cannot be negative, got {warmup}")
        if self.timeout is not None and self.timeout <= 0:
            raise ValueError(f"timeout must be positive seconds or None, got {timeout}")
        self.metric_name, self._metric_function, self.higher_is_better = resolve_metric(metric)

    def __repr__(self) -> str:
        return (
            f"Benchmark(metric={self.metric_name!r}, warmup={self.warmup}, "
            f"repeats={self.repeats}, batch_sizes={self.batch_sizes!r}, timeout={self.timeout!r})"
        )

    def run(self, models: Any, data: Any = None) -> BenchmarkReport:
        """Measure every model in `models` on `data` and return a report."""
        started = perf_counter()
        prepared = _normalise_models(models)

        if self._metric_function is not None:
            if not (isinstance(data, tuple) and len(data) == 2):
                raise ValueError(
                    "when metric= is given, data must be the tuple (X, y_true); got "
                    f"{type(data).__name__}"
                )
            x_data, y_true = data
        else:
            x_data, y_true = data, None

        _check_columns(x_data)

        report_warnings: List[str] = []
        n_samples = _row_count(x_data)
        batch_sizes = _normalise_batch_sizes(self.batch_sizes, n_samples, report_warnings)
        batches = _build_batches(x_data, batch_sizes, n_samples)
        pass_nothing = x_data is None

        if self._metric_function is not None and n_samples is None:
            report_warnings.append(
                "X has no row count, so the score is computed on whatever the model "
                "returns for the input as given"
            )

        results: "Dict[str, ModelResult]" = {}
        failures: "Dict[str, str]" = {}
        for name, model in prepared.items():
            try:
                function = _as_callable(name, model)
            except TypeError as exc:
                message = f"{type(exc).__name__}: {exc}"
                failures[name] = message
                results[name] = ModelResult(
                    name=name, ok=False, error=message, metric_name=self.metric_name
                )
                continue

            deadline = None if self.timeout is None else perf_counter() + self.timeout

            def work(
                _name: str = name,
                _function: Callable[..., Any] = function,
                _deadline: Optional[float] = deadline,
            ) -> ModelResult:
                """Measure this one model; run under the timeout when there is one."""
                return _measure_model(
                    name=_name,
                    function=_function,
                    batches=batches,
                    batch_sizes=batch_sizes,
                    x_data=x_data,
                    y_true=y_true,
                    metric_function=self._metric_function,
                    metric_name=self.metric_name,
                    warmup=self.warmup,
                    repeats=self.repeats,
                    pass_nothing=pass_nothing,
                    deadline=_deadline,
                )

            try:
                results[name] = _run_with_timeout(work, self.timeout)
            except (_ModelTimeout, _Abandoned):
                message = f"timed out after {self.timeout:g} s"
                failures[name] = message
                results[name] = ModelResult(
                    name=name,
                    ok=False,
                    error=message,
                    timed_out=True,
                    metric_name=self.metric_name,
                )
                report_warnings.append(
                    f"{name} was abandoned after {self.timeout:g} s; the call it was in "
                    "finishes on a daemon background thread and the rest of its schedule "
                    "is skipped, so later measurements may be slightly noisier"
                )
                log.debug("model %s timed out after %s s", name, self.timeout)
            except BaseException as exc:  # one broken model must not end the run
                if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                    raise
                message = f"{type(exc).__name__}: {exc}"
                failures[name] = message
                results[name] = ModelResult(
                    name=name, ok=False, error=message, metric_name=self.metric_name
                )
                log.debug("model %s failed: %s", name, message)

        return BenchmarkReport(
            results=results,
            failures=failures,
            metric_name=self.metric_name,
            higher_is_better=self.higher_is_better,
            repeats=self.repeats,
            warmup=self.warmup,
            batch_sizes=batch_sizes,
            n_samples=n_samples,
            timeout=self.timeout,
            elapsed_s=perf_counter() - started,
            warnings=report_warnings,
        )


def benchmark(
    models: Any,
    data: Any = None,
    *,
    metric: Any = None,
    warmup: int = DEFAULT_WARMUP,
    repeats: int = DEFAULT_REPEATS,
    batch_sizes: Any = DEFAULT_BATCH_SIZES,
    timeout: Optional[float] = None,
) -> BenchmarkReport:
    """Benchmark several models on the same task and compare them side by side.

    Args:
        models: ``{name: model}``, where a model is either a callable or an
            object with a ``.predict`` method. They all take the same input.
        data: the input handed to every model. When `metric` is given this is
            the tuple ``(X, y_true)``. ``None`` means the models take no argument.
        metric: ``callable(y_true, y_pred) -> float``, or one of ``"accuracy"``,
            ``"f1"``, ``"rmse"``, ``"mae"``, ``"r2"``.
        warmup: untimed calls made before the measured ones.
        repeats: timed calls per batch size. 1 is allowed.
        batch_sizes: how many rows of `X` go into each timed call. Use ``"all"``
            for the whole input in one call.
        timeout: seconds allowed for one model's whole measurement, or None.

    Returns:
        A :class:`~model_benchmark.BenchmarkReport`. A model that raises is
        recorded in ``report.failures`` and the rest of the run continues.
    """
    return Benchmark(
        metric=metric,
        warmup=warmup,
        repeats=repeats,
        batch_sizes=batch_sizes,
        timeout=timeout,
    ).run(models, data)


def compare(models: Any, data: Any = None, **kwargs: Any) -> pd.DataFrame:
    """Benchmark `models` on `data` and return just the table, one row per model."""
    return benchmark(models, data, **kwargs).to_frame()
