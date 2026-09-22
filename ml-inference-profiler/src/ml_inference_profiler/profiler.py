"""The :class:`Profiler`: time stages, nested stages, decorated functions, pipelines.

Thread safety, honestly
-----------------------
A profiler can be used from several threads without corrupting itself: the nesting stack
is thread-local, so two threads never see each other's parent stage, and every write to
the shared totals happens under a lock. That is as far as the guarantee goes.

What it does **not** give you is a meaningful breakdown of elapsed time when threads run
at the same time. Each stage measures its own wall clock, so two stages running in
parallel both count their full duration and the totals add up to more than the run really
took. Read a multi-threaded profile as "how long each stage took", never as "where the
elapsed time went". :attr:`ProfileReport.threads` reports how many threads contributed,
and the summary says so out loud when it is more than one.

A stage entered on one thread must be left on the same thread. Entering on one and
leaving on another is not supported; the mismatched exit is logged and ignored.

Warmup passes are silenced per thread, never globally. While :meth:`Profiler.run` is in a
warmup pass it stops recording **on the thread that called it** and nowhere else, so a
worker thread that happens to open a stage at that moment keeps its measurement instead of
having it dropped without a word. The other side of that promise: a pipeline step which
fans work out to worker threads will see those threads' stages counted even during warmup,
because the pause does not follow the work across the thread boundary. When that matters,
use ``warmup=0`` and throw the first report away yourself.
"""

from __future__ import annotations

import functools
import logging
import threading
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

from ._timing import clock, measure_stage_overhead, median_ms, percentile_ms, to_ms
from .report import ProfileReport, Stage

logger = logging.getLogger(__name__)

#: A stage is addressed by the tuple of labels from the root down to it.
Path = Tuple[str, ...]

#: A pipeline step is ``(label, callable)`` or a bare callable.
Step = Union[Callable[[Any], Any], Sequence[Any]]


class _Node:
    """Accumulated measurements for one position in the stage tree."""

    __slots__ = ("path", "durations", "child_seconds", "errors")

    def __init__(self, path: Path) -> None:
        self.path = path
        self.durations: List[float] = []
        self.child_seconds = 0.0
        self.errors = 0


class _Frame:
    """One live ``with`` block on a thread's stack."""

    __slots__ = ("path", "node", "start", "child_seconds")

    def __init__(self, path: Path, node: _Node, start: float) -> None:
        self.path = path
        self.node = node
        self.start = start
        self.child_seconds = 0.0


class _StageContext:
    """Context manager returned by :meth:`Profiler.stage`. Nestable and re-usable."""

    __slots__ = ("_profiler", "label")

    def __init__(self, profiler: "Profiler", label: str) -> None:
        self._profiler = profiler
        self.label = label

    def __enter__(self) -> "_StageContext":
        self._profiler._push(self.label)
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self._profiler._pop(failed=exc_type is not None)
        return False  # never swallow: the caller must see the real error

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<stage {self.label!r}>"


class _NullStage:
    """The no-op stage handed out during a warmup pass, so warmup records nothing.

    Only the thread inside :meth:`Profiler.run` gets these: the pause flag lives on the
    profiler's thread-local, so a stage opened on any other thread is recorded normally.
    """

    __slots__ = ("label",)

    def __init__(self, label: str) -> None:
        self.label = label

    def __enter__(self) -> "_NullStage":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False


class Profiler:
    """Time the steps of an inference pipeline and say which one is slow.

    Args:
        name: Shown at the top of the report. Defaults to ``"pipeline"``.

    Example:
        >>> profiler = Profiler("resnet")
        >>> with profiler.stage("preprocess"):
        ...     pass
        >>> profiler.report().stages[0].label
        'preprocess'

    Measurements accumulate until :meth:`reset` is called, so the same label used twice
    aggregates into one stage (two calls, summed time) instead of overwriting.

    Threads: a profiler may be used from several threads at once without corrupting
    itself - the nesting stack is thread-local and the totals are written under a lock -
    but every stage measures its own wall clock, so stages running in parallel add up to
    more than the elapsed time. A warmup pass in :meth:`run` pauses recording only on the
    thread running it, so no other thread's measurements are thrown away. See this
    module's docstring for the full caveat.
    """

    def __init__(self, name: str = "pipeline") -> None:
        self.name = str(name) if name is not None else "pipeline"
        self._lock = threading.Lock()
        self._local = threading.local()
        self._nodes: Dict[Path, _Node] = {}
        self._children: Dict[Path, List[Path]] = {(): []}
        self._thread_ids: set = set()
        self._repeats = 1
        self._warmup = 0

    # -- recording ------------------------------------------------------------------
    def stage(self, label: str) -> Any:
        """Time one stage. Use it as a context manager; nesting works to any depth.

        Args:
            label: Name of the stage. The same label used twice at the same position
                aggregates rather than overwriting.

        Returns:
            A context manager. Leaving it records the elapsed time - including when the
            block raises, in which case the error is re-raised unchanged and the call is
            counted in :attr:`Stage.errors`.

        Raises:
            ValueError: If ``label`` is not a non-empty string.
        """
        if not isinstance(label, str) or not label.strip():
            raise ValueError(
                f"stage label must be a non-empty string, got {label!r}"
            )
        if getattr(self._local, "paused", False):
            return _NullStage(label)
        return _StageContext(self, label)

    def profile(self, func_or_label: Union[Callable[..., Any], str]) -> Any:
        """Decorator that records every call to a function as a stage.

        Use it bare (``@profiler.profile``) to take the function's own name as the label,
        or with one (``@profiler.profile("model")``) to choose it.

        Args:
            func_or_label: The function being decorated, or the label to use.

        Returns:
            The wrapped function, or a decorator when a label was given.
        """
        if isinstance(func_or_label, str):
            label = func_or_label
            if not label.strip():
                raise ValueError("stage label must be a non-empty string, got ''")

            def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
                return self._wrap(func, label)

            return decorator
        if callable(func_or_label):
            name = getattr(func_or_label, "__name__", None) or "function"
            return self._wrap(func_or_label, name)
        raise TypeError(
            "profile() takes a function or a label string, got "
            + type(func_or_label).__name__
        )

    def _wrap(self, func: Callable[..., Any], label: str) -> Callable[..., Any]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            with self.stage(label):
                return func(*args, **kwargs)

        return wrapper

    def _stack(self) -> List[_Frame]:
        stack = getattr(self._local, "stack", None)
        if stack is None:
            stack = []
            self._local.stack = stack
        return stack

    def _node_for(self, path: Path, parent: Path) -> _Node:
        with self._lock:
            node = self._nodes.get(path)
            if node is None:
                node = _Node(path)
                self._nodes[path] = node
                self._children.setdefault(parent, []).append(path)
                self._children.setdefault(path, [])
            self._thread_ids.add(threading.get_ident())
            return node

    def _push(self, label: str) -> None:
        stack = self._stack()
        parent = stack[-1].path if stack else ()
        path = parent + (label,)
        node = self._node_for(path, parent)
        stack.append(_Frame(path, node, clock()))

    def _pop(self, failed: bool = False) -> None:
        end = clock()
        stack = self._stack()
        if not stack:
            logger.warning(
                "%s: a stage was left on a thread that never entered one; ignoring it",
                self.name,
            )
            return
        frame = stack.pop()
        elapsed = max(end - frame.start, 0.0)
        with self._lock:
            frame.node.durations.append(elapsed)
            frame.node.child_seconds += frame.child_seconds
            if failed:
                frame.node.errors += 1
        if stack:
            stack[-1].child_seconds += elapsed

    def reset(self) -> None:
        """Throw away every measurement and start again."""
        with self._lock:
            self._nodes.clear()
            self._children = {(): []}
            self._thread_ids.clear()
            self._repeats = 1
            self._warmup = 0
        self._local = threading.local()

    # -- running a pipeline ---------------------------------------------------------
    def run(
        self,
        pipeline: Sequence[Step],
        data: Any = None,
        *,
        repeats: int = 5,
        warmup: int = 1,
    ) -> ProfileReport:
        """Time a list of steps, feeding each step's output into the next.

        Args:
            pipeline: Steps as ``(label, callable)`` pairs, or bare callables (their
                ``__name__`` becomes the label). An empty list is allowed and gives an
                empty report.
            data: The input handed to the first step. Every repeat starts from this same
                value. A step that returns ``None`` passes its input along unchanged, so
                in-place steps work.
            repeats: How many timed passes to make (at least 1).
            warmup: How many untimed passes to make first (0 or more). Warmup runs pay
                the cold-cache and lazy-import costs so the timed runs measure steady
                state - set it to 0 when the cold cost is what you want to see. The pause
                is thread-local: it silences this thread only, so stages recorded on other
                threads during a warmup pass are kept rather than silently dropped, and
                work a step fans out to worker threads is counted even during warmup.

        Returns:
            The :class:`ProfileReport` for everything recorded on this profiler so far.

        Raises:
            ValueError: If the pipeline is not a sequence of steps, or the counts are
                not positive integers.
        """
        steps = _normalise_pipeline(pipeline)
        repeats = _positive_int(repeats, "repeats", minimum=1)
        warmup = _positive_int(warmup, "warmup", minimum=0)
        self._repeats = repeats
        self._warmup = warmup
        for _ in range(warmup):
            # Thread-local on purpose: pausing globally would discard stages that other
            # threads record while this one happens to be warming up.
            previous = getattr(self._local, "paused", False)
            self._local.paused = True
            try:
                self._execute(steps, data)
            finally:
                self._local.paused = previous
        for _ in range(repeats):
            self._execute(steps, data)
        return self.report()

    def _execute(self, steps: List[Tuple[str, Callable[[Any], Any]]], data: Any) -> Any:
        value = data
        for label, func in steps:
            with self.stage(label):
                produced = func(value)
            if produced is not None:
                value = produced
        return value

    # -- results --------------------------------------------------------------------
    def report(self) -> ProfileReport:
        """Build the :class:`ProfileReport` for everything recorded so far.

        Safe to call at any time, including with nothing recorded: that gives an empty
        report whose ``total_ms`` is 0.0 and whose ``bottleneck`` is ``None``, never a
        division by zero.
        """
        with self._lock:
            snapshot = {
                path: (list(node.durations), node.child_seconds, node.errors)
                for path, node in self._nodes.items()
            }
            children = {path: list(kids) for path, kids in self._children.items()}
            threads = len(self._thread_ids) or 1
            repeats, warmup = self._repeats, self._warmup

        roots = children.get((), [])
        totals = {path: sum(snapshot[path][0]) for path in snapshot}
        run_total = sum(totals[path] for path in roots)

        stages: List[Stage] = []
        self._collect(roots, children, snapshot, totals, run_total, None, stages)
        return ProfileReport(
            name=self.name,
            stages=stages,
            total_ms=to_ms(run_total),
            repeats=repeats,
            warmup=warmup,
            overhead_ms_per_stage=to_ms(measure_stage_overhead()),
            threads=threads,
        )

    def _collect(
        self,
        paths: List[Path],
        children: Dict[Path, List[Path]],
        snapshot: Dict[Path, Tuple[List[float], float, int]],
        totals: Dict[Path, float],
        run_total: float,
        parent: Optional[str],
        out: List[Stage],
    ) -> None:
        """Walk one level of the tree in the order the stages were first seen."""
        level_total = sum(totals[path] for path in paths)
        for path in paths:
            durations, child_seconds, errors = snapshot[path]
            total = totals[path]
            calls = len(durations)
            self_seconds = max(total - child_seconds, 0.0)
            if level_total > 0:
                share = 100.0 * total / level_total
            else:  # every stage measured as zero: split the level evenly, never 0/0
                share = 100.0 / len(paths)
            out.append(
                Stage(
                    label=path[-1],
                    calls=calls,
                    total_ms=to_ms(total),
                    mean_ms=to_ms(total / calls) if calls else 0.0,
                    p95_ms=percentile_ms(durations, 95),
                    share=share,
                    depth=len(path) - 1,
                    parent=parent,
                    self_ms=to_ms(self_seconds),
                    pct_of_total=(100.0 * total / run_total) if run_total > 0 else 0.0,
                    self_pct_of_total=(
                        100.0 * self_seconds / run_total if run_total > 0 else 0.0
                    ),
                    median_ms=median_ms(durations),
                    min_ms=to_ms(min(durations)) if durations else 0.0,
                    max_ms=to_ms(max(durations)) if durations else 0.0,
                    first_ms=to_ms(durations[0]) if durations else 0.0,
                    errors=errors,
                    path="/".join(path),
                )
            )
            kids = children.get(path, [])
            if kids:
                self._collect(
                    kids, children, snapshot, totals, run_total, "/".join(path), out
                )

    def __repr__(self) -> str:
        return f"<Profiler {self.name!r}: {len(self._nodes)} stage(s) recorded>"


def _positive_int(value: Any, field: str, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field} must be an integer, got {value!r}")
    if value < minimum:
        raise ValueError(f"{field} must be >= {minimum}, got {value}")
    return value


def _normalise_pipeline(pipeline: Any) -> List[Tuple[str, Callable[[Any], Any]]]:
    """Accept ``[(label, func), ...]``, bare callables, or a mix of both."""
    if pipeline is None:
        raise ValueError("pipeline must be a list of steps, got None")
    if callable(pipeline) or isinstance(pipeline, (str, bytes, dict)):
        raise ValueError(
            "pipeline must be a list of (label, function) steps, got "
            + type(pipeline).__name__
        )
    try:
        items = list(pipeline)
    except TypeError as exc:
        raise ValueError(
            "pipeline must be a list of (label, function) steps, got "
            + type(pipeline).__name__
        ) from exc

    steps: List[Tuple[str, Callable[[Any], Any]]] = []
    for position, item in enumerate(items):
        if callable(item):
            label = getattr(item, "__name__", None) or f"step{position + 1}"
            steps.append((str(label), item))
            continue
        if isinstance(item, (tuple, list)) and len(item) == 2 and callable(item[1]):
            label = item[0]
            if not isinstance(label, str) or not label.strip():
                raise ValueError(
                    f"step {position + 1}: the label must be a non-empty string, got {label!r}"
                )
            steps.append((label, item[1]))
            continue
        raise ValueError(
            f"step {position + 1} must be a (label, function) pair or a callable, got {item!r}"
        )
    return steps


__all__ = ["Profiler"]
