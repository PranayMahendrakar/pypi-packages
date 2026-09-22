"""Regressions for the issues the independent review of 0.1.0 turned up.

The big one is ``timeout=``: it reported promptly, but the process could not exit
until the abandoned model had worked through its ENTIRE remaining schedule,
because the worker lived in a ``concurrent.futures`` pool whose atexit hook joins
every worker thread. A model that genuinely never returns never let the process
exit at all.
"""
from __future__ import annotations

import subprocess
import sys
import textwrap
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import model_benchmark
from model_benchmark import benchmark
from model_benchmark._result import BenchmarkReport, ModelResult, _display_width, _pad, _shorten

_CORE = Path(model_benchmark.__file__).parent / "_core.py"


# ------------------------------------------------ timeout: bounded, not endless
def test_an_abandoned_model_does_not_run_the_rest_of_its_schedule():
    """The call already in flight finishes; the remaining repeats do not run."""
    release = threading.Event()
    calls = []

    def hangs(rows):
        calls.append(time.perf_counter())
        release.wait(timeout=10)
        return rows

    report = benchmark({"hangs": hangs}, [1, 2, 3], warmup=0, repeats=3, timeout=0.2)
    assert report.results["hangs"].timed_out is True

    release.set()
    # With the pool the worker went on to run repeats 2 and 3 plus the memory
    # pass, four calls in all. It must stop after the one it was already in.
    deadline = time.perf_counter() + 2.0
    while time.perf_counter() < deadline and len(calls) < 2:
        time.sleep(0.02)
    assert len(calls) == 1


def test_the_timeout_worker_is_a_daemon_thread():
    release = threading.Event()

    def hangs(rows):
        release.wait(timeout=10)
        return rows

    try:
        benchmark({"hangs": hangs}, [1, 2, 3], repeats=2, timeout=0.2)
        workers = [t for t in threading.enumerate() if "model-benchmark" in t.name]
        assert workers, "the abandoned worker should still be running"
        # A non-daemon worker is what kept the interpreter from exiting.
        assert all(worker.daemon for worker in workers)
    finally:
        release.set()


def test_a_model_that_outruns_its_budget_mid_schedule_is_a_clean_timeout():
    """The worker can notice its own deadline before join() gives up on it.

    Both routes have to end in the same place: a timed-out model, never an
    internal exception leaking out as the failure string.
    """

    def slowish(rows):
        time.sleep(0.002)
        return rows

    messages = set()
    for _ in range(10):
        report = benchmark({"s": slowish}, [1, 2, 3], warmup=0, repeats=200, timeout=0.06)
        assert report.results["s"].timed_out is True
        messages.add(report.failures["s"])
    assert messages == {"timed out after 0.06 s"}


def test_the_timeout_path_does_not_use_a_thread_pool():
    """concurrent.futures.thread registers the atexit hook that caused the hang."""
    source = _CORE.read_text(encoding="utf-8")
    assert "ThreadPoolExecutor" not in source
    assert "concurrent.futures" not in source


def test_the_process_exits_promptly_even_when_the_model_never_returns(tmp_path):
    """The end-to-end complaint: the shell prompt has to come back.

    The model here blocks on an event nothing ever sets, which is the realistic
    reason to pass ``timeout=`` at all. Before the fix this process never exited.
    """
    script = tmp_path / "hangs_forever.py"
    script.write_text(
        textwrap.dedent(
            """
            import threading
            from model_benchmark import benchmark

            never = threading.Event()

            def hangs(rows):
                never.wait(timeout=60)   # a backstop; nothing sets this
                return rows

            report = benchmark(
                {"hangs": hangs, "quick": lambda rows: rows},
                [1, 2, 3],
                repeats=2,
                timeout=0.3,
            )
            print("FAILURES", sorted(report.failures))
            """
        ),
        encoding="utf-8",
    )
    started = time.perf_counter()
    try:
        done = subprocess.run(
            [sys.executable, str(script)],
            capture_output=True,
            text=True,
            timeout=40,
        )
    except subprocess.TimeoutExpired:  # pragma: no cover - only on a regression
        pytest.fail("the process never exited: the abandoned worker is holding it open")
    elapsed = time.perf_counter() - started
    assert done.returncode == 0, done.stderr
    assert "FAILURES ['hangs']" in done.stdout
    # Generous on purpose: importing pandas is most of this. The model would have
    # blocked for 60 s, so anything well under that proves it is not being joined.
    assert elapsed < 25.0, "took %.1f s to exit" % elapsed


# ------------------------------------------- to_frame on an unbenchmarked size
def test_to_frame_refuses_a_batch_size_that_was_never_benchmarked():
    report = benchmark(
        {"m": lambda rows: np.zeros(len(rows))},
        pd.DataFrame({"x": np.arange(100.0)}),
        repeats=3,
        batch_sizes=(1, 10),
    )
    with pytest.raises(ValueError, match="batch size 999 was not benchmarked"):
        report.to_frame(batch_size=999)
    with pytest.raises(ValueError, match="this report has 1, 10"):
        report.to_frame(batch_size=999)
    # The sizes that were benchmarked still work.
    assert len(report.to_frame(batch_size=1)) == 1
    assert len(report.to_frame(batch_size=10)) == 1
    assert report.to_frame().equals(report.to_frame(batch_size=1))


def test_to_frame_on_a_report_with_no_batch_sizes_says_so():
    report = BenchmarkReport(results={"a": ModelResult(name="a")})
    with pytest.raises(ValueError, match="this report has none"):
        report.to_frame(batch_size=4)


# ------------------------------------------------- warmup at every batch size
def test_every_batch_size_is_warmed_up_before_it_is_timed():
    seen = []

    def watcher(rows):
        seen.append(len(rows))
        return np.zeros(len(rows))

    benchmark(
        {"w": watcher},
        pd.DataFrame({"x": np.arange(100.0)}),
        warmup=2,
        repeats=2,
        batch_sizes=(1, 50),
    )
    # 2 warmup + 2 timed at each size, then the one memory call at the primary size.
    assert seen == [1, 1, 1, 1, 50, 50, 50, 50, 1]


def test_warmup_zero_still_means_no_untimed_calls_at_any_batch_size():
    seen = []

    def watcher(rows):
        seen.append(len(rows))
        return rows

    benchmark(
        {"w": watcher},
        pd.DataFrame({"x": np.arange(20.0)}),
        warmup=0,
        repeats=1,
        batch_sizes=(1, 5),
    )
    assert seen == [1, 5, 1]


# -------------------------------------------------------- degenerate messages
def _empty_report():
    frame = pd.DataFrame({"x": pd.Series(dtype=float)})
    return benchmark(
        {"a": lambda rows: np.zeros(len(rows), dtype=int)},
        (frame, []),
        metric="accuracy",
        repeats=2,
    )


def test_the_empty_input_warning_reads_forwards():
    report = _empty_report()
    assert report.warnings == ["data has 0 rows, so there is nothing to time"]
    assert not any("larger than" in note for note in report.warnings)
    assert report.summary().isascii()


def test_best_score_does_not_tell_you_to_pass_a_metric_you_already_passed():
    report = _empty_report()
    with pytest.raises(ValueError, match="no model produced a usable accuracy score") as caught:
        report.best("score")
    assert "pass metric=" not in str(caught.value)

    # Without a metric, the original advice is still the right advice.
    plain = benchmark({"a": lambda rows: rows}, [1, 2], repeats=1)
    with pytest.raises(ValueError, match="pass metric="):
        plain.best("score")


# ----------------------------------------------------------- column alignment
def test_summary_columns_line_up_for_wide_model_names():
    report = benchmark(
        {
            "零モデル": lambda rows: np.zeros(len(rows), dtype=int),
            "ascii_name": lambda rows: np.zeros(len(rows), dtype=int),
        },
        (pd.DataFrame({"x": [1.0, 2.0]}), [0, 0]),
        metric="accuracy",
        repeats=2,
    )
    lines = report.summary().splitlines()
    index = next(i for i, line in enumerate(lines) if "latency_ms" in line)
    header = lines[index]
    body = lines[index + 1 : index + 1 + len(report.results)]
    assert len(body) == 2
    assert "零モデル" in body[0] and "ascii_name" in body[1]
    # Every row has to be as wide as the header, counted in terminal columns.
    for line in body:
        assert _display_width(line) == _display_width(header), line
    # str.ljust would have made the CJK row two columns short of the others.
    assert len(body[0]) != len(body[1])


def test_display_width_counts_terminal_cells_not_code_points():
    assert _display_width("零モデル") == 8  # 4 characters, 8 columns
    assert _display_width("ascii") == 5
    assert _display_width("é") == 1  # a combining accent takes no column
    assert _display_width(_pad("零モデル", 12)) == 12
    assert _display_width(_pad("ascii", 12)) == 12
    assert _display_width(_shorten("零モデル" * 3, 8)) <= 8
    assert _shorten("short", 8) == "short"


# ----------------------------------------------------------------- CLI --demo
def test_demo_refuses_to_silently_ignore_data(capsys):
    from model_benchmark import cli

    with pytest.raises(SystemExit) as caught:
        cli.main(["--data", "missing.csv", "--demo"])
    assert caught.value.code == 2
    message = capsys.readouterr().err
    assert "--data" in message and "--demo" in message


def test_demo_refuses_to_silently_ignore_target_or_models(capsys):
    from model_benchmark import cli

    with pytest.raises(SystemExit):
        cli.main(["--demo", "--target", "label"])
    assert "--target" in capsys.readouterr().err

    with pytest.raises(SystemExit):
        cli.main(["--demo", "mine=model_benchmark.cli:demo_case"])
    assert "MODEL" in capsys.readouterr().err

    # --demo on its own still works, and --metric is still allowed with it.
    assert cli.main(["--demo", "--repeats", "2"]) == 0
    assert cli.main(["--demo", "--repeats", "2", "--metric", "f1"]) == 0


# --------------------------------------------------------------- f1 behaviour
def test_f1_documented_conventions_hold():
    from model_benchmark._metrics import f1

    # The positive class is the last label in sorted order.
    assert round(f1(["no", "yes", "yes", "no"], ["no", "yes", "no", "no"]), 4) == 0.6667
    # The ordinary binary case matches sklearn.
    assert round(f1([0, 0, 1, 1, 1, 0], [0, 1, 1, 1, 0, 0]), 4) == 0.6667
    # One label present throughout and predicted perfectly scores 1.0 here.
    assert f1([0, 0, 0, 0], [0, 0, 0, 0]) == 1.0
    assert f1([1, 1, 1], [1, 1, 1]) == 1.0
