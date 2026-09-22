"""The README quickstart must keep working, exactly as written."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

import ml_inference_profiler as mip

README = Path(__file__).resolve().parents[1] / "README.md"


def test_quickstart_runs():
    steps = [
        ("preprocess", lambda r: [x.strip().lower() for x in r]),
        ("model", lambda r: [len(x) ** 0.5 for x in r]),
        ("postprocess", lambda s: sum(s) / len(s)),
    ]
    report = mip.profile_pipeline(steps, ["  Ünicode text  "] * 2000, repeats=5)

    assert [s.label for s in report.stages] == ["preprocess", "model", "postprocess"]
    assert all(s.calls == 5 for s in report.stages)
    assert report.total_ms > 0
    assert report.bottleneck is not None
    assert report.bottleneck.label in {"preprocess", "model", "postprocess"}
    assert "Bottleneck" in report.summary()
    assert report.suggestions


def _run_readme_blocks() -> dict:
    """Run the exact code blocks the README shows, so the two cannot drift apart.

    The first two blocks are the worked examples (the third is the API reference, which
    is signatures, not runnable code). They share one namespace, exactly as a reader
    pasting them one after the other would.
    """
    blocks = re.findall(r"```python\n(.*?)```", README.read_text(encoding="utf-8"), re.S)
    assert len(blocks) >= 2, "the README must contain the quickstart and the Profiler example"
    namespace = {"__name__": "__readme__"}
    for index, block in enumerate(blocks[:2]):
        exec(compile(block, f"README-block-{index + 1}", "exec"), namespace)
    return namespace


@pytest.mark.skipif(not README.exists(), reason="README.md is not installed with the wheel")
def test_readme_examples_execute():
    assert _run_readme_blocks()["profiler"] is not None


@pytest.mark.skipif(not README.exists(), reason="README.md is not installed with the wheel")
def test_readme_profiler_example_keeps_the_model_beside_preprocessing():
    """Regression: the worked example nested run_model inside the preprocess stage.

    For a package whose pitch is separating the data path from the model, the two must
    read as siblings. Nesting them made the model look like a part of preprocessing.
    """
    namespace = _run_readme_blocks()
    report = namespace["profiler"].report()

    run_model = report.find("run_model")
    preprocess = report.find("preprocess")
    assert run_model is not None and preprocess is not None
    assert run_model.depth == 0 and run_model.parent is None, "the model is nested again"
    assert preprocess.depth == 0
    assert report.find("resize").parent == "preprocess"  # resize is still a child
    # and the example is no longer so small that the shares are rounding noise
    assert report.total_ms > 0.5
