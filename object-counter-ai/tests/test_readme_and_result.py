"""The README examples run as written, and results explain themselves."""
from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path

import numpy as np

import object_counter_ai as oc
from synth_images import discs_image

README = Path(__file__).resolve().parent.parent / "README.md"


def _blocks(lang: str):
    """Fenced blocks whose info string is exactly ``lang`` ("" for bare fences)."""
    blocks, current, info = [], None, None
    for line in README.read_text(encoding="utf-8").splitlines():
        if line.startswith("```"):
            if current is None:
                current, info = [], line[3:].strip()
            else:
                if info == lang:
                    blocks.append("\n".join(current) + "\n")
                current = None
        elif current is not None:
            current.append(line)
    return blocks


def test_quickstart_runs_verbatim_and_prints_what_the_readme_shows():
    python_blocks = _blocks("python")
    quickstart = python_blocks[0]
    assert 3 <= len([ln for ln in quickstart.splitlines() if ln.strip()]) <= 6
    out = io.StringIO()
    namespace: dict = {}
    with contextlib.redirect_stdout(out):
        exec(compile(quickstart, "README-quickstart", "exec"), namespace)
    shown = _blocks("")[1]        # the output block right after the quickstart
    assert out.getvalue().strip() == shown.strip()
    assert "4 blobs counted" in out.getvalue()


def test_every_readme_python_example_runs_and_its_comments_are_true():
    namespace: dict = {}
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        for block in _blocks("python"):
            if block.lstrip().startswith(("count(", "Counter(")):
                continue           # signatures, not examples
            exec(compile(block, "README", "exec"), namespace)
    printed = out.getvalue()
    assert "2 {'bolt': 1, 'nut': 1} 0.875" in printed
    assert "{'line 1': 1}" in printed


def test_readme_first_paragraph_is_honest_about_the_classical_counter():
    text = README.read_text(encoding="utf-8")
    first = text.split("\n\n")[1]
    for phrase in ("heuristic", "blobs, not objects", "parts on a conveyor",
                   "cells on", "bolts on a tray", "useless for people in a street"):
        assert phrase in first.replace("\n", " "), phrase
    assert "object detection" not in text.lower()


def test_summary_is_plain_ascii_for_every_kind_of_result():
    results = [
        oc.count(discs_image(4, seed=1)),
        oc.count(np.zeros((0, 0))),
        oc.count(discs_image(2), detector=lambda im: [((1, 2, 30, 40), "café été", 0.5)]),
        oc.count(discs_image(2), detector=lambda im: 1 / 0),
        oc.count(discs_image(3, seed=2), region=[(0, 0), (150, 0), (0, 150)]),
    ]
    for result in results:
        text = result.summary()
        # Only user-supplied labels may carry non-ASCII; the package's own text never does.
        own = text.replace("café été", "")
        assert own.isascii(), text


def test_to_dict_is_json_safe_and_complete():
    result = oc.count(discs_image(3, seed=3), region=(0, 0, 320, 240))
    data = result.to_dict()
    text = json.dumps(data, ensure_ascii=False)
    back = json.loads(text)
    assert back["count"] == 3 and back["method"] == "classical" and back["ok"] is True
    assert len(back["objects"]) == 3
    assert set(back["objects"][0]) >= {"box", "label", "score", "area", "centre"}
    assert back["details"]["confidence_factors"] is not None
    assert back["region"] == {"kind": "box", "box": [0, 0, 320, 240]}


def test_str_is_the_summary():
    result = oc.count(discs_image(2, seed=5))
    assert str(result) == result.summary()
    counter = oc.Counter()
    counter.update(discs_image(2, seed=5))
    assert str(counter) == counter.summary()


def test_public_api():
    assert oc.__version__ == "0.1.0"
    assert set(oc.__all__) == {"count", "Counter", "CountResult", "__version__"}
    assert isinstance(oc.count(discs_image(1)), oc.CountResult)
