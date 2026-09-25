"""The documentation is code too: run it, do not just read it."""
from __future__ import annotations

import doctest
import pathlib
import re

import pytest

import vision_anomaly
from vision_anomaly import _detector, _features, _loading, _profile, _result

README = pathlib.Path(__file__).resolve().parents[1] / "README.md"


@pytest.mark.parametrize(
    "module",
    [vision_anomaly, _detector, _features, _loading, _profile, _result],
    ids=lambda module: module.__name__,
)
def test_every_docstring_example_is_true(module):
    results = doctest.testmod(module, verbose=False, report=True)

    assert results.failed == 0


def _fenced_blocks() -> list:
    """Every fenced code block in the README, as ``(language, body)`` pairs."""
    text = README.read_text(encoding="utf-8")
    blocks = []
    for raw in text.split("```")[1::2]:
        language, _, body = raw.partition("\n")
        blocks.append((language.strip(), body))
    return blocks


def _quickstart() -> str:
    """The first python block of the README, exactly as a reader would copy it."""
    for language, body in _fenced_blocks():
        if language == "python":
            return body
    raise AssertionError("the README has lost its quickstart block")


@pytest.mark.skipif(not README.exists(), reason="README is not installed beside the package")
def test_the_readme_quickstart_runs_verbatim(capsys):
    """Copy-paste is the first thing every user does. It has to work."""
    code = _quickstart()

    exec(compile(code, "README.md quickstart", "exec"), {"__name__": "__main__"})

    printed = capsys.readouterr().out
    assert "1 anomalous" in printed
    assert "ANOMALOUS" in printed


@pytest.mark.skipif(not README.exists(), reason="README is not installed beside the package")
def test_the_readme_shows_the_output_the_quickstart_actually_produces(capsys):
    """A sample output that drifted from the code is worse than none at all."""
    exec(compile(_quickstart(), "README.md quickstart", "exec"), {"__name__": "__main__"})
    printed = capsys.readouterr().out.strip()

    samples = [body.strip() for language, body in _fenced_blocks() if not language]
    assert printed in samples


@pytest.mark.skipif(not README.exists(), reason="README is not installed beside the package")
def test_the_readme_sections_are_in_the_house_order():
    headings = re.findall(r"^## (.+)$", README.read_text(encoding="utf-8"), re.M)

    assert headings[0] == "Install"
    assert headings[1] == "Quickstart"
    assert headings[-1] == "License"
    assert "API" in headings
    assert "CLI" in headings


@pytest.mark.skipif(not README.exists(), reason="README is not installed beside the package")
def test_the_readme_does_not_promise_a_copy_scores_exactly_zero():
    """It said "exactly zero" two lines from its own output printing 0.02.

    Fitted images can score a little above zero against the profile they built.
    The spec asks for near zero and that is what happens; only the wording was
    wrong, and a wrong promise in a README is a bug report waiting to be filed.
    """
    text = README.read_text(encoding="utf-8")

    assert "exactly zero" not in text
    assert "near zero" in text
    assert "fit_worst" in text


def test_a_copy_of_a_fitted_image_scores_near_zero_and_never_above_fit_worst():
    """The honest version of the claim, checked on every fitted image."""
    from conftest import normal_set
    from vision_anomaly import Detector

    good = normal_set(12)
    detector = Detector().fit(good)
    worst = detector.profile.worst_fit_score

    scores = [detector.score(image.copy()) for image in good]

    assert max(scores) <= worst + 1e-9
    assert worst < detector.sensitivity
    assert all(score >= 0.0 for score in scores)
