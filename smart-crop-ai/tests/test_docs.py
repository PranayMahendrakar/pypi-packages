"""Keep the documentation honest: run what it tells people to run."""
from __future__ import annotations

import doctest
import io
import os
import re
import runpy

import pytest

import smart_crop_ai
from smart_crop_ai import _core, _energy, _images, _result

PACKAGE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
README = os.path.join(PACKAGE_ROOT, "README.md")


def read_readme() -> str:
    with io.open(README, encoding="utf-8") as handle:
        return handle.read()


def first_python_block(text: str) -> str:
    """The first ```python fenced block in the README: the quickstart."""
    match = re.search(r"```python\n(.*?)```", text, re.DOTALL)
    assert match, "the README has no python block"
    return match.group(1)


@pytest.mark.parametrize(
    "module", [smart_crop_ai, _core, _energy, _images, _result]
)
def test_the_docstring_examples_are_true(module):
    """Every >>> in the source must produce what it claims to produce."""
    results = doctest.testmod(module, verbose=False, report=False)
    assert results.failed == 0, "{0} doctest failure(s) in {1}".format(
        results.failed, module.__name__
    )


def test_the_readme_quickstart_runs_verbatim(tmp_path, capsys):
    """Paste the README quickstart into a file, run it, and it must work.

    This is the block QA runs, so it is extracted from the README rather than
    copied into the test: the two cannot drift apart.
    """
    script = tmp_path / "quickstart.py"
    with io.open(script, "w", encoding="utf-8") as handle:
        handle.write(first_python_block(read_readme()))

    runpy.run_path(str(script), run_name="__main__")

    printed = capsys.readouterr().out
    assert "smart-crop-ai" in printed
    assert "confidence" in printed


def test_the_quickstart_output_in_the_readme_is_what_it_prints(tmp_path, capsys):
    """The sample output block under the quickstart must not be aspirational."""
    text = read_readme()
    script = tmp_path / "quickstart.py"
    with io.open(script, "w", encoding="utf-8") as handle:
        handle.write(first_python_block(text))

    runpy.run_path(str(script), run_name="__main__")
    printed = capsys.readouterr().out.strip()

    shown = re.search(r"```\n(smart-crop-ai:.*?)```", text, re.DOTALL)
    assert shown, "the README shows no quickstart output"
    assert printed == shown.group(1).strip()


def test_every_name_the_readme_promises_is_importable():
    for name in ("crop", "crop_to_file", "thumbnail", "CropResult", "STRATEGIES"):
        assert hasattr(smart_crop_ai, name)
        assert name in smart_crop_ai.__all__


def test_every_strategy_in_the_readme_table_exists():
    text = read_readme()
    for strategy in smart_crop_ai.STRATEGIES:
        assert '`"{0}"`'.format(strategy) in text


def test_every_result_attribute_the_readme_lists_exists(subject_image):
    result = smart_crop_ai.crop(subject_image, ratio=1.0)
    for name in (
        "image", "box", "strategy_used", "confidence", "confidence_label",
        "notes", "scores", "size", "offset", "covers", "moved",
    ):
        assert hasattr(result, name), name
    for method in ("summary", "to_dict", "to_json", "save"):
        assert callable(getattr(result, method)), method


def test_the_readme_has_the_sections_the_conventions_require():
    text = read_readme()
    expected = ["## Install", "## What it does", "## API", "## CLI", "## License"]
    positions = [text.find(heading) for heading in expected]

    assert all(position >= 0 for position in positions), expected
    assert positions == sorted(positions), "README sections are out of order"


def test_the_public_functions_all_have_docstrings():
    for name in smart_crop_ai.__all__:
        attribute = getattr(smart_crop_ai, name)
        if callable(attribute) and not isinstance(attribute, type):
            assert attribute.__doc__, "{0} has no docstring".format(name)
