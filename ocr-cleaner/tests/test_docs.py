"""The README has to be true. This runs the quickstart in it, exactly as printed."""
from __future__ import annotations

import os
import re

import pytest

import ocr_cleaner

README = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "README.md")


@pytest.fixture(scope="module")
def readme() -> str:
    with open(README, encoding="utf-8") as handle:
        return handle.read()


def python_blocks(text: str):
    """Every ```python block in the README, in order."""
    return re.findall(r"```python\n(.*?)```", text, re.DOTALL)


def test_the_quickstart_block_runs_exactly_as_printed(readme, capsys):
    blocks = python_blocks(readme)
    assert blocks, "the README has no python block"
    exec(compile(blocks[0], "README.md", "exec"), {})
    printed = capsys.readouterr().out
    assert "ocr-cleaner:" in printed
    assert "document" in printed


def test_the_quickstart_output_block_matches_what_it_prints(readme, capsys):
    """The block under the quickstart is its real output, not a sketch of one."""
    exec(compile(python_blocks(readme)[0], "README.md", "exec"), {})
    printed = capsys.readouterr().out.strip().splitlines()
    shown = re.search(r"```\n(ocr-cleaner:.*?)```", readme, re.DOTALL)
    assert shown, "the quickstart output block is missing"
    expected = shown.group(1).strip().splitlines()
    assert printed == expected


def test_every_documented_section_is_present(readme):
    for heading in (
        "# ocr-cleaner", "## Install", "## Quickstart", "## What it does",
        "## API", "## CLI", "## License",
    ):
        assert heading in readme, "{0} is missing".format(heading)


def test_the_sections_are_in_the_documented_order(readme):
    order = ["## Install", "## Quickstart", "## What it does", "## API", "## CLI", "## License"]
    positions = [readme.index(heading) for heading in order]
    assert positions == sorted(positions)


def test_the_install_line_names_the_package(readme):
    assert "pip install ocr-cleaner" in readme


def test_every_name_the_readme_promises_exists(readme):
    for name in ("clean", "clean_file", "estimate_skew", "CleanResult", "Step"):
        assert name in readme
        assert hasattr(ocr_cleaner, name)


def test_the_public_api_is_all_importable():
    for name in ocr_cleaner.__all__:
        assert hasattr(ocr_cleaner, name), name


def test_every_public_function_has_a_docstring():
    for name in ocr_cleaner.__all__:
        thing = getattr(ocr_cleaner, name)
        if callable(thing):
            assert thing.__doc__, "{0} has no docstring".format(name)


def test_the_result_attributes_the_readme_lists_are_real():
    result = ocr_cleaner.clean(
        __import__("numpy").full((200, 160), 250, dtype="uint8")
    )
    for attribute in (
        "image", "steps", "skew_corrected_degrees", "estimated_text_height_px",
        "page_kind", "estimated_skew_degrees", "applied", "skipped",
        "is_blank", "is_photograph", "binary",
    ):
        assert hasattr(result, attribute), attribute
    assert callable(result.summary)
    assert callable(result.to_dict)
    assert callable(result.to_json)
    assert callable(result.save)


def test_the_version_is_the_first_release():
    assert ocr_cleaner.__version__ == "0.1.0"


def test_the_module_docstring_example_is_true():
    """The doctest-shaped example at the top of __init__ has to hold."""
    import numpy as np

    page = np.full((1100, 850), 246, dtype=np.uint8)
    for top in range(120, 1000, 34):
        page[top:top + 11, 90:760] = 50
    result = ocr_cleaner.clean(page)
    assert result.page_kind == "document"
    assert "threshold" in result.applied
