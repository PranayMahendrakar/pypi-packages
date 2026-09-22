"""The command line: every flag, every exit code, and output that survives a
console which cannot encode the filenames it is printing."""
from __future__ import annotations

import io
import json
import re
import subprocess
import sys

import numpy as np
import pytest
from PIL import Image

from conftest import as_rgb, blurred
from image_quality_ai import cli


@pytest.fixture
def shots(tmp_path, good):
    """A folder with one good photo under a non-ASCII name, one blurred, and a
    subfolder holding a third, so --recursive has something to find."""
    Image.fromarray(as_rgb(good)).save(tmp_path / "写真-café.png")
    Image.fromarray(as_rgb(blurred(good, 8.0))).save(tmp_path / "blurred.png")
    (tmp_path / "more").mkdir()
    Image.fromarray(as_rgb(good)).save(tmp_path / "more" / "nested.png")
    (tmp_path / "notes.txt").write_text("not an image", encoding="utf-8")
    return tmp_path


#: Images in the ``corpus`` fixture. More than the summary's default five, so
#: ``--worst N`` has something to cut on both sides of it.
CORPUS_SIZE = 7


@pytest.fixture
def corpus(tmp_path, good):
    """Seven photos, each softer than the last, so every score is different."""
    for index in range(CORPUS_SIZE):
        Image.fromarray(as_rgb(blurred(good, index * 1.1))).save(
            tmp_path / "shot{0}.png".format(index)
        )
    return tmp_path


# ---------------------------------------------------------------------------
# the happy paths
# ---------------------------------------------------------------------------
def test_help_exits_zero_and_names_the_program(capsys):
    with pytest.raises(SystemExit) as caught:
        cli.main(["--help"])

    assert caught.value.code == 0
    out = capsys.readouterr().out
    assert "image-quality-ai" in out
    assert "--fail-under" in out


def test_version_exits_zero(capsys):
    with pytest.raises(SystemExit) as caught:
        cli.main(["--version"])

    assert caught.value.code == 0
    assert "image-quality-ai" in capsys.readouterr().out


def test_one_file_prints_the_full_report(shots, capsys):
    code = cli.main([str(shots / "写真-café.png")])

    out = capsys.readouterr().out
    assert code == 0
    assert "Image quality:" in out
    assert "sharpness" in out and "framing" in out
    assert "写真-café.png" in out


def test_a_directory_prints_the_batch_summary(shots, capsys):
    code = cli.main([str(shots)])

    out = capsys.readouterr().out
    assert code == 0
    assert "2 image(s) assessed" in out
    assert "Worst first:" in out


def test_recursive_finds_the_nested_photo_and_skips_the_text_file(shots, capsys):
    code = cli.main([str(shots), "--recursive", "--quiet"])

    out = capsys.readouterr().out
    assert code == 0
    assert len(out.strip().splitlines()) == 3
    assert "nested.png" in out
    assert "notes.txt" not in out


def test_quiet_is_one_line_per_image(shots, capsys):
    cli.main([str(shots), "--quiet"])

    lines = capsys.readouterr().out.strip().splitlines()
    assert len(lines) == 2
    assert any("not usable" in line for line in lines)
    assert any(" usable" in line for line in lines)


def test_json_output_is_valid_json_and_keeps_unicode(shots, capsys):
    code = cli.main([str(shots), "--json"])

    data = json.loads(capsys.readouterr().out)
    assert code == 0
    assert data["count"] == 2
    sources = " ".join(item["source"] for item in data["results"])
    assert "写真" in sources
    assert data["results"][0]["metrics"]["sharpness"]["value"] >= 0.0


def test_output_writes_utf8_to_a_file(shots, tmp_path, capsys):
    target = tmp_path / "report.json"

    code = cli.main([str(shots), "--json", "--output", str(target)])

    assert code == 0
    assert capsys.readouterr().out == ""
    data = json.loads(target.read_text(encoding="utf-8"))
    assert data["count"] == 2


#: A row of the worst-first list: four spaces, a score, a grade, two spaces.
_WORST_ROW = re.compile(r"^ +\d+\.\d [A-F]  \S")


def _worst_rows(text: str):
    """Just the worst-first entries, without their wrapped continuation lines."""
    return [line for line in text.splitlines() if _WORST_ROW.match(line)]


@pytest.mark.parametrize("asked", [0, 1, 2, 5, 7, 20])
def test_worst_lists_exactly_what_was_asked_for_and_only_once(corpus, capsys, asked):
    """--help promises "how many of the worst images to list", for every N.

    Anything from 0 to 5 used to be accepted and then ignored, because the
    summary printed its own hard-coded five; anything above 5 printed a second
    block that repeated those five rows before continuing.
    """
    code = cli.main([str(corpus), "--worst", str(asked)])
    out = capsys.readouterr().out

    assert code == 0
    assert out.count("Worst") <= 1, "one worst-first list, not two:\n{0}".format(out)
    assert len(_worst_rows(out)) == min(asked, CORPUS_SIZE)


def test_worst_lists_the_lowest_scores_in_order(corpus, capsys):
    cli.main([str(corpus), "--worst", "3"])
    rows = _worst_rows(capsys.readouterr().out)

    scores = [float(row.split()[0]) for row in rows]
    assert len(scores) == 3
    assert scores == sorted(scores)


def test_list_thresholds_prints_the_whole_table(capsys):
    code = cli.main(["--list-thresholds"])

    out = capsys.readouterr().out
    assert code == 0
    assert "sharpness_blurry" in out and "100" in out
    assert "usable_score" in out


def test_a_threshold_override_changes_the_verdict(shots, capsys):
    """The defaults are tuned for ordinary photographs. Somebody grading soft
    imagery - scans, astro frames - moves the sharpness band down and the same
    pixels come back with the opposite verdict."""
    strict = cli.main([str(shots / "blurred.png"), "--quiet"])
    assert strict == cli.EXIT_ERROR or "not usable" in capsys.readouterr().out

    lenient = cli.main([
        str(shots / "blurred.png"), "--quiet",
        "--threshold", "sharpness_floor=0.05",
        "--threshold", "sharpness_blurry=0.2",
        "--threshold", "sharpness_good=2",
    ])
    out = capsys.readouterr().out

    assert lenient == 0
    assert "not usable" not in out
    assert " usable" in out


def test_list_thresholds_shows_the_override(capsys):
    cli.main(["--list-thresholds", "--threshold", "sharpness_blurry=42"])

    rows = [
        line.split()
        for line in capsys.readouterr().out.splitlines()
        if line.startswith(("sharpness_blurry", "sharpness_good"))
    ]
    assert ["sharpness_blurry", "42"] in rows       # the override
    assert ["sharpness_good", "600"] in rows        # everything else untouched


# ---------------------------------------------------------------------------
# exit codes
# ---------------------------------------------------------------------------
def test_fail_under_returns_two_when_something_scores_low(shots, capsys):
    code = cli.main([str(shots), "--quiet", "--fail-under", "60"])

    assert code == cli.EXIT_BELOW_THRESHOLD
    assert "scored below 60" in capsys.readouterr().err


def test_fail_under_returns_zero_when_everything_clears_it(shots, capsys):
    code = cli.main([str(shots / "写真-café.png"), "--quiet", "--fail-under", "60"])

    assert code == 0
    assert capsys.readouterr().err == ""


def test_no_arguments_prints_help_and_returns_one(capsys):
    code = cli.main([])

    assert code == cli.EXIT_ERROR
    assert "usage:" in capsys.readouterr().out


def test_a_missing_path_returns_one(tmp_path, capsys):
    code = cli.main([str(tmp_path / "nowhere.png")])

    assert code == cli.EXIT_ERROR
    assert "no such file or directory" in capsys.readouterr().err


def test_a_folder_with_no_images_returns_one(tmp_path, capsys):
    code = cli.main([str(tmp_path)])

    assert code == cli.EXIT_ERROR
    assert "no images found" in capsys.readouterr().err


def test_only_unreadable_files_returns_one(tmp_path, capsys):
    (tmp_path / "broken.png").write_bytes(b"not an image")

    code = cli.main([str(tmp_path), "--quiet"])

    assert code == cli.EXIT_ERROR
    assert "unreadable" in capsys.readouterr().out


def test_a_bad_threshold_is_a_usage_error_not_a_traceback(capsys):
    with pytest.raises(SystemExit) as caught:
        cli.main(["--list-thresholds", "--threshold", "nonsense=1"])

    assert caught.value.code == cli.EXIT_ERROR
    assert "unknown threshold" in capsys.readouterr().err


def test_a_threshold_without_a_number_is_a_usage_error(capsys):
    with pytest.raises(SystemExit) as caught:
        cli.main(["--list-thresholds", "--threshold", "sharpness_blurry=sharp"])

    assert caught.value.code == cli.EXIT_ERROR
    assert "needs a number" in capsys.readouterr().err


def test_an_unparseable_flag_exits_one_not_two(capsys):
    """Exit code 2 belongs to --fail-under, so a typo must not claim it."""
    with pytest.raises(SystemExit) as caught:
        cli.main(["--nonsense"])

    assert caught.value.code == cli.EXIT_ERROR


def test_workers_and_worst_are_validated(shots, capsys):
    for argv in ([str(shots), "--workers", "0"], [str(shots), "--worst", "-1"]):
        with pytest.raises(SystemExit) as caught:
            cli.main(argv)
        assert caught.value.code == cli.EXIT_ERROR


def test_an_unwritable_output_path_returns_one(shots, tmp_path, capsys):
    code = cli.main([str(shots), "--quiet", "--output", str(tmp_path / "gone" / "x.txt")])

    assert code == cli.EXIT_ERROR
    assert "cannot write output" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def test_parse_overrides_reads_pairs():
    assert cli.parse_overrides(["sharpness_blurry=40", "usable_score=70"]) == {
        "sharpness_blurry": 40.0,
        "usable_score": 70.0,
    }
    with pytest.raises(ValueError, match="NAME=VALUE"):
        cli.parse_overrides(["sharpness_blurry"])


def test_collect_paths_sorts_and_filters(shots):
    found = cli.collect_paths([str(shots)], recursive=False)

    assert [item.rsplit("\\", 1)[-1].rsplit("/", 1)[-1] for item in found] == sorted(
        ["blurred.png", "写真-café.png"]
    )
    with pytest.raises(FileNotFoundError):
        cli.collect_paths([str(shots / "gone")], recursive=False)


# ---------------------------------------------------------------------------
# encoding safety
# ---------------------------------------------------------------------------
def test_a_console_that_cannot_encode_the_filename_does_not_crash(shots, monkeypatch):
    """The Windows default console is cp1252 and cannot encode 写真. The CLI
    retunes its own streams to UTF-8 rather than dying halfway through a report."""
    raw = io.BytesIO()
    monkeypatch.setattr(
        sys, "stdout", io.TextIOWrapper(raw, encoding="cp1252", errors="strict")
    )

    code = cli.main([str(shots), "--quiet"])
    sys.stdout.flush()

    assert code == 0
    printed = raw.getvalue().decode("utf-8")
    assert "写真-café.png" in printed


def test_the_module_entry_point_runs_with_output_piped(shots):
    """`python -m image_quality_ai` through a pipe: no console, no encoding help
    from the terminal, non-ASCII in the data."""
    finished = subprocess.run(
        [sys.executable, "-m", "image_quality_ai", str(shots), "--quiet"],
        capture_output=True,
    )

    assert finished.returncode == 0, finished.stderr.decode("utf-8", "replace")
    assert "写真-café.png" in finished.stdout.decode("utf-8")


def test_the_module_entry_point_reports_its_version():
    finished = subprocess.run(
        [sys.executable, "-m", "image_quality_ai", "--version"], capture_output=True
    )

    assert finished.returncode == 0
    assert "image-quality-ai" in finished.stdout.decode("utf-8")
