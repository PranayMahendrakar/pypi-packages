"""The command line: exit codes, encoding, and writing only when asked."""
from __future__ import annotations

import json
import os
import subprocess
import sys

import numpy as np
import pytest
from PIL import Image

from smart_crop_ai import cli
from conftest import subject_array

#: A filename with CJK, Cyrillic, an accent and an emoji in it.
UNICODE_NAME = "東京-привет-café-\U0001f4f7.png"


@pytest.fixture
def photo(tmp_path):
    """One image on disk to run the CLI against."""
    path = tmp_path / "photo.png"
    Image.fromarray(subject_array()).save(path)
    return str(path)


@pytest.fixture
def folder(tmp_path):
    """A directory of three images, one of them under a subdirectory."""
    root = tmp_path / "shots"
    (root / "inner").mkdir(parents=True)
    for name in ("a.png", "b.png"):
        Image.fromarray(subject_array()).save(root / name)
    Image.fromarray(subject_array()).save(root / "inner" / "c.png")
    (root / "notes.txt").write_text("not an image", encoding="utf-8")
    return str(root)


# ------------------------------------------------------------------- basics


def test_help_exits_zero(capsys):
    with pytest.raises(SystemExit) as caught:
        cli.main(["--help"])

    assert caught.value.code == 0
    assert "smart-crop-ai" in capsys.readouterr().out


def test_version_exits_zero(capsys):
    with pytest.raises(SystemExit) as caught:
        cli.main(["--version"])

    assert caught.value.code == 0
    assert "0.1.0" in capsys.readouterr().out


def test_no_arguments_prints_help_and_fails(capsys):
    assert cli.main([]) == cli.EXIT_ERROR
    assert "usage" in capsys.readouterr().out


def test_a_plain_run_prints_the_summary_and_writes_nothing(photo, tmp_path, capsys):
    before = sorted(p.name for p in tmp_path.iterdir())

    assert cli.main([photo, "--ratio", "1:1"]) == 0

    out = capsys.readouterr().out
    assert "confidence" in out
    assert "strategy" in out
    assert sorted(p.name for p in tmp_path.iterdir()) == before


def test_width_and_height_work(photo, capsys):
    assert cli.main([photo, "--width", "300", "--height", "200"]) == 0
    assert "300 x 200" in capsys.readouterr().out


def test_json_output_parses(photo, capsys):
    assert cli.main([photo, "--ratio", "16:9", "--json"]) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["size"] == [711, 400]
    assert 0.0 <= payload["confidence"] <= 1.0


def test_quiet_is_one_line_per_image(folder, capsys):
    assert cli.main([folder, "--ratio", "1:1", "--quiet"]) == 0

    lines = [line for line in capsys.readouterr().out.splitlines() if line.strip()]
    assert len(lines) == 2          # a.png and b.png, not the one in inner/


def test_recursive_walks_subdirectories(folder, capsys):
    assert cli.main([folder, "--ratio", "1:1", "--quiet", "--recursive"]) == 0

    lines = [line for line in capsys.readouterr().out.splitlines() if line.strip()]
    assert len(lines) == 3


@pytest.mark.parametrize("strategy", ["auto", "saliency", "entropy", "edges", "center"])
def test_every_strategy_is_reachable_from_the_cli(photo, strategy, capsys):
    assert cli.main([photo, "--ratio", "1:1", "--strategy", strategy]) == 0
    assert "confidence" in capsys.readouterr().out


# ------------------------------------------------------------------ writing


def test_output_writes_one_file(photo, tmp_path, capsys):
    destination = tmp_path / "hero.jpg"

    assert cli.main([photo, "--ratio", "1:1", "--output", str(destination)]) == 0

    assert destination.exists()
    assert Image.open(destination).size == (400, 400)
    assert "written to" in capsys.readouterr().out


def test_out_dir_writes_every_image_with_a_suffix(folder, tmp_path, capsys):
    out = tmp_path / "square"

    code = cli.main(
        [folder, "--ratio", "1:1", "--out-dir", str(out), "--suffix", "-crop"]
    )

    assert code == 0
    assert sorted(p.name for p in out.iterdir()) == ["a-crop.png", "b-crop.png"]


def test_a_suffix_starting_with_a_dash_is_taken_as_a_value(folder, tmp_path):
    """The README writes --suffix -crop; argparse alone would call that an option."""
    out = tmp_path / "dashed"

    code = cli.main(
        [folder, "--ratio", "1:1", "--out-dir", str(out), "--suffix", "-crop"]
    )

    assert code == 0
    assert sorted(p.name for p in out.iterdir()) == ["a-crop.png", "b-crop.png"]


def test_the_equals_spelling_of_a_dashed_suffix_still_works(folder, tmp_path):
    out = tmp_path / "equals"

    assert cli.main([folder, "--ratio", "1:1", "--out-dir", str(out), "--suffix=-x"]) == 0
    assert sorted(p.name for p in out.iterdir()) == ["a-x.png", "b-x.png"]


def test_a_real_option_after_suffix_is_still_a_missing_value(photo, capsys):
    """Gluing must not swallow a genuine flag and hide the user's mistake."""
    with pytest.raises(SystemExit) as caught:
        cli.main([photo, "--ratio", "1:1", "--suffix", "--quiet"])

    assert caught.value.code == cli.EXIT_ERROR
    assert "--suffix" in capsys.readouterr().err


@pytest.mark.parametrize(
    "argv, expected",
    [
        (["a.png", "--suffix", "-crop"], ["a.png", "--suffix=-crop"]),
        (["a.png", "--suffix", "crop"], ["a.png", "--suffix", "crop"]),
        (["a.png", "--suffix", "--quiet"], ["a.png", "--suffix", "--quiet"]),
        (["a.png", "--suffix"], ["a.png", "--suffix"]),
        (["--quiet", "-r"], ["--quiet", "-r"]),
    ],
)
def test_glue_dash_values_only_touches_what_it_should(argv, expected):
    assert cli.glue_dash_values(argv, cli.build_parser()) == expected


def test_thumb_resizes_after_cropping(photo, tmp_path):
    destination = tmp_path / "thumb.png"

    code = cli.main(
        [photo, "--ratio", "1:1", "--thumb", "64x48", "--output", str(destination)]
    )

    assert code == 0
    assert Image.open(destination).size == (64, 48)


def test_thumb_on_a_dry_run_says_it_did_nothing(photo, capsys):
    assert cli.main([photo, "--ratio", "1:1", "--thumb", "64"]) == 0
    assert "was not applied" in capsys.readouterr().out


def test_report_is_written_as_utf8(photo, tmp_path, capsys):
    report = tmp_path / "deep" / "report.txt"

    assert cli.main([photo, "--ratio", "1:1", "--report", str(report)]) == 0

    text = report.read_text(encoding="utf-8")
    assert "confidence" in text
    assert text == capsys.readouterr().out.rstrip("\n") + "\n"


def test_output_refuses_more_than_one_image(folder, tmp_path, capsys):
    code = cli.main([folder, "--ratio", "1:1", "--output", str(tmp_path / "x.png")])

    assert code == cli.EXIT_ERROR
    assert "use --out-dir" in capsys.readouterr().err


# -------------------------------------------------------------- exit codes


def test_min_confidence_exits_two_when_the_crop_is_a_guess(tmp_path, capsys):
    flat = tmp_path / "flat.png"
    Image.fromarray(np.full((200, 400, 3), 128, dtype=np.uint8)).save(flat)

    code = cli.main([str(flat), "--ratio", "1:1", "--min-confidence", "0.5"])

    assert code == cli.EXIT_LOW_CONFIDENCE
    assert "scored below" in capsys.readouterr().err


def test_min_confidence_passes_a_confident_crop(photo):
    assert cli.main([photo, "--ratio", "1:1", "--min-confidence", "0.2"]) == 0


def test_a_missing_path_exits_one(tmp_path, capsys):
    code = cli.main([str(tmp_path / "nope.png"), "--ratio", "1:1"])

    assert code == cli.EXIT_ERROR
    assert "no such file" in capsys.readouterr().err


def test_a_directory_with_no_images_exits_one(tmp_path, capsys):
    empty = tmp_path / "empty"
    empty.mkdir()

    assert cli.main([str(empty), "--ratio", "1:1"]) == cli.EXIT_ERROR
    assert "no images found" in capsys.readouterr().err


def test_asking_for_no_size_exits_one(photo, capsys):
    with pytest.raises(SystemExit) as caught:
        cli.main([photo])

    assert caught.value.code == cli.EXIT_ERROR
    assert "nothing to crop to" in capsys.readouterr().err


def test_ratio_and_width_together_exits_one(photo, capsys):
    with pytest.raises(SystemExit) as caught:
        cli.main([photo, "--ratio", "1:1", "--width", "100"])

    assert caught.value.code == cli.EXIT_ERROR
    assert "not both" in capsys.readouterr().err


def test_a_bad_thumb_exits_one(photo, capsys):
    with pytest.raises(SystemExit) as caught:
        cli.main([photo, "--ratio", "1:1", "--thumb", "big"])

    assert caught.value.code == cli.EXIT_ERROR
    assert "--thumb" in capsys.readouterr().err


def test_an_unreadable_file_is_reported_not_crashed(tmp_path, capsys):
    broken = tmp_path / "broken.png"
    broken.write_bytes(b"this is not a PNG")

    assert cli.main([str(broken), "--ratio", "1:1"]) == cli.EXIT_ERROR
    assert "broken.png" in capsys.readouterr().err


# --------------------------------------------------------------- encoding


def test_a_unicode_filename_does_not_break_the_report(tmp_path, capsys):
    path = tmp_path / UNICODE_NAME
    Image.fromarray(subject_array()).save(path)

    assert cli.main([str(path), "--ratio", "1:1"]) == 0
    assert "café" in capsys.readouterr().out


def test_a_unicode_filename_survives_json_and_a_report_file(tmp_path, capsys):
    path = tmp_path / UNICODE_NAME
    Image.fromarray(subject_array()).save(path)
    report = tmp_path / "r.json"

    assert cli.main([str(path), "--ratio", "1:1", "--json", "--report", str(report)]) == 0

    payload = json.loads(report.read_text(encoding="utf-8"))
    assert "東京" in payload["source"]
    assert "東京" in capsys.readouterr().out


def test_the_cli_survives_a_pipe_that_cannot_encode_unicode(tmp_path):
    """The real bug this guards: UnicodeEncodeError under subprocess capture.

    The child is started with a legacy ANSI code page for its streams, which is
    exactly what a piped Windows console looks like. Without the reconfigure in
    ``main()`` this raises and the process dies with a traceback.
    """
    path = tmp_path / UNICODE_NAME
    Image.fromarray(subject_array()).save(path)

    environment = dict(os.environ)
    environment["PYTHONIOENCODING"] = "cp1252"

    completed = subprocess.run(
        [sys.executable, "-m", "smart_crop_ai.cli", str(path), "--ratio", "1:1"],
        capture_output=True,
        env=environment,
    )

    assert completed.returncode == 0, completed.stderr.decode("utf-8", "replace")
    assert b"Traceback" not in completed.stderr
    assert b"UnicodeEncodeError" not in completed.stderr
    assert b"confidence" in completed.stdout


def test_the_installed_console_script_runs(tmp_path):
    """``smart-crop-ai --help`` via the module entry point, as a real process."""
    completed = subprocess.run(
        [sys.executable, "-m", "smart_crop_ai.cli", "--help"],
        capture_output=True,
    )

    assert completed.returncode == 0
    assert b"--ratio" in completed.stdout


# ------------------------------------------------------------- helper units


@pytest.mark.parametrize(
    "text, expected",
    [("64", (64, 64)), ("200x150", (200, 150)), ("200X150", (200, 150)),
     ("200,150", (200, 150)), ("200:150", (200, 150)), (None, None)],
)
def test_parse_thumb(text, expected):
    assert cli.parse_thumb(text) == expected


@pytest.mark.parametrize("text", ["big", "0x10", "1x2x3", "-5", ""])
def test_parse_thumb_rejects_nonsense(text):
    with pytest.raises(ValueError):
        cli.parse_thumb(text)


def test_destination_for_keeps_the_name_and_extension():
    assert cli.destination_for("a/b/photo.jpg", "out", "-crop") == os.path.join(
        "out", "photo-crop.jpg"
    )


def test_collect_paths_sorts_and_skips_non_images(folder):
    found = cli.collect_paths([folder], recursive=False)

    assert [os.path.basename(p) for p in found] == ["a.png", "b.png"]


def test_build_parser_is_importable():
    parser = cli.build_parser()
    assert parser.prog == "smart-crop-ai"
