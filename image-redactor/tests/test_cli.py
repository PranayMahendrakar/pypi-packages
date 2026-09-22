"""The command line: argument parsing, exit codes, JSON, and encoding safety."""
from __future__ import annotations

import json
import os
import subprocess
import sys

import numpy as np
import pytest
from PIL import Image

from image_redactor import __version__
from image_redactor.cli import build_parser, main, parse_region

from conftest import noise

UNICODE_NAME = "स्कूल-校園-café"


def _write(path, array=None):
    """Save a small deterministic image at ``path`` and return it as a string."""
    Image.fromarray(noise(60, 80, seed=41) if array is None else array).save(path)
    return str(path)


# --------------------------------------------------------------------------- #
# parsing
# --------------------------------------------------------------------------- #

def test_parse_region_accepts_commas_and_spaces():
    assert parse_region("40,20,100,90") == [40.0, 20.0, 100.0, 90.0]
    assert parse_region("40 20 100 90") == [40.0, 20.0, 100.0, 90.0]
    assert parse_region(" 40 , 20 , 100 , 90 ") == [40.0, 20.0, 100.0, 90.0]


@pytest.mark.parametrize("raw", ["40,20,100", "40,20,100,90,5", "a,b,c,d", ""])
def test_parse_region_rejects_anything_that_is_not_four_numbers(raw):
    with pytest.raises(ValueError, match="LEFT,TOP,RIGHT,BOTTOM"):
        parse_region(raw)


def test_the_parser_advertises_every_method():
    help_text = build_parser().format_help()
    for method in ("blur", "pixelate", "fill", "blackout"):
        assert method in help_text


def test_help_exits_cleanly_and_warns_about_the_heuristics(capsys):
    with pytest.raises(SystemExit) as caught:
        main(["--help"])
    assert caught.value.code == 0
    out = capsys.readouterr().out
    assert "image-redactor" in out
    assert "weak" in out                      # the caveat is in the help, not buried


def test_version_flag_prints_the_version(capsys):
    with pytest.raises(SystemExit) as caught:
        main(["--version"])
    assert caught.value.code == 0
    assert __version__ in capsys.readouterr().out


# --------------------------------------------------------------------------- #
# doing the work
# --------------------------------------------------------------------------- #

def test_redacting_one_region_writes_the_file_and_prints_the_summary(tmp_path, capsys):
    src = _write(tmp_path / "in.png")
    dst = tmp_path / "out.png"
    code = main([src, "-o", str(dst), "--region", "10,10,50,40", "--method", "blackout"])
    assert code == 0
    assert dst.is_file()
    assert np.asarray(Image.open(dst))[10:40, 10:50].max() == 0
    out = capsys.readouterr().out
    assert "1 region(s) redacted with blackout" in out
    assert "saved to:" in out


def test_region_repeats_for_more_than_one_box(tmp_path, capsys):
    src = _write(tmp_path / "in.png")
    dst = tmp_path / "out.png"
    code = main([src, "-o", str(dst), "--region", "5,5,20,20", "--region", "40,30,70,50",
                 "--method", "blackout", "--expand", "0"])
    assert code == 0
    assert "2 region(s) redacted" in capsys.readouterr().out
    written = np.asarray(Image.open(dst))
    assert written[5:20, 5:20].max() == 0
    assert written[30:50, 40:70].max() == 0


def test_json_output_is_valid_json_and_carries_the_boxes(tmp_path, capsys):
    src = _write(tmp_path / "in.png")
    dst = tmp_path / "out.png"
    code = main([src, "-o", str(dst), "--region", "10,10,50,40", "--json"])
    assert code == 0
    data = json.loads(capsys.readouterr().out)
    assert data["count"] == 1
    assert data["method"] == "blur"
    assert data["saved_to"] == str(dst)


def test_detect_only_writes_nothing_and_says_so(tmp_path, capsys):
    src = _write(tmp_path / "in.png")
    code = main([src, "--detect-only"])
    assert code == 0
    out = capsys.readouterr().out
    assert "nothing was written" in out
    assert "weak pixel heuristics" in out     # the fallback owns up in the output
    assert [item.name for item in tmp_path.iterdir()] == [os.path.basename(src)]


def test_quiet_prints_nothing_on_success(tmp_path, capsys):
    src = _write(tmp_path / "in.png")
    code = main([src, "-o", str(tmp_path / "out.png"), "--region", "5,5,30,30", "--quiet"])
    assert code == 0
    assert capsys.readouterr().out == ""
    assert (tmp_path / "out.png").is_file()   # quiet, but it still did the work


def test_strength_and_expand_reach_the_result(tmp_path, capsys):
    src = _write(tmp_path / "in.png")
    code = main([src, "-o", str(tmp_path / "o.png"), "--region", "10,10,50,40",
                 "--method", "pixelate", "--strength", "0.5", "--expand", "0.2", "--json"])
    assert code == 0
    data = json.loads(capsys.readouterr().out)
    assert data["strength"] == 0.5 and data["expand"] == 0.2
    # a 40x30 box grown by 20 percent is 8 wider and 6 taller on every edge
    assert data["boxes"] == [[2, 4, 58, 46]]
    assert data["irreversible"] is True


# --------------------------------------------------------------------------- #
# failing usefully
# --------------------------------------------------------------------------- #

def test_forgetting_the_output_path_is_an_error_not_a_silent_no_op(tmp_path, capsys):
    src = _write(tmp_path / "in.png")
    code = main([src, "--region", "5,5,30,30"])
    assert code == 1
    assert "pass -o PATH" in capsys.readouterr().err


def test_a_bad_region_string_exits_one_with_the_hint(tmp_path, capsys):
    src = _write(tmp_path / "in.png")
    code = main([src, "-o", str(tmp_path / "o.png"), "--region", "10,10,50"])
    assert code == 1
    assert "LEFT,TOP,RIGHT,BOTTOM" in capsys.readouterr().err


def test_an_inverted_region_exits_one_with_a_clear_message(tmp_path, capsys):
    src = _write(tmp_path / "in.png")
    code = main([src, "-o", str(tmp_path / "o.png"), "--region", "90,10,20,60"])
    assert code == 1
    assert "inverted" in capsys.readouterr().err


def test_a_missing_file_exits_one_and_names_it(tmp_path, capsys):
    code = main([str(tmp_path / "nope.png"), "--detect-only"])
    assert code == 1
    assert "nope.png" in capsys.readouterr().err


def test_an_unknown_method_is_rejected_by_argparse(tmp_path):
    src = _write(tmp_path / "in.png")
    with pytest.raises(SystemExit) as caught:
        main([src, "-o", "o.png", "--method", "smudge"])
    assert caught.value.code == 2


# --------------------------------------------------------------------------- #
# encoding: a non-ASCII name must survive a pipe under any console codepage
# --------------------------------------------------------------------------- #

def test_a_non_ascii_path_prints_without_crashing(tmp_path, capsys):
    src = _write(tmp_path / (UNICODE_NAME + ".png"))
    code = main([src, "--detect-only"])
    assert code == 0
    assert UNICODE_NAME in capsys.readouterr().out


@pytest.mark.parametrize("flags", [["--detect-only"], ["--detect-only", "--json"]])
def test_piped_output_survives_a_legacy_console_codepage(tmp_path, flags):
    """The real regression: stdout is a pipe whose codec cannot encode the data.

    Without the reconfigure at the top of ``main()`` this dies with a
    UnicodeEncodeError, which is exactly the failure QA found in other packages.
    """
    src = _write(tmp_path / (UNICODE_NAME + ".png"))
    env = {**os.environ, "PYTHONIOENCODING": "cp1252"}
    proc = subprocess.run(
        [sys.executable, "-m", "image_redactor.cli", src, *flags],
        capture_output=True, env=env,
    )
    assert proc.returncode == 0, proc.stderr.decode("utf-8", "replace")
    assert b"UnicodeEncodeError" not in proc.stderr
    assert proc.stdout.decode("utf-8", "replace").strip()


def test_running_as_a_module_produces_the_same_json(tmp_path):
    src = _write(tmp_path / "in.png")
    proc = subprocess.run(
        [sys.executable, "-m", "image_redactor.cli", src, "-o",
         str(tmp_path / "out.png"), "--region", "5,5,30,30", "--json"],
        capture_output=True, text=True, encoding="utf-8",
    )
    assert proc.returncode == 0, proc.stderr
    assert json.loads(proc.stdout)["count"] == 1
