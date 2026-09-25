"""The command line: text, JSON, files, frames, detectors and console encoding."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap

import pytest
from PIL import Image

from object_counter_ai.cli import main, parse_line_spec, parse_region_spec
from synth_images import discs_image, moving_disc_frames

UNICODE_NAME = "zählung_部品_счёт.png"


@pytest.fixture()
def tray(tmp_path):
    path = tmp_path / "tray.png"
    Image.fromarray(discs_image(6, seed=2)).save(path)
    return path


def test_help_exits_zero(capsys):
    with pytest.raises(SystemExit) as exit_info:
        main(["--help"])
    assert exit_info.value.code == 0
    out = capsys.readouterr().out
    assert "blobs, not objects" in out and "--detector" in out


def test_no_arguments_prints_help_and_fails(capsys):
    assert main([]) == 1
    assert "usage" in capsys.readouterr().out


def test_summary_for_one_image(tray, capsys):
    assert main([str(tray)]) == 0
    out = capsys.readouterr().out
    assert "6 blobs counted" in out and str(tray) in out


def test_json_output_and_output_file_with_unicode_name(tmp_path, capsys):
    img = tmp_path / UNICODE_NAME
    Image.fromarray(discs_image(4, seed=1)).save(img)
    target = tmp_path / "out" / "résultat.json"
    assert main([str(img), "--json", "--output", str(target)]) == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["count"] == 4 and printed["path"] == str(img)
    saved = json.loads(target.read_text(encoding="utf-8"))
    assert saved == printed
    assert UNICODE_NAME in target.read_text(encoding="utf-8")     # ensure_ascii=False


def test_folder_quiet_and_total(tmp_path, capsys):
    for i, n in enumerate((2, 3, 5)):
        Image.fromarray(discs_image(n, seed=i)).save(tmp_path / f"img{i}.png")
    (tmp_path / "notes.txt").write_text("not an image", encoding="utf-8")
    assert main([str(tmp_path), "--quiet"]) == 0
    lines = capsys.readouterr().out.strip().splitlines()
    assert [ln.split("\t")[0] for ln in lines] == ["2", "3", "5"]
    assert main([str(tmp_path), "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["total"] == 10


def test_expect_mismatch_exits_two(tray, capsys):
    assert main([str(tray), "--expect", "6"]) == 0
    assert main([str(tray), "--expect", "7"]) == 2
    assert "expected 7" in capsys.readouterr().err


def test_region_min_area_and_bad_arguments(tray, capsys):
    assert main([str(tray), "--region", "0,0,1,1", "--quiet"]) == 0
    assert capsys.readouterr().out.startswith("0\t")
    with pytest.raises(SystemExit) as exit_info:
        main([str(tray), "--region", "1,2,3"])
    assert exit_info.value.code == 1
    with pytest.raises(SystemExit):
        main([str(tray), "--line", "0,0,5"])


def test_missing_file_is_an_error(tmp_path, capsys):
    assert main([str(tmp_path / "missing.png")]) == 1
    assert "does not exist" in capsys.readouterr().err


def test_frames_with_a_line(tmp_path, capsys):
    path = [(50.0, float(y)) for y in range(20, 101, 5)]
    for i, frame in enumerate(moving_disc_frames(path)):
        Image.fromarray(frame).save(tmp_path / f"f{i:03d}.png")
    assert main([str(tmp_path), "--line", "0,60,200,60"]) == 0
    assert "1 crossed (1 forward, 0 backward)" in capsys.readouterr().out
    assert main([str(tmp_path), "--line", "0,60,200,60", "--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["totals"] == {"line 1": 1} and data["frames"] == len(path)


def test_detector_from_a_module(tmp_path, tray, capsys, monkeypatch):
    (tmp_path / "my_models.py").write_text(textwrap.dedent("""
        def detect(image):
            return [((1, 1, 20, 20), "bolt", 0.9), ((30, 30, 60, 60), "bolt", 0.7)]

        def broken(image):
            raise RuntimeError("no GPU")
    """), encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.syspath_prepend(str(tmp_path))
    assert main([str(tray), "--detector", "my_models:detect"]) == 0
    assert "bolt 2" in capsys.readouterr().out
    assert main([str(tray), "--detector", "my_models:broken"]) == 1
    assert "no GPU" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        main([str(tray), "--detector", "my_models:nothing_here"])


def test_parse_specs():
    assert parse_region_spec("1,2,3,4") == (1.0, 2.0, 3.0, 4.0)
    assert parse_region_spec("0,0 10,0 0,10") == [(0.0, 0.0), (10.0, 0.0), (0.0, 10.0)]
    assert parse_line_spec("0,5,10,5") == ((0.0, 5.0), (10.0, 5.0))


def test_piped_output_with_non_ascii_does_not_crash(tmp_path):
    """Run the real entry point with a hostile console encoding and a piped stdout."""
    img = tmp_path / UNICODE_NAME
    Image.fromarray(discs_image(3, seed=4)).save(img)
    env = dict(os.environ, PYTHONIOENCODING="ascii", PYTHONUTF8="0")
    code = "import sys; from object_counter_ai.cli import main; sys.exit(main())"
    for extra in ([], ["--json"], ["--quiet"]):
        proc = subprocess.run([sys.executable, "-c", code, str(img), *extra],
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env, timeout=60)
        assert proc.returncode == 0, proc.stderr.decode("utf-8", "replace")
        out = proc.stdout.decode("utf-8")
        assert UNICODE_NAME in out
        assert "UnicodeEncodeError" not in proc.stderr.decode("utf-8", "replace")
