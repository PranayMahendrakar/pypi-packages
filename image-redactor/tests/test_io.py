"""Reading and writing real files, including paths with non-ASCII characters."""
from __future__ import annotations

import json

import numpy as np
import pytest
from PIL import Image

from image_redactor import redact, redact_file

from conftest import noise

UNICODE_NAME = "स्कूल-校園-café"


def _write(path, array):
    Image.fromarray(array).save(path)
    return path


def test_a_path_in_gives_a_pil_image_out(tmp_path):
    src = _write(tmp_path / "in.png", noise(60, 80, seed=21))
    result = redact(src, regions=[(10, 10, 50, 40)], method="blackout")
    assert isinstance(result.image, Image.Image)
    assert result.source == str(src)
    assert result.summary().count(str(src)) == 1


def test_a_path_may_be_a_plain_string(tmp_path):
    src = _write(tmp_path / "in.png", noise(40, 40, seed=22))
    result = redact(str(src), regions=[(5, 5, 30, 30)], method="blackout")
    assert result.count == 1


def test_redact_file_writes_the_result_and_reports_where(tmp_path):
    src = _write(tmp_path / "in.png", noise(60, 80, seed=23))
    dst = tmp_path / "out" / "clean.png"
    result = redact_file(src, dst, regions=[(10, 10, 50, 40)], method="blackout")

    assert dst.is_file()
    assert result.saved_to == str(dst)
    assert "saved to:" in result.summary()
    assert result.to_dict()["saved_to"] == str(dst)

    written = np.asarray(Image.open(dst))
    assert written[10:40, 10:50].max() == 0


def test_redact_file_creates_missing_folders(tmp_path):
    src = _write(tmp_path / "in.png", noise(40, 40, seed=24))
    dst = tmp_path / "a" / "b" / "c" / "out.png"
    redact_file(src, dst, regions=[(5, 5, 30, 30)])
    assert dst.is_file()


def test_redact_file_leaves_the_source_file_alone(tmp_path):
    src = _write(tmp_path / "in.png", noise(40, 40, seed=25))
    before = src.read_bytes()
    redact_file(src, tmp_path / "out.png", regions=[(5, 5, 30, 30)], method="blackout")
    assert src.read_bytes() == before


def test_saving_rgba_to_jpeg_warns_instead_of_failing(tmp_path):
    source = noise(40, 40, channels=4, seed=26)
    result = redact(source, regions=[(5, 5, 30, 30)], method="blackout")
    dst = tmp_path / "out.jpg"
    result.save(dst)
    assert dst.is_file()
    assert any("transparency" in warning for warning in result.warnings)
    assert Image.open(dst).mode == "RGB"


def test_save_works_when_the_input_was_a_numpy_array(tmp_path):
    result = redact(noise(40, 40, seed=27), regions=[(5, 5, 30, 30)], method="blackout")
    path = result.save(tmp_path / "out.png")
    assert path.is_file()
    assert np.asarray(Image.open(path))[5:30, 5:30].max() == 0


def test_save_works_for_a_greyscale_array(tmp_path):
    result = redact(noise(40, 40, channels=1, seed=28), regions=[(5, 5, 30, 30)],
                    method="blackout")
    path = result.save(tmp_path / "grey.png")
    assert Image.open(path).mode == "L"


def test_an_rgba_png_keeps_its_alpha_on_disk(tmp_path):
    source = noise(40, 40, channels=4, seed=29)
    src = tmp_path / "in.png"
    Image.fromarray(source, mode="RGBA").save(src)
    dst = tmp_path / "out.png"
    redact_file(src, dst, regions=[(5, 5, 30, 30)], method="blackout", expand=0.0)
    written = np.asarray(Image.open(dst))
    assert written.shape[2] == 4
    assert np.array_equal(written[:, :, 3], source[:, :, 3])


def test_a_non_ascii_path_round_trips(tmp_path):
    src = tmp_path / (UNICODE_NAME + ".png")
    _write(src, noise(40, 40, seed=30))
    dst = tmp_path / (UNICODE_NAME + "-safe.png")
    result = redact_file(src, dst, regions=[(5, 5, 30, 30)], method="blackout")

    assert dst.is_file()
    assert UNICODE_NAME in result.source
    assert UNICODE_NAME in result.summary()
    # JSON keeps the characters as-is rather than escaping them
    assert UNICODE_NAME in result.to_json()
    assert json.loads(result.to_json())["source"] == str(src)


def test_a_non_ascii_summary_survives_a_utf8_encode(tmp_path):
    src = tmp_path / (UNICODE_NAME + ".png")
    _write(src, noise(40, 40, seed=31))
    result = redact(src, regions=[(5, 5, 30, 30)])
    assert result.summary().encode("utf-8").decode("utf-8") == result.summary()


def test_a_directory_passed_as_an_image_is_rejected(tmp_path):
    with pytest.raises(FileNotFoundError):
        redact(tmp_path)


def test_a_file_that_is_not_an_image_fails_clearly(tmp_path):
    bad = tmp_path / "notes.txt"
    bad.write_text("this is not a picture", encoding="utf-8")
    with pytest.raises(Exception):
        redact(bad, regions=[(1, 1, 5, 5)])
