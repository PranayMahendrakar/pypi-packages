"""Reading images: every spelling that works, and every failure that must be clear."""
from __future__ import annotations

import os

import numpy as np
import pytest
from PIL import Image

from conftest import as_grey, held_out_scene, normal_scene, normal_set, save_png
from vision_anomaly import Detector
from vision_anomaly._loading import iter_images, label_for, load_image


def test_a_corrupt_file_raises_a_clear_error_naming_the_path(tmp_path):
    """The path has to be in the message: a batch run reports nothing else."""
    broken = tmp_path / "broken.png"
    broken.write_bytes(b"\x89PNG\r\n\x1a\n" + b"this is not an image at all")

    with pytest.raises(ValueError) as raised:
        load_image(str(broken))

    message = str(raised.value)
    assert str(broken) in message
    assert "cannot read image" in message


def test_a_truncated_image_raises_a_clear_error(tmp_path):
    whole = tmp_path / "whole.png"
    save_png(normal_scene(0), whole)
    cut = tmp_path / "cut.png"
    cut.write_bytes(whole.read_bytes()[: len(whole.read_bytes()) // 3])

    with pytest.raises(ValueError) as raised:
        load_image(str(cut))
    assert str(cut) in str(raised.value)


def test_a_corrupt_file_in_a_batch_names_itself(tmp_path, good_images):
    broken = tmp_path / "不良.png"
    broken.write_bytes(b"not an image")
    detector = Detector().fit(good_images)

    with pytest.raises(ValueError) as raised:
        detector.predict_batch([str(broken)])
    assert "不良.png" in str(raised.value)


def test_a_missing_file_raises_a_clear_error(tmp_path):
    missing = str(tmp_path / "nowhere.png")

    with pytest.raises(ValueError) as raised:
        load_image(missing)
    assert missing in str(raised.value)
    assert "no such image file" in str(raised.value)


def test_a_directory_is_not_an_image(tmp_path):
    with pytest.raises(ValueError) as raised:
        load_image(str(tmp_path))
    assert "is a directory" in str(raised.value)


def test_a_non_image_object_raises_a_type_error():
    for bad in (42, 4.0, {"pixels": 1}, object()):
        with pytest.raises(TypeError) as raised:
            load_image(bad)
        assert "path" in str(raised.value)


@pytest.mark.parametrize(
    "array, expected",
    [
        (np.zeros((8,), dtype=np.uint8), "HxW"),
        (np.zeros((2, 3, 4, 5), dtype=np.uint8), "HxW"),
        (np.zeros((8, 8, 2), dtype=np.uint8), "3 or 4 channels"),
        (np.zeros((0, 8, 3), dtype=np.uint8), "no pixels"),
        (np.zeros((8, 8, 3), dtype=np.complex128), "unsupported dtype"),
    ],
)
def test_an_array_that_is_not_an_image_says_why(array, expected):
    with pytest.raises(ValueError) as raised:
        load_image(array)
    assert expected in str(raised.value)


def test_nan_pixels_are_refused():
    array = np.zeros((16, 16, 3), dtype=np.float32)
    array[3, 3, 0] = np.nan

    with pytest.raises(ValueError) as raised:
        load_image(array)
    assert "NaN" in str(raised.value)


def test_a_single_channel_third_axis_is_greyscale():
    plane = as_grey(normal_scene(0))
    flat = load_image(plane)
    stacked = load_image(plane[:, :, None])

    assert stacked.channels == 1
    assert np.array_equal(flat.luminance, stacked.luminance)


@pytest.mark.parametrize("mode", ["L", "RGB", "RGBA", "P", "LA", "1", "I", "CMYK"])
def test_every_common_pil_mode_loads(mode):
    source = Image.fromarray(normal_scene(0)).resize((64, 64)).convert(mode)

    loaded = load_image(source)

    assert loaded.rgb.shape == (256, 256, 3)
    assert float(loaded.rgb.min()) >= 0.0
    assert float(loaded.rgb.max()) <= 1.0


def test_a_transparent_png_is_composited_over_white(tmp_path):
    rgba = np.zeros((64, 64, 4), dtype=np.uint8)
    rgba[..., :3] = 10
    rgba[..., 3] = 0                        # fully transparent

    loaded = load_image(rgba)

    assert float(loaded.luminance.mean()) > 0.9
    assert any("alpha" in note for note in loaded.notes)


@pytest.mark.parametrize(
    "array, expected_note",
    [
        (np.full((32, 32, 3), 0.5, dtype=np.float32), None),
        (np.full((32, 32, 3), 200.0, dtype=np.float32), "0-255"),
        (np.full((32, 32, 3), -0.5, dtype=np.float32), "-1..1"),
        (np.full((32, 32, 3), 9000.0, dtype=np.float32), "maximum"),
    ],
)
def test_float_pixel_conventions_are_read_and_reported(array, expected_note):
    loaded = load_image(array)

    assert 0.0 <= float(loaded.rgb.min()) <= 1.0
    if expected_note is None:
        assert loaded.notes == ()
    else:
        assert any(expected_note in note for note in loaded.notes)


def test_a_sixteen_bit_image_keeps_its_scale():
    array = np.full((32, 32), 30000, dtype=np.uint16)

    loaded = load_image(array)

    assert float(loaded.luminance.mean()) == pytest.approx(30000 / 65535.0, abs=1e-3)


def test_a_boolean_mask_is_readable():
    mask = np.zeros((32, 32), dtype=bool)
    mask[8:24, 8:24] = True

    loaded = load_image(mask)

    assert float(loaded.luminance.max()) == pytest.approx(1.0)


def test_the_same_picture_from_disk_and_from_an_array_agree(tmp_path):
    """A path and an array must not be two different opinions of one file."""
    image = normal_scene(0)
    path = save_png(image, tmp_path / "same.png")

    assert np.allclose(load_image(path).rgb, load_image(image).rgb, atol=1e-6)


def test_an_exif_rotation_is_applied(tmp_path):
    upright = Image.fromarray(normal_scene(0)).resize((128, 96))
    sideways = upright.transpose(Image.Transpose.ROTATE_90)
    path = tmp_path / "rotated.jpg"
    exif = sideways.getexif()
    exif[274] = 8                           # "rotate 90 CCW to display upright"
    sideways.save(path, exif=exif)

    loaded = load_image(str(path))

    assert (loaded.width, loaded.height) == (128, 96)


def test_a_tiny_image_is_scaled_up_rather_than_refused():
    tiny = np.array([[10, 200], [200, 10]], dtype=np.uint8)

    loaded = load_image(tiny)

    assert loaded.rgb.shape == (256, 256, 3)
    assert loaded.resized is True
    assert loaded.size == (2, 2)


def test_iter_images_finds_files_and_walks_when_asked(tmp_path):
    save_png(normal_scene(0), tmp_path / "a.png")
    save_png(normal_scene(1), tmp_path / "b.jpg")
    (tmp_path / "notes.txt").write_text("ignore me", encoding="utf-8")
    deep = tmp_path / "more"
    deep.mkdir()
    save_png(normal_scene(2), deep / "c.png")

    assert len(iter_images(str(tmp_path))) == 2
    assert len(iter_images(str(tmp_path), recursive=True)) == 3
    assert iter_images(str(tmp_path / "notes.txt")) == [str(tmp_path / "notes.txt")]
    with pytest.raises(ValueError):
        iter_images(str(tmp_path / "nope"))


def test_labels_are_readable_before_anything_is_loaded():
    assert label_for("a/b.png") == "a/b.png"
    assert label_for(np.zeros((2, 3, 3), np.uint8)) == "<array 2x3x3>"
    assert label_for(Image.new("RGB", (4, 4))) == "<PIL.Image>"
    assert label_for(42) == "<int>"


def test_a_unicode_path_round_trips(tmp_path, good_images):
    name = tmp_path / "pièce-部品-часть.png"
    save_png(held_out_scene(0), name)
    detector = Detector().fit(good_images)

    result = detector.predict(str(name))

    assert "部品" in result.source
    assert "部品" in result.summary()
    assert "部品" in result.to_dict()["source"]


def test_images_read_from_disk_score_the_same_as_the_arrays(tmp_path):
    images = normal_set(6)
    paths = [save_png(image, tmp_path / "p{0}.png".format(i)) for i, image in enumerate(images)]

    from_arrays = Detector().fit(images)
    from_disk = Detector().fit(paths)

    assert from_disk.score(paths[0]) == pytest.approx(from_arrays.score(images[0]), abs=1e-6)
