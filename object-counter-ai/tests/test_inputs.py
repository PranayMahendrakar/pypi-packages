"""Image types, empty images, and the promise never to touch the caller's image."""
from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

import object_counter_ai as oc
from synth_images import blank, discs_image, draw_disc


def _five_discs_grey() -> np.ndarray:
    return discs_image(5, seed=11, h=160, w=200, r=10)


# -- colour modes ----------------------------------------------------------------------

def test_greyscale_rgb_and_rgba_agree():
    grey = _five_discs_grey()
    rgb = np.stack([grey] * 3, axis=2)
    rgba = np.dstack([rgb, np.full(grey.shape, 255, np.uint8)])
    results = [oc.count(img) for img in (grey, rgb, rgba)]
    assert [r.count for r in results] == [5, 5, 5]
    assert [r.image_kind for r in results] == ["grey", "rgb", "rgba"]
    assert results[0].boxes == results[1].boxes == results[2].boxes


def test_rgba_with_transparent_background_counts_the_opaque_things():
    img = np.zeros((120, 160, 4), dtype=np.uint8)
    rng = np.random.default_rng(0)
    img[..., :3] = rng.integers(0, 255, (120, 160, 3))   # junk colour under alpha 0
    for x in (30, 80, 130):
        draw_disc(img, x, 60, 12, (20, 20, 20, 255))
    assert oc.count(img).count == 3


def test_grey_plus_alpha_and_single_channel_arrays():
    grey = _five_discs_grey()
    la = np.dstack([grey, np.full(grey.shape, 255, np.uint8)])
    assert oc.count(la).count == 5
    assert oc.count(la).image_kind == "grey+alpha"
    assert oc.count(grey[:, :, None]).count == 5


def test_pil_modes():
    grey = _five_discs_grey()
    base = Image.fromarray(grey)
    for mode in ("L", "RGB", "RGBA", "P", "1"):
        img = base.convert(mode) if mode != "1" else base.point(lambda v: 255 if v > 128 else 0).convert("1")
        assert oc.count(img).count == 5, mode
    wide = Image.fromarray((grey.astype(np.uint16) * 257))
    assert oc.count(wide).count == 5


def test_sixteen_bit_float_and_bool_arrays():
    grey = _five_discs_grey()
    assert oc.count(grey.astype(np.uint16) * 257).count == 5
    assert oc.count(grey.astype(np.float32) / 255.0).count == 5
    assert oc.count(grey < 128).count == 5


def test_out_of_range_values_are_stretched_and_noted():
    grey = _five_discs_grey().astype(np.float64) * 40.0 - 3000.0
    result = oc.count(grey)
    assert result.count == 5
    assert any("stretched" in note for note in result.notes)


def test_nan_pixels_are_ignored_and_noted():
    grey = _five_discs_grey().astype(np.float64)
    grey[0:5, :] = np.nan
    result = oc.count(grey)
    assert result.count == 5
    assert any("NaN" in note for note in result.notes)


def test_channels_first_array_is_read():
    rgb = np.stack([_five_discs_grey()] * 3, axis=0)       # (3, h, w)
    result = oc.count(rgb)
    assert result.count == 5
    assert result.image_size == (200, 160)


def test_nested_list_input():
    img = blank(30, 40, 200)
    img[10:20, 10:20] = 0
    assert oc.count(img.tolist()).count == 1


def test_file_path_input(tmp_path):
    path = tmp_path / "tray.png"
    Image.fromarray(_five_discs_grey()).save(path)
    assert oc.count(str(path)).count == 5
    assert oc.count(path).count == 5


def test_missing_file_raises_clearly(tmp_path):
    with pytest.raises(FileNotFoundError):
        oc.count(str(tmp_path / "nope.png"))


@pytest.mark.parametrize("bad, error", [
    (np.zeros((2, 3, 4, 5)), ValueError),
    (np.zeros((20, 30, 7)), ValueError),
    (np.array([["a", "b"], ["c", "d"]]), TypeError),
    (12345, TypeError),
])
def test_unreadable_images_raise_a_clear_error(bad, error):
    with pytest.raises(error):
        oc.count(bad)


def test_detector_that_is_not_callable_is_rejected():
    with pytest.raises(TypeError, match="callable"):
        oc.count(blank(), detector="yolo")


# -- empty images count 0 ----------------------------------------------------------------

@pytest.mark.parametrize("empty", [
    np.zeros((0, 0), np.uint8),
    np.zeros((0, 50), np.uint8),
    np.zeros((40, 0, 3), np.uint8),
    np.zeros((0, 0, 4), np.uint8),
    np.array([]),
    [],
])
def test_empty_image_counts_zero_rather_than_raising(empty):
    result = oc.count(empty)
    assert result.count == 0
    assert result.ok
    assert result.boxes == []
    assert any("no pixels" in note for note in result.notes)
    assert "0 blobs" in result.summary()


def test_empty_pil_image_counts_zero():
    assert oc.count(Image.new("L", (0, 0))).count == 0


def test_empty_image_with_a_detector_counts_zero_without_calling_it():
    calls = []

    def detector(image):
        calls.append(1)
        return [(0, 0, 1, 1)]

    result = oc.count(np.zeros((0, 10, 3), np.uint8), detector=detector)
    assert result.count == 0 and result.ok and calls == []


def test_empty_frame_in_a_stream_is_counted_as_zero():
    counter = oc.Counter()
    counter.update(np.zeros((0, 0)))
    counter.update(_five_discs_grey())
    assert [r.count for r in counter.history] == [0, 5]


# -- the caller's image is never modified -------------------------------------------------

def test_caller_array_is_not_modified():
    img = discs_image(6, seed=4)
    before = img.copy()
    oc.count(img)
    oc.count(img, region=(0, 0, 100, 100))
    counter = oc.Counter().line((0, 50), (320, 50))
    counter.update(img)
    assert np.array_equal(img, before)
    assert img.flags.writeable


def test_read_only_array_is_accepted():
    img = discs_image(6, seed=4)
    img.setflags(write=False)
    assert oc.count(img).count == 6


def test_caller_float_and_rgba_arrays_are_not_modified():
    img = np.dstack([np.stack([_five_discs_grey()] * 3, axis=2),
                     np.full((160, 200), 128, np.uint8)]).astype(np.float64)
    img[0, 0, 0] = np.nan
    before = img.copy()
    oc.count(img)
    assert np.array_equal(img, before, equal_nan=True)


def test_caller_pil_image_is_not_modified():
    img = Image.fromarray(np.stack([_five_discs_grey()] * 3, axis=2))
    before = img.tobytes()
    oc.count(img)
    assert img.tobytes() == before and img.mode == "RGB"


def test_detector_writing_into_its_input_cannot_touch_the_caller_image():
    img = discs_image(3, seed=1)
    before = img.copy()

    def vandal(image):
        image[:] = 0
        return [(0, 0, 10, 10)]

    assert oc.count(img, detector=vandal).count == 1
    assert np.array_equal(img, before)

    pil = Image.fromarray(img)

    def pil_vandal(image):
        image.paste(0, (0, 0, 50, 50))
        return []

    oc.count(pil, detector=pil_vandal)
    assert np.array_equal(np.array(pil), before)


def test_single_pixel_and_one_pixel_strips():
    assert oc.count(np.array([[7]], dtype=np.uint8)).count == 0
    strip = np.full((1, 60), 220, dtype=np.uint8)
    strip[0, 20:35] = 10
    assert oc.count(strip).count == 1
    assert oc.count(strip.T.copy()).count == 1


def test_all_nan_image_counts_zero_and_says_why():
    result = oc.count(np.full((20, 30), np.nan))
    assert result.count == 0
    assert any("NaN" in n for n in result.notes)
