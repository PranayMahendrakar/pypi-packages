"""Reading frames: every input type, every awkward shape, every dtype."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from camera_health import Frame, check, list_images, load_frame
from conftest import live_frame, save_png, scene


def test_a_numpy_array_a_pil_image_and_a_path_all_work(tmp_path, base):
    path = save_png(tmp_path / "view.png", base)

    from_array = check(base)
    from_pil = check(Image.fromarray(base))
    from_path = check(path)

    assert from_array.score == from_pil.score == from_path.score
    assert from_path.source == "view.png"
    assert from_array.source is None


def test_rgb_rgba_and_single_channel_shapes_are_accepted(base):
    rgb = np.stack([base, base, base], axis=-1)
    rgba = np.concatenate([rgb, np.full(base.shape + (1,), 255, np.uint8)], axis=-1)
    single = base[:, :, None]

    for frame in (rgb, rgba, single):
        assert check(frame).metrics["brightness"] == pytest.approx(
            check(base).metrics["brightness"], abs=0.5
        )


def test_mixed_dtypes_land_on_the_same_scale(base):
    as_float_0_1 = base.astype(np.float32) / 255.0
    as_float_0_255 = base.astype(np.float64)
    as_uint16 = (base.astype(np.uint16) * 257)
    wanted = check(base).metrics["brightness"]

    assert check(as_float_0_1).metrics["brightness"] == pytest.approx(wanted, abs=1.0)
    assert check(as_float_0_255).metrics["brightness"] == pytest.approx(wanted, abs=0.5)
    assert check(as_uint16).metrics["brightness"] == pytest.approx(wanted, abs=1.5)


def test_a_boolean_mask_is_read_as_black_and_white():
    mask = np.zeros((64, 64), dtype=bool)
    mask[:, 32:] = True

    health = check(mask)

    assert health.metrics["brightness"] == pytest.approx(127.5, abs=1.0)


def test_nan_and_infinity_do_not_crash_the_checks(base):
    frame = base.astype(np.float32)
    frame[0, 0] = np.nan
    frame[0, 1] = np.inf
    frame[0, 2] = -np.inf

    health = check(frame)

    assert np.isfinite(health.score)
    assert np.isfinite(health.metrics["brightness"])


def test_an_empty_frame_is_refused_with_a_useful_message():
    with pytest.raises(ValueError, match="the frame is empty"):
        check(np.zeros((0, 320), dtype=np.uint8))


def test_a_one_pixel_frame_still_produces_a_report():
    """The smallest possible input: no detail checks, but a verdict all the same."""
    health = check(np.full((1, 1), 130, dtype=np.uint8))

    assert health.size == (1, 1)
    assert not health.was_checked("obstruction")
    assert not health.was_checked("defocus")
    assert any("3x3" in note for note in health.notes)
    assert health.summary()


def test_a_tiny_frame_says_it_is_too_small_to_tile():
    health = check(np.tile(np.array([[10, 240], [240, 10]], np.uint8), (5, 5)))

    assert not health.was_checked("obstruction")
    assert "too small to split into tiles" in health.not_checkable["obstruction"]


def test_frames_of_different_sizes_are_compared_after_resizing(base):
    half = scene(120, 160, seed=4)

    health = check(base, reference=half, previous=half)

    assert any("was resized before comparing" in note for note in health.notes)
    assert any("reduced to a 32x32 signature" in note for note in health.notes)
    assert "mean_diff" in health.metrics
    assert "scene_correlation" in health.metrics


def test_a_resized_previous_frame_of_the_same_scene_is_not_a_freeze(base):
    """Resizing must not manufacture a match - or a mismatch."""
    half = np.asarray(Image.fromarray(base).resize((160, 120), Image.BILINEAR))

    health = check(base, reference=half, previous=half)

    assert not health.has("frozen")
    assert not health.has("tampering")
    assert health.metrics["scene_correlation"] > 0.85


def test_a_badly_shaped_array_is_refused():
    with pytest.raises(ValueError, match="HxW"):
        check(np.zeros((4, 4, 5), dtype=np.uint8))
    with pytest.raises(ValueError, match="HxW"):
        check(np.zeros((4,), dtype=np.uint8))


def test_none_and_nonsense_inputs_are_refused():
    with pytest.raises(ValueError, match="the frame is None"):
        check(None)
    with pytest.raises(ValueError, match="cannot read a frame of type"):
        check({"not": "a frame"})


def test_a_missing_file_says_so(tmp_path):
    with pytest.raises(FileNotFoundError, match="no such image file"):
        check(str(tmp_path / "gone.png"))


def test_a_directory_passed_to_check_says_to_use_check_stream(tmp_path):
    with pytest.raises(ValueError, match="check_stream"):
        check(str(tmp_path))


def test_a_file_that_is_not_an_image_says_so(tmp_path):
    bad = tmp_path / "frame.png"
    bad.write_bytes(b"this is not a PNG")

    with pytest.raises(ValueError, match="cannot read"):
        check(str(bad))


def test_unicode_file_names_survive_the_round_trip(tmp_path, base):
    path = save_png(tmp_path / "カメラ-廀下-2.png", base)

    health = check(path)

    assert health.source == "カメラ-廀下-2.png"
    assert health.to_dict()["source"] == health.source
    assert health.ok


def test_list_images_sorts_naturally_and_skips_other_files(tmp_path, base):
    for name in ("b10.png", "b2.png", "b1.png"):
        save_png(tmp_path / name, base)
    (tmp_path / "readme.txt").write_text("x", encoding="utf-8")

    names = [path.rsplit("\\", 1)[-1].rsplit("/", 1)[-1] for path in list_images(str(tmp_path))]

    assert names == ["b1.png", "b2.png", "b10.png"]


def test_list_images_refuses_a_file(tmp_path, base):
    path = save_png(tmp_path / "one.png", base)

    with pytest.raises(ValueError, match="not a directory"):
        list_images(path)


def test_big_frames_are_decimated_to_the_pixel_budget():
    big = np.zeros((1080, 1920), dtype=np.uint8)

    frame = load_frame(big)

    assert isinstance(frame, Frame)
    assert frame.size == (1920, 1080)
    assert frame.stride > 1
    assert frame.gray.size <= 409_600
    assert frame.describe() == "1920x1080"


def test_decimation_can_be_turned_off(base):
    frame = load_frame(base, analysis_pixels=0)

    assert frame.stride == 1
    assert frame.shape == base.shape


def test_an_already_loaded_frame_passes_straight_through(base):
    frame = load_frame(base)

    assert check(frame).score == check(base).score
