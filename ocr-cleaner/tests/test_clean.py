"""The pipeline: what each step does, and what it reports when it does nothing."""
from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

import ocr_cleaner

from conftest import (
    PAPER,
    add_grain,
    add_scanner_border,
    add_specks,
    colour_page,
    count_bands,
    light_unevenly,
    low_contrast_uneven_page,
    rotate,
    text_page,
)


def test_clean_returns_every_step_in_order(page):
    result = ocr_cleaner.clean(page)
    assert [step.name for step in result.steps] == list(ocr_cleaner.STEP_NAMES)
    assert all(step.detail for step in result.steps)


def test_clean_binarises_by_default(page):
    result = ocr_cleaner.clean(page)
    assert result.binary
    assert set(np.unique(np.array(result.image))) <= {0, 255}
    assert result.threshold_mode == "adaptive"


def test_threshold_none_leaves_the_page_in_greyscale(page):
    result = ocr_cleaner.clean(page, threshold="none")
    assert not result.binary
    assert not result.step("threshold").applied
    assert "greyscale" in result.step("threshold").detail
    assert set(np.unique(np.array(result.image))) == {50, PAPER}


def test_otsu_reports_the_level_it_chose(page):
    result = ocr_cleaner.clean(page, threshold="otsu")
    assert result.step("threshold").applied
    assert result.threshold_level is not None
    assert 0 < result.threshold_level < 255
    assert set(np.unique(np.array(result.image))) <= {0, 255}


def test_unknown_threshold_mode_is_rejected(page):
    with pytest.raises(ValueError, match="threshold must be one of"):
        ocr_cleaner.clean(page, threshold="sauvola")


def test_adaptive_reads_a_page_no_single_cut_can_split():
    """The case for thresholding locally: ink at the bright end is paler than
    paper at the dark end, so every global level is wrong somewhere."""
    page = low_contrast_uneven_page()
    drawn = len(range(80, 830, 30))
    adaptive = np.array(ocr_cleaner.clean(page, threshold="adaptive").image)
    otsu = np.array(ocr_cleaner.clean(page, threshold="otsu").image)
    assert count_bands(adaptive) == drawn
    assert count_bands(otsu) < drawn
    # the dark end floods to solid ink under one global cut, and does not here
    assert (adaptive[:, -200:] == 0).mean() < 0.45
    assert (otsu[:, -200:] == 0).mean() > 0.9


def test_adaptive_window_follows_the_text_height():
    small = ocr_cleaner.clean(text_page(620, 480, pitch=18, bar=6, top=60))
    large = ocr_cleaner.clean(text_page(900, 700, pitch=60, bar=26, left=70, right=630, top=100))
    assert small.estimated_text_height_px < large.estimated_text_height_px
    assert "x the" in small.step("threshold").detail


def test_denoise_runs_on_a_grainy_page_and_removes_specks(page):
    speckled = add_specks(add_grain(page, 9.0, seed=11), 0.006, seed=12)
    result = ocr_cleaner.clean(speckled, threshold="none")
    assert result.step("denoise").applied
    assert "median filter 3 x 3" in result.step("denoise").detail
    cleaned = np.array(result.image)
    # count isolated black specks sitting in the top margin, which holds no text
    assert (cleaned[8:60, :] < 100).sum() < (speckled[8:60, :] < 100).sum() / 10


def test_denoise_stands_down_on_a_page_that_is_already_clean(page):
    result = ocr_cleaner.clean(page, threshold="none")
    assert not result.step("denoise").applied
    assert "already clean" in result.step("denoise").detail
    assert np.array_equal(np.array(result.image), page)


def test_denoise_window_grows_with_the_text():
    big = text_page(2200, 1700, pitch=120, bar=60, left=180, right=1520, top=240)
    result = ocr_cleaner.clean(add_grain(big, 9.0, seed=13), threshold="none")
    assert result.estimated_text_height_px > 40
    assert "median filter 5 x 5" in result.step("denoise").detail


def test_border_removes_the_black_bands(page):
    scanned = add_scanner_border(page)
    result = ocr_cleaner.clean(scanned, threshold="none")
    assert result.step("border").applied
    assert result.output_size[0] < scanned.shape[1]
    cleaned = np.array(result.image)
    assert (cleaned[:, :4] < 60).mean() == 0.0
    assert (cleaned[:, -4:] < 60).mean() == 0.0


@pytest.mark.parametrize("angle", (-2.4, 3.1))
def test_border_removes_the_bands_on_a_skewed_page(angle, page):
    scanned = rotate(add_scanner_border(page), angle)
    result = ocr_cleaner.clean(scanned, threshold="none")
    assert result.step("deskew").applied
    assert result.step("border").applied
    cleaned = np.array(result.image)
    assert (cleaned[:, :4] < 60).mean() == 0.0
    assert (cleaned[:, -4:] < 60).mean() == 0.0


def test_border_stands_down_when_there_is_no_border(page):
    result = ocr_cleaner.clean(page)
    assert not result.step("border").applied
    assert "already paper" in result.step("border").detail


def test_every_step_can_be_turned_off(page):
    scanned = add_scanner_border(add_grain(rotate(page, -2.0), 8.0, seed=14))
    result = ocr_cleaner.clean(
        scanned, deskew=False, denoise=False, border=False, threshold="none"
    )
    assert result.applied == []
    for name in ("deskew", "denoise", "border"):
        assert "turned off" in result.step(name).detail
    assert np.array_equal(np.array(result.image), scanned)


def test_deskew_fills_the_corners_with_paper_not_white():
    result = ocr_cleaner.clean(rotate(text_page(), 4.0), threshold="none", border=False)
    corner = np.array(result.image)[:6, :6]
    assert corner.max() <= PAPER + 2
    assert "paper level" in result.step("deskew").detail


def test_upscale_needs_both_resolutions(page):
    both = ocr_cleaner.clean(page, dpi=150, upscale_to_dpi=300)
    assert both.step("upscale").applied
    assert both.output_size[0] == pytest.approx(page.shape[1] * 2, abs=2)
    assert both.output_dpi == 300.0

    target_only = ocr_cleaner.clean(page, upscale_to_dpi=300)
    assert not target_only.step("upscale").applied
    assert "dpi is unknown" in target_only.step("upscale").detail
    assert target_only.notes

    source_only = ocr_cleaner.clean(page, dpi=150)
    assert not source_only.step("upscale").applied
    assert "no upscale_to_dpi" in source_only.step("upscale").detail


def test_upscale_stands_down_when_the_page_is_already_big_enough(page):
    result = ocr_cleaner.clean(page, dpi=600, upscale_to_dpi=300)
    assert not result.step("upscale").applied
    assert "already 600 dpi" in result.step("upscale").detail
    assert result.output_size == (page.shape[1], page.shape[0])


def test_upscaled_binary_page_stays_binary(page):
    result = ocr_cleaner.clean(page, dpi=150, upscale_to_dpi=300)
    assert set(np.unique(np.array(result.image))) <= {0, 255}
    assert "black and white" in result.step("upscale").detail


@pytest.mark.parametrize("bad", (0, -1, "many"))
def test_bad_dpi_is_rejected(page, bad):
    with pytest.raises(ValueError):
        ocr_cleaner.clean(page, dpi=bad)


def test_colour_input_becomes_luminance():
    result = ocr_cleaner.clean(colour_page())
    assert result.step("grayscale").applied
    assert "luminance" in result.step("grayscale").detail
    assert result.image.mode == "L"
    assert result.page_kind == "document"


def test_greyscale_input_says_it_had_nothing_to_convert(page):
    result = ocr_cleaner.clean(page)
    assert not result.step("grayscale").applied
    assert "already 8-bit greyscale" in result.step("grayscale").detail


@pytest.mark.parametrize("mode", ("RGB", "RGBA", "LA", "P", "1", "I", "F", "CMYK"))
def test_every_pillow_mode_is_accepted(mode, page):
    source = Image.fromarray(page, mode="L").convert(mode)
    result = ocr_cleaner.clean(source)
    assert result.image.mode == "L"
    assert result.output_size == (page.shape[1], page.shape[0])


def test_clean_file_writes_and_records_the_destination(tmp_path, page_file):
    destination = tmp_path / "out" / "clean.png"
    result = ocr_cleaner.clean_file(page_file, destination)
    assert destination.exists()
    assert result.destination == str(destination)
    assert result.source == page_file
    reopened = Image.open(destination)
    assert reopened.size == result.output_size


def test_clean_file_carries_the_output_resolution(tmp_path, page_file):
    destination = tmp_path / "big.png"
    ocr_cleaner.clean_file(page_file, destination, dpi=150, upscale_to_dpi=300)
    written = Image.open(destination)
    assert written.info.get("dpi", (0, 0))[0] == pytest.approx(300, abs=1)


def test_clean_accepts_a_path_and_names_it(page_file):
    result = ocr_cleaner.clean(page_file)
    assert result.source == page_file
    assert result.page_kind == "document"


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        ocr_cleaner.clean(str(tmp_path / "nope.png"))


def test_a_wrong_kind_of_input_raises():
    with pytest.raises(TypeError):
        ocr_cleaner.clean(42)


def test_the_whole_repair_on_one_bad_scan(bad_scan):
    result = ocr_cleaner.clean(bad_scan)
    assert result.page_kind == "document"
    assert {"deskew", "denoise", "threshold"} <= set(result.applied)
    assert abs(ocr_cleaner.estimate_skew(result.image)) < 0.5
    assert count_bands(np.array(result.image)) >= 18


def test_clean_is_deterministic(bad_scan):
    first = np.array(ocr_cleaner.clean(bad_scan).image)
    second = np.array(ocr_cleaner.clean(bad_scan).image)
    assert np.array_equal(first, second)


def test_uneven_lighting_does_not_make_a_document_look_like_a_photograph(page):
    result = ocr_cleaner.clean(light_unevenly(page, -90.0))
    assert result.page_kind == "document"
    assert result.binary


# --- regression: the border step read a lighting shadow as a scanner band -----------

def _text_page(height=1200, width=900, seed=0):
    import numpy as np

    rng = np.random.default_rng(seed)
    page = np.full((height, width), 235.0)
    for y in range(80, height - 80, 42):
        for x in range(60, width - 60, 28):
            if rng.random() < 0.8:
                page[y:y + 14, x:x + 18] = 30
    return page


def _as_image(page):
    import numpy as np
    from PIL import Image

    return Image.fromarray(page.clip(0, 255).astype(np.uint8))


def test_a_lighting_shadow_is_not_cropped_off_the_page():
    """Darkness was measured against ONE page-wide paper level, so under a soft shadow
    the paper itself read as dark, a line of text there looked almost wholly dark, and
    the trim silently cut away up to a quarter of the page - text and all."""
    import numpy as np

    import ocr_cleaner as oc

    page = _text_page()
    page[:, :260] *= np.linspace(0.45, 1.0, 260)[None, :]
    result = oc.clean(_as_image(page))
    assert result.image.size[0] == page.shape[1], "no text may be cropped out of a shadow"


def test_a_real_scanner_band_is_still_removed():
    import ocr_cleaner as oc

    page = _text_page()
    page[:, :90] = 8
    result = oc.clean(_as_image(page))
    removed = page.shape[1] - result.image.size[0]
    assert 85 <= removed <= 100, removed
    assert result.step("border").applied


def test_a_band_beside_a_shadow_stops_at_the_band():
    """Remove the band, and not a pixel of the shadowed text beyond it."""
    import numpy as np

    import ocr_cleaner as oc

    page = _text_page()
    page[:, :260] *= np.linspace(0.45, 1.0, 260)[None, :]
    page[:, :90] = 8
    removed = page.shape[1] - oc.clean(_as_image(page)).image.size[0]
    assert 85 <= removed <= 100, removed
