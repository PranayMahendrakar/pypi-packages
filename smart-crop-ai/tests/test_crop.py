"""The main path: does it actually find the subject, and say so honestly."""
from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

import smart_crop_ai
from conftest import (
    SUBJECT_COLS,
    SUBJECT_ROWS,
    SUBJECT_SIZE,
    box_contains_subject,
    subject_array,
)


def test_quickstart_from_the_readme():
    """The exact block the README tells people to paste."""
    wall = np.full((400, 800, 3), 200, dtype=np.uint8)
    subject = np.random.default_rng(0).integers(0, 255, (140, 140, 3))
    wall[140:280, 560:700] = subject.astype(np.uint8)
    result = smart_crop_ai.crop(Image.fromarray(wall), ratio=(1, 1))

    text = result.summary()
    assert "smart-crop-ai" in text
    assert "confidence" in text
    assert result.size == (400, 400)
    assert result.confidence > 0.5


def test_the_crop_box_actually_contains_the_planted_subject(subject_image):
    """The whole point: a high-detail patch off centre must land inside the box.

    The patch sits at columns 560-700 of an 800 wide image. A square centre crop
    spans 200-600, so it cuts the patch in half. Anything that claims to find
    subjects has to do better than that.
    """
    result = smart_crop_ai.crop(subject_image, ratio=1.0)

    assert box_contains_subject(result.box), (
        "box {0} misses the subject at rows {1} cols {2}".format(
            result.box, SUBJECT_ROWS, SUBJECT_COLS
        )
    )
    assert result.strategy_used == "saliency"
    assert result.confidence > 0.3


@pytest.mark.parametrize("strategy", ["auto", "saliency", "entropy", "edges"])
def test_every_searching_strategy_finds_the_subject(subject_image, strategy):
    result = smart_crop_ai.crop(subject_image, ratio=1.0, strategy=strategy)

    assert box_contains_subject(result.box)
    assert result.confidence > 0.0


def test_center_strategy_does_not_find_it_and_admits_that(subject_image):
    """The baseline. It must be honest that it looked at nothing."""
    result = smart_crop_ai.crop(subject_image, ratio=1.0, strategy="center")

    assert not box_contains_subject(result.box)
    assert result.strategy_used == "center"
    assert result.confidence == 0.0
    assert not result.moved
    assert any("without looking at it" in note for note in result.notes)


def test_confidence_really_compares_against_a_centre_crop(subject_image, centred_image):
    """Same detail, centred instead of off to the side, must score far lower.

    If confidence were a plain "how much detail is in here" score, both of these
    would be high. It is a comparison, so only the off-centre one may be.
    """
    off_centre = smart_crop_ai.crop(subject_image, ratio=1.0)
    centred = smart_crop_ai.crop(centred_image, ratio=1.0)

    assert off_centre.confidence > 0.3
    assert centred.confidence < 0.1
    assert off_centre.confidence > centred.confidence


def test_confidence_stays_inside_zero_and_one(subject_image):
    for strategy in smart_crop_ai.STRATEGIES:
        result = smart_crop_ai.crop(subject_image, ratio=1.0, strategy=strategy)
        assert 0.0 <= result.confidence <= 1.0


def test_explicit_width_and_height_are_honoured(subject_image):
    result = smart_crop_ai.crop(subject_image, 300, 200)

    assert result.size == (300, 200)
    assert result.image.size == (300, 200)
    assert result.target_size == (300, 200)


def test_one_side_alone_keeps_the_source_aspect_ratio(subject_image):
    by_width = smart_crop_ai.crop(subject_image, width=400)
    by_height = smart_crop_ai.crop(subject_image, height=200)

    assert by_width.size == (400, 200)
    assert by_height.size == (400, 200)


@pytest.mark.parametrize(
    "ratio, expected",
    [
        ((1, 1), (400, 400)),
        (1.0, (400, 400)),
        ("1:1", (400, 400)),
        ((16, 9), (711, 400)),
        ("16:9", (711, 400)),
        ([2, 1], (800, 400)),
        (0.5, (200, 400)),
    ],
)
def test_ratio_is_accepted_in_every_spelling(subject_image, ratio, expected):
    result = smart_crop_ai.crop(subject_image, ratio=ratio)
    assert result.size == expected


def test_padding_pushes_the_subject_away_from_the_border(subject_image):
    """More padding means more room between the subject and the window edge."""
    tight = smart_crop_ai.crop(subject_image, ratio=1.0, padding=0.0)
    roomy = smart_crop_ai.crop(subject_image, ratio=1.0, padding=0.4)

    assert box_contains_subject(tight.box)
    assert box_contains_subject(roomy.box)
    # The subject is right of centre, so more padding shifts the window right,
    # leaving the subject further from the right-hand edge.
    assert roomy.box[0] >= tight.box[0]
    assert roomy.padding == pytest.approx(0.4)


def test_thumbnail_keeps_the_subject_and_hits_the_exact_size(subject_image):
    thumb = smart_crop_ai.thumbnail(subject_image, (120, 120))

    assert isinstance(thumb, Image.Image)
    assert thumb.size == (120, 120)
    # A plain PIL thumbnail of the whole 800x400 frame would squash the subject
    # into a corner; this one crops to it first, so the result is not flat.
    assert np.asarray(thumb.convert("L"), dtype=float).std() > 10.0


def test_thumbnail_takes_one_number_as_a_square(subject_image):
    assert smart_crop_ai.thumbnail(subject_image, 64).size == (64, 64)


def test_thumbnail_passes_options_through(subject_image):
    assert smart_crop_ai.thumbnail(subject_image, 64, strategy="edges").size == (64, 64)


def test_crop_to_file_writes_and_reports_where(tmp_path, subject_image):
    source = tmp_path / "in.png"
    subject_image.save(source)
    destination = tmp_path / "nested" / "out.png"

    result = smart_crop_ai.crop_to_file(str(source), str(destination), ratio=1.0)

    assert destination.exists()
    assert result.destination == str(destination)
    assert result.source == str(source)
    assert Image.open(destination).size == result.size
    assert "written to" in result.summary()


def test_a_path_is_accepted_wherever_an_image_is(tmp_path, subject_image):
    source = tmp_path / "photo.png"
    subject_image.save(source)

    from_path = smart_crop_ai.crop(str(source), ratio=1.0)
    from_image = smart_crop_ai.crop(subject_image, ratio=1.0)

    assert from_path.box == from_image.box
    assert from_path.source == str(source)


def test_a_numpy_array_is_accepted_too():
    result = smart_crop_ai.crop(subject_array(), ratio=1.0)

    assert box_contains_subject(result.box)
    assert result.source_size == SUBJECT_SIZE


def test_a_missing_path_says_so(tmp_path):
    with pytest.raises(FileNotFoundError):
        smart_crop_ai.crop(str(tmp_path / "nope.png"), ratio=1.0)


def test_version_is_exposed():
    assert smart_crop_ai.__version__ == "0.1.0"
