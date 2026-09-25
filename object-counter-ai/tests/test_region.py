"""Counting inside a box, a polygon or a mask."""
from __future__ import annotations

import numpy as np
import pytest

import object_counter_ai as oc
from synth_images import blank, draw_disc


def _row_of_discs() -> np.ndarray:
    img = blank(100, 260, 230)
    for x in (30, 80, 130, 180, 230):
        draw_disc(img, x, 50, 10, 30)
    return img


def test_box_region_counts_only_inside():
    result = oc.count(_row_of_discs(), region=(0, 0, 155, 100))
    assert result.count == 3
    assert result.region == {"kind": "box", "box": [0, 0, 155, 100]}
    assert "region" in result.summary()


def test_two_corner_box_region():
    assert oc.count(_row_of_discs(), region=[(155, 0), (260, 100)]).count == 2


def test_polygon_region():
    triangle = [(0, 0), (160, 0), (0, 100)]     # covers the discs at x=30 and x=80 only
    result = oc.count(_row_of_discs(), region=triangle)
    assert result.count == 2
    assert result.region["kind"] == "polygon"


def test_boolean_mask_region():
    mask = np.zeros((100, 260), dtype=bool)
    mask[:, 200:] = True
    assert oc.count(_row_of_discs(), region=mask).count == 1


def test_region_mask_must_match_the_image():
    with pytest.raises(ValueError, match="region mask"):
        oc.count(_row_of_discs(), region=np.ones((10, 10), dtype=bool))


def test_a_blob_cut_by_the_region_boundary_is_flagged():
    result = oc.count(_row_of_discs(), region=(0, 0, 130, 100))
    assert result.count == 3              # the third disc is cut in half, still counted
    assert any("region boundary" in n for n in result.notes)


def test_region_containing_nothing_counts_zero_without_false_alarms():
    result = oc.count(_row_of_discs(), region=(0, 0, 18, 18))
    assert result.count == 0


def test_background_is_measured_inside_the_region():
    img = blank(100, 200, 230)
    img[:, 100:] = 40                      # the right half is a dark table, not objects
    draw_disc(img, 50, 50, 10, 30)
    draw_disc(img, 150, 50, 10, 200)       # a light part on the dark table
    assert oc.count(img, region=(100, 0, 200, 100)).count == 1


@pytest.mark.parametrize("bad", [(0, 0, 0, 10), [(0, 0), (1, 1), (2, 2)], "left half", (1, 2, 3)])
def test_bad_regions_raise(bad):
    with pytest.raises(ValueError):
        oc.count(_row_of_discs(), region=bad)


def test_detector_boxes_count_when_their_centre_is_inside():
    def detector(image):
        return [(10, 10, 30, 30), (90, 10, 130, 30), (140, 60, 150, 70)]

    result = oc.count(_row_of_discs(), detector=detector, region=(0, 0, 100, 100))
    assert result.count == 1
    assert result.details["dropped_by_region"] == 2
    polygon = oc.count(_row_of_discs(), detector=detector,
                       region=[(0, 0), (200, 0), (200, 100), (0, 100)])
    assert polygon.count == 3


def test_detector_path_checks_a_mask_region_size_too():
    with pytest.raises(ValueError, match="region mask"):
        oc.count(_row_of_discs(), detector=lambda im: [], region=np.ones((5, 5), dtype=bool))
