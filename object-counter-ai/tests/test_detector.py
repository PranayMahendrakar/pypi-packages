"""Plugging in a detector: output formats, clipping, filtering and failures."""
from __future__ import annotations

import json

import numpy as np
import pytest

import object_counter_ai as oc
from synth_images import blank

IMG = np.zeros((100, 200, 3), dtype=np.uint8)      # 200 wide, 100 high


def _count(output, **kw):
    return oc.count(IMG, detector=lambda image: output, **kw)


def test_plain_boxes():
    result = _count([(10, 10, 30, 40), (50, 20, 90, 60)])
    assert result.method == "detector"
    assert result.count == 2
    assert result.labels == ["object", "object"]
    assert result.confidence is None
    assert "no scores" in result.summary()


def test_box_label_score_tuples_give_labels_and_confidence():
    result = _count([((10, 10, 30, 40), "bolt", 0.9), ((50, 20, 90, 60), "nut", 0.7),
                     ((100, 20, 120, 60), "bolt", 0.8)])
    assert result.count == 3
    assert result.by_label == {"bolt": 2, "nut": 1}
    assert list(result.by_label) == ["bolt", "nut"]
    assert result.confidence == pytest.approx(0.8)
    assert result.scores == [0.9, 0.7, 0.8]


def test_two_part_tuples_and_dicts():
    result = _count([((10, 10, 30, 40), "bolt"), ((50, 20, 90, 60), 0.5),
                     {"box": (100, 10, 120, 30), "label": "washer", "score": 0.25},
                     {"bbox": [130, 10, 150, 30]}])
    assert result.labels == ["bolt", "object", "washer", "object"]
    assert result.scores == [None, 0.5, 0.25, None]
    assert result.confidence == pytest.approx(0.375)
    assert any("had no score" in n for n in result.notes)


def test_torchvision_style_dict():
    output = {"boxes": np.array([[10, 10, 30, 40], [50, 20, 90, 60]], dtype=np.float32),
              "labels": np.array([3, 3]), "scores": np.array([0.9, 0.6], dtype=np.float32)}
    result = _count(output)
    assert result.count == 2
    assert result.by_label == {"3": 2}
    assert result.confidence == pytest.approx(0.75, abs=1e-4)


@pytest.mark.parametrize("columns", [4, 5, 6])
def test_numpy_arrays_of_detections(columns):
    rows = np.array([[10, 10, 30, 40, 0.9, 1], [50, 20, 90, 60, 0.5, 2]], dtype=np.float64)
    result = _count(rows[:, :columns])
    assert result.count == 2
    if columns >= 5:
        assert result.confidence == pytest.approx(0.7)
    if columns == 6:
        assert result.labels == ["1", "2"]


def test_empty_outputs_count_zero():
    for output in ([], (), np.zeros((0, 4)), {"boxes": []}):
        result = _count(output)
        assert result.count == 0 and result.ok


def test_boxes_outside_the_image_are_clipped_or_dropped():
    result = _count([(-20, -5, 30, 40), (180, 80, 260, 140), (300, 300, 350, 350),
                     (-50, 10, -10, 30)])
    assert result.count == 2
    assert result.boxes == [(0.0, 0.0, 30.0, 40.0), (180.0, 80.0, 200.0, 100.0)]
    assert result.details["clipped"] == 2
    assert result.details["outside"] == 2
    assert "clipped to the image" in result.summary()


def test_swapped_corners_and_nan_boxes():
    result = _count([(30, 40, 10, 10), (float("nan"), 0, 10, 10)])
    assert result.count == 1
    assert result.boxes == [(10.0, 10.0, 30.0, 40.0)]
    assert result.details["swapped"] == 1
    assert result.details["non_finite"] == 1


def test_min_and_max_area_apply_to_box_area():
    boxes = [(0, 0, 5, 5), (10, 10, 30, 30), (40, 0, 140, 100)]    # 25, 400, 10000 px
    assert _count(boxes).count == 3
    result = _count(boxes, min_area=100, max_area=5000)
    assert result.count == 1
    assert result.details["dropped_by_area"] == 2


def test_detector_that_raises_is_caught_and_reported():
    def broken(image):
        raise RuntimeError("model weights not loaded")

    result = oc.count(IMG, detector=broken)
    assert result.ok is False
    assert result.count == 0
    assert result.confidence == 0.0
    assert result.error == "detector raised RuntimeError: model weights not loaded"
    assert "NOT counted" in result.summary()
    data = result.to_dict()
    assert data["ok"] is False and data["error"].startswith("detector raised")
    json.dumps(data)


@pytest.mark.parametrize("output, fragment", [
    (None, "returned None"),
    ("boxes", "returned text"),
    (42, "returned int"),
    ([(1, 2, 3)], "detection 0"),
    ([((1, 2, 3, 4), "a", "high")], "score must be a number"),
    ({"scores": [0.4]}, "'boxes' key"),
    ({"boxes": [(1, 2, 3, 4)], "scores": [0.1, 0.2]}, "1 boxes but 2 scores"),
    (np.zeros((3, 7)), "(n, 4)"),
])
def test_unreadable_detector_output_is_reported_not_raised(output, fragment):
    result = _count(output)
    assert not result.ok
    assert result.error.startswith("detector output could not be read")
    assert fragment in result.error


def test_detector_gets_a_copy_of_the_image_in_the_type_given():
    seen = []

    def spy(image):
        seen.append(image)
        return []

    img = blank(20, 30, 100)
    oc.count(img, detector=spy)
    assert isinstance(seen[0], np.ndarray) and seen[0] is not img
    assert np.array_equal(seen[0], img)


def test_detector_gets_an_opened_image_for_a_path(tmp_path):
    from PIL import Image

    path = tmp_path / "frame.png"
    Image.fromarray(blank(20, 30, 100)).save(path)
    seen = []

    def spy(image):
        seen.append(image)
        return [(0, 0, 100, 100)]

    result = oc.count(str(path), detector=spy)
    assert isinstance(seen[0], Image.Image)
    assert result.boxes == [(0.0, 0.0, 30.0, 20.0)]


def test_detector_results_are_deterministic():
    boxes = [((5, 5, 25, 25), "a", 0.5), ((30, 30, 60, 60), "b", 0.75)]
    assert _count(boxes).to_dict() == _count(boxes).to_dict()
