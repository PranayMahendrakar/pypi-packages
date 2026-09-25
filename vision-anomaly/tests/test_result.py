"""The result objects: what they say, and whether it survives a JSON round trip."""
from __future__ import annotations

import json

import pytest

from conftest import (
    colour_fault,
    different_scene,
    held_out_scene,
    missing_part,
    normal_scene,
)
from vision_anomaly import Box, Detector, detect_anomalies
from vision_anomaly._features import GROUP_NAMES
from vision_anomaly._result import confidence_for


@pytest.fixture(scope="module")
def fitted(request):
    from conftest import normal_set

    return Detector().fit(normal_set(8))


def test_a_flagged_result_says_what_moved_and_where(fitted):
    result = fitted.predict(colour_fault(0))

    assert result.anomalous
    assert result.verdict == "anomalous"
    assert result.reasons
    assert result.reasons[0].startswith("colour")
    assert "sigmas" in result.reasons[0]
    assert result.regions
    assert result.regions[0].score >= result.regions[-1].score


def test_a_clean_result_points_nowhere_rather_than_somewhere_arbitrary(fitted):
    """Padding the region list would invent a defect in a good image."""
    result = fitted.predict(held_out_scene(0))

    assert not result.anomalous
    assert result.regions == []
    assert result.reasons == []
    assert "nothing moved" in result.summary()


def test_the_regions_land_on_the_fault_not_somewhere_else(fitted):
    """The stain in colour_fault covers the middle of the frame."""
    result = fitted.predict(colour_fault(0))

    worst = result.regions[0]
    assert worst.row in (1, 2)
    assert worst.col in (1, 2)


def test_a_box_maps_onto_the_callers_own_picture(fitted):
    result = fitted.predict(colour_fault(0))
    box = result.regions[0]

    assert (box.x, box.y, box.width, box.height) == (
        box.col * 64,
        box.row * 64,
        64,
        64,
    )

    drawn = box.scaled(512, 384)
    assert drawn.width == 128 and drawn.height == 96
    assert drawn.x == box.col * 128 and drawn.y == box.row * 96
    assert drawn.score == box.score
    assert (box.x, box.y) == (box.col * 64, box.row * 64)   # the original is untouched


def test_a_box_scaled_to_an_odd_size_still_covers_the_picture():
    grid = 3
    boxes = [Box(row=r, col=c, x=0, y=0, width=1, height=1, score=1.0, grid=grid)
             for r in range(grid) for c in range(grid)]

    scaled = [box.scaled(100, 70) for box in boxes]

    assert min(box.x for box in scaled) == 0
    assert max(box.x + box.width for box in scaled) == 100
    assert max(box.y + box.height for box in scaled) == 70
    assert all(box.width > 0 and box.height > 0 for box in scaled)


def test_every_family_is_reported_even_the_quiet_ones(fitted):
    result = fitted.predict(colour_fault(0))

    assert set(result.group_scores) == set(GROUP_NAMES)
    assert result.score == pytest.approx(max(result.group_scores.values()))
    assert min(result.group_scores.values()) >= 0.0


def test_confidence_is_confidence_in_the_verdict_not_in_the_anomaly(fitted):
    clean = fitted.predict(held_out_scene(0))
    obvious = fitted.predict(different_scene(0))

    assert 0.0 <= clean.confidence <= 1.0
    assert clean.confidence > 0.9            # confidently normal
    assert obvious.confidence > 0.99         # confidently not
    assert confidence_for(3.0, 3.0) == pytest.approx(0.5)
    assert confidence_for(0.0, 0.0) == 1.0


def test_confidence_is_lowest_on_the_line():
    on_the_line = confidence_for(3.0, 3.0)
    nearby = confidence_for(3.3, 3.0)
    far = confidence_for(30.0, 3.0)

    assert on_the_line < nearby < far


def test_a_result_is_json_safe_all_the_way_down(fitted):
    result = fitted.predict(colour_fault(0))

    text = json.dumps(result.to_dict(), ensure_ascii=False)
    data = json.loads(text)

    assert data["anomalous"] is True
    assert isinstance(data["score"], float)
    assert isinstance(data["regions"][0]["row"], int)
    assert set(data["group_scores"]) == set(GROUP_NAMES)
    assert data["n_fitted"] == 8


def test_a_summary_is_plain_ascii(fitted):
    """Arrows and box characters are what break a cp1252 console."""
    text = fitted.predict(colour_fault(0)).summary()

    assert text.isascii()
    assert "ANOMALOUS" in text
    assert "where it moved most" in text
    assert "row 1, column" in text or "row 2, column" in text


def test_a_batch_report_iterates_indexes_and_totals(fitted):
    images = [held_out_scene(0), colour_fault(0), different_scene(0), held_out_scene(1)]

    report = fitted.predict_batch(images)

    assert len(report) == 4
    assert len(list(report)) == 4
    assert report[1].anomalous
    assert report.n_anomalous == 2
    assert len(report.anomalous) == 2
    assert len(report.normal) == 2
    assert report.anomalous[0].score > report.anomalous[1].score
    assert report.worst is report.anomalous[0]
    assert report.scores == [result.score for result in report]


def test_an_empty_batch_is_an_empty_report(fitted):
    report = fitted.predict_batch([])

    assert len(report) == 0
    assert report.n_anomalous == 0
    assert report.worst is None
    assert "0 images checked" in report.summary()
    assert report.to_dict()["results"] == []


def test_a_batch_summary_says_so_when_nothing_is_wrong(fitted):
    report = fitted.predict_batch([held_out_scene(0), held_out_scene(1)])

    text = report.summary()
    assert "0 anomalous" in text
    assert "nothing unusual" in text
    assert text.isascii()


def test_a_batch_summary_limits_how_much_it_prints(fitted):
    images = [colour_fault(i) for i in range(3)] + [missing_part(i) for i in range(3)]
    report = fitted.predict_batch(images)

    short = report.summary(limit=2)
    assert "and 4 more anomalous images" in short
    assert len(short) < len(report.summary(limit=10))


def test_a_report_serialises_to_json_with_non_ascii_intact(fitted, tmp_path):
    from conftest import save_png

    path = save_png(different_scene(0), tmp_path / "异常.png")
    report = fitted.predict_batch([path])

    text = report.to_json()

    assert "异常" in text            # not escaped away
    assert json.loads(text)["n_anomalous"] == 1


def test_detect_anomalies_report_carries_the_fit_yardstick(good_images):
    report = detect_anomalies(good_images, [held_out_scene(0)])

    assert report.n_fitted == 8
    assert report.fit_worst >= report.fit_typical
    assert "the known-good images themselves score" in report.summary()


def test_the_result_records_how_the_image_was_read(fitted):
    result = fitted.predict(colour_fault(0))

    assert (result.width, result.height) == (512, 384)
    assert result.channels == 3
    assert result.threshold == fitted.sensitivity


def _mixed_size_set():
    """Six good frames, every one a different shape."""
    shapes = [(320, 240), (256, 256), (400, 300), (200, 260), (512, 384), (300, 300)]
    return [normal_scene(index, size) for index, size in enumerate(shapes)]


def test_the_resize_caveat_reaches_the_result_not_only_the_profile():
    """A predict_batch caller never reads detector.summary().

    The README promises the note lands on ``.notes``; it was landing on the
    profile alone whenever the fitted set was itself mixed-size, so the caveat
    behind every score went unseen.
    """
    mixed = _mixed_size_set()
    detector = Detector().fit(mixed)

    result = detector.predict(mixed[1])

    assert any("resampled onto the 256x256 analysis grid" in note for note in result.notes)
    assert any("different sizes" in note for note in result.notes)


def test_the_resize_caveat_survives_to_json_and_to_a_batch():
    mixed = _mixed_size_set()
    detector = Detector().fit(mixed)

    report = detector.predict_batch(mixed)

    assert all(result.notes for result in report)
    assert any("different sizes" in note for note in report.results[0].to_dict()["notes"])


def test_a_single_size_fit_set_says_nothing_about_sizes(good_images, clean_image):
    """The note must not become noise on the ordinary path."""
    detector = Detector().fit(good_images)

    assert detector.predict(good_images[0]).notes == []
