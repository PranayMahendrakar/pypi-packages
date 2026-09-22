"""assess_batch: many images, ranked, and one bad file does not stop the rest."""
from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

import image_quality_ai
from conftest import as_rgb, blurred, noisy


@pytest.fixture
def folder(tmp_path, good):
    """A directory holding one good photo, one blurred, one dark and one corrupt."""
    Image.fromarray(as_rgb(good)).save(tmp_path / "good.png")
    Image.fromarray(as_rgb(blurred(good, 8.0))).save(tmp_path / "blurred.png")
    Image.fromarray(as_rgb((good // 9))).save(tmp_path / "dark.png")
    (tmp_path / "corrupt.png").write_bytes(b"not an image")
    return tmp_path


def test_a_batch_of_arrays_is_assessed_in_order(good):
    batch = image_quality_ai.assess_batch([good, blurred(good, 8.0), noisy(good, 18.0)])

    assert len(batch) == 3
    assert len(batch.results) == 3
    assert batch.failures == []
    assert batch[0].usable is True
    assert batch[1].usable is False


def test_a_batch_iterates_and_indexes(good):
    batch = image_quality_ai.assess_batch([good, good])

    assert [report.score for report in batch] == [batch[0].score, batch[1].score]


def test_one_corrupt_file_does_not_stop_the_batch(folder):
    paths = sorted(str(item) for item in folder.iterdir())
    batch = image_quality_ai.assess_batch(paths)

    assert len(batch.results) == 3
    assert len(batch.failures) == 1
    assert "corrupt.png" in batch.failures[0]["source"]
    assert "cannot read image" in batch.failures[0]["error"]


def test_a_missing_file_is_recorded_rather_than_raised(tmp_path, good):
    present = tmp_path / "there.png"
    Image.fromarray(as_rgb(good)).save(present)

    batch = image_quality_ai.assess_batch([str(present), str(tmp_path / "gone.png")])

    assert len(batch.results) == 1
    assert len(batch.failures) == 1
    assert "no such image file" in batch.failures[0]["error"]


def test_worst_ranks_the_lowest_scores_first(good):
    batch = image_quality_ai.assess_batch(
        [good, blurred(good, 8.0), (good // 9), noisy(good, 18.0)]
    )

    worst = batch.worst(2)

    assert len(worst) == 2
    assert worst[0].score <= worst[1].score
    assert worst[0].score == min(report.score for report in batch.results)


def test_worst_defaults_to_five_and_never_asks_for_more_than_there_is(good):
    batch = image_quality_ai.assess_batch([good, blurred(good, 8.0)])

    assert len(batch.worst()) == 2
    assert batch.worst(0) == []
    with pytest.raises(ValueError, match="zero or more"):
        batch.worst(-1)


def test_worst_is_deterministic_when_scores_tie(good):
    batch = image_quality_ai.assess_batch([good] * 4)

    assert [report.source for report in batch.worst(4)] == [
        report.source for report in batch.worst(4)
    ]


def test_usable_and_rejected_split_the_batch(good):
    batch = image_quality_ai.assess_batch([good, blurred(good, 8.0), good])

    assert len(batch.usable) == 2
    assert len(batch.rejected) == 1
    assert len(batch.usable) + len(batch.rejected) == len(batch)


def test_workers_change_nothing_but_the_speed(folder):
    paths = sorted(str(item) for item in folder.iterdir())

    single = image_quality_ai.assess_batch(paths, workers=1)
    threaded = image_quality_ai.assess_batch(paths, workers=4)

    assert single.to_dict() == threaded.to_dict()


def test_workers_must_be_at_least_one(good):
    with pytest.raises(ValueError, match="workers must be 1 or more"):
        image_quality_ai.assess_batch([good], workers=0)


def test_a_single_image_is_accepted_as_a_batch_of_one(good):
    assert len(image_quality_ai.assess_batch(good)) == 1


def test_an_empty_batch_is_not_an_error():
    batch = image_quality_ai.assess_batch([])

    assert len(batch) == 0
    assert batch.mean_score() == 0.0
    assert batch.summary() == "No images were assessed."
    assert batch.to_dict()["count"] == 0


def test_batch_summary_names_the_worst_and_the_unreadable(folder):
    paths = sorted(str(item) for item in folder.iterdir())
    text = image_quality_ai.assess_batch(paths).summary()

    assert "3 image(s) assessed" in text
    assert "1 could not be read" in text
    assert "Worst first:" in text
    assert "Could not be read:" in text
    assert text.isprintable() or "\n" in text


def test_batch_grade_counts_and_mean(good):
    batch = image_quality_ai.assess_batch([good, blurred(good, 8.0)])

    counts = batch.grade_counts()

    assert sum(counts.values()) == 2
    assert counts["F"] == 1
    assert 0.0 <= batch.mean_score() <= 100.0


def test_batch_rows_are_one_flat_dict_per_image(good):
    rows = image_quality_ai.assess_batch([good, blurred(good, 8.0)]).rows()

    assert len(rows) == 2
    assert set(rows[0]) >= {"source", "score", "grade", "usable", "sharpness", "top_issue"}
    assert rows[0]["sharpness_value"] > rows[1]["sharpness_value"]


def test_to_frame_uses_pandas_when_it_is_there_and_says_so_when_it_is_not(good):
    batch = image_quality_ai.assess_batch([good, blurred(good, 8.0)])

    pandas = pytest.importorskip("pandas") if _has_pandas() else None
    if pandas is None:
        with pytest.raises(ImportError, match="needs pandas"):
            batch.to_frame()
        return

    frame = batch.to_frame()
    assert list(frame.columns) == list(batch.rows()[0])
    assert len(frame) == 2


def _has_pandas() -> bool:
    try:
        import pandas  # noqa: F401
    except ImportError:
        return False
    return True


def test_batch_json_is_safe_for_any_alphabet(tmp_path, good):
    target = tmp_path / "写真.png"
    Image.fromarray(as_rgb(good)).save(target)

    text = image_quality_ai.assess_batch([str(target)]).to_json()

    assert "写真" in text


def test_batch_thresholds_are_passed_through(good):
    strict = image_quality_ai.assess_batch([good], thresholds={"usable_score": 99.0})

    assert strict[0].usable is False
    assert strict[0].changed_thresholds() == {"usable_score": 99.0}


def test_the_assessor_object_batches_too(good):
    assessor = image_quality_ai.ImageAssessor({"usable_score": 99.0})

    batch = assessor.assess_batch([good, good], workers=2)

    assert len(batch) == 2
    assert batch.usable == []
