"""Saving, loading and the robust statistics underneath."""
from __future__ import annotations

import json

import numpy as np
import pytest

from conftest import different_scene, held_out_scene, normal_set
from vision_anomaly import Detector, Profile
from vision_anomaly._profile import (
    NOISE_FLOOR,
    PROFILE_FORMAT,
    RELATIVE_FLOOR,
    small_sample_factor,
)


def test_a_saved_profile_scores_exactly_like_the_detector_that_wrote_it(
    tmp_path, good_images, odd_image, clean_image
):
    original = Detector(sensitivity=4.0).fit(good_images)
    path = original.save(tmp_path / "profile.json")

    reloaded = Detector.load(path, sensitivity=4.0)

    assert reloaded.score(odd_image) == pytest.approx(original.score(odd_image))
    assert reloaded.score(clean_image) == pytest.approx(original.score(clean_image))
    assert reloaded.profile.n_images == original.profile.n_images
    assert reloaded.predict(odd_image).to_dict() == original.predict(odd_image).to_dict()


def test_a_profile_is_readable_json_and_not_a_pickle(tmp_path, good_images):
    path = Detector().fit(good_images).save(tmp_path / "profile.json")

    text = open(path, "r", encoding="utf-8").read()
    data = json.loads(text)

    assert data["format"] == PROFILE_FORMAT
    assert data["n_images"] == 8
    assert len(data["median"]) == len(data["scale"]) == 216
    assert data["features"]["grid"] == 4
    assert text.lstrip().startswith("{")


def test_a_saved_profile_keeps_its_own_layout_not_the_current_default(
    tmp_path, good_images, odd_image
):
    """Loading must not quietly re-measure with different settings."""
    original = Detector(grid=8, analysis_size=128).fit(good_images)
    path = original.save(tmp_path / "eight.json")

    reloaded = Detector.load(path)

    assert reloaded.config.grid == 8
    assert reloaded.config.analysis_size == 128
    assert reloaded.score(odd_image) == pytest.approx(original.score(odd_image))


def test_saving_creates_missing_directories(tmp_path, good_images):
    path = Detector().fit(good_images).save(tmp_path / "deep" / "down" / "p.json")

    assert Profile.load(path).n_images == 8


def test_loading_something_that_is_not_a_profile_says_so(tmp_path):
    missing = tmp_path / "gone.json"
    with pytest.raises(ValueError) as raised:
        Detector.load(str(missing))
    assert "no such profile file" in str(raised.value)

    with pytest.raises(ValueError) as raised:
        Detector.load(str(tmp_path))
    assert "is a directory" in str(raised.value)

    broken = tmp_path / "broken.json"
    broken.write_text("{not json at all", encoding="utf-8")
    with pytest.raises(ValueError) as raised:
        Detector.load(str(broken))
    assert "not valid JSON" in str(raised.value)
    assert str(broken) in str(raised.value)

    stranger = tmp_path / "stranger.json"
    stranger.write_text(json.dumps({"hello": "world"}), encoding="utf-8")
    with pytest.raises(ValueError) as raised:
        Detector.load(str(stranger))
    assert "not a vision-anomaly profile" in str(raised.value)


def test_a_damaged_profile_is_refused_rather_than_used(tmp_path, good_images):
    path = Detector().fit(good_images).save(tmp_path / "p.json")
    data = json.loads(open(path, encoding="utf-8").read())

    short = dict(data, median=data["median"][:10])
    with pytest.raises(ValueError) as raised:
        Profile.from_dict(short)
    assert "damaged" in str(raised.value)

    zeroed = dict(data, scale=[0.0] * len(data["scale"]))
    with pytest.raises(ValueError) as raised:
        Profile.from_dict(zeroed)
    assert "zero or negative scale" in str(raised.value)

    wrong_version = dict(data, format="vision-anomaly/profile/99")
    with pytest.raises(ValueError) as raised:
        Profile.from_dict(wrong_version)
    assert "expected format" in str(raised.value)

    unknown_layout = dict(data, features=dict(data["features"], wobble=1))
    with pytest.raises(ValueError) as raised:
        Profile.from_dict(unknown_layout)
    assert "unknown feature settings" in str(raised.value)


def test_a_profile_carries_its_own_yardstick(good_images):
    profile = Detector().fit(good_images).profile

    assert len(profile.fit_scores) == 8
    assert profile.typical_fit_score == pytest.approx(0.0, abs=1e-9)
    assert profile.worst_fit_score >= profile.typical_fit_score
    assert "8 images" in profile.summary()
    assert "216 features" in profile.summary()


def test_no_feature_spread_is_ever_below_the_floors(good_images):
    profile = Detector().fit(good_images).profile

    assert float(profile.scale.min()) >= NOISE_FLOOR
    floor = np.maximum(np.abs(profile.median) * RELATIVE_FLOOR, NOISE_FLOOR)
    assert np.all(profile.scale >= floor - 1e-12)


def test_the_small_sample_correction_shrinks_as_the_sample_grows():
    assert small_sample_factor(1) == 1.0
    assert small_sample_factor(5) > small_sample_factor(20) > small_sample_factor(100)
    assert small_sample_factor(100) == pytest.approx(1.0, abs=0.01)


def test_one_bad_image_in_the_normal_set_does_not_blind_the_profile():
    """The whole reason for medians. A mean would swallow the anomaly whole."""
    dirty = list(normal_set(9)) + [different_scene(1)]

    detector = Detector().fit(dirty)

    assert detector.score(held_out_scene(0)) < detector.sensitivity
    assert detector.predict(different_scene(0)).anomalous


def test_a_deviation_against_the_wrong_number_of_features_is_refused(good_images):
    profile = Detector().fit(good_images).profile

    with pytest.raises(ValueError) as raised:
        profile.deviation(np.zeros(5))
    assert "5 numbers" in str(raised.value)
    assert "216" in str(raised.value)


def test_profile_round_trips_through_to_dict(good_images):
    profile = Detector().fit(good_images).profile

    again = Profile.from_dict(profile.to_dict())

    assert again.median.tolist() == pytest.approx(profile.median.tolist())
    assert again.scale.tolist() == pytest.approx(profile.scale.tolist())
    assert again.notes == profile.notes
    assert again.config == profile.config
