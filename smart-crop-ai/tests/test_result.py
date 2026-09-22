"""CropResult: the part that has to explain itself."""
from __future__ import annotations

import json

import numpy as np
import pytest
from PIL import Image

import smart_crop_ai
from smart_crop_ai import CropResult, confidence_label
from conftest import subject_array


def test_summary_names_everything_that_matters(subject_image):
    text = smart_crop_ai.crop(subject_image, ratio=1.0).summary()

    for expected in ("source", "asked for", "crop", "strategy", "confidence", "keeps"):
        assert expected in text
    assert "800 x 400" in text
    assert "400 x 400" in text


def test_summary_is_plain_ascii(subject_image):
    """CONVENTIONS: no arrows, bullets or box characters in summary text."""
    for strategy in smart_crop_ai.STRATEGIES:
        result = smart_crop_ai.crop(subject_image, ratio=1.0, strategy=strategy)
        text = result.summary()
        text.encode("ascii")  # raises if anything fancy crept in
        assert not any(ch in text for ch in "→•─“’")


def test_summary_survives_a_non_ascii_source_name(subject_image):
    """A unicode filename may appear in the report without breaking it."""
    result = smart_crop_ai.crop(subject_image, ratio=1.0)
    result.source = "東京-привет-café.png"

    text = result.summary()
    assert "café" in text
    text.encode("utf-8")


def test_summary_mentions_the_strategy_that_was_asked_for_when_it_differs(flat_image):
    text = smart_crop_ai.crop(flat_image, ratio=1.0).summary()
    assert "center" in text
    assert "asked for: auto" in text


def test_summary_includes_the_notes(subject_image):
    result = smart_crop_ai.crop(subject_image, 5000, 5000)
    assert "notes" in result.summary()
    assert "Nothing was upscaled" in result.summary()


def test_to_dict_is_json_safe_and_holds_no_pixels(subject_image):
    payload = smart_crop_ai.crop(subject_image, ratio=1.0).to_dict()

    text = json.dumps(payload)            # raises on anything numpy or PIL
    assert "image" not in payload
    assert json.loads(text) == payload
    for key in (
        "source", "box", "size", "offset", "strategy_used", "strategy_requested",
        "confidence", "confidence_label", "padding", "covers", "moved", "scores",
        "notes", "source_size", "target_size", "destination",
    ):
        assert key in payload


def test_to_dict_values_are_plain_python_types(subject_image):
    payload = smart_crop_ai.crop(subject_image, ratio=1.0).to_dict()

    assert isinstance(payload["confidence"], float)
    assert isinstance(payload["moved"], bool)
    assert isinstance(payload["box"], list)
    assert all(isinstance(value, int) for value in payload["box"])
    assert all(isinstance(value, float) for value in payload["scores"].values())


def test_to_json_keeps_non_ascii_as_is(subject_image):
    result = smart_crop_ai.crop(subject_image, ratio=1.0)
    result.source = "東京.png"

    text = result.to_json()
    assert "東京" in text
    assert json.loads(text)["source"] == "東京.png"


def test_box_and_size_and_offset_agree(subject_image):
    result = smart_crop_ai.crop(subject_image, 300, 200)
    left, top, right, bottom = result.box

    assert result.size == (right - left, bottom - top)
    assert result.offset == (left, top)
    assert result.image.size == result.size


def test_covers_is_the_share_of_the_source_kept(subject_image):
    result = smart_crop_ai.crop(subject_image, 400, 400)
    assert result.covers == pytest.approx((400 * 400) / (800 * 400))


def test_moved_says_whether_the_box_left_the_centre(subject_image):
    found = smart_crop_ai.crop(subject_image, ratio=1.0)
    centred = smart_crop_ai.crop(subject_image, ratio=1.0, strategy="center")

    assert found.moved
    assert not centred.moved


@pytest.mark.parametrize(
    "confidence, expected",
    [(1.0, "strong"), (0.35, "strong"), (0.2, "moderate"), (0.05, "weak"), (0.0, "none")],
)
def test_confidence_label_table(confidence, expected):
    assert confidence_label(confidence) == expected


def test_confidence_sentence_explains_what_was_measured(subject_image):
    result = smart_crop_ai.crop(subject_image, ratio=1.0)
    sentence = result.confidence_sentence()

    assert "centre crop" in sentence
    assert "%" in sentence
    assert result.confidence_label in sentence


def test_a_zero_confidence_sentence_says_it_bought_nothing(flat_image):
    sentence = smart_crop_ai.crop(flat_image, ratio=1.0).confidence_sentence()
    assert "no better than a centre crop" in sentence


def test_scores_carry_the_numbers_behind_the_decision(subject_image):
    scores = smart_crop_ai.crop(subject_image, ratio=1.0).scores

    for key in ("best_window", "center_window", "energy_peak", "energy_spread",
                "window_spread"):
        assert key in scores
    assert scores["best_window"] >= scores["center_window"]
    assert 0.0 <= scores["energy_peak"] <= 1.0


def test_the_confidence_matches_the_scores_it_came_from(subject_image):
    """Confidence is not a mood; it is arithmetic on two window scores."""
    result = smart_crop_ai.crop(subject_image, ratio=1.0)
    best = result.scores["best_window"]
    centre = result.scores["center_window"]

    assert result.confidence == pytest.approx((best - centre) / best, abs=1e-9)


def test_save_writes_the_crop(tmp_path, subject_image):
    result = smart_crop_ai.crop(subject_image, ratio=1.0)
    path = tmp_path / "sub" / "out.png"

    written = result.save(str(path))

    assert written == str(path)
    assert result.destination == str(path)
    assert Image.open(path).size == result.size


def test_save_flattens_alpha_for_jpeg_and_records_it(tmp_path):
    array = np.zeros((200, 400, 4), dtype=np.uint8)
    array[..., :3] = subject_array()[:200, :400]
    array[..., 3] = 255
    array[20:60, 20:60, 3] = 0

    result = smart_crop_ai.crop(Image.fromarray(array, "RGBA"), ratio=1.0)
    path = tmp_path / "out.jpg"
    result.save(str(path))

    assert Image.open(path).mode == "RGB"
    assert any("composited onto white" in note for note in result.notes)


def test_save_does_not_repeat_the_flatten_note(tmp_path):
    array = np.zeros((120, 240, 4), dtype=np.uint8)
    array[..., :3] = 180
    array[..., 3] = 255
    result = smart_crop_ai.crop(Image.fromarray(array, "RGBA"), ratio=1.0)

    result.save(str(tmp_path / "a.jpg"))
    result.save(str(tmp_path / "b.jpg"))

    flatten_notes = [n for n in result.notes if "composited onto white" in n]
    assert len(flatten_notes) == 1


def test_the_result_can_be_built_by_hand():
    """It is a plain dataclass, so anyone can construct one in a test."""
    result = CropResult(
        image=Image.new("RGB", (10, 10)),
        box=(0, 0, 10, 10),
        strategy_used="center",
        confidence=0.0,
        source_size=(20, 20),
        target_size=(10, 10),
    )

    assert result.size == (10, 10)
    assert result.covers == pytest.approx(0.25)
    assert result.confidence_label == "none"
    assert "smart-crop-ai" in result.summary()


def test_repr_is_short_and_useful(subject_image):
    text = repr(smart_crop_ai.crop(subject_image, ratio=1.0))
    assert text.startswith("CropResult(")
    assert "confidence=" in text
