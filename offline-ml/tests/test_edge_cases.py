"""The awkward inputs, and the cases a user hits on a real machine."""
from __future__ import annotations

import io
import json

import pytest

import offline_ml
from offline_ml import ModelSpec, recommend

from conftest import machine


# --------------------------------------------------------------------------- #
# nothing fits
# --------------------------------------------------------------------------- #
def test_model_larger_than_total_ram_is_rejected_not_recommended(tiny_box):
    pick = recommend([{"name": "llama-70b-q4", "size_gb": 39.0}], hardware=tiny_box)
    assert pick.fits is False
    assert pick.model is None
    assert pick.name is None
    why = pick.rejected["llama-70b-q4"]
    assert "RAM" in why and "0.5 GiB in total" in why
    assert "None of the 1 model fits" in pick.reason
    assert pick.summary().startswith("offline-ml: nothing fits")
    assert pick.device == "cpu"


def test_nothing_fits_still_produces_a_usable_result(tiny_box, models):
    pick = recommend(models, hardware=tiny_box)
    assert pick.fits is False and pick.alternatives == []
    assert set(pick.rejected) == {spec["name"] for spec in models}
    assert "smaller or more heavily quantized" in pick.reason
    assert json.loads(json.dumps(pick.to_dict(), ensure_ascii=False))["model"] is None


def test_no_model_matches_the_task(cpu_box):
    pick = recommend(
        [{"name": "whisper", "size_gb": 0.5, "task": "asr"}], task="chat", hardware=cpu_box
    )
    assert pick.fits is False
    assert "is a 'chat' model" in pick.reason
    assert "leave task out" in pick.reason


# --------------------------------------------------------------------------- #
# bad input
# --------------------------------------------------------------------------- #
def test_empty_model_list_raises_value_error(cpu_box):
    with pytest.raises(ValueError, match="no models to choose from"):
        recommend([], hardware=cpu_box)


def test_bad_prefer_names_the_choices(cpu_box):
    with pytest.raises(ValueError, match="prefer must be one of"):
        recommend([{"name": "m", "size_gb": 1.0}], prefer="cheapest", hardware=cpu_box)


@pytest.mark.parametrize("headroom", [-0.1, 99.0, "lots", None])
def test_bad_headroom_is_rejected(cpu_box, headroom):
    with pytest.raises(ValueError, match="headroom must be"):
        recommend([{"name": "m", "size_gb": 1.0}], headroom=headroom, hardware=cpu_box)


@pytest.mark.parametrize("size", [0, -4.0, float("nan"), float("inf"), "big", None])
def test_bad_size_is_rejected(size):
    with pytest.raises(ValueError, match="size_gb"):
        ModelSpec(name="m", size_gb=size)


def test_missing_keys_are_named():
    with pytest.raises(ValueError, match="missing 'size_gb'"):
        ModelSpec.coerce({"name": "m"})
    with pytest.raises(ValueError, match="missing 'name'"):
        ModelSpec.coerce({"size_gb": 1.0})


def test_unknown_keys_are_named():
    with pytest.raises(ValueError, match="unknown model keys: 'sizegb'"):
        ModelSpec.coerce({"name": "m", "size_gb": 1.0, "sizegb": 2})


def test_empty_name_is_rejected():
    with pytest.raises(ValueError, match="non-empty name"):
        ModelSpec(name="   ", size_gb=1.0)


def test_duplicate_model_names_are_rejected(cpu_box):
    twice = [{"name": "same", "size_gb": 1.0}, {"name": "same", "size_gb": 2.0}]
    with pytest.raises(ValueError, match="duplicate model names: 'same'"):
        recommend(twice, hardware=cpu_box)


@pytest.mark.parametrize("models", ["mistral", {"name": "m", "size_gb": 1.0}, 7])
def test_a_single_model_instead_of_a_list_is_explained(cpu_box, models):
    with pytest.raises(ValueError, match="models must be a list"):
        recommend(models, hardware=cpu_box)


def test_a_non_model_item_is_explained(cpu_box):
    with pytest.raises(ValueError, match="must be a ModelSpec or a dict"):
        recommend(["mistral-7b"], hardware=cpu_box)


def test_negative_minimums_are_rejected():
    with pytest.raises(ValueError, match="min_ram_gb cannot be negative"):
        ModelSpec(name="m", size_gb=1.0, min_ram_gb=-8)


def test_bad_task_type_is_explained():
    with pytest.raises(ValueError, match="task must be a string"):
        ModelSpec(name="m", size_gb=1.0, task=3)


# --------------------------------------------------------------------------- #
# text and encoding
# --------------------------------------------------------------------------- #
def test_unicode_model_names_survive(cpu_box):
    exotic = [
        {"name": "模型-7b-q4", "size_gb": 4.1, "quality": 8},
        {"name": "Café-Mötör-tiny", "size_gb": 0.5, "quality": 2},
    ]
    pick = recommend(exotic, prefer="quality", hardware=cpu_box)
    assert pick.model.name == "模型-7b-q4"
    assert "模型-7b-q4" in pick.summary()
    text = json.dumps(pick.to_dict(), ensure_ascii=False)
    assert "模型-7b-q4" in text
    assert json.loads(text)["model"]["name"] == "模型-7b-q4"


def _print_through(text: str, encoding: str, errors: str) -> bytes:
    """Print `text` to a stream encoded like a hostile console, return the bytes.

    This is what ``cli.main()`` does after it reconfigures stdout, so it is the
    behaviour worth asserting on - unlike ``str.encode(errors="replace")``,
    which cannot raise and therefore cannot fail this test.
    """
    buffer = io.BytesIO()
    stream = io.TextIOWrapper(buffer, encoding=encoding, errors=errors, newline="")
    try:
        print(text, file=stream)
        stream.flush()
        return buffer.getvalue()
    finally:
        stream.detach()


def test_summaries_print_through_a_console_that_cannot_encode_them(cpu_box):
    """Whatever is in the data, printing must never be able to raise."""
    pick = recommend([{"name": "Ωmega-ß", "size_gb": 1.0}], hardware=cpu_box)
    texts = [pick.summary(), cpu_box.summary(), pick.reason]
    assert any(ord(char) > 127 for text in texts for char in text), (
        "the fixture lost its non-ASCII text, so this test would prove nothing"
    )
    for text in texts:
        for encoding in ("ascii", "cp1252"):
            written = _print_through(text, encoding, "replace")
            decoded = written.decode(encoding)  # raises if a stray byte got out
            assert decoded.endswith("\n")
            # errors="replace" swaps one character for one '?', never drops any.
            assert len(decoded) == len(text) + 1
    with pytest.raises(UnicodeEncodeError):
        _print_through(pick.reason, "ascii", "strict")


# --------------------------------------------------------------------------- #
# odd but legal machines
# --------------------------------------------------------------------------- #
def test_unknown_cpu_frequency_is_reported_not_faked():
    box = machine()
    box.cpu_freq_mhz = None
    assert "unknown speed" in box.summary()
    assert box.to_dict()["cpu_freq_mhz"] is None


def test_gpu_with_unreadable_vram_falls_back_to_cpu():
    box = machine(gpus=[offline_ml.GPU(name="NVIDIA GPU", vram_gb=None, backend="cuda")])
    assert box.has_gpu is True and box.vram_gb == 0.0
    pick = recommend([{"name": "m", "size_gb": 1.0}], hardware=box)
    assert pick.device == "cpu"
    assert "could not be read" in pick.reason
    assert "VRAM unknown" in box.summary()


def test_zero_headroom_is_allowed(cpu_box):
    pick = recommend([{"name": "m", "size_gb": 4.0}], headroom=0.0, hardware=cpu_box)
    assert pick.requirements["m"]["ram_gb"] == 4.0


def test_every_public_name_is_exported():
    for name in offline_ml.__all__:
        assert hasattr(offline_ml, name), name
    assert offline_ml.__version__ == "0.1.0"


# --------------------------------------------------------------------------- #
# free space that could not be read is missing information, not a full disk
# --------------------------------------------------------------------------- #
def blind_box(**kwargs):
    """A roomy machine whose free space psutil refused to report."""
    box = machine(ram_total_gb=128.0, ram_available_gb=120.0, **kwargs)
    box.disk_free_gb = None
    return box


def test_unreadable_free_space_does_not_reject_every_model():
    pick = recommend([{"name": "tinyllama-1.1b-q4", "size_gb": 0.7}], hardware=blind_box())
    assert pick.fits is True
    assert pick.model.name == "tinyllama-1.1b-q4"
    assert "0.0 GiB is free" not in pick.reason
    assert "smaller or more heavily quantized" not in pick.reason


def test_unreadable_free_space_does_not_stop_the_ram_gate():
    """Skipping the disk check must not weaken the check that still works."""
    pick = recommend([{"name": "big-download", "size_gb": 300.0}], hardware=blind_box())
    assert pick.fits is False
    assert "360.0 GiB of RAM" in pick.rejected["big-download"]
    assert "128.0 GiB in total" in pick.rejected["big-download"]


def test_unreadable_free_space_says_the_disk_check_was_skipped():
    box = blind_box()
    pick = recommend([{"name": "phi-3-mini-q4", "size_gb": 2.3}], hardware=box)
    assert "could not be read" in pick.reason
    assert "disk check was skipped" in pick.reason
    assert "2.3" in pick.reason
    assert "could not be read" in box.summary()
    assert box.to_dict()["disk_free_gb"] is None
    assert json.loads(json.dumps(pick.to_dict(), ensure_ascii=False))["fits"] is True


def test_fits_says_yes_when_the_disk_cannot_be_checked():
    assert offline_ml.fits(2.0, hardware=blind_box()) is True
    assert offline_ml.fits(5000.0, hardware=blind_box()) is False  # RAM still decides


def test_a_readable_disk_still_rejects_what_will_not_download():
    full = machine(ram_total_gb=128.0, ram_available_gb=120.0, disk_free_gb=1.0)
    pick = recommend([{"name": "big-download", "size_gb": 20.0}], hardware=full)
    assert pick.fits is False
    assert "only 1.0 GiB is free" in pick.rejected["big-download"]
    assert "Free up disk space" in pick.reason


# --------------------------------------------------------------------------- #
# the "nothing fits" sentence reads as a sentence
# --------------------------------------------------------------------------- #
def test_nothing_fits_reason_has_one_subject_per_clause():
    box = machine(ram_total_gb=4.0, ram_available_gb=3.0)
    pick = recommend(
        [
            {"name": "whisper", "size_gb": 0.5, "task": "asr"},
            {"name": "llama-70b", "size_gb": 39.0, "task": "chat"},
        ],
        task="chat",
        hardware=box,
    )
    reason = pick.reason
    assert "'whisper', it is" not in reason      # the double subject
    assert "a 'asr'" not in reason               # the wrong article
    assert "The smallest one, 'whisper', is an 'asr' model" in reason
    # the advice must match why it was actually rejected
    assert "Pass a different task" in reason
    assert "smaller or more heavily quantized" not in reason
    assert pick.rejected["whisper"].startswith("is an 'asr' model")


def test_nothing_fits_on_ram_still_advises_a_smaller_build(tiny_box, models):
    reason = recommend(models, hardware=tiny_box).reason
    assert "The smallest one, 'tinyllama-1.1b-q4', needs about" in reason
    assert "smaller or more heavily quantized" in reason


def test_task_rejection_uses_the_right_article(cpu_box):
    for task, article in (("asr", "an"), ("chat", "a"), ("embedding", "an")):
        pick = recommend(
            [{"name": "m", "size_gb": 1.0, "task": task}], task="other", hardware=cpu_box
        )
        assert pick.rejected["m"] == f"is {article} {task!r} model and you asked for 'other'"


def test_negative_quality_is_accepted_but_negative_minimums_are_not():
    assert ModelSpec(name="m", size_gb=1.0, quality=-3, speed=-2).speed == -2.0
    with pytest.raises(ValueError, match="min_vram_gb cannot be negative"):
        ModelSpec(name="m", size_gb=1.0, min_vram_gb=-1)
