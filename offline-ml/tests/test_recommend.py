"""recommend(), fits() and the Recommendation object."""
from __future__ import annotations

import json

import pytest

import offline_ml
from offline_ml import GPU, ModelSpec, recommend

from conftest import machine

FANCY = "→←•·▸─│–—"


def test_quickstart_shape(cpu_box, models):
    pick = recommend(models, hardware=cpu_box)
    assert pick.fits is True
    assert isinstance(pick.model, ModelSpec)
    assert pick.device == "cpu"
    assert pick.reason.startswith("'" + pick.model.name + "'")
    assert pick.name == pick.model.name


def test_prefer_smallest_picks_the_smallest(cpu_box, models):
    assert recommend(models, prefer="smallest", hardware=cpu_box).model.name == "tinyllama-1.1b-q4"


def test_prefer_quality_picks_the_best_that_fits(cpu_box, models):
    # llama-70b needs ~47 GiB and does not fit in 32 GiB, so mistral wins.
    pick = recommend(models, prefer="quality", hardware=cpu_box)
    assert pick.model.name == "mistral-7b-q4"
    assert "llama-70b-q4" in pick.rejected


def test_prefer_speed_picks_the_fastest(cpu_box, models):
    assert recommend(models, prefer="speed", hardware=cpu_box).model.name == "tinyllama-1.1b-q4"


def test_prefer_quality_uses_size_when_quality_is_not_declared(cpu_box):
    plain = [{"name": "small", "size_gb": 1.0}, {"name": "big", "size_gb": 8.0}]
    assert recommend(plain, prefer="quality", hardware=cpu_box).model.name == "big"
    assert recommend(plain, prefer="speed", hardware=cpu_box).model.name == "small"


def test_alternatives_are_ranked_and_exclude_the_winner(cpu_box, models):
    pick = recommend(models, prefer="smallest", hardware=cpu_box)
    assert [spec.name for spec in pick.alternatives] == ["mistral-7b-q4"]
    assert pick.model.name not in [spec.name for spec in pick.alternatives]


def test_gpu_machine_runs_on_cuda(cuda_box, models):
    pick = recommend(models, hardware=cuda_box)
    assert pick.device == "cuda"
    assert pick.model.name == "mistral-7b-q4"
    assert "GPU (cuda)" in pick.reason and "24.0 GiB" in pick.reason


def test_prefer_quality_will_choose_a_cpu_model_over_a_smaller_gpu_one(cuda_box, models):
    """Asking for quality means quality; the reason says where it will run."""
    pick = recommend(models, prefer="quality", hardware=cuda_box)
    assert pick.model.name == "llama-70b-q4"  # 47 GiB of RAM fits, 47 of VRAM does not
    assert pick.device == "cpu"
    assert "run on the CPU" in pick.reason


def test_model_too_big_for_vram_falls_back_to_cpu(cuda_box):
    small_gpu = machine(
        ram_total_gb=64.0,
        ram_available_gb=60.0,
        gpus=[GPU(name="NVIDIA GeForce GTX 1050", vram_gb=4.0, backend="cuda")],
    )
    pick = recommend([{"name": "mistral-7b-q4", "size_gb": 4.1}], hardware=small_gpu)
    assert pick.fits is True
    assert pick.device == "cpu"
    assert "run on the CPU" in pick.reason


def test_apple_silicon_reports_mps(mps_box):
    pick = recommend([{"name": "phi-3-mini-q4", "size_gb": 2.3}], hardware=mps_box)
    assert pick.device == "mps"
    assert offline_ml.best_device(mps_box) == "mps"


def test_rocm_reports_cuda_as_the_device_string():
    amd = machine(gpus=[GPU(name="AMD Radeon RX 7900", vram_gb=20.0, backend="rocm")])
    assert offline_ml.best_device(amd) == "cuda"
    assert recommend([{"name": "m", "size_gb": 4.0}], hardware=amd).device == "cuda"


def test_task_filter(cpu_box):
    catalogue = [
        {"name": "whisper-small", "size_gb": 0.5, "task": "asr"},
        {"name": "phi-3-mini", "size_gb": 2.3, "task": "chat"},
        {"name": "anything", "size_gb": 1.0},
    ]
    pick = recommend(catalogue, task="chat", prefer="quality", hardware=cpu_box)
    assert pick.model.name == "phi-3-mini"
    assert "whisper-small" in pick.rejected
    assert "asr" in pick.rejected["whisper-small"]
    assert "anything" not in pick.rejected  # no declared task means any task


def test_headroom_changes_what_fits():
    box = machine(ram_total_gb=8.0, ram_available_gb=7.0)
    model = [{"name": "seven", "size_gb": 7.0}]
    assert recommend(model, headroom=0.0, hardware=box).fits is True
    assert recommend(model, headroom=0.5, hardware=box).fits is False


def test_declared_minimums_are_honoured(cpu_box):
    hungry = [{"name": "hungry", "size_gb": 1.0, "min_ram_gb": 64.0}]
    pick = recommend(hungry, hardware=cpu_box)
    assert pick.fits is False
    assert "64.0 GiB of RAM" in pick.rejected["hungry"]


def test_min_vram_keeps_a_model_off_a_small_gpu():
    box = machine(gpus=[GPU(name="NVIDIA T600", vram_gb=4.0, backend="cuda")])
    pick = recommend([{"name": "m", "size_gb": 2.0, "min_vram_gb": 10.0}], hardware=box)
    assert pick.fits is True and pick.device == "cpu"


def test_tight_ram_is_flagged_but_still_recommended():
    busy = machine(ram_total_gb=16.0, ram_available_gb=2.0)
    pick = recommend([{"name": "mid", "size_gb": 6.0}], hardware=busy)
    assert pick.fits is True
    assert "free right now" in pick.reason and "close something" in pick.reason


def test_disk_space_is_checked():
    full = machine(disk_free_gb=1.0)
    pick = recommend([{"name": "big-download", "size_gb": 20.0}], hardware=full)
    assert pick.fits is False
    assert "on disk" in pick.rejected["big-download"]


def test_modelspec_objects_and_dicts_are_interchangeable(cpu_box):
    as_dicts = [{"name": "a", "size_gb": 1.0}, {"name": "b", "size_gb": 2.0}]
    as_specs = [ModelSpec(**item) for item in as_dicts]
    assert (
        recommend(as_dicts, hardware=cpu_box).model.name
        == recommend(as_specs, hardware=cpu_box).model.name
    )


def test_summary_and_to_dict(cpu_box, models):
    pick = recommend(models, hardware=cpu_box)
    text = pick.summary()
    assert text.startswith("offline-ml: run '")
    assert "machine" in text and "rejected" in text
    assert not any(char in text for char in FANCY)
    assert str(pick) == text
    payload = pick.to_dict()
    round_tripped = json.loads(json.dumps(payload, ensure_ascii=False))
    assert round_tripped["model"]["name"] == pick.model.name
    assert round_tripped["fits"] is True
    assert round_tripped["hardware"]["ram_total_gb"] == cpu_box.ram_total_gb
    assert round_tripped["requirements"][pick.model.name]["ram_gb"] > 0


def test_requirements_cover_every_model(cpu_box, models):
    pick = recommend(models, hardware=cpu_box)
    assert set(pick.requirements) == {spec["name"] for spec in models}
    for values in pick.requirements.values():
        assert set(values) == {"disk_gb", "ram_gb", "vram_gb"}


def test_fits_accepts_specs_dicts_and_numbers(cpu_box):
    assert offline_ml.fits(1.0, hardware=cpu_box) is True
    assert offline_ml.fits(1000.0, hardware=cpu_box) is False
    assert offline_ml.fits({"name": "m", "size_gb": 2.0}, hardware=cpu_box) is True
    assert offline_ml.fits(ModelSpec(name="m", size_gb=2.0), hardware=cpu_box) is True


def test_fits_on_the_real_machine():
    assert offline_ml.fits(0.001) is True
    assert offline_ml.fits(1_000_000.0) is False


def test_result_is_deterministic(cpu_box, models):
    first = recommend(models, hardware=cpu_box).to_dict()
    second = recommend(list(reversed(models)), hardware=cpu_box).to_dict()
    assert first["model"] == second["model"]
    assert first["device"] == second["device"]


def test_single_model_says_so(cpu_box):
    pick = recommend([{"name": "only", "size_gb": 1.0}], hardware=cpu_box)
    assert "the only model that fits" in pick.reason
    assert pick.alternatives == []


# --------------------------------------------------------------------------- #
# more than one GPU: the decision follows the best card
# --------------------------------------------------------------------------- #
def test_a_big_model_goes_to_the_big_card_not_the_cpu():
    """A 70B on a box with a T400 and an idle A100 must not be sent to the CPU."""
    two_cards = machine(
        ram_total_gb=256.0,
        ram_available_gb=200.0,
        gpus=[
            GPU(name="NVIDIA T400 4GB", vram_gb=4.0, backend="cuda"),
            GPU(name="NVIDIA A100-SXM4-80GB", vram_gb=80.0, backend="cuda"),
        ],
    )
    pick = recommend([{"name": "llama-70b-q4", "size_gb": 39.0}], hardware=two_cards)
    assert pick.device == "cuda"
    assert "NVIDIA A100-SXM4-80GB" in pick.reason
    assert "NVIDIA T400 4GB" not in pick.reason
    assert "run on the GPU" in pick.reason


def test_a_model_too_big_for_every_card_still_names_the_best_one():
    two_cards = machine(
        ram_total_gb=256.0,
        ram_available_gb=200.0,
        gpus=[
            GPU(name="NVIDIA T400 4GB", vram_gb=4.0, backend="cuda"),
            GPU(name="NVIDIA A100-SXM4-80GB", vram_gb=80.0, backend="cuda"),
        ],
    )
    pick = recommend([{"name": "huge", "size_gb": 120.0}], hardware=two_cards)
    assert pick.device == "cpu"
    assert "NVIDIA A100-SXM4-80GB has only 80.0 GiB" in pick.reason


# --------------------------------------------------------------------------- #
# summary(limit=...) counts what it actually withheld
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("limit", [-1, 0])
def test_a_limit_below_one_lists_nothing_and_counts_correctly(cpu_box, limit):
    """`--limit -1` used to print one alternative and then 'and 3 more'."""
    four = [{"name": name, "size_gb": size} for name, size in
            (("a", 1.0), ("b", 2.0), ("c", 3.0), ("d", 4.0))]
    pick = recommend(four, hardware=cpu_box)
    text = pick.summary(limit=limit)
    assert len(pick.alternatives) == 3
    assert f"and {len(pick.alternatives)} more" in text
    for spec in pick.alternatives:
        assert f"    {spec}" not in text.splitlines()


def test_the_and_n_more_line_always_matches_what_was_withheld(cpu_box):
    six = [{"name": f"m{index}", "size_gb": 1.0 + index} for index in range(6)]
    pick = recommend(six, hardware=cpu_box)
    for limit in range(0, 8):
        text = pick.summary(limit=limit)
        listed = sum(1 for spec in pick.alternatives if f"    {spec}" in text.splitlines())
        withheld = len(pick.alternatives) - listed
        if withheld:
            assert f"and {withheld} more" in text
        else:
            assert "more" not in text.split("alternatives:")[-1].split("rejected:")[0]


# --------------------------------------------------------------------------- #
# quality and speed on a scale of the caller's choosing
# --------------------------------------------------------------------------- #
def test_a_centred_quality_scale_is_accepted_and_ranked(cpu_box):
    """The README promises any numeric scale; -5..+5 is a numeric scale."""
    centred = [
        {"name": "poor", "size_gb": 1.0, "quality": -5, "speed": 5},
        {"name": "fine", "size_gb": 2.0, "quality": 0, "speed": 0},
        {"name": "great", "size_gb": 3.0, "quality": 5, "speed": -5},
    ]
    assert recommend(centred, prefer="quality", hardware=cpu_box).model.name == "great"
    assert recommend(centred, prefer="speed", hardware=cpu_box).model.name == "poor"
    assert ModelSpec(name="m", size_gb=1.0, quality=-3).quality == -3.0
    assert ModelSpec(name="m", size_gb=1.0, speed=-3).speed == -3.0
