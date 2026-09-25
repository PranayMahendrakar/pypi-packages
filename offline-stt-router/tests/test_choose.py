"""choose(): the rule for each preference, memory limits, GPUs and languages."""

from __future__ import annotations

import json

import pytest

import offline_stt_router as stt
from offline_stt_router import GPU, Engine, LocalModel, Machine
from offline_stt_router._models import FAMILIES, whisper_languages_for
from offline_stt_router._plan import describe_speed, plan

from conftest import roomy_machine


def ascii_only(text: str) -> bool:
    return all(ord(ch) < 128 for ch in text)


def fw_world(world, *names):
    world.faster_whisper()
    for name in names:
        world.ct2_model("Systran/faster-whisper-" + name, languages=100 if "v3" in name else 99)
    return world


def cuda_machine(free_gb: float, **kw) -> Machine:
    return roomy_machine(gpus=[GPU(name="Test GPU", vendor="nvidia", backend="cuda",
                                   vram_total_gb=max(free_gb, 1.0) + 1.0, vram_free_gb=free_gb)], **kw)


# --------------------------------------------------------------------------- nothing installed


def test_nothing_installed_returns_a_clear_choice(world):
    choice = world.router().choose(language="en")
    assert choice.ok is False and not choice
    assert choice.engine is None and choice.model is None and choice.device is None
    assert "None of the engines" in choice.reason and "installed" in choice.reason
    assert choice.missing and choice.missing[0].startswith("pip install faster-whisper")
    assert "never downloads" in choice.summary()
    assert ascii_only(choice.summary())
    data = json.loads(json.dumps(choice.to_dict()))
    assert data["ok"] is False and data["missing"] == choice.missing
    assert [e["status"] for e in data["engines"]] == ["not-installed"] * 4


def test_installed_but_no_model_says_what_to_fetch(world):
    world.faster_whisper()
    choice = world.router().choose(language="en")
    assert not choice.ok
    assert "has no model on disk" in choice.reason
    assert any(m.startswith("for faster-whisper, fetch the") for m in choice.missing)


def test_models_on_disk_without_the_engine(world):
    world.ct2_model("Systran/faster-whisper-small")
    choice = world.router().choose(language="en")
    assert not choice.ok
    assert any("already on disk" in m and "pip install faster-whisper" in m for m in choice.missing)


def test_failing_engine_gets_a_repair_line(world):
    world.faster_whisper(deps=("av",))
    world.ct2_model("Systran/faster-whisper-small")
    choice = world.router().choose()
    assert not choice.ok
    assert "failing" in choice.reason
    assert any(m.startswith("repair faster-whisper") for m in choice.missing)


# --------------------------------------------------------------------------- memory


def test_model_bigger_than_ram_is_rejected_with_the_reason(world):
    fw_world(world, "small", "large-v3")
    small_box = roomy_machine(ram_total_gb=4.0, ram_available_gb=3.0)
    choice = world.router(machine=small_box).choose(prefer="accurate")
    assert choice.ok and choice.model == "small"
    [rejected] = [r for r in choice.rejected if r.model == "large-v3"]
    assert "needs about 3.6 GB of RAM" in rejected.reason
    assert "only 3.0 GB is available right now (of 4.0 GB)" in rejected.reason
    assert "large-v3" in choice.summary() and "Set aside:" in choice.summary()


def test_no_false_rejection_when_ram_is_plenty(world):
    fw_world(world, "small", "large-v3")
    choice = world.router(machine=roomy_machine(ram_total_gb=64.0, ram_available_gb=48.0)).choose(prefer="accurate")
    assert choice.model == "large-v3"
    assert choice.rejected == []


def test_max_ram_gb_caps_but_never_raises_the_limit(world):
    fw_world(world, "small", "large-v3")
    router = world.router(machine=roomy_machine(ram_available_gb=12.0))
    capped = router.choose(prefer="accurate", max_ram_gb=1.0)
    assert capped.model == "small"
    assert "you allowed 1.0 GB (max_ram_gb)" in capped.rejected[0].reason
    assert capped.ram_budget_gb == 1.0
    generous = router.choose(prefer="accurate", max_ram_gb=500)
    assert generous.ram_budget_gb == 12.0 and generous.model == "large-v3"
    for bad in (0, -2):
        with pytest.raises(ValueError):
            router.choose(max_ram_gb=bad)
    for bad in ("4", True):
        with pytest.raises(TypeError):
            router.choose(max_ram_gb=bad)


def test_everything_too_big_explains_and_suggests_smaller(world):
    fw_world(world, "large-v3")
    choice = world.router(machine=roomy_machine(ram_total_gb=2.0, ram_available_gb=1.0)).choose()
    assert not choice.ok
    assert "needs about 3.6 GB of RAM" in choice.reason
    assert any("fetch the" in m for m in choice.missing)


def test_unknown_ram_is_noted_not_fatal(world):
    fw_world(world, "small")
    choice = world.router(machine=Machine(ram_total_gb=None, cpu_physical=4)).choose()
    assert choice.ok and choice.ram_budget_gb is None
    assert any("could not be read" in n for n in choice.notes)


# --------------------------------------------------------------------------- the three rules


def test_each_preference_follows_its_rule(world):
    fw_world(world, "tiny", "small", "medium", "large-v3")
    router = world.router(machine=roomy_machine())
    fast, balanced, accurate = (router.choose(prefer=p) for p in ("fast", "balanced", "accurate"))
    assert fast.model == "tiny"
    assert balanced.model == "medium" and balanced.realtime_factor <= 0.5
    assert "at least 2x faster than real time" in balanced.reason
    assert accurate.model == "large-v3"
    assert "accuracy was put ahead of speed" in accurate.reason
    assert accurate.realtime_factor > balanced.realtime_factor > fast.realtime_factor
    assert [a.model for a in balanced.alternatives][:1] == ["small"]
    assert router.choose(prefer="BALANCED").model == "medium"
    with pytest.raises(ValueError, match="prefer must be one of"):
        router.choose(prefer="cheap")


def test_balanced_on_a_slow_machine_takes_the_fastest(world):
    fw_world(world, "medium", "large-v3")
    choice = world.router(machine=roomy_machine(cpu_physical=1, cpu_logical=1)).choose()
    assert choice.model == "medium"
    assert "Nothing on disk is expected to reach 2x real time" in choice.reason


def test_more_cores_means_faster_estimates(world):
    fw_world(world, "small")
    slow = world.router(machine=roomy_machine(cpu_physical=2)).choose().realtime_factor
    quick = world.router(machine=roomy_machine(cpu_physical=16)).choose().realtime_factor
    assert quick < slow


def test_better_model_suggested_when_only_tiny_is_on_disk(world):
    fw_world(world, "tiny")
    choice = world.router().choose(prefer="balanced")
    assert choice.model == "tiny"
    assert any("fetch the" in m and "more accurate than tiny" in m for m in choice.missing)


def test_choice_is_deterministic(world):
    fw_world(world, "tiny", "small")
    world.vosk()
    world.vosk_model(world.home / ".cache" / "vosk", "vosk-model-small-en-us-0.15")
    router = world.router()
    first, second = router.choose().to_dict(), router.choose().to_dict()
    for data in (first, second):
        data.pop("machine")
    assert first == second


# --------------------------------------------------------------------------- GPUs


def test_gpu_with_room_runs_the_big_model(world):
    fw_world(world, "small", "large-v3")
    choice = world.router(machine=cuda_machine(8.0)).choose()
    assert (choice.model, choice.device, choice.compute_type) == ("large-v3", "cuda", "float16")
    assert choice.estimated_vram_gb == pytest.approx(4.5)
    assert "Test GPU" in choice.reason


def test_small_gpu_uses_int8_then_falls_back_to_cpu(world):
    fw_world(world, "large-v3")
    tight = world.router(machine=cuda_machine(3.2)).choose(prefer="accurate")
    assert (tight.device, tight.compute_type) == ("cuda", "int8_float16")
    tiny_gpu = world.router(machine=cuda_machine(1.0)).choose(prefer="accurate")
    assert tiny_gpu.device == "cpu"
    assert any("does not fit in the GPU" in n and "runs on the CPU" in n for n in tiny_gpu.notes)


def test_apple_silicon_uses_metal_for_whisper_cpp():
    model = LocalModel(name="small", engine="whisper.cpp", family="small", path="/m/ggml-small.bin",
                       size_gb=0.46, languages=whisper_languages_for("small", False))
    engine = Engine(name="whisper.cpp", installed=True, models_on_disk=[model], can_use_gpu=True,
                    location="/opt/homebrew/bin/whisper-cli").finish()
    mac = Machine(os="Darwin", arch="arm64", cpu_physical=8, ram_total_gb=16, ram_available_gb=10,
                  gpus=[GPU(name="Apple Silicon GPU", vendor="apple", backend="metal",
                            vram_total_gb=16, vram_free_gb=10, shared_memory=True)])
    choice = plan([engine], mac, language="de")
    assert choice.device == "metal" and "Mac" in choice.reason
    linux = plan([engine], Machine(os="Linux", arch="x86_64", cpu_physical=8, ram_total_gb=16), language="de")
    windows = plan([engine], Machine(os="Windows", arch="AMD64", cpu_physical=8, ram_total_gb=16), language="de")
    assert linux.device == "cpu" and windows.device == "cpu"
    assert "shares system RAM" in mac.describe()


def test_engine_that_needs_a_gpu_is_rejected_without_one(world):
    router = world.router()
    router.register("gpu-only", lambda: {"needs_gpu": True, "quality": 0.9}, lambda audio: "x")
    choice = router.choose()
    assert not choice.ok
    assert "needs a GPU and none was found" in choice.reason
    with_gpu = world.router(machine=cuda_machine(8.0))
    with_gpu.register("gpu-only", lambda: {"needs_gpu": True, "quality": 0.9}, lambda audio: "x")
    assert with_gpu.choose().device == "gpu"


# --------------------------------------------------------------------------- languages


def test_english_only_model_is_never_offered_for_hindi(world):
    fw_world(world, "small.en", "base")
    choice = world.router().choose(language="hi")
    assert choice.model == "base"
    [rejected] = choice.rejected
    assert rejected.model == "small.en" and rejected.reason == "English only, not Hindi"
    assert "smaller than Hindi needs (small or larger)" in choice.reason
    assert "Whisper saw far less Hindi than English" in choice.reason


def test_english_prefers_the_english_only_model(world):
    fw_world(world, "base", "base.en")
    choice = world.router().choose(language="English")
    assert choice.model == "base.en"
    assert "English-only models are more accurate" in choice.reason
    assert choice.alternatives[0].model == "base"


def test_auto_detect_needs_a_multilingual_model(world):
    fw_world(world, "tiny.en")
    world.vosk()
    world.vosk_model(world.home / ".cache" / "vosk", "vosk-model-small-fr-0.22")
    choice = world.router().choose(language=None)
    assert not choice.ok
    reasons = " ".join(r.reason for r in choice.rejected)
    assert "cannot detect the language" in reasons
    assert "Vosk models cannot detect the language" in reasons
    assert choice.language_name == "any language (auto-detect)"
    assert world.router().choose(language="auto").language is None


def test_vosk_is_matched_by_the_language_in_its_name(world):
    world.vosk()
    root = world.home / ".cache" / "vosk"
    world.vosk_model(root, "vosk-model-small-hi-0.22")
    router = world.router()
    hindi = router.choose(language="hi-IN")
    assert hindi.engine == "vosk" and hindi.model == "vosk-model-small-hi-0.22"
    english = router.choose(language="en")
    assert not english.ok and "Hindi only, not English" in english.reason


def test_language_whisper_does_not_know_goes_to_an_engine_that_does(world):
    fw_world(world, "small")
    router = world.router()
    assert not router.choose(language="eo").ok  # Esperanto: not a Whisper language
    world.vosk()
    world.vosk_model(world.home / ".cache" / "vosk", "vosk-model-small-eo-0.42")
    assert world.router().choose(language="Esperanto").engine == "vosk"
    router.register("tlh-asr", lambda: {"languages": ["tlh"]}, lambda audio: "Qapla'")
    assert router.choose(language="tlh").engine == "tlh-asr"


def test_native_language_names_are_understood(world):
    fw_world(world, "small")
    choice = world.router().choose(language="हिन्दी")
    assert choice.language == "hi" and choice.language_name == "Hindi"
    with pytest.raises(ValueError, match="unknown language"):
        world.router().choose(language="not a language at all")


def test_custom_engines_compete_on_their_declared_figures(world):
    fw_world(world, "small")
    router = world.router()
    router.register("great", lambda: {"quality": 0.99, "realtime_factor": 0.05, "ram_gb": 0.5}, lambda audio: "x")
    router.register("slow", lambda: {"quality": 0.99, "realtime_factor": 5.0, "ram_gb": 0.5}, lambda audio: "x")
    assert router.choose(prefer="accurate").engine == "great"
    assert router.choose(prefer="balanced").engine == "great"
    models = {"models_on_disk": [{"name": "m1", "quality": 0.3}, {"name": "m2", "quality": 0.8}, "m3"]}
    router.register("zoo", lambda: models, lambda audio, model=None: model)
    zoo = [e for e in router.available() if e.name == "zoo"][0]
    assert [m.name for m in zoo.models_on_disk] == ["m1", "m2", "m3"]


def test_summary_and_dict_are_complete_for_a_real_choice(world):
    fw_world(world, "tiny", "small", "large-v3")
    choice = world.router(machine=roomy_machine(ram_available_gb=3.0)).choose()
    text = choice.summary()
    for part in ("Use faster-whisper with the", "Why:", "Estimated:", "Alternatives:",
                 "Set aside:", "Checked:", "Machine:"):
        assert part in text
    assert ascii_only(text)
    data = json.loads(json.dumps(choice.to_dict(), ensure_ascii=False))
    assert data["ok"] and data["engine"] == "faster-whisper" and data["model_path"]
    assert data["alternatives"] and data["rejected"]
    assert choice.speed.startswith("about ")


def test_describe_speed_wording():
    assert describe_speed(0.05) == "about 20x faster than real time"
    assert describe_speed(0.4) == "about 2.5x faster than real time"
    assert describe_speed(2.0) == "about 2.0x slower than real time"
    assert describe_speed(0) == "speed unknown"


def test_every_family_has_complete_figures():
    from offline_stt_router._models import RAM_CPU, VRAM_GPU

    for family in FAMILIES.values():
        for table in list(RAM_CPU.values()) + list(VRAM_GPU.values()):
            assert family.size_class in table


def test_top_level_choose_on_the_real_environment():
    choice = stt.choose(language="en")
    assert isinstance(choice.reason, str) and choice.reason
    json.dumps(choice.to_dict())
