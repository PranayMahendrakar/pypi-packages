"""Probing: what is installed, what is broken, and that probing is inert."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

import offline_stt_router as stt
from offline_stt_router import Engine, Router
from offline_stt_router._engines import vosk_library_name

from conftest import TINY_DIMS, roomy_machine

HEAVY = ("faster_whisper", "whisper", "vosk", "torch", "ctranslate2")


def by_name(engines):
    return {e.name: e for e in engines}


def test_nothing_installed_is_the_normal_case(world):
    engines = world.router().available()
    names = [e.name for e in engines]
    assert names == ["faster-whisper", "openai-whisper", "whisper.cpp", "vosk"]
    for engine in engines:
        assert engine.installed is False
        assert engine.status == "not-installed"
        assert engine.error is None
        assert engine.models_on_disk == []
        assert engine.install_hint
        assert "not installed" in engine.describe()


def test_top_level_available_runs_on_the_real_environment():
    engines = stt.available()
    assert {"faster-whisper", "openai-whisper", "whisper.cpp", "vosk"} <= {e.name for e in engines}
    for engine in engines:
        assert engine.status in ("ready", "no-model", "failing", "not-installed")
        engine.to_dict()


def test_installed_engine_is_found_without_importing_it(world):
    world.faster_whisper()
    world.ct2_model("Systran/faster-whisper-small")
    before = {m for m in HEAVY if m in sys.modules}
    engine = by_name(world.router().available())["faster-whisper"]
    assert engine.installed and engine.ready
    assert engine.version == "1.1.0"
    assert engine.error is None
    assert [m.name for m in engine.models_on_disk] == ["small"]
    assert "en" in engine.languages and "hi" in engine.languages
    # the fake package raises on import, so reaching here proves it was never imported
    assert {m for m in HEAVY if m in sys.modules} == before


def test_missing_dependency_is_installed_but_failing(world):
    world.faster_whisper(deps=("av", "tokenizers", "huggingface_hub", "numpy"))  # no ctranslate2
    world.ct2_model("Systran/faster-whisper-small")
    engine = by_name(world.router().available())["faster-whisper"]
    assert engine.installed is True
    assert engine.status == "failing"
    assert "ctranslate2" in engine.error
    assert "pip install --force-reinstall faster-whisper" in engine.error


def test_complete_install_has_no_false_alarm(world):
    world.faster_whisper()
    world.openai_whisper()
    world.vosk()
    for engine in world.router().available():
        assert engine.error is None, engine.error
        assert engine.status != "failing"


def test_metadata_without_module_is_failing(world):
    world.metadata("faster-whisper", "1.0.3")
    engine = by_name(world.router().available())["faster-whisper"]
    assert engine.status == "failing"
    assert "module is missing" in engine.error


def test_leftover_empty_folder_is_not_an_install(world):
    (world.site / "vosk").mkdir()
    engine = by_name(world.router().available())["vosk"]
    assert engine.installed is False
    assert any("empty vosk folder" in n for n in engine.notes)


def test_vosk_without_native_library_is_failing(world):
    world.vosk(with_library=False)
    world.vosk_model(world.home / ".cache" / "vosk", "vosk-model-small-en-us-0.15")
    engine = by_name(world.router().available())["vosk"]
    assert engine.status == "failing"
    assert vosk_library_name() in engine.error


def test_vosk_with_library_and_model_is_ready(world):
    world.vosk()
    world.vosk_model(world.home / ".cache" / "vosk", "vosk-model-small-en-us-0.15")
    engine = by_name(world.router().available())["vosk"]
    assert engine.ready and engine.version == "0.3.45"
    assert engine.languages == ["en"]


def test_vosk_library_name_per_platform():
    assert vosk_library_name("win32") == "libvosk.dll"
    assert vosk_library_name("linux") == "libvosk.so"
    assert vosk_library_name("darwin") == "libvosk.dyld"


def test_installed_without_models_is_no_model(world):
    world.vosk()
    engine = by_name(world.router().available())["vosk"]
    assert engine.installed and engine.status == "no-model"
    assert "no model on disk" in engine.describe()


def test_a_different_package_named_whisper_is_not_openai_whisper(world):
    (world.site / "whisper.py").write_text("raise RuntimeError('graphite whisper')\n", encoding="utf-8")
    engine = by_name(world.router().available())["openai-whisper"]
    assert engine.installed is False
    assert any("not openai-whisper" in n for n in engine.notes)


def test_openai_whisper_reads_pt_checkpoints(world):
    world.openai_whisper()
    world.pt_model(world.home / ".cache" / "whisper", "tiny", TINY_DIMS)
    engine = by_name(world.router().available())["openai-whisper"]
    assert engine.ready
    assert engine.models_on_disk[0].family == "tiny"
    assert len(engine.languages) == 99


def test_models_are_listed_even_when_the_engine_is_missing(world):
    world.ct2_model("Systran/faster-whisper-medium")
    engine = by_name(world.router().available())["faster-whisper"]
    assert engine.installed is False
    assert [m.name for m in engine.models_on_disk] == ["medium"]


def test_whisper_cpp_binary_on_path(world):
    world.script("whisper-cli", "print('never run by the probe')")
    world.ggml_model(world.home / ".cache" / "whisper.cpp", "base.en", n_vocab=51864)
    engine = by_name(world.router().available())["whisper.cpp"]
    assert engine.installed and engine.ready
    assert engine.location and "whisper-cli" in os.path.basename(engine.location)
    assert engine.models_on_disk[0].languages == ["en"]


def test_whisper_cpp_bin_env_pointing_nowhere_is_failing(world, monkeypatch):
    monkeypatch.setenv("WHISPER_CPP_BIN", str(world.bin / "nothing-here"))
    engine = by_name(world.router().available())["whisper.cpp"]
    assert engine.status == "failing"
    assert "does not exist" in engine.error


def test_whisper_cpp_empty_binary_is_failing(world, monkeypatch):
    empty = world.bin / ("whisper-cli.exe" if os.name == "nt" else "whisper-cli")
    empty.write_bytes(b"")
    if os.name != "nt":
        empty.chmod(0o755)
    monkeypatch.setenv("WHISPER_CPP_BIN", str(empty))
    engine = by_name(world.router().available())["whisper.cpp"]
    assert engine.status == "failing"
    assert "empty" in engine.error


# --------------------------------------------------------------------------- custom


def test_custom_probe_that_raises_is_reported_not_raised(world):
    router = world.router()

    def broken():
        raise OSError("driver exploded")

    router.register("my-asr", broken)
    engine = by_name(router.available())["my-asr"]
    assert engine.installed is True
    assert engine.status == "failing"
    assert "driver exploded" in engine.error


def test_custom_probe_missing_import_means_not_installed(world):
    router = world.router()

    def probe():
        import definitely_not_a_real_module_xyz  # noqa: F401

    router.register("my-asr", probe)
    engine = by_name(router.available())["my-asr"]
    assert engine.status == "not-installed"
    assert any("definitely_not_a_real_module_xyz" in n for n in engine.notes)


def test_custom_probe_shapes(world):
    router = world.router()
    router.register("yes", lambda: True)
    router.register("no", lambda: False)
    router.register("odd", lambda: 42)
    router.register("rich", lambda: {"version": "2.0", "languages": ["en", "Hindi", "klingon language"],
                                     "quality": 3.0, "surprise": 1})
    router.register("object", lambda: Engine(name="ignored", installed=True, version="9"))
    engines = by_name(router.available())
    assert engines["yes"].ready and engines["yes"].languages == ["*"]
    assert engines["no"].status == "not-installed"
    assert engines["odd"].status == "failing" and "int" in engines["odd"].error
    rich = engines["rich"]
    assert rich.ready and rich.version == "2.0"
    assert rich.languages == ["en", "hi"]
    assert rich.quality == 1.0
    assert any("klingon" in n for n in rich.notes)
    assert any("surprise" in n for n in rich.notes)
    assert engines["object"].name == "object" and engines["object"].ready


def test_register_validation_and_unregister(world):
    router = world.router()
    with pytest.raises(ValueError):
        router.register("  ", lambda: True)
    with pytest.raises(TypeError):
        router.register("x", "not callable")
    with pytest.raises(TypeError):
        router.register("x", lambda: True, transcribe=5)
    router.register("x", lambda: True)
    assert router.registered == ["x"]
    assert router.unregister("x") is True
    assert router.unregister("x") is False


def test_registering_a_builtin_name_replaces_it(world):
    router = world.router()
    router.register("vosk", lambda: {"version": "mine"})
    engines = [e for e in router.available() if e.name == "vosk"]
    assert len(engines) == 1 and engines[0].kind == "custom" and engines[0].version == "mine"


def test_top_level_register_and_unregister():
    stt.register("unit-test-engine", lambda: False)
    try:
        assert any(e.name == "unit-test-engine" for e in stt.available())
    finally:
        assert stt.unregister("unit-test-engine") is True
    assert not any(e.name == "unit-test-engine" for e in stt.available())


def test_engine_lookup_by_name(world):
    router = world.router()
    assert router.engine("VOSK").name == "vosk"
    with pytest.raises(KeyError):
        router.engine("nope")


# --------------------------------------------------------------------------- inert


def _tree(path: Path):
    return sorted((str(p.relative_to(path)), p.stat().st_size) for p in path.rglob("*") if p.is_file())


def test_probing_and_choosing_with_the_network_blocked(world, no_network):
    world.faster_whisper()
    world.openai_whisper()
    world.vosk()
    world.ct2_model("Systran/faster-whisper-large-v3", languages=100)
    world.ct2_model("Systran/faster-whisper-large-v2", complete=False)
    world.pt_model(world.home / ".cache" / "whisper", "tiny", TINY_DIMS)
    world.vosk_model(world.home / ".cache" / "vosk", "vosk-model-small-hi-0.22")
    world.script("whisper-cli", "raise SystemExit('the probe must never run this')")
    world.ggml_model(world.home / ".cache" / "whisper.cpp", "base")
    before = _tree(world.home)
    heavy_before = {m for m in HEAVY if m in sys.modules}

    router = world.router()
    engines = router.available()
    choice = router.choose(language="hi", prefer="accurate")
    stt.machine()

    assert no_network == []  # nothing even tried to connect
    assert _tree(world.home) == before  # nothing was downloaded or written
    assert {m for m in HEAVY if m in sys.modules} == heavy_before
    assert all(e.status == "ready" for e in engines)
    assert choice.ok and choice.model == "large-v3"
    assert any("download not finished" in n for n in by_name(engines)["faster-whisper"].notes)


def test_real_machine_detection_never_raises():
    found = stt.machine()
    assert found.cpu_physical >= 1 and found.cpu_logical >= found.cpu_physical
    assert isinstance(found.describe(), str)
    found.to_dict()


def test_machine_what_if_defaults():
    what_if = stt.Machine(ram_total_gb=4, cpu_physical=2)
    assert what_if.ram_available_gb == 4 and what_if.cpu_logical == 2
    assert what_if.threads == 2
    assert "4.0 GB RAM" in what_if.describe()
    assert roomy_machine().summary().startswith("Linux")


def test_cpu_only_pytorch_is_noticed(world):
    world.openai_whisper(deps=("numpy", "tiktoken", "numba"))
    torch = world.package("torch", "torch", "2.4.0", files={"lib/torch_cpu.dll": "<binary>"})
    engine = by_name(world.router().available())["openai-whisper"]
    assert engine.can_use_gpu is False
    if sys.platform == "win32" or sys.platform.startswith("linux"):
        assert any("CPU-only build" in n for n in engine.notes)
    (torch / "lib" / "torch_cuda.dll").write_bytes(b"x")
    engine = by_name(world.router().available())["openai-whisper"]
    assert engine.can_use_gpu is (sys.platform == "win32" or sys.platform.startswith("linux"))
    assert not any("CPU-only build" in n for n in engine.notes)


def test_apple_silicon_detection():
    from offline_stt_router._machine import is_apple_silicon

    assert is_apple_silicon("darwin", "arm64") is True
    assert is_apple_silicon("darwin", "x86_64") is False
    assert is_apple_silicon("linux", "aarch64") is False
    assert is_apple_silicon("win32", "AMD64") is False


def test_gpu_detection_can_be_switched_off(monkeypatch):
    from offline_stt_router._machine import detect_gpus

    monkeypatch.setenv("OFFLINE_STT_ROUTER_NO_GPU", "1")
    notes = []
    assert detect_gpus(notes=notes) == []
    assert "switched off" in notes[0]


def test_an_engine_installed_after_a_probe_is_seen_on_the_next(world):
    router = world.router()
    assert by_name(router.available())["vosk"].status == "not-installed"
    world.vosk()  # within the same clock tick as the probe above, on most filesystems
    assert by_name(router.available())["vosk"].status == "no-model"
