"""detect() on the real machine, and the GPU probes on fake ones."""
from __future__ import annotations

import json
import logging
import platform
import subprocess
import sys
import warnings

import pytest

import offline_ml
from offline_ml import _gpu
from offline_ml._gpu import BACKENDS, GPU, parse_nvidia_smi, probe_gpus

FANCY = "→←•·▸─│–—"


def test_detect_reports_this_machine():
    hardware = offline_ml.detect()
    assert isinstance(hardware, offline_ml.Hardware)
    assert hardware.cpu_count >= 1
    assert hardware.cpu_freq_mhz is None or hardware.cpu_freq_mhz > 0
    assert hardware.ram_total_gb > 0
    assert 0 <= hardware.ram_available_gb <= hardware.ram_total_gb + 1
    assert hardware.disk_free_gb is None or hardware.disk_free_gb >= 0
    expected = {"windows": "windows", "linux": "linux", "darwin": "macos"}.get(
        platform.system().lower(), platform.system().lower()
    )
    assert hardware.platform == expected
    assert hardware.python_version.startswith(str(sys.version_info[0]))
    assert isinstance(hardware.gpus, list)


def test_gpu_facts_are_consistent():
    hardware = offline_ml.detect()
    for gpu in hardware.gpus:
        assert isinstance(gpu, GPU)
        assert gpu.backend in BACKENDS
        assert gpu.vram_gb is None or gpu.vram_gb >= 0
    assert hardware.has_gpu == any(gpu.is_accelerator for gpu in hardware.gpus)
    assert hardware.gpu_backend in BACKENDS
    if not hardware.has_gpu:
        assert hardware.gpu_backend == "none"
        assert hardware.vram_gb == 0.0
        assert hardware.device == "cpu"


def test_no_gpu_is_silent_not_an_error(caplog):
    """The normal case must produce no exception, no warning, no log noise."""
    probe_gpus(refresh=True)  # warm the cache outside the assertions
    with caplog.at_level(logging.WARNING, logger="offline_ml"):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            hardware = offline_ml.detect(refresh=True)
            _ = hardware.summary()
    assert caught == []
    assert caplog.records == []


def test_summary_and_to_dict():
    hardware = offline_ml.detect()
    text = hardware.summary()
    assert text.startswith("offline-ml:")
    assert "ram" in text and "python" in text
    assert not any(char in text for char in FANCY), "summary must be plain ASCII punctuation"
    payload = hardware.to_dict()
    assert json.loads(json.dumps(payload, ensure_ascii=False))["cpu_count"] == hardware.cpu_count
    assert payload["gpus"] == [gpu.to_dict() for gpu in hardware.gpus]
    assert "GiB" in payload["units"]


def test_best_device():
    assert offline_ml.best_device() in ("cuda", "mps", "cpu")
    assert offline_ml.best_device() == offline_ml.detect().device


def test_detect_needs_no_network(monkeypatch):
    """Anything reaching for a socket would be a bug."""
    import socket

    def refuse(*args, **kwargs):
        raise AssertionError("offline-ml must not open a socket")

    monkeypatch.setattr(socket, "socket", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)
    assert offline_ml.detect(refresh=True).cpu_count >= 1


def test_torch_is_never_imported_at_import_time():
    assert "torch" not in sys.modules, "importing offline_ml must not import torch"


# --------------------------------------------------------------------------- #
# the GPU probes
# --------------------------------------------------------------------------- #
def test_parse_nvidia_smi_two_cards():
    gpus = parse_nvidia_smi("NVIDIA GeForce RTX 4090, 24564\nNVIDIA A100-SXM4, 81920\n")
    assert [gpu.name for gpu in gpus] == ["NVIDIA GeForce RTX 4090", "NVIDIA A100-SXM4"]
    assert gpus[0].vram_gb == pytest.approx(23.99, abs=0.05)
    assert all(gpu.backend == "cuda" and gpu.is_accelerator for gpu in gpus)


def test_parse_nvidia_smi_handles_junk():
    assert parse_nvidia_smi("") == []
    assert parse_nvidia_smi("\n  \n") == []
    assert parse_nvidia_smi("no commas here") == []
    unknown = parse_nvidia_smi("Some GPU, [N/A]")
    assert unknown[0].vram_gb is None
    assert unknown[0].usable_vram_gb == 0.0


def test_missing_nvidia_smi_is_not_an_error(monkeypatch):
    def missing(*args, **kwargs):
        raise FileNotFoundError("nvidia-smi")

    monkeypatch.setattr(subprocess, "run", missing)
    monkeypatch.setattr(_gpu, "_from_torch", lambda: [])
    monkeypatch.setattr(_gpu, "_from_platform", lambda: [])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert probe_gpus(refresh=True) == []
    assert caught == []


def test_failing_nvidia_smi_is_not_an_error(monkeypatch):
    class Failed:
        returncode = 9
        stdout = ""
        stderr = "NVIDIA-SMI has failed because it couldn't communicate with the driver"

    monkeypatch.setattr(subprocess, "run", lambda *a, **k: Failed())
    monkeypatch.setattr(_gpu, "_from_torch", lambda: [])
    monkeypatch.setattr(_gpu, "_from_platform", lambda: [])
    assert probe_gpus(refresh=True) == []


def test_timed_out_nvidia_smi_is_not_an_error(monkeypatch):
    def hang(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd="nvidia-smi", timeout=5)

    monkeypatch.setattr(subprocess, "run", hang)
    monkeypatch.setattr(_gpu, "_from_torch", lambda: [])
    monkeypatch.setattr(_gpu, "_from_platform", lambda: [])
    assert probe_gpus(refresh=True) == []


def test_nvidia_smi_present_is_used(monkeypatch):
    class Ok:
        returncode = 0
        stdout = "NVIDIA GeForce RTX 3060, 12288\n"
        stderr = ""

    monkeypatch.setattr(subprocess, "run", lambda *a, **k: Ok())
    gpus = probe_gpus(refresh=True)
    assert gpus[0].backend == "cuda"
    assert gpus[0].vram_gb == pytest.approx(12.0, abs=0.05)
    assert "12.0 GiB VRAM" in str(gpus[0])


def test_torch_probe_skipped_when_torch_absent(monkeypatch):
    monkeypatch.setitem(_gpu.os.environ, "OFFLINE_ML_NO_TORCH", "1")
    assert _gpu._from_torch() == []


def test_backend_guess_from_name():
    assert _gpu.backend_for_name("NVIDIA GeForce RTX 4090") == "cuda"
    assert _gpu.backend_for_name("AMD Radeon RX 7900", "linux") == "rocm"
    assert _gpu.backend_for_name("AMD Radeon RX 7900", "windows") == "none"
    assert _gpu.backend_for_name("Intel(R) UHD Graphics", "windows") == "none"
    assert _gpu.backend_for_name("") == "none"


def test_display_only_adapter_does_not_count_as_a_gpu(monkeypatch):
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: (_ for _ in ()).throw(OSError()))
    monkeypatch.setattr(_gpu, "_from_torch", lambda: [])
    monkeypatch.setattr(
        _gpu,
        "_from_platform",
        lambda: [GPU(name="Intel(R) UHD Graphics 770", vram_gb=0.12, backend="none")],
    )
    hardware = offline_ml.detect(refresh=True)
    assert hardware.gpus and hardware.has_gpu is False
    assert hardware.device == "cpu"
    assert "CPU" in hardware.summary()


def test_probe_result_is_cached(monkeypatch):
    calls = []

    def once(*args, **kwargs):
        calls.append(1)
        raise FileNotFoundError

    monkeypatch.setattr(subprocess, "run", once)
    monkeypatch.setattr(_gpu, "_from_torch", lambda: [])
    monkeypatch.setattr(_gpu, "_from_platform", lambda: [])
    probe_gpus(refresh=True)
    probe_gpus()
    probe_gpus()
    assert len(calls) == 1


def test_windows_linux_macos_paths_all_return_a_list(monkeypatch):
    for system, machine_name in (
        ("Windows", "AMD64"),
        ("Linux", "x86_64"),
        ("Darwin", "arm64"),
        ("Haiku", "x86_64"),
    ):
        monkeypatch.setattr(_gpu.platform, "system", lambda s=system: s)
        monkeypatch.setattr(_gpu.platform, "machine", lambda m=machine_name: m)
        found = _gpu._from_platform()
        assert isinstance(found, list)
        assert all(gpu.backend in BACKENDS for gpu in found)


def test_apple_silicon_is_reported_as_mps(monkeypatch):
    monkeypatch.setattr(_gpu.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(_gpu.platform, "machine", lambda: "arm64")
    gpus = _gpu._from_platform()
    assert gpus[0].backend == "mps"
    assert gpus[0].vram_gb is None or gpus[0].vram_gb > 0


# --------------------------------------------------------------------------- #
# more than one GPU: the best card decides, not the first one on the bus
# --------------------------------------------------------------------------- #
TWO_CARDS = "".join(
    ["NVIDIA T400 4GB, 4096\n", "NVIDIA A100-SXM4-80GB, 81920\n"]
)


def fake_nvidia_smi(stdout):
    """A subprocess.run stand-in feeding the package's own nvidia-smi parser."""

    class Ok:
        returncode = 0
        stderr = ""

    Ok.stdout = stdout
    return lambda *args, **kwargs: Ok()


def box_with(gpus, **kwargs):
    """A Hardware record with these cards and otherwise roomy numbers."""
    defaults = dict(
        cpu_count=8,
        cpu_freq_mhz=3000.0,
        ram_total_gb=128.0,
        ram_available_gb=120.0,
        platform="linux",
        python_version="3.10.11",
        disk_free_gb=500.0,
        disk_path="/home/tester",
    )
    defaults.update(kwargs)
    return offline_ml.Hardware(gpus=list(gpus), **defaults)


def test_best_gpu_is_the_largest_not_the_first(monkeypatch):
    """A small display card in a lower PCI slot must not shadow the big one."""
    monkeypatch.setattr(subprocess, "run", fake_nvidia_smi(TWO_CARDS))
    hardware = offline_ml.detect(refresh=True)
    assert [gpu.name for gpu in hardware.gpus] == [
        "NVIDIA T400 4GB",
        "NVIDIA A100-SXM4-80GB",
    ]
    assert hardware.gpu.name == "NVIDIA A100-SXM4-80GB"
    assert hardware.vram_gb == pytest.approx(80.0, abs=0.05)
    assert hardware.to_dict()["vram_gb"] == hardware.vram_gb
    assert hardware.device == "cuda"


def test_best_gpu_from_a_plain_fixture():
    """The same rule, without going through nvidia-smi at all."""
    box = box_with(
        [
            GPU(name="NVIDIA T400", vram_gb=4.0, backend="cuda"),
            GPU(name="NVIDIA A100", vram_gb=80.0, backend="cuda"),
        ]
    )
    assert box.gpu.name == "NVIDIA A100"
    assert box.vram_gb == 80.0
    assert box.short().endswith("NVIDIA A100 (80.0 GiB)")


def test_summary_header_and_gpu_lines_agree_on_the_chosen_card(monkeypatch):
    """The header line used to name one card while the list showed another."""
    monkeypatch.setattr(subprocess, "run", fake_nvidia_smi(TWO_CARDS))
    text = offline_ml.detect(refresh=True).summary()
    head = text.splitlines()[0]
    assert "NVIDIA A100-SXM4-80GB" in head and "NVIDIA T400" not in head
    chosen = [line for line in text.splitlines() if "[chosen]" in line]
    assert len(chosen) == 1 and "NVIDIA A100-SXM4-80GB" in chosen[0]


def test_a_display_only_card_never_shadows_a_real_one():
    """Backend 'none' means zero usable VRAM, whatever the driver reported."""
    box = box_with(
        [
            GPU(name="Intel(R) UHD Graphics 770", vram_gb=128.0, backend="none"),
            GPU(name="NVIDIA RTX 4090", vram_gb=24.0, backend="cuda"),
        ],
        platform="windows",
    )
    assert box.gpu.name == "NVIDIA RTX 4090"
    assert box.vram_gb == 24.0
    assert box.gpu_backend == "cuda"


def test_a_card_with_unreadable_vram_loses_to_one_that_reported_its_size():
    box = box_with(
        [
            GPU(name="NVIDIA unknown", vram_gb=None, backend="cuda"),
            GPU(name="NVIDIA RTX 4090", vram_gb=24.0, backend="cuda"),
        ]
    )
    assert box.gpu.name == "NVIDIA RTX 4090"
    assert box.vram_gb == 24.0


def test_equal_cards_keep_the_first_one_so_the_answer_is_stable():
    twins = [
        GPU(name="NVIDIA A6000 #0", vram_gb=48.0, backend="cuda"),
        GPU(name="NVIDIA A6000 #1", vram_gb=48.0, backend="cuda"),
    ]
    box = box_with(twins)
    assert box.gpu is twins[0]
    assert box.gpu.name == "NVIDIA A6000 #0"
