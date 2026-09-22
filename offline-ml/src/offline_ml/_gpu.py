"""Best-effort GPU discovery that never requires a GPU library.

Having no GPU is the normal case, so every probe in here is written so that a
missing tool, a failing driver, a permission error or an unusual platform reads
as "no GPU found" rather than as an error. Nothing is printed and nothing is
warned about; the details go to ``logging.getLogger("offline_ml._gpu")`` at
DEBUG level for anyone who wants to look.

Probe order:

1. ``nvidia-smi`` - present whenever an NVIDIA driver is installed, and cheap.
2. ``torch`` - only when it already happens to be importable. It is never
   imported at package import time, and never installed by this package.
3. Platform APIs - the Windows registry, Linux sysfs/procfs, the macOS machine
   type. No downloads, no network, no shell.

Set ``OFFLINE_ML_NO_TORCH=1`` to skip step 2 (useful in test suites, where
importing torch would cost seconds).
"""
from __future__ import annotations

import logging
import os
import platform
import re
import subprocess
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

log = logging.getLogger(__name__)

BYTES_PER_GB = 1024 ** 3
"""One binary gigabyte (GiB). Every ``*_gb`` number in this package uses it."""

#: Backends that can actually run a model. Anything else is display-only.
ACCELERATED_BACKENDS = ("cuda", "rocm", "mps")
#: Every value ``GPU.backend`` can take.
BACKENDS = ACCELERATED_BACKENDS + ("none",)

_PROBE_TIMEOUT = 5.0
_NVIDIA_WORDS = ("nvidia", "geforce", "quadro", "tesla", "rtx", "gtx", "titan")
_AMD_WORDS = ("radeon", "firepro", "instinct", "amd")
_WINDOWS_DISPLAY_CLASS = (
    r"SYSTEM\CurrentControlSet\Control\Class"
    r"\{4d36e968-e325-11ce-bfc1-08002be10318}"
)


@dataclass(frozen=True)
class GPU:
    """One graphics device that was found.

    Attributes:
        name: What the driver calls it, e.g. ``"NVIDIA GeForce RTX 4090"``.
        vram_gb: Dedicated video memory in GiB, or ``None`` when the device was
            found but its memory could not be read. On Apple Silicon this is the
            unified system memory, which the GPU really can use.
        backend: ``"cuda"``, ``"rocm"``, ``"mps"``, or ``"none"`` for a display
            adapter with no usable compute backend.
    """

    name: str
    vram_gb: Optional[float] = None
    backend: str = "none"

    @property
    def is_accelerator(self) -> bool:
        """True when this device can actually run a model."""
        return self.backend in ACCELERATED_BACKENDS

    @property
    def usable_vram_gb(self) -> float:
        """VRAM in GiB, counting "unknown" and "display only" as zero."""
        if not self.is_accelerator or self.vram_gb is None:
            return 0.0
        return float(self.vram_gb)

    def to_dict(self) -> Dict[str, Any]:
        """JSON-safe dict of this device."""
        return {"name": self.name, "vram_gb": self.vram_gb, "backend": self.backend}

    def __str__(self) -> str:
        vram = "VRAM unknown" if self.vram_gb is None else f"{self.vram_gb:.1f} GiB VRAM"
        return f"{self.name}, {vram} ({self.backend})"


def backend_for_name(name: str, system: Optional[str] = None) -> str:
    """Guess the compute backend from a device name. Never raises."""
    lowered = (name or "").lower()
    system = (system or platform.system()).lower()
    if any(word in lowered for word in _NVIDIA_WORDS):
        return "cuda"
    if any(word in lowered for word in _AMD_WORDS):
        # ROCm is a Linux story; the same card on Windows has no usable backend.
        return "rocm" if system == "linux" else "none"
    return "none"


# --------------------------------------------------------------------------- #
# probe 1: nvidia-smi
# --------------------------------------------------------------------------- #
def _run(argv: List[str]) -> Optional[str]:
    """Run a probe command. Returns its stdout, or None for any failure."""
    kwargs: Dict[str, Any] = {}
    if os.name == "nt":  # keep a console window from flashing up
        kwargs["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)
    try:
        proc = subprocess.run(
            argv,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=_PROBE_TIMEOUT,
            check=False,
            encoding="utf-8",
            errors="replace",
            **kwargs,
        )
    except Exception as exc:  # missing binary, timeout, permissions, anything
        log.debug("probe %s unavailable: %s", argv[0], exc)
        return None
    if proc.returncode != 0:
        log.debug("probe %s exited %s", argv[0], proc.returncode)
        return None
    return proc.stdout


def parse_nvidia_smi(text: str) -> List[GPU]:
    """Parse ``nvidia-smi --query-gpu=name,memory.total`` CSV output."""
    gpus: List[GPU] = []
    for line in (text or "").splitlines():
        line = line.strip()
        if not line:
            continue
        name, sep, raw_mem = line.rpartition(",")
        name = name.strip()
        if not sep or not name:  # a line with no comma is not a row we understand
            continue
        vram: Optional[float] = None
        match = re.search(r"\d+(?:\.\d+)?", raw_mem)
        if match:  # nvidia-smi reports MiB with --format=...,nounits
            vram = round(float(match.group()) / 1024.0, 2)
        gpus.append(GPU(name=name, vram_gb=vram, backend="cuda"))
    return gpus


def _from_nvidia_smi() -> List[GPU]:
    out = _run(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total",
            "--format=csv,noheader,nounits",
        ]
    )
    if not out:
        return []
    try:
        return parse_nvidia_smi(out)
    except Exception as exc:  # pragma: no cover - defensive
        log.debug("could not parse nvidia-smi output: %s", exc)
        return []


# --------------------------------------------------------------------------- #
# probe 2: torch, only if it is already installed
# --------------------------------------------------------------------------- #
def _torch_installed() -> bool:
    if os.environ.get("OFFLINE_ML_NO_TORCH"):
        return False
    if "torch" in sys.modules:
        return True
    try:
        import importlib.util

        return importlib.util.find_spec("torch") is not None
    except Exception as exc:  # pragma: no cover - broken import machinery
        log.debug("could not look for torch: %s", exc)
        return False


def _from_torch() -> List[GPU]:
    """Ask torch, but only when torch is already part of the environment."""
    if not _torch_installed():
        return []
    try:
        import torch  # deliberately lazy: never imported at package import time

        gpus: List[GPU] = []
        if torch.cuda.is_available():
            for index in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(index)
                total = float(getattr(props, "total_memory", 0) or 0)
                gpus.append(
                    GPU(
                        name=str(getattr(props, "name", "GPU %d" % index)),
                        vram_gb=round(total / BYTES_PER_GB, 2) if total else None,
                        backend="cuda",
                    )
                )
            if gpus:
                return gpus
        mps = getattr(getattr(torch, "backends", None), "mps", None)
        if mps is not None and mps.is_available():
            return [apple_gpu()]
    except Exception as exc:  # torch present but unhappy is still "no GPU"
        log.debug("torch probe did not find a GPU: %s", exc)
    return []


# --------------------------------------------------------------------------- #
# probe 3: platform APIs
# --------------------------------------------------------------------------- #
def _unified_memory_gb() -> Optional[float]:
    try:
        import psutil

        return round(psutil.virtual_memory().total / BYTES_PER_GB, 2)
    except Exception:  # pragma: no cover - psutil is a hard dependency
        return None


def apple_gpu() -> GPU:
    """The Apple Silicon GPU, whose VRAM is the machine's unified memory."""
    return GPU(
        name="Apple Silicon GPU (Metal)",
        vram_gb=_unified_memory_gb(),
        backend="mps",
    )


def _reg_value(key: Any, name: str) -> Optional[int]:
    import winreg

    try:
        value, _ = winreg.QueryValueEx(key, name)
    except OSError:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, (bytes, bytearray)):
        return int.from_bytes(bytes(value), "little")
    return None


def _from_windows_registry() -> List[GPU]:
    """Read display adapters out of the registry. No shell, no WMI, no wmic."""
    try:
        import winreg
    except ImportError:  # not Windows
        return []
    gpus: List[GPU] = []
    try:
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, _WINDOWS_DISPLAY_CLASS) as root:
            count = winreg.QueryInfoKey(root)[0]
            for index in range(count):
                sub = winreg.EnumKey(root, index)
                if not re.fullmatch(r"\d{4}", sub):
                    continue
                with winreg.OpenKey(root, sub) as key:
                    try:
                        desc, _ = winreg.QueryValueEx(key, "DriverDesc")
                    except OSError:
                        continue
                    if not isinstance(desc, str) or not desc.strip():
                        continue
                    raw = _reg_value(key, "HardwareInformation.qwMemorySize")
                    if not raw:
                        raw = _reg_value(key, "HardwareInformation.MemorySize")
                    vram = round(raw / BYTES_PER_GB, 2) if raw else None
                    gpus.append(
                        GPU(
                            name=desc.strip(),
                            vram_gb=vram,
                            backend=backend_for_name(desc, "windows"),
                        )
                    )
    except Exception as exc:
        log.debug("registry GPU probe found nothing: %s", exc)
        return []
    return gpus


def _read_text(path: str) -> Optional[str]:
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as handle:
            return handle.read().strip()
    except OSError:
        return None


def _from_linux() -> List[GPU]:
    """Look at /proc and /sys. Reached only when nvidia-smi and torch found nothing."""
    gpus: List[GPU] = []
    proc_root = "/proc/driver/nvidia/gpus"
    try:
        entries = sorted(os.listdir(proc_root))
    except OSError:
        entries = []
    for entry in entries:
        info = _read_text(os.path.join(proc_root, entry, "information")) or ""
        match = re.search(r"^Model:\s*(.+)$", info, re.MULTILINE)
        name = match.group(1).strip() if match else "NVIDIA GPU"
        gpus.append(GPU(name=name, vram_gb=None, backend="cuda"))
    if gpus:
        return gpus
    drm_root = "/sys/class/drm"
    try:
        cards = sorted(os.listdir(drm_root))
    except OSError:
        return []
    for entry in cards:
        if not re.fullmatch(r"card\d+", entry):
            continue
        device = os.path.join(drm_root, entry, "device")
        vendor = (_read_text(os.path.join(device, "vendor")) or "").lower()
        if vendor != "0x1002":  # only AMD is identifiable this way
            continue
        raw = _read_text(os.path.join(device, "mem_info_vram_total"))
        vram: Optional[float] = None
        if raw and raw.isdigit():
            vram = round(int(raw) / BYTES_PER_GB, 2)
        gpus.append(GPU(name="AMD Radeon GPU", vram_gb=vram, backend="rocm"))
    return gpus


def _from_platform() -> List[GPU]:
    system = platform.system().lower()
    if system == "windows":
        return _from_windows_registry()
    if system == "darwin":
        if platform.machine().lower() in ("arm64", "aarch64"):
            return [apple_gpu()]
        return []
    if system == "linux":
        return _from_linux()
    return []


# --------------------------------------------------------------------------- #
# the public probe
# --------------------------------------------------------------------------- #
_cache: Optional[List[GPU]] = None


def probe_gpus(refresh: bool = False) -> List[GPU]:
    """Find the GPUs on this machine. Returns ``[]`` when there are none.

    The answer is cached after the first call, because hardware does not change
    while a process runs; pass ``refresh=True`` to probe again.
    """
    global _cache
    if _cache is not None and not refresh:
        return list(_cache)
    gpus: List[GPU] = []
    for probe in (_from_nvidia_smi, _from_torch, _from_platform):
        try:
            found = probe()
        except Exception as exc:  # pragma: no cover - probes swallow their own
            log.debug("GPU probe %s failed: %s", probe.__name__, exc)
            found = []
        if found:
            gpus = found
            break
    _cache = list(gpus)
    return list(gpus)
