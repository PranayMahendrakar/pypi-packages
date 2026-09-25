"""What this computer can run: RAM, CPU cores and GPUs.

NVIDIA GPUs are read through the driver's own NVML library with ``ctypes``,
which needs no PyTorch, no CUDA toolkit and no subprocess. Apple Silicon is
recognised from the platform, and its GPU shares the system RAM. Other GPUs
(AMD, Intel) are not detected, because none of the supported Python engines
can use them without a special build.
"""

from __future__ import annotations

import ctypes
import logging
import os
import platform
import sys
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_GB = 1024.0 ** 3

#: Set to 1 to skip GPU detection entirely (a broken driver can make NVML hang).
NO_GPU_ENV = "OFFLINE_STT_ROUTER_NO_GPU"


@dataclass
class GPU:
    """One GPU an engine could use."""

    name: str
    vendor: str
    backend: str
    vram_total_gb: Optional[float] = None
    vram_free_gb: Optional[float] = None
    shared_memory: bool = False

    @property
    def usable_vram_gb(self) -> Optional[float]:
        """Free VRAM when known, otherwise the total."""
        if self.vram_free_gb is not None:
            return self.vram_free_gb
        return self.vram_total_gb

    def describe(self) -> str:
        """One line, plain ASCII."""
        if self.shared_memory:
            return "{0} (shares system RAM)".format(self.name)
        if self.vram_total_gb is None:
            return self.name
        if self.vram_free_gb is None:
            return "{0} ({1:.1f} GB VRAM)".format(self.name, self.vram_total_gb)
        return "{0} ({1:.1f} GB VRAM, {2:.1f} GB free)".format(
            self.name, self.vram_total_gb, self.vram_free_gb
        )

    def to_dict(self) -> Dict[str, Any]:
        """JSON-safe form."""
        return {
            "name": self.name,
            "vendor": self.vendor,
            "backend": self.backend,
            "vram_total_gb": _round(self.vram_total_gb),
            "vram_free_gb": _round(self.vram_free_gb),
            "shared_memory": self.shared_memory,
        }


def _round(value: Optional[float]) -> Optional[float]:
    return None if value is None else round(float(value), 2)


@dataclass
class Machine:
    """The hardware the router weighs engines against.

    Build one by hand to ask "what would it pick on a 4 GB laptop?"::

        Machine(ram_total_gb=4, ram_available_gb=3, cpu_physical=2)

    ``Machine.detect()`` (or ``offline_stt_router.machine()``) reads this one.
    """

    os: str = "unknown"
    arch: str = "unknown"
    cpu_logical: Optional[int] = None
    cpu_physical: Optional[int] = None
    ram_total_gb: Optional[float] = 8.0
    ram_available_gb: Optional[float] = None
    gpus: List[GPU] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.cpu_logical is None and self.cpu_physical is None:
            self.cpu_logical = self.cpu_physical = 4
        self.cpu_logical = max(1, int(self.cpu_logical or self.cpu_physical or 1))
        self.cpu_physical = max(1, int(self.cpu_physical or self.cpu_logical))
        self.cpu_logical = max(self.cpu_logical, self.cpu_physical)
        if self.ram_available_gb is None and self.ram_total_gb is not None:
            self.ram_available_gb = float(self.ram_total_gb)
        self.gpus = [g if isinstance(g, GPU) else GPU(**g) for g in self.gpus]

    @classmethod
    def detect(cls) -> "Machine":
        """Read RAM, cores and GPUs from this computer. Never raises."""
        notes: List[str] = []
        logical = os.cpu_count() or 1
        physical: Optional[int] = None
        total: Optional[float] = None
        available: Optional[float] = None
        try:
            import psutil

            logical = psutil.cpu_count(logical=True) or logical
            physical = psutil.cpu_count(logical=False)
            memory = psutil.virtual_memory()
            total = memory.total / _GB
            available = memory.available / _GB
        except Exception as exc:  # psutil missing or unsupported platform
            notes.append("could not read RAM and cores ({0}); limits are unknown".format(exc))
            logger.warning("psutil could not read this machine: %s", exc)
        if not physical:
            physical = logical
            notes.append("physical core count unknown; using the logical count")
        gpus = detect_gpus(total_ram_gb=total, available_ram_gb=available, notes=notes)
        return cls(
            os=platform.system() or sys.platform,
            arch=platform.machine() or "unknown",
            cpu_logical=int(logical),
            cpu_physical=int(physical),
            ram_total_gb=total,
            ram_available_gb=available,
            gpus=gpus,
            notes=notes,
        )

    @property
    def cuda_gpu(self) -> Optional[GPU]:
        """The NVIDIA GPU with the most usable VRAM, if any."""
        cuda = [g for g in self.gpus if g.backend == "cuda"]
        if not cuda:
            return None
        return max(cuda, key=lambda g: g.usable_vram_gb or 0.0)

    @property
    def metal_gpu(self) -> Optional[GPU]:
        """The Apple Silicon GPU, if any."""
        for gpu in self.gpus:
            if gpu.backend == "metal":
                return gpu
        return None

    @property
    def threads(self) -> int:
        """Threads an engine should use: the physical cores, at most 16."""
        return max(1, min(16, self.cpu_physical))

    def describe(self) -> str:
        """One line, plain ASCII."""
        cores = "{0} cores".format(self.cpu_physical)
        if self.cpu_logical != self.cpu_physical:
            cores += " / {0} threads".format(self.cpu_logical)
        if self.ram_total_gb is None:
            ram = "RAM unknown"
        elif self.ram_available_gb is not None:
            ram = "{0:.1f} GB RAM ({1:.1f} GB available)".format(
                self.ram_total_gb, self.ram_available_gb
            )
        else:
            ram = "{0:.1f} GB RAM".format(self.ram_total_gb)
        if self.gpus:
            gpu = "GPU: " + "; ".join(g.describe() for g in self.gpus)
        else:
            gpu = "no usable GPU found"
        return "{0} ({1}), {2}, {3}, {4}".format(self.os, self.arch, cores, ram, gpu)

    def summary(self) -> str:
        """Human-readable text, plain ASCII."""
        lines = [self.describe()]
        for note in self.notes:
            lines.append("  note: " + note)
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """JSON-safe form."""
        return {
            "os": self.os,
            "arch": self.arch,
            "cpu_logical": self.cpu_logical,
            "cpu_physical": self.cpu_physical,
            "ram_total_gb": _round(self.ram_total_gb),
            "ram_available_gb": _round(self.ram_available_gb),
            "gpus": [g.to_dict() for g in self.gpus],
            "notes": list(self.notes),
        }


_gpu_lock = threading.Lock()
_gpu_cache: Optional[List[GPU]] = None


def detect_gpus(
    total_ram_gb: Optional[float] = None,
    available_ram_gb: Optional[float] = None,
    notes: Optional[List[str]] = None,
) -> List[GPU]:
    """NVIDIA GPUs through NVML, plus Apple Silicon. Cached for the process."""
    global _gpu_cache
    if os.environ.get(NO_GPU_ENV, "").strip() not in ("", "0"):
        if notes is not None:
            notes.append("GPU detection switched off by {0}".format(NO_GPU_ENV))
        return []
    found: List[GPU] = []
    with _gpu_lock:
        if _gpu_cache is None:
            try:
                _gpu_cache = _nvidia_gpus()
            except Exception as exc:  # a driver library that misbehaves
                logger.info("NVML query failed: %s", exc)
                _gpu_cache = []
        found.extend(GPU(**g.__dict__) for g in _gpu_cache)
    if is_apple_silicon():
        found.append(
            GPU(
                name="Apple Silicon GPU",
                vendor="apple",
                backend="metal",
                vram_total_gb=total_ram_gb,
                vram_free_gb=available_ram_gb,
                shared_memory=True,
            )
        )
    return found


def is_apple_silicon(system: Optional[str] = None, arch: Optional[str] = None) -> bool:
    """True on an M-series Mac, where the GPU shares system memory."""
    system = sys.platform if system is None else system
    arch = platform.machine() if arch is None else arch
    return system == "darwin" and arch.lower() in ("arm64", "aarch64")


class _NvmlMemory(ctypes.Structure):
    _fields_ = [
        ("total", ctypes.c_ulonglong),
        ("free", ctypes.c_ulonglong),
        ("used", ctypes.c_ulonglong),
    ]


def _nvml_library() -> Optional[Any]:
    if sys.platform == "win32":
        root = os.environ.get("SystemRoot", r"C:\Windows")
        program_files = os.environ.get("ProgramFiles", r"C:\Program Files")
        candidates = [
            os.path.join(root, "System32", "nvml.dll"),
            os.path.join(program_files, "NVIDIA Corporation", "NVSMI", "nvml.dll"),
        ]
    elif sys.platform.startswith("linux"):
        candidates = ["libnvidia-ml.so.1", "libnvidia-ml.so"]
    else:
        return None
    for candidate in candidates:
        if os.path.isabs(candidate) and not os.path.isfile(candidate):
            continue
        try:
            return ctypes.CDLL(candidate)
        except OSError:
            continue
    return None


def _nvml_call(lib: Any, *names: str) -> Any:
    for name in names:
        function = getattr(lib, name, None)
        if function is not None:
            return function
    raise AttributeError(names[0])


def _nvidia_gpus() -> List[GPU]:
    lib = _nvml_library()
    if lib is None:
        return []
    init = _nvml_call(lib, "nvmlInit_v2", "nvmlInit")
    if init() != 0:
        return []
    gpus: List[GPU] = []
    try:
        count = ctypes.c_uint(0)
        if _nvml_call(lib, "nvmlDeviceGetCount_v2", "nvmlDeviceGetCount")(ctypes.byref(count)) != 0:
            return []
        get_handle = _nvml_call(lib, "nvmlDeviceGetHandleByIndex_v2", "nvmlDeviceGetHandleByIndex")
        for index in range(count.value):
            handle = ctypes.c_void_p()
            if get_handle(ctypes.c_uint(index), ctypes.byref(handle)) != 0:
                continue
            name_buffer = ctypes.create_string_buffer(96)
            name = "NVIDIA GPU {0}".format(index)
            if lib.nvmlDeviceGetName(handle, name_buffer, ctypes.c_uint(96)) == 0:
                name = name_buffer.value.decode("utf-8", "replace") or name
            memory = _NvmlMemory()
            total = free = None
            if lib.nvmlDeviceGetMemoryInfo(handle, ctypes.byref(memory)) == 0:
                total = memory.total / _GB
                free = memory.free / _GB
            gpus.append(
                GPU(name=name, vendor="nvidia", backend="cuda", vram_total_gb=total, vram_free_gb=free)
            )
    finally:
        try:
            lib.nvmlShutdown()
        except Exception:  # pragma: no cover - shutdown failure changes nothing
            pass
    return gpus
