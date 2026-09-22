"""What machine am I on? ``detect()`` and the ``Hardware`` record it returns.

Units, stated once and used everywhere: every ``*_gb`` number is a binary
gigabyte (GiB, 1024**3 bytes) and every frequency is in MHz.
"""
from __future__ import annotations

import logging
import os
import platform
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import psutil

from ._gpu import BYTES_PER_GB, GPU, probe_gpus

log = logging.getLogger(__name__)

#: ``platform.system()`` values mapped to the names this package reports.
_PLATFORM_NAMES = {"windows": "windows", "linux": "linux", "darwin": "macos"}


def _round(value: float, digits: int = 2) -> float:
    return round(float(value), digits)


def _disk_probe_path() -> str:
    """Where model files normally land: the user's home directory."""
    for candidate in (os.path.expanduser("~"), os.getcwd(), os.path.abspath(os.sep)):
        if candidate and os.path.isdir(candidate):
            return candidate
    return os.path.abspath(os.sep)  # pragma: no cover - no readable directory


def _free_disk_gb(path: str) -> Optional[float]:
    """Free space in GiB, or None when it could not be read.

    None means "cannot check", not "the disk is full". An unreadable mount
    point, a sandbox or an exotic filesystem is missing information, and
    reporting it as zero would reject every model on a machine with room to
    spare.
    """
    try:
        return _round(psutil.disk_usage(path).free / BYTES_PER_GB)
    except OSError as exc:  # unreadable mount point, sandbox, exotic filesystem
        log.debug("could not read free space on %s: %s", path, exc)
        return None


def _cpu_freq_mhz() -> Optional[float]:
    """Current CPU frequency in MHz, or None where the OS does not expose one."""
    try:
        freq = psutil.cpu_freq()
    except Exception as exc:  # not available in containers, some ARM kernels, VMs
        log.debug("cpu frequency unavailable: %s", exc)
        return None
    if freq is None:
        return None
    value = getattr(freq, "max", 0.0) or getattr(freq, "current", 0.0) or 0.0
    return _round(value, 1) if value else None


@dataclass
class Hardware:
    """The facts about this machine that decide what a model can do here.

    Every ``*_gb`` value is a binary gigabyte (GiB, 1024**3 bytes).
    """

    cpu_count: int
    cpu_freq_mhz: Optional[float]
    ram_total_gb: float
    ram_available_gb: float
    gpus: List[GPU] = field(default_factory=list)
    platform: str = "unknown"
    python_version: str = ""
    disk_free_gb: Optional[float] = None
    disk_path: str = ""

    # -- derived facts ----------------------------------------------------- #
    @property
    def has_gpu(self) -> bool:
        """True when a GPU with a usable compute backend was found.

        A display-only adapter is still listed in :attr:`gpus`, with backend
        ``"none"``, but does not count as a GPU you can run a model on.
        """
        return any(gpu.is_accelerator for gpu in self.gpus)

    @property
    def gpu(self) -> Optional[GPU]:
        """The GPU a model would run on: the accelerator with the most VRAM.

        Cards are enumerated in PCI bus order, not by size, so a small display
        card in a lower slot must not shadow the big compute card beside it. A
        tie keeps the first one listed, so the answer never changes between
        runs.
        """
        accelerators = [gpu for gpu in self.gpus if gpu.is_accelerator]
        if not accelerators:
            return None
        return max(accelerators, key=lambda gpu: gpu.usable_vram_gb)

    @property
    def gpu_backend(self) -> str:
        """``"cuda"``, ``"rocm"``, ``"mps"``, or ``"none"`` when there is no GPU."""
        gpu = self.gpu
        return gpu.backend if gpu is not None else "none"

    @property
    def vram_gb(self) -> float:
        """Usable VRAM in GiB on the best GPU; 0.0 when there is none."""
        gpu = self.gpu
        return gpu.usable_vram_gb if gpu is not None else 0.0

    @property
    def device(self) -> str:
        """The torch-style device string for this machine.

        ``"cuda"``, ``"mps"`` or ``"cpu"``. An AMD ROCm card also reports
        ``"cuda"``, because that is the device string PyTorch uses for ROCm.
        """
        backend = self.gpu_backend
        if backend in ("cuda", "rocm"):
            return "cuda"
        if backend == "mps":
            return "mps"
        return "cpu"

    # -- output ------------------------------------------------------------ #
    def short(self) -> str:
        """One line: cores, RAM, GPU. Used inside other summaries."""
        gpu = self.gpu
        if gpu is None:
            tail = "no GPU"
        elif gpu.vram_gb is None:
            tail = f"{gpu.name} (VRAM unknown)"
        else:
            tail = f"{gpu.name} ({gpu.vram_gb:.1f} GiB)"
        return f"{self.cpu_count} cores, {self.ram_total_gb:.1f} GiB RAM, {tail}"

    def summary(self) -> str:
        """The human-readable report. Plain ASCII punctuation only."""
        freq = "unknown speed" if self.cpu_freq_mhz is None else f"{self.cpu_freq_mhz:.0f} MHz"
        if self.disk_free_gb is None:
            disk = f"free space could not be read on {self.disk_path}"
        else:
            disk = f"{self.disk_free_gb:.1f} GiB free on {self.disk_path}"
        lines = [
            f"offline-ml: {self.platform}, {self.short()}",
            f"  cpu       : {self.cpu_count} logical cores at {freq}",
            f"  ram       : {self.ram_total_gb:.1f} GiB total, "
            f"{self.ram_available_gb:.1f} GiB available now",
            f"  disk      : {disk}",
        ]
        if not self.gpus:
            lines.append("  gpu       : none found - models will run on the CPU")
        else:
            best = self.gpu
            many = sum(1 for gpu in self.gpus if gpu.is_accelerator) > 1
            for index, gpu in enumerate(self.gpus):
                label = "  gpu       :" if index == 0 else "             "
                chosen = " [chosen]" if many and gpu is best else ""
                lines.append(f"{label} {gpu}{chosen}")
            if not self.has_gpu:
                lines.append("             no usable compute backend - models run on the CPU")
        lines.append(f"  python    : {self.python_version}")
        lines.append(f"  device    : {self.device}")
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """JSON-safe dict of everything above."""
        return {
            "cpu_count": self.cpu_count,
            "cpu_freq_mhz": self.cpu_freq_mhz,
            "ram_total_gb": self.ram_total_gb,
            "ram_available_gb": self.ram_available_gb,
            "gpus": [gpu.to_dict() for gpu in self.gpus],
            "has_gpu": self.has_gpu,
            "gpu_backend": self.gpu_backend,
            "vram_gb": self.vram_gb,
            "platform": self.platform,
            "python_version": self.python_version,
            "disk_free_gb": self.disk_free_gb,
            "disk_path": self.disk_path,
            "device": self.device,
            "units": "all *_gb values are binary gigabytes (GiB, 1024**3 bytes)",
        }


def detect(*, refresh: bool = False) -> Hardware:
    """Look at this machine and report what it can run.

    Needs no network, no GPU and no optional libraries. Finding no GPU is the
    normal case and is not an error, a warning, or anything you have to handle.

    Args:
        refresh: probe the GPUs again instead of reusing the first answer.
            RAM and disk numbers are always read fresh.

    Returns:
        A :class:`Hardware` record.
    """
    memory = psutil.virtual_memory()
    disk_path = _disk_probe_path()
    system = platform.system().lower()
    return Hardware(
        cpu_count=int(psutil.cpu_count(logical=True) or os.cpu_count() or 1),
        cpu_freq_mhz=_cpu_freq_mhz(),
        ram_total_gb=_round(memory.total / BYTES_PER_GB),
        ram_available_gb=_round(memory.available / BYTES_PER_GB),
        gpus=probe_gpus(refresh=refresh),
        platform=_PLATFORM_NAMES.get(system, system or "unknown"),
        python_version=platform.python_version(),
        disk_free_gb=_free_disk_gb(disk_path),
        disk_path=disk_path,
    )


def best_device(hardware: Optional[Hardware] = None) -> str:
    """The device string a model should use here: ``"cuda"``, ``"mps"`` or ``"cpu"``.

    An AMD ROCm card reports ``"cuda"``, because that is the device string
    PyTorch uses for ROCm builds.
    """
    return (hardware or detect()).device
