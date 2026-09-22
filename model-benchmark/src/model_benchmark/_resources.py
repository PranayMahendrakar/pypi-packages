"""Allocation and resident-memory probes.

Two independent numbers are collected for every model:

* the peak *Python allocation* delta, from :mod:`tracemalloc`, which is exact
  and comparable between models but only sees Python objects;
* the process *RSS* delta, which also sees memory allocated by C extensions but
  is noisy and can legitimately come back as ``0`` or ``None``.

No GPU is assumed or queried anywhere.
"""
from __future__ import annotations

import sys
import tracemalloc
from contextlib import contextmanager
from typing import Iterator, Optional

__all__ = ["MemoryReading", "measure_allocations", "rss_bytes"]


class MemoryReading:
    """Filled in when the :func:`measure_allocations` block exits."""

    __slots__ = ("peak_bytes",)

    def __init__(self) -> None:
        self.peak_bytes: float = 0.0


@contextmanager
def measure_allocations() -> Iterator[MemoryReading]:
    """Peak Python allocation growth inside the block, in bytes.

    Tracing is started and stopped around the block so nothing leaks into the
    next model. If the caller was already tracing, that tracing is left running
    and only the peak counter is reset.
    """
    reading = MemoryReading()
    was_tracing = tracemalloc.is_tracing()
    if was_tracing:
        tracemalloc.reset_peak()
    else:
        tracemalloc.start()
    baseline, _ = tracemalloc.get_traced_memory()
    try:
        yield reading
    finally:
        try:
            _current, peak = tracemalloc.get_traced_memory()
            reading.peak_bytes = float(max(0.0, peak - baseline))
        except Exception:
            reading.peak_bytes = 0.0
        finally:
            try:
                if not was_tracing and tracemalloc.is_tracing():
                    tracemalloc.stop()
            except Exception:
                pass


def _rss_psutil() -> Optional[int]:
    try:
        import psutil  # type: ignore
    except Exception:
        return None
    try:
        return int(psutil.Process().memory_info().rss)
    except Exception:
        return None


def _rss_proc() -> Optional[int]:
    try:
        with open("/proc/self/statm", "r", encoding="utf-8") as handle:
            fields = handle.read().split()
        import os

        return int(fields[1]) * os.sysconf("SC_PAGE_SIZE")
    except Exception:
        return None


def _rss_windows() -> Optional[int]:
    try:
        import ctypes
        from ctypes import wintypes

        class _Counters(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        counters = _Counters()
        counters.cb = ctypes.sizeof(_Counters)
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        handle = kernel32.GetCurrentProcess()
        ok = psapi.GetProcessMemoryInfo(
            ctypes.c_void_p(handle), ctypes.byref(counters), counters.cb
        )
        if not ok:
            return None
        return int(counters.WorkingSetSize)
    except Exception:
        return None


def _rss_rusage() -> Optional[int]:
    try:
        import resource  # type: ignore
    except Exception:
        return None
    try:
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    except Exception:
        return None
    # Linux reports kilobytes, macOS and the BSDs report bytes.
    return int(peak) if sys.platform == "darwin" else int(peak) * 1024


def rss_bytes() -> Optional[int]:
    """Resident set size of this process in bytes, or None when unavailable."""
    for probe in (_rss_psutil, _rss_windows, _rss_proc, _rss_rusage):
        value = probe()
        if value is not None:
            return value
    return None
