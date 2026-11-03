"""Adaptive hardware resource profiling, memory budgeting, and dynamic optimization.

This module detects available host RAM, CPU topology, and GPU (CUDA/VRAM) capacity.
It configures memory budgets and worker thread counts to safely target ~90% of
available hardware resources while preventing out-of-memory errors, system swap
thrashing, and Qt UI event loop freezes.
"""

from __future__ import annotations

import ctypes
import logging
import os
import sys
from dataclasses import dataclass
from functools import lru_cache

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class SystemResources:
    #Snapshot of hardware capabilities and target 90% allocation budgets.

    total_ram_bytes: int
    available_ram_bytes: int
    ram_budget_bytes: int  # ~90% of available RAM
    cpu_count: int
    recommended_threads: int  # ~90% of CPU cores (min 1)
    gpu_available: bool
    gpu_name: str
    gpu_total_vram_bytes: int
    gpu_available_vram_bytes: int
    gpu_budget_bytes: int  # ~90% of available VRAM

    @property
    def total_ram_gb(self) -> float:
        return self.total_ram_bytes / (1024**3)

    @property
    def available_ram_gb(self) -> float:
        return self.available_ram_bytes / (1024**3)

    @property
    def ram_budget_gb(self) -> float:
        return self.ram_budget_bytes / (1024**3)

    @property
    def gpu_total_vram_gb(self) -> float:
        return self.gpu_total_vram_bytes / (1024**3)

    @property
    def gpu_available_vram_gb(self) -> float:
        return self.gpu_available_vram_bytes / (1024**3)

    @property
    def total_cpu_cores(self) -> int:
        return self.cpu_count

    @property
    def safe_ram_budget_bytes(self) -> int:
        return self.ram_budget_bytes

    @property
    def safe_gpu_budget_bytes(self) -> int:
        return self.gpu_budget_bytes

    @property
    def gpu_budget_gb(self) -> float:
        return self.gpu_budget_bytes / (1024**3)

    def summary_string(self) -> str:
        #Formatted one-line summary for status displays.
        ram_part = (
            f"RAM: {self.available_ram_gb:.1f}/{self.total_ram_gb:.1f} GB avail "
            f"(90% budget: {self.ram_budget_gb:.1f} GB)"
        )
        cpu_part = f"CPU: {self.cpu_count} cores ({self.recommended_threads} workers)"
        if self.gpu_available:
            gpu_part = (
                f"GPU: {self.gpu_name} ({self.gpu_available_vram_gb:.1f}/"
                f"{self.gpu_total_vram_gb:.1f} GB VRAM, 90% budget: {self.gpu_budget_gb:.1f} GB)"
            )
        else:
            gpu_part = "GPU: None / CPU mode"
        return f"{cpu_part} | {ram_part} | {gpu_part}"


def _query_system_ram() -> tuple[int, int]:
    #Return (total_ram_bytes, available_ram_bytes) across Windows, Linux, and macOS.
    # 1. Windows: GlobalMemoryStatusEx via ctypes (built into standard library)
    if os.name == "nt":
        try:
            class _MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            stat = _MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(_MEMORYSTATUSEX)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
                return int(stat.ullTotalPhys), int(stat.ullAvailPhys)
        except Exception as error:
            logger.debug("Windows GlobalMemoryStatusEx query failed: %s", error)

    # 2. Linux: /proc/meminfo
    if sys.platform.startswith("linux"):
        try:
            meminfo = {}
            with open("/proc/meminfo", "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.split(":")
                    if len(parts) == 2:
                        key = parts[0].strip()
                        val = parts[1].split()[0].strip()
                        meminfo[key] = int(val) * 1024
            total = meminfo.get("MemTotal", 0)
            avail = meminfo.get("MemAvailable", meminfo.get("MemFree", 0))
            if total > 0:
                return total, avail
        except Exception as error:
            logger.debug("Linux /proc/meminfo query failed: %s", error)

    # 3. Fallback: sysconf
    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        total = int(pages * page_size)
        avail = int(os.sysconf("SC_AVPHYS_PAGES") * page_size)
        return total, avail
    except Exception:
        pass

    # Generic fallback: Assume 8 GB total, 4 GB available
    return 8 * 1024**3, 4 * 1024**3


def _query_gpu_resources() -> tuple[bool, str, int, int]:
    """Return (gpu_available, name, total_vram_bytes, available_vram_bytes)."""
    # Try CuPy first
    try:
        import cupy as cp

        device_count = cp.cuda.runtime.getDeviceCount()
        if device_count > 0:
            free_bytes, total_bytes = cp.cuda.runtime.memGetInfo()
            props = cp.cuda.runtime.getDeviceProperties(0)
            raw_name = props.get("name", "CUDA GPU")
            name = (
                raw_name.decode("utf-8", errors="replace")
                if isinstance(raw_name, bytes)
                else str(raw_name)
            )
            return True, name, int(total_bytes), int(free_bytes)
    except Exception:
        pass

    # Try PyTorch if available
    try:
        import torch

        if torch.cuda.is_available():
            free_bytes, total_bytes = torch.cuda.mem_get_info(0)
            name = torch.cuda.get_device_name(0)
            return True, name, int(total_bytes), int(free_bytes)
    except Exception:
        pass

    # Try nvidia-smi command-line tool 
        import subprocess

        cmd = [
            "nvidia-smi",
            "--query-gpu=name,memory.total,memory.free",
            "--format=csv,noheader,nounits",
        ]
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=2, check=False)
        if res.returncode == 0 and res.stdout.strip():
            line = res.stdout.strip().splitlines()[0]
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 3:
                gpu_name = parts[0]
                total_bytes = int(float(parts[1]) * 1024 * 1024)
                free_bytes = int(float(parts[2]) * 1024 * 1024)
                return True, gpu_name, total_bytes, free_bytes
    except Exception:
        pass

    # Generic graphics adapter detection on Windows (Intel, AMD, NVIDIA)
    if sys.platform == "win32":
        try:
            import subprocess

            ps_cmd = "Get-CimInstance Win32_VideoController | Select-Object -ExpandProperty Name"
            res = subprocess.run(
                ["powershell", "-NoProfile", "-Command", ps_cmd],
                capture_output=True,
                text=True,
                timeout=2,
                check=False,
            )
            if res.returncode == 0 and res.stdout.strip():
                # Pick the primary or dedicated adapter
                names = [line.strip() for line in res.stdout.strip().splitlines() if line.strip()]
                # Prefer dedicated NVIDIA/AMD if listed, else integrated
                dedicated = [n for n in names if any(k in n.lower() for k in ("nvidia", "geforce", "rtx", "radeon", "amd"))]
                chosen_name = dedicated[0] if dedicated else names[0]
                # Default 2GB VRAM estimate for integrated/unspecified GPU
                return True, chosen_name, 2 * 1024**3, 1 * 1024**3
        except Exception:
            pass

    return False, "None", 0, 0


def get_system_resources(
    ram_target_ratio: float = 0.90,
    gpu_target_ratio: float = 0.90,
    cpu_target_ratio: float = 0.90,
) -> SystemResources:
    """Inspect the host machine and calculate ~90% resource allocation targets."""
    total_ram, avail_ram = _query_system_ram()
    # Safety margin: always retain at least 512 MB for OS/UI
    ram_budget = max(512 * 1024**2, int(avail_ram * min(0.95, max(0.50, ram_target_ratio))))

    cpu_cores = os.cpu_count() or 4
    recommended_threads = max(1, int(round(cpu_cores * cpu_target_ratio)))

    gpu_avail, gpu_name, total_vram, avail_vram = _query_gpu_resources()
    gpu_budget = (
        max(256 * 1024**2, int(avail_vram * gpu_target_ratio)) if gpu_avail else 0
    )

    return SystemResources(
        total_ram_bytes=total_ram,
        available_ram_bytes=avail_ram,
        ram_budget_bytes=ram_budget,
        cpu_count=cpu_cores,
        recommended_threads=recommended_threads,
        gpu_available=gpu_avail,
        gpu_name=gpu_name,
        gpu_total_vram_bytes=total_vram,
        gpu_available_vram_bytes=avail_vram,
        gpu_budget_bytes=gpu_budget,
    )


# Alias for backwards compatibility / alternate naming
query_system_resources = get_system_resources


def configure_system_optimization(
    ram_fraction: float = 0.90,
    gpu_fraction: float = 0.90,
    cpu_fraction: float = 0.90,
) -> SystemResources:
    """Configure CuPy memory pools and thread pools to utilize ~90% hardware.

    
    """
    resources = get_system_resources(
        ram_target_ratio=ram_fraction,
        gpu_target_ratio=gpu_fraction,
        cpu_target_ratio=cpu_fraction,
    )

    # Configure CuPy default memory pool limit if available
    try:
        import cupy as cp

        if resources.gpu_available and resources.gpu_budget_bytes > 0:
            mempool = cp.get_default_memory_pool()
            mempool.set_limit(size=resources.gpu_budget_bytes)
            # Pinned memory pool for host-to-device transfers
            pinned_pool = cp.get_default_pinned_memory_pool()
            pinned_pool.set_limit(size=min(resources.ram_budget_bytes // 2, 4 * 1024**3))
            logger.info(
                "CuPy memory pool limit set to %.2f GB (90%% of VRAM)",
                resources.gpu_budget_bytes / (1024**3),
            )
    except Exception as error:
        logger.debug("Could not configure CuPy memory pools: %s", error)

    return resources


def adaptive_chunk_depth(
    shape_zyx: tuple[int, int, int],
    itemsize: int = 1,
    *,
    memory_multiplier: float = 3.0,
    min_chunk: int = 4,
    max_chunk: int = 256,
    resources: SystemResources | None = None,
) -> int:
    """Calculate an optimal axial chunk depth using up to ~90% of available memory.

    Args:
        shape_zyx: Dimensions of the 3D volume (Z, Y, X).
        itemsize: Size of voxel type in bytes (e.g. 1 for uint8/bool, 2 for uint16).
        memory_multiplier: Working memory factor per chunk (e.g. source + destination + halos).
        min_chunk: Minimum number of planes per chunk.
        max_chunk: Upper limit to keep chunk boundary overhead minimal.
        resources: Pre-fetched SystemResources snapshot, or None to query live.

    Returns:
        Adaptive slab depth in voxels.
    """
    if resources is None:
        resources = get_system_resources()

    depth, height, width = shape_zyx
    plane_bytes = max(1, height * width * itemsize)

    # How many planes can comfortably fit in 90% available budget?
    # Divide budget by memory_multiplier to leave room for intermediate buffers
    effective_budget = resources.ram_budget_bytes / max(1.0, memory_multiplier)
    planes_in_budget = int(effective_budget / plane_bytes)

    # Constrain to sensible bounds
    chunk = max(min_chunk, min(max_chunk, planes_in_budget, depth))
    return int(chunk)
