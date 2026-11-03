#Optional CUDA discovery and OpenGL GPU selection shared by the toolkit.

from __future__ import annotations

import logging
import os
import sys
from functools import lru_cache
from pathlib import Path

logger = logging.getLogger(__name__)

_GPU_PREFERENCE_KEY = r"Software\Microsoft\DirectX\UserGpuPreferences"
_HIGH_PERFORMANCE = "GpuPreference=2;"


def opengl_renderer_name(capabilities: str) -> str:
    """Extract the OpenGL device name from VTK's ReportCapabilities() text."""
    for line in capabilities.splitlines():
        if "OpenGL renderer string" in line:
            return line.split(":", 1)[1].strip()
    return ""


def classify_gpu(renderer_name: str) -> str:
    """Return 'nvidia', 'intel', 'amd', or 'other' for an OpenGL renderer string."""
    name = renderer_name.casefold()
    if "nvidia" in name or "geforce" in name or "quadro" in name or "rtx" in name:
        return "nvidia"
    if "intel" in name:
        return "intel"
    if "amd" in name or "radeon" in name:
        return "amd"
    return "other"


def _windows_executables() -> list[Path]:
    """Every executable that can own the OpenGL context for this environment."""
    candidates: list[Path] = [Path(sys.executable)]
    base = getattr(sys, "_base_executable", None)
    if base:
        candidates.append(Path(base))
    candidates.append(Path(sys.base_prefix) / "python.exe")
    scripts = Path(sys.executable).parent
    candidates.append(scripts / "ctseg.exe")
    expanded: list[Path] = []
    for path in candidates:
        expanded.append(path)
        if path.name.casefold() == "python.exe":
            expanded.append(path.with_name("pythonw.exe"))
    unique: list[Path] = []
    for path in expanded:
        if path.is_file() and path not in unique:
            unique.append(path)
    return unique


def windows_gpu_preference(high_performance: bool = True) -> list[Path]:
    """Select the high-performance (NVIDIA) GPU, or reset to automatic, for this Python.

    This writes the same per-user setting as Windows Settings > Display > Graphics, so it
    needs no administrator rights and can be reverted with ``ctseg --reset-gpu-preference``.
    The application must be restarted afterwards.
    """
    if os.name != "nt":
        raise OSError("GPU preference is a Windows graphics setting")
    import winreg

    changed: list[Path] = []
    with winreg.CreateKeyEx(
        winreg.HKEY_CURRENT_USER, _GPU_PREFERENCE_KEY, 0, winreg.KEY_SET_VALUE
    ) as key:
        for executable in _windows_executables():
            if high_performance:
                winreg.SetValueEx(key, str(executable), 0, winreg.REG_SZ, _HIGH_PERFORMANCE)
            else:
                try:
                    winreg.DeleteValue(key, str(executable))
                except FileNotFoundError:
                    continue
            changed.append(executable)
    return changed


@lru_cache(maxsize=1)
def get_cupy():
    #Return CuPy when a working CUDA device is available, otherwise use CPU.
    try:
        import cupy as cp
    except ImportError as error:
        logger.info("CuPy is not installed; whole-volume operations will use the CPU: %s", error)
        return None

    try:
        device_count = cp.cuda.runtime.getDeviceCount()
    except Exception as error:
        logger.warning("CUDA is unavailable; whole-volume operations will use the CPU: %s", error)
        return None
    if device_count == 0:
        logger.info("No CUDA device found; whole-volume operations will use the CPU")
        return None

    try:
        properties = cp.cuda.runtime.getDeviceProperties(0)
        name = properties["name"]
        if isinstance(name, bytes):
            name = name.decode("utf-8", errors="replace")
        logger.info("CUDA acceleration available on %s", name)
    except Exception as error:
        logger.warning("Could not query CUDA device properties; using CPU: %s", error)
        return None
    return cp
