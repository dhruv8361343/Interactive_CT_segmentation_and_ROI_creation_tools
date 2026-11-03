"""
Command-line entry point for the optional PyQt6 application.

FILE PURPOSE:
    This file is registered as the `ctseg` CLI command in pyproject.toml
    (`ctseg = "ctseg.app:main"`). It handles two modes:

    1. `ctseg --inspect <path>` — Metadata-only TIFF inspection (no GUI needed).
       Useful for quickly checking a TIFF stack's shape, dtype, and axes without
       loading gigabytes of pixel data into memory.

    2. `ctseg [path]` — Launch the full PyQt6 desktop application with the optional
       path pre-loaded. If PyQt6 is not installed, it raises a clear error message
       instead of a cryptic ImportError.

    The PyQt6 import is deferred (inside the function body) so that the core
    package can be imported without installing Qt. This lets headless scripts
    use `from ctseg import CTVolume` without needing a display server.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    """Launch the desktop application or print a metadata-only TIFF inspection.

    Args:
        argv: Command-line arguments to parse. Defaults to sys.argv if None.

    Returns:
        Exit code: 0 for success, nonzero for errors.
    """
    # to taKE The argument from the use and then parse it 
    parser = argparse.ArgumentParser(prog="ctseg", description="CT segmentation toolkit")

    # Optional positional argument
    parser.add_argument("path", nargs="?", help="TIFF image, slice directory, or supported volume")

    # inspect flag
    parser.add_argument(
        "--inspect", action="store_true", help="inspect TIFF metadata without loading pixels"
    )

    parser.add_argument(
        "--prefer-nvidia-gpu",
        action="store_true",
        help="Windows: make 3D rendering use the high-performance (NVIDIA) GPU",
    )
    parser.add_argument(
        "--reset-gpu-preference",
        action="store_true",
        help="Windows: undo --prefer-nvidia-gpu and let Windows choose the GPU",
    )

    arguments = parser.parse_args(argv)

    #  GPU selection 
    if arguments.prefer_nvidia_gpu or arguments.reset_gpu_preference:
        from ctseg.core.gpu import windows_gpu_preference

        enable = bool(arguments.prefer_nvidia_gpu)
        for executable in windows_gpu_preference(enable):
            print(
                f"{'High-performance GPU set for' if enable else 'Reset GPU preference for'}: "
                f"{executable}"
            )
        print("Restart ctseg for the change to take effect.")
        return 0

    # Inspect mode
    if arguments.inspect:
        if not arguments.path:
            parser.error("--inspect requires a TIFF file or directory")

       
        from ctseg.io.tiff import inspect_tiff

        # inspect_tiff reads TIFF headers/metadata without decoding pixel data
        summary = inspect_tiff(Path(arguments.path))
        print(
            f"Layout: {summary.layout}\nShape: {summary.shape}\n"
            f"Axes: {summary.axes}\nChannels: {summary.num_channels}\nDtype: {summary.dtype}\nPages: {summary.pages}\n"
            f"Source: {summary.path}"
        )
        return 0

  #launching the application
    try:
        from PyQt6.QtWidgets import QApplication
    except ImportError as error:
       
        raise RuntimeError(
            "Install ctseg-toolkit[ui] before starting the desktop application"
        ) from error

    from ctseg.core.resources import configure_system_optimization
    configure_system_optimization()

   
    from ctseg.ui.main_window import MainWindow

    # this returns the existing app if one was already created
    
   
    application = QApplication.instance() or QApplication(sys.argv[:1])

    # Create the main window, 
    window = MainWindow(arguments.path)
    window.show()

    # application.exec() starts the Qt event loop and blocks until the window closes.
    return application.exec()


if __name__ == "__main__":
    raise SystemExit(main())
