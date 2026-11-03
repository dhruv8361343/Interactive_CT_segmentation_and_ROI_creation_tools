"""Optional PyQt6/VTK widgets; core package imports do not require the UI stack."""

from __future__ import annotations

__all__ = ["MainWindow", "SliceView"]


def __getattr__(name: str):
    """Load Qt only when a UI class is requested by an application."""
    if name == "MainWindow":
        from ctseg.ui.main_window import MainWindow

        return MainWindow
    if name == "SliceView":
        from ctseg.ui.slice_view import SliceView

        return SliceView
    raise AttributeError(name)
