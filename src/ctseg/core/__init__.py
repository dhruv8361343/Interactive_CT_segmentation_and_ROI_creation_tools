"""
UI-independent image, label, and editing primitives.

FILE PURPOSE:
    This is the `core` sub-package init. It re-exports the three foundational
    classes so that code can write `from ctseg.core import CTVolume` instead
    of the longer `from ctseg.core.volume import CTVolume`.

    Everything in the `core` package works with just `numpy` — no PyQt6, no VTK.
    This is the foundation layer that both `io/` and `ui/` build upon.
"""

from ctseg.core.geometry import (
    VolumeGeometry,  # Spatial metadata (spacing, origin, direction cosines)
)
from ctseg.core.label_volume import (  # Editable segmentation mask + per-label properties
    LabelInfo,
    LabelVolume,
)
from ctseg.core.volume import CTVolume  # 3D grayscale image container

__all__ = ["CTVolume", "LabelInfo", "LabelVolume", "VolumeGeometry"]
