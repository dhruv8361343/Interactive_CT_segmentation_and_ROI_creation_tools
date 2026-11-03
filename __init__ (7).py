"""
ctseg — Reusable CT image viewing and segmentation toolkit.

FILE PURPOSE:
     By importing the core classes here, users can
    write `from ctseg import CTVolume` instead of `from ctseg.core.volume import CTVolume`.

    The UI layer (PyQt6/VTK) is NOT imported here to keep the core package
    lightweight and usable in headless/server environments.
"""

# Public API imports
# These are the four foundational classes that external code will use most often.
# They come from the `core` sub-package which has zero UI dependencies.
from ctseg.core.geometry import VolumeGeometry  # Spatial metadata (spacing, origin, direction)
from ctseg.core.label_volume import (  # Editable segmentation mask + label metadata
    LabelInfo,
    LabelVolume,
)
from ctseg.core.volume import CTVolume  # 3D grayscale image container

# Controls what `from ctseg import *` exports
__all__ = ["CTVolume", "LabelInfo", "LabelVolume", "VolumeGeometry"]

# Semantic version — should match pyproject.toml [project].version
__version__ = "0.1.0"
