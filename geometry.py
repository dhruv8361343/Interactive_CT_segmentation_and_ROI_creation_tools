"""
Image geometry helpers shared by readers, algorithms, and viewers.

FILE PURPOSE:
    This file defines `VolumeGeometry`, the single source of truth for how
    a 3D voxel grid maps to physical (real-world) space. Every CT image has:

    - spacing_xyz: voxel size in millimeters along X, Y, Z axes
    - origin_xyz: the physical position of voxel [0,0,0]
    - direction: a 3x3 rotation matrix (as a flat 9-tuple) that describes
                 how the voxel axes align with the patient LPS coordinate system

   
created in: Phase 1 — Core Data Model

"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import isfinite

import numpy as np

# Default identity direction matrix: X, Y, Z axes align with patient LPS
# (Left-Posterior-Superior) coordinate system without any rotation.
IDENTITY_DIRECTION = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)


@dataclass(frozen=True, slots=True)
class VolumeGeometry:
    """Describe voxel spacing and orientation without guessing data provenance.

    TIFF often lacks slice spacing and origin. Default identity metadata keeps
    the image viewable, while the explicit confirmation flags stop those
    display defaults being mistaken for measurement-grade geometry.
    """

    # Spatial metadata 
    # Voxel size in mm: (pixel_width, pixel_height, slice_thickness)
    spacing_xyz: tuple[float, float, float] = (1.0, 1.0, 1.0)

    # Physical position of voxel [0,0,0] in LPS coordinates
    origin_xyz: tuple[float, float, float] = (0.0, 0.0, 0.0)

    # 3x3 direction cosine matrix stored as a flat 9-tuple (row-major).
    # Describes how voxel axes map to LPS patient coordinates.
    direction: tuple[float, ...] = IDENTITY_DIRECTION

    # Confirmation flags 
    spacing_confirmed: bool = False
    origin_confirmed: bool = False
    direction_confirmed: bool = False

    # Physical measurement unit (typically "mm" for medical images)
    units: str = "mm"

    def __post_init__(self) -> None:
        """Reject malformed metadata at the boundary, before it can skew edits.

        This runs automatically when a VolumeGeometry is created. It validates:
        1. spacing must be three positive finite values (can't have zero or negative voxel size)
        2. origin must be three finite values
        3. direction must be nine finite values forming an orthonormal rotation matrix
        4. units must not be empty
        """
        # Convert to tuples of float (handles numpy scalars, ints, etc.)
        spacing = tuple(float(value) for value in self.spacing_xyz)
        origin = tuple(float(value) for value in self.origin_xyz)
        direction = tuple(float(value) for value in self.direction)

        # Validate spacing: 3 values, all positive and finite (no NaN, no Inf)
        if len(spacing) != 3 or any(not isfinite(value) or value <= 0 for value in spacing):
            raise ValueError("spacing_xyz must contain three finite positive values")

        # Validate origin: 3 values, all finite
        if len(origin) != 3 or any(not isfinite(value) for value in origin):
            raise ValueError("origin_xyz must contain three finite values")

        # Validate direction: 9 values, all finite
        if len(direction) != 9 or any(not isfinite(value) for value in direction):
            raise ValueError("direction must contain nine finite values")

        # Check that the direction matrix is orthonormal:
        
        matrix = np.asarray(direction, dtype=np.float64).reshape(3, 3)
        if not np.allclose(matrix @ matrix.T, np.eye(3), atol=1e-5):
            raise ValueError("direction must be an orthonormal direction-cosine matrix")

        # Check that the determinant has unit magnitude (not a scaling/shearing matrix)
        if not np.isclose(abs(np.linalg.det(matrix)), 1.0, atol=1e-5):
            raise ValueError("direction matrix must be invertible with unit determinant magnitude")

        # Units must be explicitly provided
        if not self.units:
            raise ValueError("units must be a non-empty string")

        # Since the dataclass is frozen, we use object.__setattr__ to set the
        # validated values. This is a standard pattern for frozen dataclasses.
        object.__setattr__(self, "spacing_xyz", spacing)
        object.__setattr__(self, "origin_xyz", origin)
        object.__setattr__(self, "direction", direction)

    @property
    def is_confirmed(self) -> bool:
        """Whether every physical transform component came from known metadata.

        Returns True only when spacing, origin, AND direction are all confirmed.
        This is used by measurement tools to refuse producing misleading results
        when the geometry is only a default placeholder.
        """
        return self.spacing_confirmed and self.origin_confirmed and self.direction_confirmed

    @property
    def direction_matrix(self) -> np.ndarray:
        """Return a fresh 3x3 numpy matrix so callers cannot mutate this immutable geometry.

        `.copy()` ensures the returned array is independent — modifying it won't
        affect the stored direction tuple.
        """
        return np.asarray(self.direction, dtype=np.float64).reshape(3, 3).copy()

    def index_to_physical(
        self, index_xyz: Sequence[float], *, allow_assumed: bool = False
    ) -> tuple[float, float, float]:
        """Map a voxel coordinate to LPS physical space.

        Formula: physical = origin + direction_matrix @ (index * spacing)

        This is the standard image-to-physical transform used in medical imaging.
        The `allow_assumed` parameter exists for display-only calculations that
        need approximate coordinates even when geometry isn't fully confirmed.

        Measurement-facing callers must not silently use unit-spaced defaults;
        `allow_assumed` exists for explicitly display-only calculations.
        """
        # Refuse to produce physical coordinates from unconfirmed geometry
        
        if not self.is_confirmed and not allow_assumed:
            raise ValueError("physical geometry is incomplete; supply or confirm metadata first")

        # Validate the input: must be exactly 3 finite values
        index = np.asarray(index_xyz, dtype=np.float64)
        if index.shape != (3,) or not np.all(np.isfinite(index)):
            raise ValueError("index_xyz must contain three finite values")

        # Apply the transform: origin + direction @ (index * spacing)
        physical = np.asarray(self.origin_xyz) + self.direction_matrix @ (
            index * np.asarray(self.spacing_xyz)
        )
        return tuple(float(value) for value in physical)

    def physical_to_index(
        self, point_lps_xyz: Sequence[float], *, allow_assumed: bool = False
    ) -> tuple[float, float, float]:
        """Map an LPS physical point to continuous voxel coordinates.

        This is the inverse of index_to_physical(). It solves:
            direction_matrix @ (index * spacing) = point - origin
        for `index`, using numpy's linalg.solve for numerical stability.

        Returns continuous (floating-point) indices — caller must round if
        they need integer voxel coordinates.
        """
        if not self.is_confirmed and not allow_assumed:
            raise ValueError("physical geometry is incomplete; supply or confirm metadata first")

        point = np.asarray(point_lps_xyz, dtype=np.float64)
        if point.shape != (3,) or not np.all(np.isfinite(point)):
            raise ValueError("point_lps_xyz must contain three finite values")

        # Solve the linear system: direction @ scaled = (point - origin)
        # Then divide by spacing to get voxel indices
        scaled = np.linalg.solve(self.direction_matrix, point - np.asarray(self.origin_xyz))
        index = scaled / np.asarray(self.spacing_xyz)
        return tuple(float(value) for value in index)
