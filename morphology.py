"""
N-dimensional binary morphology with explicit per-axis voxel radii.

FILE PURPOSE:
    This file implements the four standard binary morphological operations:
    dilate, erode, opening, and closing. These are used to refine segmentation
    boundaries:

    - dilate():
    - erode():  
    - opening(): 
    - closing(): 

   
CreateD IN: Phase 3 — Segmentation Operations
"""

from __future__ import annotations

from collections.abc import Sequence
from math import ceil, prod

import numpy as np


def structuring_element(radius_voxels: Sequence[float] | float, ndim: int) -> np.ndarray:
    """Build an ellipsoidal/ball structuring element that respects voxel spacing.
    Args:
        radius_voxels: Radius in voxels. Scalar = isotropic, sequence = per-axis.
        ndim: Number of dimensions (2 or 3).

    Returns:
        A boolean numpy array marking voxels inside the ellipsoid.
    """
    # Normalize scalar radius to per-axis tuple
    radii = (
        (float(radius_voxels),) * ndim
        if isinstance(radius_voxels, (int, float))
        else tuple(float(value) for value in radius_voxels)
    )

    # Validate radii
    if len(radii) != ndim or any(value < 0 or not np.isfinite(value) for value in radii):
        raise ValueError(f"radius_voxels must contain {ndim} non-negative finite values")

    # Integer extents (ceiling to ensure coverage)
    extents = tuple(ceil(value) for value in radii)
    kernel_voxels = prod(2 * extent + 1 for extent in extents)
    if kernel_voxels > 4_000_000:
        raise ValueError(
            f"morphology kernel would contain {kernel_voxels:,} voxels; reduce the radius"
        )

    # Build coordinate grids centered at 0
    grid = np.ogrid[tuple(slice(-extent, extent + 1) for extent in extents)]

    # Compute normalized ellipsoidal distance: sum((coord/radius)^2) <= 1
    distance = np.zeros(tuple(2 * extent + 1 for extent in extents), dtype=np.float64)
    for axis, radius in enumerate(radii):
        if radius == 0:
            # Zero radius: only the center voxel counts along this axis
            distance += np.where(grid[axis] == 0, 0.0, np.inf)
        else:
            distance += (grid[axis] / radius) ** 2

    # 1e-12 tolerance avoids boundary floating-point exclusions
    return distance <= 1.0 + 1e-12


def _binary_operation(
    mask: np.ndarray, radius_voxels: Sequence[float] | float, operation: str
) -> np.ndarray:
    """Internal dispatcher: builds kernel then calls SciPy or shift fallback.

    Args:
        mask: Binary input mask (True = foreground).
        radius_voxels: Kernel radii.
        operation: "dilate" or "erode".
    """
    source = np.asarray(mask, dtype=bool)
    if source.ndim not in (2, 3):
        raise ValueError("morphology accepts only 2D or 3D masks")

    # Build the structuring element (kernel)
    structure = structuring_element(radius_voxels, source.ndim)

    # Try SciPy first (much faster for large arrays)
    try:
        from scipy import ndimage
    except ImportError:
        return _shift_morphology(source, structure, operation)

    function = ndimage.binary_dilation if operation == "dilate" else ndimage.binary_erosion
    # border_value=0 means out-of-bounds voxels are treated as background
    return function(source, structure=structure, border_value=0)


def _shift_morphology(source: np.ndarray, structure: np.ndarray, operation: str) -> np.ndarray:
    """Fallback implementation without SciPy using shifted copies.
    """
    # Initialize: zeros for dilation (OR accumulator), ones for erosion (AND accumulator)
    result = np.zeros_like(source) if operation == "dilate" else np.ones_like(source)

    # Center of the structuring element
    center = tuple(length // 2 for length in structure.shape)

    # For each True voxel in the kernel, create a shifted version of the source
    for point in zip(*np.nonzero(structure), strict=True):
        # Calculate the shift offset relative to center
        offset = tuple(point[axis] - center[axis] for axis in range(source.ndim))

        shifted = np.zeros_like(source)
        src_slices = []
        dst_slices = []

        for axis, delta in enumerate(offset):
            if abs(delta) >= source.shape[axis]:
                # Shift is larger than the array — all zeros
                src_slices = []
                break
            if delta >= 0:
                # Shift right/down: source[0:N-delta] → dest[delta:N]
                src_slices.append(slice(0, source.shape[axis] - delta))
                dst_slices.append(slice(delta, source.shape[axis]))
            else:
                # Shift left/up: source[-delta:N] → dest[0:N+delta]
                src_slices.append(slice(-delta, source.shape[axis]))
                dst_slices.append(slice(0, source.shape[axis] + delta))

        if not src_slices:
            shifted.fill(False)
        else:
            shifted[tuple(dst_slices)] = source[tuple(src_slices)]

        # Combine shifted copy into result
        if operation == "dilate":
            result |= shifted  # Dilation: any shifted copy with True → True
        else:
            result &= shifted  # Erosion: all shifted copies must be True

    return result


def dilate(mask: np.ndarray, radius_voxels: Sequence[float] | float) -> np.ndarray:
    """Expand foreground regions by the given radius.
    """
    return _binary_operation(mask, radius_voxels, "dilate")


def erode(mask: np.ndarray, radius_voxels: Sequence[float] | float) -> np.ndarray:
    """Shrink foreground regions by the given radius.
    """
    return _binary_operation(mask, radius_voxels, "erode")


def opening(mask: np.ndarray, radius_voxels: Sequence[float] | float) -> np.ndarray:
    """Opening = erosion followed by dilation.
    """
    return dilate(erode(mask, radius_voxels), radius_voxels)


def closing(mask: np.ndarray, radius_voxels: Sequence[float] | float) -> np.ndarray:
    """Closing = dilation followed by erosion.
    """
    return erode(dilate(mask, radius_voxels), radius_voxels)
