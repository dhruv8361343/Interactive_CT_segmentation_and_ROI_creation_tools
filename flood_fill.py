"""
Deterministic 2D/3D seeded flood fill.

FILE PURPOSE:
    This file implements intensity-bounded flood fill — the operation that
    finds all connected voxels within an intensity range, starting from a
    seed point clicked by the user.

    The output is always a NEW boolean mask — it never mutates the source
    image or a committed labelmap.

created in: Phase 3 — Segmentation Operations

"""

from __future__ import annotations

from collections import deque
from itertools import product

import numpy as np


def flood_fill(
    image: np.ndarray,
    seed: tuple[int, ...],
    *,
    connectivity: int | None = None,
    lower: float | None = None,
    upper: float | None = None,
    tolerance: float | None = None,
) -> np.ndarray:
    """Return the connected component of eligible pixels containing `seed`.

    Args:
        image: 2D or 3D array of intensity values.
        seed: Tuple of indices (row, col) or (z, y, x) marking the start point.
        connectivity: 1 = face-connected only, 2 = edges, 3 = corners (3D).
                      Defaults to the number of dimensions.
        lower: Minimum intensity threshold (inclusive).
        upper: Maximum intensity threshold (inclusive).
        tolerance: Symmetric tolerance around the seed value.
    """
    source = np.asarray(image)

    # Must be 2D or 3D, and seed dimensionality must match
    if source.ndim not in (2, 3) or len(seed) != source.ndim:
        raise ValueError("flood fill accepts a 2D or 3D array and matching seed")

    # Seed must be within image bounds
    if any(index < 0 or index >= source.shape[axis] for axis, index in enumerate(seed)):
        raise IndexError("seed lies outside the image")

    # Default connectivity: N for N-D (face-connected in all dimensions)
    if connectivity is None:
        connectivity = source.ndim
    if connectivity < 1 or connectivity > source.ndim:
        raise ValueError(f"connectivity must be between 1 and {source.ndim}")

    
    seed_value = float(source[seed])

    # tolerance overrides lower/upper: [seed-tol, seed+tol]
    low = (
        seed_value - tolerance
        if tolerance is not None
        else (float(lower) if lower is not None else seed_value)
    )
    high = (
        seed_value + tolerance
        if tolerance is not None
        else (float(upper) if upper is not None else seed_value)
    )
    if low > high:
        raise ValueError("lower intensity threshold must not exceed upper threshold")

    
    # A pixel is eligible if it's finite and within [low, high]
    eligible = np.isfinite(source) & (source >= low) & (source <= high)

    # If the seed itself isn't eligible, return an empty mask
    if not eligible[seed]:
        return np.zeros(source.shape, dtype=bool)

    
    try:
        from scipy import ndimage
    except ImportError:
        ndimage = None

    if ndimage is not None:
        # SciPy approach: label all connected components in the eligible mask,
        # then return only the component that contains the seed
        structure = ndimage.generate_binary_structure(source.ndim, connectivity)
        components, _ = ndimage.label(eligible, structure=structure)
        component_id = components[seed]
        return components == component_id

   
    # Build neighbor offset list based on connectivity level
    # connectivity=1: only face neighbors (offsets with exactly 1 nonzero)
    # connectivity=2: face + edge neighbors (up to 2 nonzero)
    # connectivity=3: face + edge + corner neighbors (up to 3 nonzero)
    offsets = [
        offset
        for offset in product((-1, 0, 1), repeat=source.ndim)
        if any(offset) and sum(value != 0 for value in offset) <= connectivity
    ]

    # Standard BFS from the seed point
    output = np.zeros(source.shape, dtype=bool)
    output[seed] = True
    queue = deque([seed])

    while queue:
        current = queue.popleft()
        for offset in offsets:
            neighbor = tuple(current[axis] + offset[axis] for axis in range(source.ndim))
            # Check bounds and eligibility
            if all(0 <= neighbor[axis] < source.shape[axis] for axis in range(source.ndim)):
                if eligible[neighbor] and not output[neighbor]:
                    output[neighbor] = True
                    queue.append(neighbor)

    return output


def flood_fill_2d(
    slice_array: np.ndarray,
    seed: tuple[int, int],
    connectivity: int = 4,
    lower: float | None = None,
    upper: float | None = None,
    tolerance: float | None = None,
) -> np.ndarray:
    """Compatibility helper for explicit 4- or 8-neighbor 2D filling.

    Maps the familiar 4/8-connectivity terminology to the internal
    connectivity parameter (1 = face-only = 4-connected, 2 = edges = 8-connected).
    """
    if connectivity not in (4, 8):
        raise ValueError("2D connectivity must be 4 or 8")
    # 4-connected = connectivity=1 (only horizontal/vertical neighbors)
    # 8-connected = connectivity=2 (includes diagonal neighbors)
    return flood_fill(
        slice_array,
        seed,
        connectivity=1 if connectivity == 4 else 2,
        lower=lower,
        upper=upper,
        tolerance=tolerance,
    )
