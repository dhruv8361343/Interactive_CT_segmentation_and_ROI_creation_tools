"""
Set operations used to combine segmentation masks.

    Simple boolean set operations on equal-shaped masks. These are utility
    functions used when combining results from different segmentation tools
    (e.g., intersecting a threshold result with a flood fill result).

creted in: Phase 3 — Segmentation Operations
"""

from __future__ import annotations

import numpy as np


def _matching_masks(mask_a: np.ndarray, mask_b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Validate that both masks have the same shape and convert to boolean.

    This is a shared helper that ensures shape compatibility before any
    set operation. Raises ValueError if shapes don't match.
    """
    a = np.asarray(mask_a, dtype=bool)
    b = np.asarray(mask_b, dtype=bool)
    if a.shape != b.shape:
        raise ValueError(f"mask shapes differ: {a.shape} != {b.shape}")
    return a, b


def combine_add(mask_a: np.ndarray, mask_b: np.ndarray) -> np.ndarray:
    """Return the union of two equal-shaped masks (A OR B).

    A voxel is True in the result if it is True in EITHER input.
    Use case: merging two separate segmentation results.
    """
    a, b = _matching_masks(mask_a, mask_b)
    return np.logical_or(a, b)


def combine_subtract(mask_a: np.ndarray, mask_b: np.ndarray) -> np.ndarray:
    #Return voxels present in A and absent from B (A AND NOT B).
    #
    #A voxel is True in the result only if it is True in A and False in B.
    #Use case: removing one region from another (e.g., subtracting a cavity).
    
    a, b = _matching_masks(mask_a, mask_b)
    return np.logical_and(a, np.logical_not(b))


def combine_intersect(mask_a: np.ndarray, mask_b: np.ndarray) -> np.ndarray:
    #Return the intersection of two equal-shaped masks (A AND B).
#
    #A voxel is True in the result only if it is True in BOTH inputs.
    #Use case: constraining a segmentation to a region of interest.
    #
    a, b = _matching_masks(mask_a, mask_b)
    return np.logical_and(a, b)
