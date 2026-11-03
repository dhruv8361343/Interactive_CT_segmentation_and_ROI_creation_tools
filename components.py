"""Connected-component labeling and small-island cleanup."""

from __future__ import annotations

from collections import deque
from itertools import product

import numpy as np


def label_components(mask: np.ndarray, connectivity: int | None = None) -> np.ndarray:
    """Label foreground components using face/edge/corner connectivity."""
    foreground = np.asarray(mask, dtype=bool)
    if foreground.ndim not in (2, 3):
        raise ValueError("connected components accept only 2D or 3D masks")
    if connectivity is None:
        connectivity = foreground.ndim
    if connectivity < 1 or connectivity > foreground.ndim:
        raise ValueError(f"connectivity must be between 1 and {foreground.ndim}")
    try:
        from scipy import ndimage
    except ImportError:
        ndimage = None
    if ndimage is not None:
        structure = ndimage.generate_binary_structure(foreground.ndim, connectivity)
        return ndimage.label(foreground, structure=structure)[0].astype(np.int32, copy=False)
    output = np.zeros(foreground.shape, dtype=np.int32)
    offsets = [
        offset
        for offset in product((-1, 0, 1), repeat=foreground.ndim)
        if any(offset) and sum(value != 0 for value in offset) <= connectivity
    ]
    component_id = 0
    for seed in zip(*np.nonzero(foreground), strict=True):
        if output[seed]:
            continue
        component_id += 1
        output[seed] = component_id
        queue = deque([seed])
        while queue:
            current = queue.popleft()
            for offset in offsets:
                neighbor = tuple(current[axis] + offset[axis] for axis in range(foreground.ndim))
                if all(
                    0 <= neighbor[axis] < foreground.shape[axis] for axis in range(foreground.ndim)
                ):
                    if foreground[neighbor] and not output[neighbor]:
                        output[neighbor] = component_id
                        queue.append(neighbor)
    return output


def keep_largest_n(labeled: np.ndarray, n: int) -> np.ndarray:
    """Retain the N largest nonzero component IDs."""
    labels = np.asarray(labeled)
    if labels.dtype.kind not in "iu" or n < 0:
        raise ValueError("component labels must be integers and n must be non-negative")
    if n == 0 or labels.size == 0:
        return np.zeros(labels.shape, dtype=bool)
    max_label = int(labels.max())
    if max_label <= 0:
        return np.zeros(labels.shape, dtype=bool)
    counts = np.bincount(labels.ravel())
    counts[0] = 0
    if len(counts) <= 1:
        return np.zeros(labels.shape, dtype=bool)
    sorted_ids = np.argsort(counts)[::-1]
    keep = [idx for idx in sorted_ids[:n] if counts[idx] > 0]
    if not keep:
        return np.zeros(labels.shape, dtype=bool)
    if len(keep) == 1:
        return labels == keep[0]
    lut = np.zeros(len(counts), dtype=bool)
    lut[keep] = True
    return lut[labels]


def select_component(
    mask: np.ndarray,
    seed: tuple[int, ...],
    connectivity: int | None = None,
) -> np.ndarray:
    """Return only the foreground component containing a selected voxel."""
    foreground = np.asarray(mask, dtype=bool)
    if foreground.ndim not in (2, 3) or len(seed) != foreground.ndim:
        raise ValueError("component selection requires a 2D/3D mask and matching seed")
    if any(index < 0 or index >= foreground.shape[axis] for axis, index in enumerate(seed)):
        raise IndexError("component seed lies outside the mask")
    if not foreground[seed]:
        return np.zeros(foreground.shape, dtype=bool)
    components = label_components(foreground, connectivity)
    component_id = components[seed]
    return components == component_id


def remove_small_islands(mask: np.ndarray, min_size_voxels: int) -> np.ndarray:
    """Remove small connected noise components common in pore/defect data."""
    if min_size_voxels < 0:
        raise ValueError("minimum component size must be non-negative")
    if min_size_voxels <= 1:
        return np.asarray(mask, dtype=bool).copy()
    labeled = label_components(mask)
    max_label = int(labeled.max())
    if max_label <= 0:
        return np.zeros(mask.shape, dtype=bool)
    counts = np.bincount(labeled.ravel())
    counts[0] = 0
    keep_ids = np.flatnonzero(counts >= min_size_voxels)
    if len(keep_ids) == 0:
        return np.zeros(mask.shape, dtype=bool)
    if len(keep_ids) == 1:
        return labeled == keep_ids[0]
    lut = np.zeros(len(counts), dtype=bool)
    lut[keep_ids] = True
    return lut[labeled]


def split_components_2d(
    mask: np.ndarray,
    *,
    min_size: int = 1,
    max_components: int | None = None,
    connectivity: int = 2,
) -> list[tuple[np.ndarray, int]]:
    """Split a 2D boolean mask into individual component masks sorted by size descending.

    Returns:
        List of (component_mask, voxel_count) tuples for components >= min_size.
    """
    foreground = np.asarray(mask, dtype=bool)
    if not np.any(foreground):
        return []
    labeled = label_components(foreground, connectivity=connectivity)
    max_label = int(labeled.max())
    if max_label <= 0:
        return []
    counts = np.bincount(labeled.ravel())
    counts[0] = 0
    sorted_ids = np.argsort(counts)[::-1]
    valid_ids = [int(idx) for idx in sorted_ids if counts[idx] >= min_size]
    if max_components is not None and max_components > 0:
        valid_ids = valid_ids[:max_components]
    return [(labeled == cid, int(counts[cid])) for cid in valid_ids]

