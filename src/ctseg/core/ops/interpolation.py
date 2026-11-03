"""Slice propagation and signed-distance contour interpolation."""

from __future__ import annotations

import uuid
from collections.abc import Sequence

import numpy as np


def _numpy_axis(axis_xyz: int, ndim: int) -> int:
    if ndim != 3 or axis_xyz not in (0, 1, 2):
        raise ValueError("propagation requires a 3D labelmap and axis 0, 1, or 2")
    return 2 - axis_xyz


def propagate_nearest(
    label_volume: object,
    label_id: int,
    source_slice: int,
    target_range: tuple[int, int],
    axis: int,
) -> np.ndarray:
    """Copy one labeled slice through an inclusive target range."""
    data = np.asarray(getattr(label_volume, "labelmap"))
    np_axis = _numpy_axis(axis, data.ndim)
    depth = data.shape[np_axis]
    start, stop = target_range
    if not 0 <= source_slice < depth or not 0 <= start <= stop < depth:
        raise IndexError("source or target slice lies outside the volume")
    source = np.take(data == label_id, source_slice, axis=np_axis)
    result = _allocate_mask(label_volume, data.shape)
    for index in range(start, stop + 1):
        selector = [slice(None)] * data.ndim
        selector[np_axis] = index
        result[tuple(selector)] = source
    return result


def propagate_morphological(
    label_volume: object,
    label_id: int,
    keyframe_slice_indices: Sequence[int],
    axis: int,
) -> np.ndarray:
    """Interpolate keyframes by linearly blending signed distance fields.

    Signed-distance interpolation avoids the abrupt area jump caused by binary
    mask blending and is stable for sparse, gradually changing contours.
    """
    data = np.asarray(getattr(label_volume, "labelmap"))
    np_axis = _numpy_axis(axis, data.ndim)
    indices = sorted(set(int(index) for index in keyframe_slice_indices))
    if len(indices) < 2:
        raise ValueError("at least two distinct keyframe slices are required")
    if indices[0] < 0 or indices[-1] >= data.shape[np_axis]:
        raise IndexError("a keyframe lies outside the volume")
    result = _allocate_mask(label_volume, data.shape)
    for z_index in range(data.shape[0]):
        result[z_index] = data[z_index] == label_id
    distance_function = _distance_map
    for first, second in zip(indices[:-1], indices[1:], strict=True):
        plane_a = np.take(data == label_id, first, axis=np_axis)
        plane_b = np.take(data == label_id, second, axis=np_axis)
        distance_a = distance_function(plane_a)
        distance_b = distance_function(plane_b)
        for index in range(first + 1, second):
            fraction = (index - first) / (second - first)
            interpolated = (1.0 - fraction) * distance_a + fraction * distance_b
            selector = [slice(None)] * data.ndim
            selector[np_axis] = index
            result[tuple(selector)] = interpolated >= 0.0
    return result


def _distance_map(mask: np.ndarray) -> np.ndarray:
    foreground = np.asarray(mask, dtype=bool)
    try:
        from scipy.ndimage import distance_transform_edt
    except ImportError:
        return foreground.astype(np.float32) - (~foreground).astype(np.float32)
    return distance_transform_edt(foreground) - distance_transform_edt(~foreground)


def _allocate_mask(label_volume: object, shape: tuple[int, ...]) -> np.ndarray:
    """Use the owning volume cache so a 3D boolean candidate is not 800 MB of RAM."""
    reference = getattr(label_volume, "reference", None)
    if reference is not None and hasattr(reference, "allocate_cache_array"):
        return reference.allocate_cache_array(f"propagation-{uuid.uuid4().hex}.bool.npy", np.bool_)
    return np.zeros(shape, dtype=bool)
