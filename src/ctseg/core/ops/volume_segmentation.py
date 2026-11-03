"""Memory-bounded thresholding and seeded 3D connected-region growth."""

from __future__ import annotations

from collections import deque
from uuid import uuid4

import numpy as np

from ctseg.core.gpu import get_cupy
from ctseg.core.volume import CTVolume


def threshold_volume(
    volume: CTVolume,
    lower: float,
    upper: float,
    *,
    calibrated: bool = True,
    mask_volume: Any | None = None,
    mask_label_id: int | None = None,
) -> np.memmap:
    """Create a whole-volume threshold mask one axial plane at a time.

    The returned boolean mask is backed by the volume's temporary cache. This
    prevents a full 3D boolean copy from consuming nearly a gigabyte for the
    professor-provided reconstruction stack. If mask_volume is provided, results
    are strictly constrained to voxels matching mask_label_id (or non-zero).
    """
    if not np.isfinite(lower) or not np.isfinite(upper) or lower > upper:
        raise ValueError("threshold bounds must be finite and lower must not exceed upper")
    mask = volume.allocate_cache_array(f"threshold-{uuid4().hex}.bool.npy", np.bool_)
    completed = False
    try:
        # Memory-bound compare: measured faster on the CPU than shipping every plane over
        # PCIe to the GPU and back, so only compute-bound morphology uses CUDA.
        for z_index in range(volume.depth):
            plane_thresh = _threshold_plane(volume, z_index, lower, upper, calibrated)
            if mask_volume is not None:
                m_arr = getattr(mask_volume, "labelmap", mask_volume)
                if mask_label_id is not None:
                    plane_thresh &= (m_arr[z_index] == mask_label_id)
                else:
                    plane_thresh &= (m_arr[z_index] != 0)
            mask[z_index] = plane_thresh
        mask.flush()
        completed = True
        return mask
    finally:
        if not completed:
            volume.release_cache_array(mask)


def grow_connected_3d(
    volume: CTVolume,
    seed_zyx: tuple[int, int, int],
    lower: float,
    upper: float,
    *,
    connectivity: int = 3,
    calibrated: bool = True,
    mask_volume: Any | None = None,
    mask_label_id: int | None = None,
) -> np.memmap:
    """Find the seed-connected 3D threshold component without labeling 3D RAM.

    
    """
    if lower > upper or not np.isfinite(lower) or not np.isfinite(upper):
        raise ValueError("threshold bounds must be finite and lower must not exceed upper")
    if connectivity not in (1, 2, 3):
        raise ValueError("3D connectivity must be 1 (6), 2 (18), or 3 (26)")
    if len(seed_zyx) != 3:
        raise ValueError("3D seed must be given in z,y,x order")
    seed = tuple(int(index) for index in seed_zyx)
    if any(index < 0 or index >= volume.shape_zyx[axis] for axis, index in enumerate(seed)):
        raise IndexError("seed lies outside the volume")
    if mask_volume is not None:
        m_arr = getattr(mask_volume, "labelmap", mask_volume)
        seed_valid = (m_arr[seed] == mask_label_id) if mask_label_id is not None else (m_arr[seed] != 0)
        if not bool(seed_valid):
            raise ValueError("seed lies outside the restricted region mask")
    seed_plane = volume.get_slice(2, seed[0])
    if calibrated and (volume.intensity_scale != 1.0 or volume.intensity_offset != 0.0):
        seed_intensity = (
            float(seed_plane[seed[1], seed[2]]) * volume.intensity_scale + volume.intensity_offset
        )
    else:
        seed_intensity = float(seed_plane[seed[1], seed[2]])
    if not np.isfinite(seed_intensity) or not lower <= seed_intensity <= upper:
        raise ValueError(
            f"seed intensity {seed_intensity:g} is outside the selected threshold "
            f"[{lower:g}, {upper:g}]"
        )

    def plane_predicate(z_index: int) -> np.ndarray:
        p = _threshold_plane(volume, z_index, lower, upper, calibrated)
        if mask_volume is not None:
            m_arr = getattr(mask_volume, "labelmap", mask_volume)
            if mask_label_id is not None:
                p &= (m_arr[z_index] == mask_label_id)
            else:
                p &= (m_arr[z_index] != 0)
        return p

    return _grow_component_planes(
        volume,
        seed,
        connectivity,
        plane_predicate,
        prefix="grow3d",
    )



def select_label_component_3d(
    label_volume,
    label_id: int,
    seed_zyx: tuple[int, int, int],
    *,
    connectivity: int = 3,
) -> np.memmap:
    """Select the active label's 3D connected component without a 3D int32 copy."""
    label_volume.get_label(label_id)
    volume = label_volume.reference
    if len(seed_zyx) != 3 or any(
        index < 0 or index >= volume.shape_zyx[axis] for axis, index in enumerate(seed_zyx)
    ):
        raise IndexError("component seed lies outside the volume")
    seed = tuple(int(value) for value in seed_zyx)
    if label_volume.labelmap[seed] != label_id:
        raise ValueError("seed is not inside the selected label")
    return _grow_component_planes(
        volume,
        seed,
        connectivity,
        lambda z_index: label_volume.labelmap[z_index] == label_id,
        prefix="component3d",
    )


def _grow_component_planes(
    volume: CTVolume,
    seed: tuple[int, int, int],
    connectivity: int,
    eligible_plane,
    *,
    prefix: str,
) -> np.memmap:
    """Trace one 3D connected component by labeling 2D planes incrementally."""
    if connectivity not in (1, 2, 3):
        raise ValueError("3D connectivity must be 1 (6), 2 (18), or 3 (26)")
    try:
        from scipy import ndimage
    except ImportError as error:
        raise RuntimeError("whole-volume connected-component selection requires SciPy") from error
    height, width = volume.shape_zyx[1:]
    in_plane_structure = ndimage.generate_binary_structure(2, min(connectivity, 2))
    between_plane_structure = ndimage.generate_binary_structure(2, max(connectivity - 1, 0))
    output = volume.allocate_cache_array(f"{prefix}-{uuid4().hex}.bool.npy", np.bool_)
    output[:] = False
    pending: dict[int, np.ndarray] = {seed[0]: np.zeros((height, width), dtype=bool)}
    pending[seed[0]][seed[1], seed[2]] = True
    queued = {seed[0]}
    worklist = deque([seed[0]])
    processed_components: dict[int, set[int]] = {}
    plane_cache: dict[int, tuple[np.ndarray, np.ndarray, int]] = {}

    def get_plane_and_labels(z: int) -> tuple[np.ndarray, np.ndarray, int]:
        cached = plane_cache.get(z)
        if cached is None:
            plane = eligible_plane(z)
            if np.any(plane):
                comp_ids, num_features = ndimage.label(plane, structure=in_plane_structure)
            else:
                comp_ids = np.zeros((height, width), dtype=np.int32)
                num_features = 0
            if len(plane_cache) > 12:
                plane_cache.pop(next(iter(plane_cache)))
            cached = (plane, comp_ids, num_features)
            plane_cache[z] = cached
        return cached

    completed = False
    try:
        while worklist:
            z_index = worklist.popleft()
            queued.discard(z_index)
            seeds = pending.pop(z_index)
            plane, component_ids, num_features = get_plane_and_labels(z_index)
            if num_features == 0:
                continue
            seed_ids = np.unique(component_ids[seeds])
            processed = processed_components.setdefault(z_index, set())
            valid_ids = [int(v) for v in seed_ids if v != 0 and int(v) not in processed]
            if not valid_ids:
                continue
            processed.update(valid_ids)
            if len(valid_ids) == 1:
                combined_component = component_ids == valid_ids[0]
            else:
                lut = np.zeros(num_features + 1, dtype=bool)
                lut[valid_ids] = True
                combined_component = lut[component_ids]
            output[z_index] |= combined_component
            reachable = ndimage.binary_dilation(
                combined_component, structure=between_plane_structure, border_value=0
            )
            for neighbor_z in (z_index - 1, z_index + 1):
                if not 0 <= neighbor_z < volume.depth:
                    continue
                neighbor_plane, _, _ = get_plane_and_labels(neighbor_z)
                new_seeds = reachable & neighbor_plane & ~output[neighbor_z]
                if not np.any(new_seeds):
                    continue
                neighbor_pending = pending.get(neighbor_z)
                if neighbor_pending is None:
                    neighbor_pending = np.zeros((height, width), dtype=bool)
                    pending[neighbor_z] = neighbor_pending
                neighbor_pending |= new_seeds
                if neighbor_z not in queued:
                    worklist.append(neighbor_z)
                    queued.add(neighbor_z)
        output.flush()
        completed = True
        return output
    finally:
        if not completed:
            volume.release_cache_array(output)


def otsu_threshold(
    data: np.ndarray,
    num_bins: int = 256,
) -> float:
    """Calculate the Otsu binarization threshold for an array of intensities."""
    sample = np.asarray(data)
    valid = sample[np.isfinite(sample)]
    if valid.size == 0:
        return 0.0
    if valid.size > 200_000:
        step = max(1, valid.size // 200_000)
        valid = valid[::step]
    min_val, max_val = float(valid.min()), float(valid.max())
    if min_val >= max_val:
        return min_val
    hist, bin_edges = np.histogram(valid, bins=num_bins, range=(min_val, max_val))
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    total = float(valid.size)

    weight1 = np.cumsum(hist) / total
    weight2 = 1.0 - weight1

    sum_total = float(np.sum(hist * bin_centers))
    sum1 = np.cumsum(hist * bin_centers)

    valid_mask = (weight1 > 0.0) & (weight2 > 0.0)
    mean1 = np.zeros_like(bin_centers)
    mean2 = np.zeros_like(bin_centers)
    mean1[valid_mask] = sum1[valid_mask] / (weight1[valid_mask] * total)
    mean2[valid_mask] = (sum_total - sum1[valid_mask]) / (weight2[valid_mask] * total)

    variance = np.zeros_like(bin_centers)
    variance[valid_mask] = (
        weight1[valid_mask] * weight2[valid_mask] * ((mean1[valid_mask] - mean2[valid_mask]) ** 2)
    )
    optimal_idx = int(np.argmax(variance))
    return float(bin_centers[optimal_idx])


def _threshold_plane(
    volume: CTVolume, z_index: int, lower: float, upper: float, calibrated: bool
) -> np.ndarray:
    """Threshold raw pixels directly when possible, avoiding float copies of TIFF planes."""
    raw = volume.get_slice(2, z_index)
    raw_lower, raw_upper = _threshold_bounds(volume, lower, upper, calibrated)
    return np.isfinite(raw) & (raw >= raw_lower) & (raw <= raw_upper)


def _threshold_bounds(
    volume: CTVolume, lower: float, upper: float, calibrated: bool
) -> tuple[float, float]:
    """Convert calibrated threshold limits into raw-pixel bounds."""
    if calibrated and (volume.intensity_scale != 1.0 or volume.intensity_offset != 0.0):
        raw_lower = (lower - volume.intensity_offset) / volume.intensity_scale
        raw_upper = (upper - volume.intensity_offset) / volume.intensity_scale
        if volume.intensity_scale < 0:
            raw_lower, raw_upper = raw_upper, raw_lower
    else:
        raw_lower, raw_upper = lower, upper
    return raw_lower, raw_upper


def label_mask_volume(label_volume, label_id: int) -> np.memmap:
    """Materialize one segment as a disk-backed boolean volume plane by plane."""
    label_volume.get_label(label_id)
    output = label_volume.reference.allocate_cache_array(
        f"label-mask-{uuid4().hex}.bool.npy", np.bool_
    )
    depth = label_volume.shape_zyx[0]
    completed = False
    try:
        for z_index in range(depth):
            output[z_index] = label_volume.labelmap[z_index] == label_id
        output.flush()
        completed = True
        return output
    finally:
        if not completed:
            label_volume.reference.release_cache_array(output)


def morphology_volume_chunked(
    mask: np.ndarray,
    radius_voxels: tuple[float, float, float],
    operation: str,
    volume: CTVolume,
    *,
    chunk_depth: int | None = None,
) -> np.memmap:
    """Run a 3D binary morphology in overlapped axial chunks.

    Overlap equals the Z radius of the structuring element, so chunk boundaries
    match a full-volume SciPy result. Input/output masks may be memory-mapped;
    working memory is bounded by one chunk plus its halo.
    """
    if operation not in {"dilate", "erode", "opening", "closing"}:
        raise ValueError(f"unsupported 3D morphology operation {operation!r}")
    if len(radius_voxels) != 3 or any(
        not np.isfinite(radius) or radius < 0 for radius in radius_voxels
    ):
        raise ValueError("3D radii must contain three finite non-negative values")
    if np.shape(mask) != volume.shape_zyx:
        raise ValueError("3D morphology mask must match the source volume")
    if chunk_depth is None:
        try:
            from ctseg.core.resources import adaptive_chunk_depth

            chunk_depth = adaptive_chunk_depth(volume.shape_zyx, itemsize=1, memory_multiplier=4.0)
        except Exception:
            chunk_depth = 8
    elif chunk_depth <= 0:
        raise ValueError("chunk_depth must be positive")
    from ctseg.core.ops.morphology import structuring_element

    structure = structuring_element(radius_voxels, 3)
    cp = get_cupy()
    if cp is None:
        try:
            from scipy import ndimage
        except ImportError as error:
            raise RuntimeError(
                "3D morphology requires SciPy; install ctseg-toolkit[all]"
            ) from error
        active_ndimage = ndimage
        active_structure = structure
    else:
        from cupyx.scipy import ndimage as active_ndimage

        active_structure = cp.asarray(structure)
    halo = int(np.ceil(radius_voxels[0]))

    def apply_once(source: np.ndarray, name: str) -> np.memmap:
        result = volume.allocate_cache_array(f"morphology-{uuid4().hex}.bool.npy", np.bool_)
        filter_function = (
            active_ndimage.binary_dilation if name == "dilate" else active_ndimage.binary_erosion
        )
        depth = volume.shape_zyx[0]
        completed = False
        try:
            for output_start in range(0, depth, chunk_depth):
                output_stop = min(depth, output_start + chunk_depth)
                input_start = max(0, output_start - halo)
                input_stop = min(depth, output_stop + halo)
                source_chunk = np.asarray(source[input_start:input_stop], dtype=bool)
                if cp is not None:
                    device_chunk = cp.asarray(source_chunk)
                    filtered = filter_function(
                        device_chunk, structure=active_structure, border_value=0
                    )
                    filtered_region = cp.asnumpy(filtered)
                else:
                    filtered_region = filter_function(
                        source_chunk, structure=active_structure, border_value=0
                    )
                local_start = output_start - input_start
                local_stop = local_start + (output_stop - output_start)
                result[output_start:output_stop] = filtered_region[local_start:local_stop]
            result.flush()
            completed = True
            return result
        finally:
            if not completed:
                volume.release_cache_array(result)

    if operation in {"dilate", "erode"}:
        return apply_once(mask, operation)
    first = apply_once(mask, "erode" if operation == "opening" else "dilate")
    try:
        return apply_once(first, "dilate" if operation == "opening" else "erode")
    finally:
        volume.release_cache_array(first)


def slice_seed_to_zyx(
    axis_xyz: int,
    slice_index: int,
    row: int,
    column: int,
) -> tuple[int, int, int]:
    """Convert a click in any orthogonal slice to a canonical 3D ZYX seed."""
    if axis_xyz == 0:
        return row, column, slice_index
    if axis_xyz == 1:
        return row, slice_index, column
    if axis_xyz == 2:
        return slice_index, row, column
    raise ValueError("slice axis must be 0 (X), 1 (Y), or 2 (Z)")


def _plane_links(
    previous: np.ndarray, following: np.ndarray, offsets: list[tuple[int, int]]
) -> np.ndarray:
    """Return unique (id_a, id_b) pairs of foreground components touching across two planes."""
    height, width = previous.shape
    pairs: list[np.ndarray] = []
    for dy, dx in offsets:
        prev_y = slice(max(0, -dy), height - max(0, dy))
        next_y = slice(max(0, dy), height - max(0, -dy))
        prev_x = slice(max(0, -dx), width - max(0, dx))
        next_x = slice(max(0, dx), width - max(0, -dx))
        a = previous[prev_y, prev_x]
        b = following[next_y, next_x]
        touching = (a > 0) & (b > 0)
        if touching.any():
            pairs.append(np.stack([a[touching], b[touching]]))
    if not pairs:
        return np.empty((2, 0), dtype=np.int64)
    return np.unique(np.concatenate(pairs, axis=1), axis=1)


def filter_components_chunked(
    mask: np.ndarray,
    volume: CTVolume,
    *,
    connectivity: int = 3,
    min_size: int = 1,
    keep_largest: int | None = None,
    slab_depth: int | None = None,
) -> np.memmap:
    """Keep 3D connected components by size without a full-volume label array.

    Args:
        connectivity: 1, 2 or 3 for 6-, 18- or 26-neighbour 3D connectivity.
        min_size: components with fewer voxels are removed.
        keep_largest: when given, only this many largest components are kept.
        slab_depth: optional slab depth, adaptively computed if None.
    """
    if np.shape(mask) != volume.shape_zyx:
        raise ValueError("component mask must match the source volume")
    if connectivity not in (1, 2, 3):
        raise ValueError("3D connectivity must be 1 (6), 2 (18), or 3 (26)")
    if slab_depth is None:
        try:
            from ctseg.core.resources import adaptive_chunk_depth

            slab_depth = adaptive_chunk_depth(volume.shape_zyx, itemsize=4, memory_multiplier=5.0)
        except Exception:
            slab_depth = 16
    elif slab_depth < 1:
        raise ValueError("slab_depth must be at least 1")
    if min_size < 1:
        raise ValueError("min_size and slab_depth must be at least 1")
    if keep_largest is not None and keep_largest < 0:
        raise ValueError("keep_largest must be non-negative")
    try:
        from scipy import ndimage
        from scipy.sparse import coo_matrix
        from scipy.sparse.csgraph import connected_components
    except ImportError as error:
        raise RuntimeError(
            "component filtering requires SciPy; install ctseg-toolkit[all]"
        ) from error

    structure = ndimage.generate_binary_structure(3, connectivity)
    # In-plane offsets that are neighbours of a voxel in the adjacent plane.
    offsets = [
        (dy, dx)
        for dy in (-1, 0, 1)
        for dx in (-1, 0, 1)
        if (dy != 0) + (dx != 0) <= connectivity - 1
    ]
    depth = volume.depth

    # Pass 1: label each slab, count component sizes, and record cross-slab contacts.
    counts: list[np.ndarray] = [np.zeros(1, dtype=np.int64)]
    links: list[np.ndarray] = []
    offset = 0
    previous_plane: np.ndarray | None = None
    for start in range(0, depth, slab_depth):
        slab = np.asarray(mask[start : start + slab_depth], dtype=bool)
        local, found = ndimage.label(slab, structure=structure)
        if found:
            counts.append(np.bincount(local.ravel(), minlength=found + 1)[1:].astype(np.int64))
        first = np.where(local[0] > 0, local[0].astype(np.int64) + offset, 0)
        if previous_plane is not None:
            links.append(_plane_links(previous_plane, first, offsets))
        previous_plane = np.where(local[-1] > 0, local[-1].astype(np.int64) + offset, 0)
        offset += found
    total = offset
    sizes = np.concatenate(counts)

    if links and sum(link.shape[1] for link in links):
        edges = np.concatenate(links, axis=1)
        graph = coo_matrix(
            (np.ones(edges.shape[1], dtype=np.int8), (edges[0], edges[1])),
            shape=(total + 1, total + 1),
        )
        _, merged = connected_components(graph, directed=False)
    else:
        merged = np.arange(total + 1)
    merged_sizes = np.bincount(merged, weights=sizes, minlength=int(merged.max()) + 1)
    keep_merged = merged_sizes >= min_size
    if keep_largest is not None:
        largest = np.zeros(len(merged_sizes), dtype=bool)
        order = np.argsort(merged_sizes)[::-1][:keep_largest]
        largest[order[merged_sizes[order] > 0]] = True
        keep_merged &= largest
    keep_node = keep_merged[merged]
    keep_node[0] = False

    # Pass 2: relabel each slab (deterministic) and write only the kept components.
    output = volume.allocate_cache_array(f"components-{uuid4().hex}.bool.npy", np.bool_)
    completed = False
    try:
        offset = 0
        for start in range(0, depth, slab_depth):
            stop = min(depth, start + slab_depth)
            slab = np.asarray(mask[start:stop], dtype=bool)
            local, found = ndimage.label(slab, structure=structure)
            lookup = np.concatenate(([False], keep_node[offset + 1 : offset + found + 1]))
            output[start:stop] = lookup[local]
            offset += found
        output.flush()
        completed = True
        return output
    finally:
        if not completed:
            volume.release_cache_array(output)


def remove_small_islands_volume(
    label_volume,
    label_id: int,
    min_size_voxels: int,
) -> np.memmap:
    """Remove 3D connected components smaller than min_size_voxels for the active label."""
    if min_size_voxels < 0:
        raise ValueError("minimum component size must be non-negative")
    label_volume.get_label(label_id)
    volume = label_volume.reference
    source = label_mask_volume(label_volume, label_id)
    try:
        return filter_components_chunked(source, volume, min_size=max(1, int(min_size_voxels)))
    finally:
        volume.release_cache_array(source)


def keep_largest_components_volume(label_volume, label_id: int, count: int) -> np.memmap:
    """Keep only the N largest 3D connected components of the active label."""
    label_volume.get_label(label_id)
    volume = label_volume.reference
    source = label_mask_volume(label_volume, label_id)
    try:
        return filter_components_chunked(source, volume, keep_largest=int(count))
    finally:
        volume.release_cache_array(source)


def boolean_volume_labels(
    label_volume,
    active_id: int,
    other_id: int,
    operation: str,
) -> np.memmap:
    """Perform whole-volume boolean addition or subtraction between two labels."""
    label_volume.get_label(active_id)
    label_volume.get_label(other_id)
    volume = label_volume.reference
    if operation not in {"add", "erase"}:
        raise ValueError(f"unsupported boolean operation {operation!r}")
    output = volume.allocate_cache_array(f"bool-{uuid4().hex}.bool.npy", np.bool_)
    completed = False
    try:
        for z_index in range(volume.depth):
            plane_active = label_volume.labelmap[z_index] == active_id
            plane_other = label_volume.labelmap[z_index] == other_id
            output[z_index] = (
                plane_active | plane_other if operation == "add" else plane_active & ~plane_other
            )
        output.flush()
        completed = True
        return output
    finally:
        if not completed:
            volume.release_cache_array(output)


def split_label_components_volume(
    label_volume,
    label_id: int | None = None,
    start_new_label_id: int = 2,
    *,
    target_label_id: int | None = None,
    connectivity: int = 3,
    min_size: int = 1,
    max_components: int | None = None,
    slab_depth: int | None = None,
) -> tuple[np.memmap, list[int], list[int]]:
    """Split 3D connected components of label_id into separate distinct labels.

    Args:
        label_volume: LabelVolume containing the label to split.
        label_id: Label ID to split (or target_label_id).
        start_new_label_id: First available label ID to assign for new components.
        connectivity: 1 (6), 2 (18), or 3 (26) neighbor connectivity.
        min_size: Minimum component voxel size to retain.
        max_components: Maximum number of components to split (None = all >= min_size).
        slab_depth: Slab chunk depth for streaming.

    Returns:
        (output_labelmap_memmap, new_label_ids, component_sizes)
    """
    if label_id is None:
        if target_label_id is not None:
            label_id = target_label_id
        else:
            raise ValueError("label_id must be provided")
    if connectivity not in (1, 2, 3):
        raise ValueError("3D connectivity must be 1 (6), 2 (18), or 3 (26)")
    if min_size < 1:
        raise ValueError("min_size must be at least 1")
    if max_components is not None and max_components < 1:
        raise ValueError("max_components must be at least 1")

    label_volume.get_label(label_id)
    volume = label_volume.reference
    mask = label_mask_volume(label_volume, label_id)

    try:
        from scipy import ndimage
        from scipy.sparse import coo_matrix
        from scipy.sparse.csgraph import connected_components
    except ImportError as error:
        volume.release_cache_array(mask)
        raise RuntimeError("component splitting requires SciPy") from error

    if slab_depth is None:
        try:
            from ctseg.core.resources import adaptive_chunk_depth

            slab_depth = adaptive_chunk_depth(volume.shape_zyx, itemsize=4, memory_multiplier=5.0)
        except Exception:
            slab_depth = 16

    structure = ndimage.generate_binary_structure(3, connectivity)
    offsets = [
        (dy, dx)
        for dy in (-1, 0, 1)
        for dx in (-1, 0, 1)
        if (dy != 0) + (dx != 0) <= connectivity - 1
    ]
    depth = volume.depth

    try:
        # Pass 1: Label slabs and link graph across boundaries
        counts: list[np.ndarray] = [np.zeros(1, dtype=np.int64)]
        links: list[np.ndarray] = []
        offset = 0
        previous_plane: np.ndarray | None = None

        for start in range(0, depth, slab_depth):
            slab = np.asarray(mask[start : start + slab_depth], dtype=bool)
            local, found = ndimage.label(slab, structure=structure)
            if found:
                counts.append(np.bincount(local.ravel(), minlength=found + 1)[1:].astype(np.int64))
            first = np.where(local[0] > 0, local[0].astype(np.int64) + offset, 0)
            if previous_plane is not None:
                links.append(_plane_links(previous_plane, first, offsets))
            previous_plane = np.where(local[-1] > 0, local[-1].astype(np.int64) + offset, 0)
            offset += found

        total = offset
        sizes = np.concatenate(counts)

        if links and sum(link.shape[1] for link in links):
            edges = np.concatenate(links, axis=1)
            graph = coo_matrix(
                (np.ones(edges.shape[1], dtype=np.int8), (edges[0], edges[1])),
                shape=(total + 1, total + 1),
            )
            _, merged = connected_components(graph, directed=False)
        else:
            merged = np.arange(total + 1)

        merged_sizes = np.bincount(merged, weights=sizes, minlength=int(merged.max()) + 1)

        valid_indices = [
            idx for idx in range(1, len(merged_sizes)) if merged_sizes[idx] >= min_size
        ]
        valid_indices.sort(key=lambda idx: merged_sizes[idx], reverse=True)

        if max_components is not None:
            valid_indices = valid_indices[:max_components]

        target_ids = {}
        new_label_ids = []
        comp_sizes = []
        for i, global_comp_id in enumerate(valid_indices):
            if i == 0:
                target_ids[global_comp_id] = label_id
                comp_sizes.append(int(merged_sizes[global_comp_id]))
            else:
                nid = start_new_label_id + (i - 1)
                target_ids[global_comp_id] = nid
                new_label_ids.append(nid)
                comp_sizes.append(int(merged_sizes[global_comp_id]))

        comp_to_label = np.zeros(len(merged_sizes), dtype=np.uint16)
        for g_id, t_id in target_ids.items():
            comp_to_label[g_id] = t_id
        node_to_label = comp_to_label[merged]
        node_to_label[0] = 0

        output = volume.allocate_cache_array(f"split-labels-{uuid4().hex}.uint16.npy", np.uint16)
        completed = False
        try:
            offset = 0
            for start in range(0, depth, slab_depth):
                stop = min(depth, start + slab_depth)
                slab = np.asarray(mask[start:stop], dtype=bool)
                local, found = ndimage.label(slab, structure=structure)
                lookup = np.concatenate(
                    ([np.uint16(0)], node_to_label[offset + 1 : offset + found + 1])
                )
                relabeled_slab = lookup[local]

                base_slab = np.array(label_volume.labelmap[start:stop], copy=True)
                mask_voxels = base_slab == label_id
                base_slab[mask_voxels] = relabeled_slab[mask_voxels]

                output[start:stop] = base_slab
                offset += found

            output.flush()
            completed = True
            return output, new_label_ids, comp_sizes
        finally:
            if not completed:
                volume.release_cache_array(output)
    finally:
        volume.release_cache_array(mask)


def generate_distinct_label_colors(
    count: int,
    existing_colors: Sequence[tuple[int, int, int]] = (),
) -> list[tuple[int, int, int]]:
    """Generate vibrant, visually distinct RGB colors using golden-ratio hue distribution."""
    import colorsys
    from collections.abc import Sequence

    colors: list[tuple[int, int, int]] = []
    golden_ratio = 0.618033988749895
    hue = 0.23 + (len(existing_colors) * 0.17)
    for _ in range(count):
        hue = (hue + golden_ratio) % 1.0
        r, g, b = colorsys.hsv_to_rgb(hue, 0.82, 0.96)
        colors.append((int(r * 255), int(g * 255), int(b * 255)))
    return colors

