"""
Spacing-aware brush stamps and command-backed label edits.

FILE PURpOSE:
    This file contains all the paint/drawing operations for the segmentation
    toolkit. It provides:

    Every paint operation ends with a call to command_stack.commit_region(),
    which creates a VoxelDiffCommand for undo/redo. This is the only path
    that mutates the labelmap — all changes go through the command system.

creted in: Phase 3 — Segmentation Operations

"""

from __future__ import annotations

from collections.abc import Sequence
from math import ceil, prod

import numpy as np

from ctseg.core.commands import CommandStack
from ctseg.core.label_volume import LabelVolume


def make_stamp(shape: str, radius_voxels: Sequence[float] | float) -> np.ndarray:
    """Return a centered boolean 2D/3D brush stamp.

    Args:
        shape: Brush shape — "circle"/"ellipse" (2D), "ball"/"sphere" (3D),
               or "box"/"square"/"rectangle" (axis-aligned).
        radius_voxels: Radius in voxels. Either a single float (isotropic)
                       or 2/3 floats for per-axis radii.

    Returns:
        A boolean numpy array (2D or 3D) representing the brush footprint.
    """
    # Normalize scalar radius to a 2D tuple
    if isinstance(radius_voxels, (float, int)):
        radii = (float(radius_voxels),) * 2
    else:
        radii = tuple(float(radius) for radius in radius_voxels)

    # Validate: 2 radii for 2D stamps, 3 for 3D; all must be finite and non-negative
    if len(radii) not in (2, 3) or any(not np.isfinite(r) or r < 0 for r in radii):
        raise ValueError("radius_voxels must contain two or three finite non-negative radii")

    # Validate shape name
    kind = shape.casefold()
    if kind not in {"circle", "ellipse", "ball", "sphere", "box", "square", "rectangle"}:
        raise ValueError(f"unsupported brush shape {shape!r}")

    # Convert floating-point radii to integer extents (ceil to ensure coverage)
    extents = tuple(ceil(radius) for radius in radii)
    stamp_voxels = prod(2 * extent + 1 for extent in extents)
    if stamp_voxels > 4_000_000:
        raise ValueError(
            f"brush stamp would contain {stamp_voxels:,} voxels; reduce the brush radius"
        )

    # Build coordinate grids centered at 0 (e.g., [-3, -2, -1, 0, 1, 2, 3] for extent=3)
    # np.ogrid creates open grids (1D arrays) that broadcast correctly for distance calculation
    coordinates = np.ogrid[tuple(slice(-extent, extent + 1) for extent in extents)]

    # Box stamps: everything inside the bounding box is True
    if kind in {"box", "square", "rectangle"}:
        return np.ones(tuple(2 * extent + 1 for extent in extents), dtype=bool)

    # Elliptical stamps: compute normalized distance from center
    # A point is inside the ellipse if sum((coord_i / radius_i)^2) <= 1
    distance = np.zeros(tuple(2 * extent + 1 for extent in extents), dtype=np.float64)
    for axis, radius in enumerate(radii):
        coordinate = coordinates[axis]
        if radius == 0:
            # Zero radius: only the center voxel is valid along this axis
            distance += np.where(coordinate == 0, 0.0, np.inf)
        else:
            # Add squared normalized distance for this axis
            distance += (coordinate / radius) ** 2

    # 1e-12 tolerance avoids floating-point boundary exclusions
    return distance <= 1.0 + 1e-12


def apply_stamp(
    label_volume: LabelVolume,
    center_index_zyx: Sequence[int],
    stamp: np.ndarray,
    value: int,
    command_stack: CommandStack,
    *,
    active_label_id: int | None = None,
    overwrite: bool = False,
) -> bool:
    """Apply one 3D stamp as a bounded, reversible region edit.

    Args:
        label_volume: The labelmap to edit.
        center_index_zyx: 3D center position for the stamp in [z,y,x] order.
        stamp: A 3D boolean brush stamp from make_stamp().
        value: Label ID to paint (0 = erase).
        command_stack: Undo/redo stack to commit to.
        active_label_id: When erasing (value=0), only erase this label's voxels.
        overwrite: If True, paint over any existing label (not just background).

    Returns:
        True if any voxels were changed.
    """
    if len(center_index_zyx) != 3 or stamp.ndim != 3 or not np.any(stamp):
        raise ValueError("a non-empty 3D stamp and a z,y,x center are required")
    if not 0 <= value <= 65535:
        raise ValueError("label value must fit uint16")

    # Check that the target label is not locked
    if value:
        label_volume.validate_edit(value)

    # Convert center to integers
    center = tuple(int(index) for index in center_index_zyx)

    # Calculate where the stamp starts in volume space (top-left corner)
    starts = tuple(center[axis] - stamp.shape[axis] // 2 for axis in range(3))

    # Clip the stamp to volume bounds
    bounds = label_volume.shape_zyx
    low = tuple(max(0, starts[axis]) for axis in range(3))
    high = tuple(min(bounds[axis], starts[axis] + stamp.shape[axis]) for axis in range(3))

    # If the stamp is entirely outside the volume, nothing to do
    if any(low[axis] >= high[axis] for axis in range(3)):
        return False

    # Build slice objects for both the volume region and the stamp sub-region
    volume_slices = tuple(slice(low[axis], high[axis]) for axis in range(3))
    stamp_slices = tuple(
        slice(low[axis] - starts[axis], high[axis] - starts[axis]) for axis in range(3)
    )

    # Extract the clipped stamp footprint
    footprint = stamp[stamp_slices]

    # Read current state of the target region
    before = np.asarray(label_volume.labelmap[volume_slices])
    after = before.copy()

    if value:
        # PAINTING: only write to voxels that are background (0), already this label,
        # or if overwrite mode is enabled
        allowed = footprint & ((before == 0) | (before == value) | bool(overwrite))
        after[allowed] = value
    else:
        # ERASING: clear voxels within the stamp footprint
        erase_mask = footprint
        if active_label_id is not None:
            # Scoped erase: only erase voxels of the active label
            label_volume.validate_edit(active_label_id)
            erase_mask = erase_mask & (before == active_label_id)
        after[erase_mask] = 0

    # Commit the change through the undo system
    return (
        command_stack.commit_region(
            label_volume, volume_slices, after, description="Paint brush stroke"
        )
        is not None
    )


def apply_stroke_3d(
    label_volume: LabelVolume,
    points_zyx: Sequence[Sequence[int]],
    stamp: np.ndarray,
    value: int,
    command_stack: CommandStack,
    *,
    active_label_id: int | None = None,
    overwrite: bool = False,
    max_interpolated_points: int = 250_000,
) -> bool:
    """Interpolate a 3D pointer stroke and commit one bounded undo command."""
    if stamp.ndim != 3 or not np.any(stamp):
        raise ValueError("3D stroke requires a non-empty 3D brush stamp")
    if not 0 <= value <= np.iinfo(np.uint16).max:
        raise ValueError("label value must fit uint16")
    points = np.asarray(points_zyx, dtype=np.int64)
    if points.ndim != 2 or points.shape[1] != 3 or points.shape[0] == 0:
        raise ValueError("3D stroke points must be a non-empty sequence of z,y,x coordinates")
    if value:
        label_volume.validate_edit(value)
    if value == 0 and active_label_id is not None:
        label_volume.validate_edit(active_label_id)

    volume_shape = np.asarray(label_volume.shape_zyx, dtype=np.int64)
    valid = np.all((points >= 0) & (points < volume_shape), axis=1)
    points = points[valid]
    if points.size == 0:
        return False

    stamp_shape = np.asarray(stamp.shape, dtype=np.int64)
    radii = stamp_shape // 2
    low = np.maximum(points.min(axis=0) - radii, 0)
    high = np.minimum(points.max(axis=0) + (stamp_shape - radii), volume_shape)
    if np.any(low >= high):
        return False
    slices = tuple(slice(int(low[axis]), int(high[axis])) for axis in range(3))
    region_shape = tuple(int(high[axis] - low[axis]) for axis in range(3))
    command_stack.ensure_capacity(int(np.prod(region_shape, dtype=np.int64)) * 4, "3D brush stroke")

    path_mask = np.zeros(region_shape, dtype=bool)
    local_points = points - low
    interpolated_count = 1
    for first, second in zip(local_points[:-1], local_points[1:], strict=False):
        distance = int(np.max(np.abs(second - first)))
        interpolated_count += distance + 1
        if interpolated_count > max_interpolated_points:
            raise ValueError("3D stroke is too long to interpolate safely; split the stroke")
        steps = max(distance, 1)
        for step in range(steps + 1):
            center = np.rint(first + (second - first) * (step / steps)).astype(np.int64)
            _stamp_into_volume(path_mask, center, stamp)
    if len(local_points) == 1:
        _stamp_into_volume(path_mask, local_points[0], stamp)

    before = np.array(label_volume.labelmap[slices], copy=True)
    after = before.copy()
    if value:
        allowed = path_mask & ((before == 0) | (before == value) | overwrite)
        after[allowed] = value
    else:
        erase_mask = path_mask
        if active_label_id is not None:
            erase_mask = erase_mask & (before == active_label_id)
        after[erase_mask] = 0
    return (
        command_stack.commit_region(
            label_volume,
            slices,
            after,
            description="3D brush stroke",
        )
        is not None
    )


def _stamp_into_volume(target: np.ndarray, center_zyx: np.ndarray, stamp: np.ndarray) -> None:
    """OR one 3D stamp into the crop-local stroke mask, clipping at its bounds."""
    starts = tuple(int(center_zyx[axis]) - stamp.shape[axis] // 2 for axis in range(3))
    low = tuple(max(0, starts[axis]) for axis in range(3))
    high = tuple(min(target.shape[axis], starts[axis] + stamp.shape[axis]) for axis in range(3))
    if any(low[axis] >= high[axis] for axis in range(3)):
        return
    target_slices = tuple(slice(low[axis], high[axis]) for axis in range(3))
    stamp_slices = tuple(
        slice(low[axis] - starts[axis], high[axis] - starts[axis]) for axis in range(3)
    )
    target[target_slices] |= stamp[stamp_slices]


def apply_stamp_2d(
    label_volume: LabelVolume,
    axis: int,
    plane_index: int,
    center_yx: Sequence[int],
    stamp: np.ndarray,
    value: int,
    command_stack: CommandStack,
    *,
    active_label_id: int | None = None,
    overwrite: bool = False,
) -> bool:
    """Apply a 2D stamp on X, Y, or Z without scanning the whole volume.

    Args:
        label_volume: The labelmap to edit.
        axis: Image axis (0=X sagittal, 1=Y coronal, 2=Z axial).
        plane_index: Which slice along that axis.
        center_yx: 2D center position [row, column] within the slice.
        stamp: A 2D boolean brush stamp from make_stamp().
        value: Label ID to paint (0 = erase).
        command_stack: Undo/redo stack.
        active_label_id: When erasing, scope to this label only.
        overwrite: If True, paint over any existing label.
    """
    if axis not in (0, 1, 2) or stamp.ndim != 2 or len(center_yx) != 2:
        raise ValueError("axis, 2D stamp, and 2D center are invalid")

    axis_lengths = label_volume.reference.shape_xyz
    if not 0 <= plane_index < axis_lengths[axis]:
        raise IndexError("slice index outside the selected image axis")

    # Get the shape of the 2D slice plane
    plane_shape = label_volume.reference.get_slice(axis, plane_index).shape

    # Calculate stamp position and clip to plane bounds
    starts = tuple(int(center_yx[i]) - stamp.shape[i] // 2 for i in range(2))
    low = tuple(max(0, starts[i]) for i in range(2))
    high = tuple(min(plane_shape[i], starts[i] + stamp.shape[i]) for i in range(2))

    if any(low[i] >= high[i] for i in range(2)):
        return False

    # Extract the clipped portion of the stamp
    stamp_slices = tuple(slice(low[i] - starts[i], high[i] - starts[i]) for i in range(2))
    footprint = stamp[stamp_slices]

    # Create a full-plane mask and place the stamp into it
    plane_mask = np.zeros(plane_shape, dtype=bool)
    plane_mask[low[0] : high[0], low[1] : high[1]] = footprint

    # Delegate to apply_mask_2d for the actual commit
    return apply_mask_2d(
        label_volume,
        axis,
        plane_index,
        plane_mask,
        value,
        command_stack,
        active_label_id=active_label_id,
        overwrite=overwrite,
        description="Paint 2D brush stroke",
    )


def apply_stroke_2d(
    label_volume: LabelVolume,
    axis: int,
    plane_index: int,
    points_yx: Sequence[Sequence[int]],
    stamp: np.ndarray,
    value: int,
    command_stack: CommandStack,
    *,
    active_label_id: int | None = None,
    overwrite: bool = False,
    mask_label_id: int | None = None,
) -> bool:
    """Interpolate a sampled mouse path and commit the whole stroke once.

    Args:
        label_volume: The labelmap to edit.
        axis: Image axis (0=X, 1=Y, 2=Z).
        plane_index: Which slice.
        points_yx: List of [row, column] mouse positions.
        stamp: 2D brush stamp.
        value: Label ID to paint (0 = erase).
        command_stack: Undo/redo stack.
        active_label_id: Active label id for scoping.
        overwrite: Allow overwriting other labels.
        mask_label_id: If provided, only edit voxels that currently belong to this label.
    """
    if axis not in (0, 1, 2) or stamp.ndim != 2:
        raise ValueError("a 2D stamp and a valid image axis are required")
    if len(points_yx) == 0:
        return False

    points = np.asarray(points_yx, dtype=np.int64)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("stroke points must be row,column pairs")

    # Get the shape of the target slice
    plane_shape = label_volume.reference.get_slice(axis, plane_index).shape

    # Build a combined mask from all interpolated stamp positions
    path_mask = np.zeros(plane_shape, dtype=bool)

    # Interpolate between consecutive mouse sample points
    for start, stop in zip(points[:-1], points[1:], strict=False):
        # Number of intermediate points = max pixel distance in either axis
        distance = int(np.max(np.abs(stop - start)))
        steps = max(distance, 1)
        for step in range(steps + 1):
            # Linear interpolation between start and stop
            point = np.rint(start + (stop - start) * (step / steps)).astype(int)
            _stamp_into(path_mask, point, stamp)

    # Handle single-click (no drag): just stamp at the one point
    if len(points) == 1:
        _stamp_into(path_mask, points[0], stamp)

    # Commit the entire stroke as one undoable operation
    return apply_mask_2d(
        label_volume,
        axis,
        plane_index,
        path_mask,
        value,
        command_stack,
        active_label_id=active_label_id,
        overwrite=overwrite,
        description="Paint 2D brush stroke",
        mask_label_id=mask_label_id,
    )


def apply_mask_2d(
    label_volume: LabelVolume,
    axis: int,
    plane_index: int,
    mask: np.ndarray,
    value: int,
    command_stack: CommandStack,
    *,
    active_label_id: int | None = None,
    overwrite: bool = False,
    description: str = "Edit 2D ROI",
    replace_label: bool = False,
    mask_label_id: int | None = None,
) -> bool:
    """Apply a 2D candidate mask using one tight bounding-box undo command.
    Args:
        label_volume: The labelmap to edit.
        axis: Image axis (0=X, 1=Y, 2=Z).
        plane_index: Which slice along that axis.
        mask: 2D boolean mask (same shape as the slice).
        value: Label ID to paint (0 = erase).
        command_stack: Undo/redo stack.
        active_label_id: When erasing, scope to this label only.
        overwrite: If True, paint over any existing label.
        description: Human-readable description for undo history.
        replace_label: If True and value != 0, replaces the label's entire mask on this slice.
        mask_label_id: If provided, restrict edits strictly inside this label's existing voxels.
    """
    if replace_label and value != 0:
        return replace_label_mask_2d(
            label_volume,
            axis,
            plane_index,
            mask,
            value,
            command_stack,
            overwrite=overwrite,
            description=description,
        )

    if axis not in (0, 1, 2) or not 0 <= value <= 65535:
        raise ValueError("invalid axis or label value")
    if value:
        label_volume.validate_edit(value)

    region_mask = np.asarray(mask, dtype=bool)
    expected_shape = label_volume.reference.get_slice(axis, plane_index).shape
    if region_mask.shape != expected_shape:
        raise ValueError(f"mask shape {region_mask.shape} must match slice {expected_shape}")

    # Find the tight bounding box of the mask (only process non-zero rows/cols)
    rows, columns = np.nonzero(region_mask)
    if rows.size == 0:
        return False  # Empty mask → no-op

    low_y, high_y = int(rows.min()), int(rows.max()) + 1
    low_x, high_x = int(columns.min()), int(columns.max()) + 1

    # Map 2D bounding box to 3D volume slices based on the axis
    # The plane_index becomes a width-1 slice along the appropriate numpy axis
    if axis == 0:
        # Sagittal (X): numpy[:, :, plane_index]
        slices = (slice(low_y, high_y), slice(low_x, high_x), slice(plane_index, plane_index + 1))
    elif axis == 1:
        # Coronal (Y): numpy[:, plane_index, :]
        slices = (slice(low_y, high_y), slice(plane_index, plane_index + 1), slice(low_x, high_x))
    else:
        # Axial (Z): numpy[plane_index, :, :]
        slices = (slice(plane_index, plane_index + 1), slice(low_y, high_y), slice(low_x, high_x))

    # Read and copy the target region (we'll modify the copy)
    region = np.array(label_volume.labelmap[slices], copy=True)

    # Get a 2D view into the 3D region (squeeze out the width-1 axis)
    if axis == 0:
        view = region[:, :, 0]
    elif axis == 1:
        view = region[:, 0, :]
    else:
        view = region[0, :, :]

    # Extract the portion of the mask that falls within the bounding box
    local_mask = region_mask[low_y:high_y, low_x:high_x]

    if mask_label_id is not None:
        # Strict constraint: only modify voxels that belong to mask_label_id
        constrained = local_mask & (view == mask_label_id)
        if value:
            view[constrained] = value
        else:
            view[constrained] = 0
    elif value:
        # PAINTING: only write where mask is True AND the voxel is background/same-label/overwrite
        view[local_mask & ((view == 0) | (view == value) | bool(overwrite))] = value
    else:
        # ERASING: clear masked voxels
        erase_mask = local_mask
        if active_label_id is not None:
            label_volume.validate_edit(active_label_id)
            # Scoped erase: only clear voxels of the active label
            erase_mask = erase_mask & (view == active_label_id)
        view[erase_mask] = 0

    # Commit the changed region through the undo system
    return (
        command_stack.commit_region(label_volume, slices, region, description=description)
        is not None
    )


def replace_label_mask_2d(
    label_volume: LabelVolume,
    axis: int,
    plane_index: int,
    desired_mask: np.ndarray,
    label_id: int,
    command_stack: CommandStack,
    *,
    overwrite: bool = False,
    description: str = "Replace label on slice",
) -> bool:
    """Replace one label's slice mask while preserving every other label.
    Args:
        label_volume: The labelmap to edit.
        axis: Image axis (0=X, 1=Y, 2=Z).
        plane_index: Which slice.
        desired_mask: The new 2D boolean mask for this label on this slice.
        label_id: Which label to replace.
        command_stack: Undo/redo stack.
        description: Human-readable description for undo history.
    """
    label_volume.validate_edit(label_id)
    desired = np.asarray(desired_mask, dtype=bool)

    # Get the current mask for this label on this slice
    current_plane = np.take(label_volume.labelmap, plane_index, axis=2 - axis) == label_id
    if desired.shape != current_plane.shape:
        raise ValueError(f"replacement mask shape must be {current_plane.shape}")

    # The "affected" region includes both old and new mask areas
    affected = current_plane | desired
    rows, columns = np.nonzero(affected)
    if rows.size == 0:
        return False

    # Build tight bounding box
    low_y, high_y = int(rows.min()), int(rows.max()) + 1
    low_x, high_x = int(columns.min()), int(columns.max()) + 1

    # Map to 3D slices (same axis logic as apply_mask_2d)
    if axis == 0:
        slices = (slice(low_y, high_y), slice(low_x, high_x), slice(plane_index, plane_index + 1))
    elif axis == 1:
        slices = (slice(low_y, high_y), slice(plane_index, plane_index + 1), slice(low_x, high_x))
    elif axis == 2:
        slices = (slice(plane_index, plane_index + 1), slice(low_y, high_y), slice(low_x, high_x))
    else:
        raise ValueError("axis must be 0, 1, or 2")

    # Read and copy the region
    region = np.array(label_volume.labelmap[slices], copy=True)
    if axis == 0:
        view = region[:, :, 0]
    elif axis == 1:
        view = region[:, 0, :]
    else:
        view = region[0, :, :]

    # Step 1: Clear all voxels that currently have this label
    view[view == label_id] = 0

    # Step 2: Paint the desired mask, but only where it won't overwrite other labels
    local_desired = desired[low_y:high_y, low_x:high_x]
    allowed = np.ones(view.shape, dtype=bool) if overwrite else ((view == 0) | (view == label_id))
    view[local_desired & allowed] = label_id

    # Commit through the undo system
    return (
        command_stack.commit_region(label_volume, slices, region, description=description)
        is not None
    )


def apply_volume_mask(
    label_volume: LabelVolume,
    desired_mask: np.ndarray,
    label_id: int,
    command_stack: CommandStack,
    *,
    replace_range: tuple[int, int, int] | None = None,
    overwrite: bool = False,
    replace_active: bool = True,
    description: str = "Apply 3D mask",
) -> bool:
    """Commit a volumetric candidate as one command over its touched bounding box.
    Args:
        label_volume: The labelmap to edit.
        desired_mask: Full-volume 3D boolean mask.
        label_id: Label ID to apply.
        command_stack: Undo/redo stack.
        replace_range: Optional (axis_xyz, first, last) to limit the operation scope.
        overwrite: If True, paint over any existing label.
        description: Human-readable description for undo history.
    """
    label_volume.validate_edit(label_id)
    candidate = np.asarray(desired_mask, dtype=bool)
    if candidate.shape != label_volume.shape_zyx:
        raise ValueError(f"3D mask shape {candidate.shape} must match {label_volume.shape_zyx}")
    if replace_range is None:
        return (
            command_stack.commit_volume_mask(
                label_volume,
                candidate,
                label_id,
                overwrite=overwrite,
                replace_active=replace_active,
                description=description,
            )
            is not None
        )

    # Initialize bounds to full volume (may be narrowed by replace_range)
    z0, z1 = 0, candidate.shape[0]
    y0, y1 = 0, candidate.shape[1]
    x0, x1 = 0, candidate.shape[2]
    selector: tuple[int, int] | None = None

    # Optionally restrict to a sub-range of slices
    if replace_range is not None:
        axis_xyz, first, last = replace_range
        if axis_xyz not in (0, 1, 2):
            raise ValueError("replacement axis must be 0, 1, or 2")
        # Convert image-space axis to numpy axis (X→2, Y→1, Z→0)
        np_axis = 2 - axis_xyz
        axis_length = label_volume.reference.shape_xyz[axis_xyz]
        if not 0 <= first <= last < axis_length:
            raise IndexError("replacement slice range is outside the image")
        selector = (np_axis, first, last + 1)
        # Narrow the scan range along the appropriate numpy axis
        if np_axis == 0:
            z0, z1 = first, last + 1
        elif np_axis == 1:
            y0, y1 = first, last + 1
        else:
            x0, x1 = first, last + 1

    # --- Chunk scan to find tight 3D bounding box ---
    # Process one Z-plane at a time to avoid creating huge index arrays
    found = False
    z_start, z_stop = z0, z1
    y_start, y_stop = y0, y1
    x_start, x_stop = x0, x1

    for z_index in range(z0, z1):
        # Read current and incoming planes (respecting replace_range sub-slicing)
        if selector is not None and selector[0] == 1:
            current = label_volume.labelmap[z_index, y0:y1, :]
            incoming = candidate[z_index, y0:y1, :]
        elif selector is not None and selector[0] == 2:
            current = label_volume.labelmap[z_index, :, x0:x1]
            incoming = candidate[z_index, :, x0:x1]
        else:
            current = label_volume.labelmap[z_index]
            incoming = candidate[z_index]

        # A voxel is "affected" if it's in the new mask OR currently has this label
        affected = incoming | (current == label_id)

        # Find which rows and columns have affected voxels
        occupied_rows = np.flatnonzero(np.any(affected, axis=1))
        occupied_columns = np.flatnonzero(np.any(affected, axis=0))

        if occupied_rows.size and occupied_columns.size:
            # Adjust for sub-slicing offsets
            row_offset = y0 if selector is not None and selector[0] == 1 else 0
            column_offset = x0 if selector is not None and selector[0] == 2 else 0

            if not found:
                # First affected plane: initialize bounding box
                z_start = z_index
                y_start = int(occupied_rows[0]) + row_offset
                x_start = int(occupied_columns[0]) + column_offset
                z_stop = z_index + 1
                y_stop = int(occupied_rows[-1]) + 1 + row_offset
                x_stop = int(occupied_columns[-1]) + 1 + column_offset
                found = True
            else:
                # Subsequent affected planes: expand bounding box
                z_stop = z_index + 1
                y_start = min(y_start, int(occupied_rows[0]) + row_offset)
                y_stop = max(y_stop, int(occupied_rows[-1]) + 1 + row_offset)
                x_start = min(x_start, int(occupied_columns[0]) + column_offset)
                x_stop = max(x_stop, int(occupied_columns[-1]) + 1 + column_offset)

    if not found:
        return False  # No affected voxels anywhere

    # Build the tight 3D bounding box
    slices = (slice(z_start, z_stop), slice(y_start, y_stop), slice(x_start, x_stop))
    region_shape = tuple(s.stop - s.start for s in slices)

    # Check memory budget before allocating (2 copies × uint16 = 4 bytes per voxel)
    command_stack.ensure_capacity(
        int(np.prod(region_shape, dtype=np.int64)) * 4,
        description,
    )

    # Read and modify the bounded region
    region = np.array(label_volume.labelmap[slices], copy=True)
    mask = candidate[slices]

    # Clear existing voxels of this label within the region
    if replace_active:
        region[region == label_id] = 0

    # Determine which voxels can be painted
    allowed = (region == 0) | (region == label_id)
    if overwrite:
        allowed = np.ones(region.shape, dtype=bool)

    # Apply the mask
    region[mask & allowed] = label_id

    # Commit through the undo system
    return (
        command_stack.commit_region(label_volume, slices, region, description=description)
        is not None
    )


def _stamp_into(target: np.ndarray, center_yx: Sequence[int], stamp: np.ndarray) -> None:
    """OR a 2D stamp into a target mask, clipping to bounds.
    """
    starts = tuple(int(center_yx[axis]) - stamp.shape[axis] // 2 for axis in range(2))
    low = tuple(max(0, starts[axis]) for axis in range(2))
    high = tuple(min(target.shape[axis], starts[axis] + stamp.shape[axis]) for axis in range(2))
    if any(low[axis] >= high[axis] for axis in range(2)):
        return
    target_slices = tuple(slice(low[axis], high[axis]) for axis in range(2))
    stamp_slices = tuple(
        slice(low[axis] - starts[axis], high[axis] - starts[axis]) for axis in range(2)
    )
    target[target_slices] |= stamp[stamp_slices]
