"""
Single-labelmap storage and metadata for editable segments.

FILE PURPSE:
    This file defines `LabelVolume`, the mutable segmentation mask that
    sits on top of a CTVolume. It stores:
    
    Labls are mutually exclusive — each voxel belongs to at most one label.
    Labl ID 0 is reserved for "background" (unlabeled).

    The labelmap can be memory-mapped for large volumes to avoid allocating
    ~1.7 GB of RAM for an 847×1000×1000 uint16 array.

Created: Phase 1 — Core Data Model

"""

from __future__ import annotations

import logging
import uuid
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass

import numpy as np

from ctseg.core.volume import CTVolume

# Logger for this module — used to report observer callback failures
logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class LabelInfo:
    """User-facing segment properties kept separately from voxel values.

    Each label has:
    - id: A nonzero uint16 value used in the labelmap array (1-65535).
    - name: Human-readable name shown in the UI (must be unique, case-insensitive).
    - color: RGB tuple (0-255) for rendering the label overlay.
    - visible: Whether the label's overlay is shown in the slice views.
    - locked: Whether the label's voxels are protected from edits.
    - order: Display position in the label list panel (0 = top).

  
    """

    id: int  # Label ID used in the uint16 labelmap (1-65535)
    name: str  # Human-readable display name
    color: tuple[int, int, int]  # RGB color for overlay rendering (0-255 each)
    visible: bool = True  # Whether overlay is shown in slice views
    locked: bool = False  # Whether voxels are protected from editing
    order: int = 0  # Display order in the label manager panel

    def __post_init__(self) -> None:
        """Validate display metadata once so renderers can trust it.

        This runs when a LabelInfo is created. It ensures:
        - ID is in the valid uint16 foreground range (1-65535)
        - Name is not empty after stripping whitespace
        - Color is a valid RGB tuple (3 values, each 0-255)
        - Display order is non-negative
        """
        # ID 0 is reserved for background; max is uint16 max (65535)
        if self.id <= 0 or self.id > np.iinfo(np.uint16).max:
            raise ValueError("label ID must be in the uint16 foreground range")

        # Name cannot be blank
        if not self.name.strip():
            raise ValueError("label name must not be empty")

        # Color must be exactly 3 byte values (R, G, B)
        if len(self.color) != 3 or any(not 0 <= value <= 255 for value in self.color):
            raise ValueError("color must be an RGB tuple of byte values")

        # Display order must be non-negative
        if self.order < 0:
            raise ValueError("label display order must be non-negative")

        # Strip whitespace from name and ensure color values are ints
        # (uses object.__setattr__ because the dataclass is frozen)
        object.__setattr__(self, "name", self.name.strip())
        object.__setattr__(self, "color", tuple(int(value) for value in self.color))


class LabelVolume:
    """One uint16 label per voxel, aligned exactly to its reference CT volume.

   

   
    """

    def __init__(self, reference: CTVolume, *, memory_map: bool = True) -> None:
        """Allocate a matching labelmap without eagerly touching gigabytes of RAM.

        Args:
            reference: The CT volume this labelmap is aligned to (same shape).
            memory_map: If True, the labelmap is stored as a memory-mapped file
                        in the volume's cache directory. This is critical for
                        large volumes to avoid ~1.7 GB of RAM usage.
        """
        # Keep a reference to the CT volume (for shape, geometry, cache allocation)
        self.reference = reference

        # Allocate the uint16 labelmap array
        if memory_map:
            # Memory-mapped: stored on disk, paged in by the OS as needed.
            
            self._labels = reference.allocate_cache_array(
                f"labels-{uuid.uuid4().hex}.uint16.npy", np.uint16
            )
        else:
            # In-RAM: for small test volumes or when disk I/O is a concern
            self._labels = np.zeros(reference.shape_zyx, dtype=np.uint16)

       
        self._labels.setflags(write=True)

        # Internal state 
        self._metadata: dict[int, LabelInfo] = {}  # label_id → LabelInfo
        self._voxel_counts: dict[int, int] = {}  # label_id → number of labeled voxels
        self._next_id = 1  # Next available label ID
        self._callbacks: list[Callable[[int], None]] = []  # Observer callbacks
        self._notification_depth = 0
        self._notification_pending = False  # Observer callbacks
        self._flush_pending = False  # Observer callbacks
        self.version = 0  # Incremented on every change

        # Create the first default label so there's always something to paint with
        self.add_label("Label 1", (255, 64, 64))

    @classmethod
    def from_array(
        cls,
        reference: CTVolume,
        labelmap_zyx: np.ndarray,
        label_info: tuple[LabelInfo, ...] = (),
        *,
        memory_map: bool = True,
    ) -> LabelVolume:
        """Build a label store from disk/import data before interactive editing.

        Used when loading a saved labelmap (e.g., from a TIFF export).
        Restores both the voxel data and the label metadata.

        Args:
            reference: The CT volume this labelmap is aligned to.
            labelmap_zyx: Imported uint16 array with label IDs.
            label_info: Pre-existing label metadata to restore.
            memory_map: Whether to use disk-backed storage.
        """
        labels = cls(reference, memory_map=memory_map)

        # Reset the default label created by __init__
        labels._metadata.clear()
        labels._voxel_counts.clear()
        labels._next_id = 1

        # Restore provided label metadata
        for info in label_info:
            labels.restore_label(info)

        # If no labels were provided, create a default one
        if not labels._metadata:
            labels.add_label("Label 1", (255, 64, 64))

        # Load the voxel data into the labelmap
        labels._load_initial_array(labelmap_zyx)
        return labels

    def _load_initial_array(self, labelmap_zyx: np.ndarray) -> None:
        """Stream imported label planes into the store without a volume-sized copy.

        Processes one Z-plane at a time to avoid creating a temporary copy of
        the entire volume in RAM. Also validates that all label IDs in the
        imported data have corresponding metadata entries.
        """
        source = np.asanyarray(labelmap_zyx)

        # The imported array must match the reference volume's shape
        if source.shape != self.shape_zyx or source.dtype.kind not in "ui":
            raise ValueError("imported labelmap must be an integer array matching the image shape")

        # Reset voxel counts for all known labels
        self._voxel_counts = {label_id: 0 for label_id in self._metadata}

        # Process one Z-plane at a time to minimize memory usage
        for z_index in range(source.shape[0]):
            plane = source[z_index]

            # Count unique label IDs in this plane
            ids, counts = np.unique(plane, return_counts=True)

            # Validate label ID range (must fit uint16)
            if int(ids.min(initial=0)) < 0 or int(ids.max(initial=0)) > np.iinfo(np.uint16).max:
                raise ValueError("imported label IDs must fit uint16")

            # Update voxel counts and validate that all non-zero IDs are known
            for label_id, count in zip(ids, counts, strict=True):
                if label_id and int(label_id) not in self._metadata:
                    raise ValueError(f"imported labelmap uses undefined label ID {int(label_id)}")
                if label_id:
                    self._voxel_counts[int(label_id)] += int(count)

            # Copy this plane into the labelmap
            self._labels[z_index] = plane

        # Flush to disk if memory-mapped
        if isinstance(self._labels, np.memmap):
            if self._notification_depth:
                self._flush_pending = True
            else:
                self._labels.flush()

        # Notify all observers that the labelmap has changed
        self._notify()

    @property
    def shape_zyx(self) -> tuple[int, int, int]:
        """Return the labelmap dimensions in numpy [z, y, x] order."""
        return tuple(int(value) for value in self._labels.shape)

    @property
    def labelmap(self) -> np.ndarray:
        """Expose a read-only view to prevent edits bypassing undo and notifications.

        All writes to the labelmap should go through `write_region()`, which
        updates voxel counts and notifies observers.
        """
        view = self._labels.view()
        view.setflags(write=False)
        return view

    @property
    def labels(self) -> tuple[LabelInfo, ...]:
        """Return all label metadata sorted by display order, then by ID.

        The UI uses this to populate the label manager panel in the correct
        display order.
        """
        return tuple(sorted(self._metadata.values(), key=lambda info: (info.order, info.id)))

    def add_label(
        self,
        name: str,
        color: tuple[int, int, int],
        label_id: int | None = None,
        *,
        target_id: int | None = None,
    ) -> int:
        """Create a unique nonzero ID, refusing uint16 wraparound.

        Args:
            name: Display name (must be unique case-insensitively).
            color: RGB display color.
            label_id: Optional explicit label ID to assign (e.g. during split operations).
            target_id: Alias for label_id.

        Returns:
            The assigned label ID.
        """
        # Validate name is not empty
        if not name.strip():
            raise ValueError("label name must not be empty")

        # Check for duplicate names (case-insensitive)
        if any(info.name.casefold() == name.casefold() for info in self._metadata.values()):
            raise ValueError(f"a label named {name!r} already exists")

        assigned_id = label_id if label_id is not None else target_id
        if assigned_id is not None:
            if assigned_id in self._metadata:
                raise ValueError(f"a label with ID {assigned_id} already exists")
            if assigned_id <= 0 or assigned_id > np.iinfo(np.uint16).max:
                raise ValueError("label ID must be in the uint16 foreground range")
        else:
            # Check that we haven't exhausted the uint16 ID space
            if self._next_id > np.iinfo(np.uint16).max:
                raise OverflowError("the uint16 label ID space is exhausted")
            assigned_id = self._next_id

        # Place new label at the end of the display order
        next_order = max((info.order for info in self._metadata.values()), default=-1) + 1

        # Create and store the label metadata
        self._metadata[assigned_id] = LabelInfo(assigned_id, name, color, order=next_order)
        self._voxel_counts[assigned_id] = 0
        self._next_id = max(self._next_id, assigned_id + 1)

        self._notify()
        return assigned_id

    def get_label(self, label_id: int) -> LabelInfo:
        """Look up label metadata by ID. Raises KeyError if not found."""
        try:
            return self._metadata[label_id]
        except KeyError as error:
            raise KeyError(f"unknown label ID {label_id}") from error

    def has_label(self, label_id: int) -> bool:
        """Return True if label_id exists in this label volume."""
        return label_id in self._metadata

    def __contains__(self, label_id: int) -> bool:
        """Support 'label_id in label_volume' syntax."""
        return label_id in self._metadata

    @property
    def next_label_id(self) -> int:
        """The next ID that will be assigned by add_label()."""
        return self._next_id

    def voxel_count(self, label_id: int) -> int:
        """Return the number of voxels currently labeled with this ID.

        Validates that the label exists before returning.
        """
        self.get_label(label_id)  # raises KeyError if unknown
        return self._voxel_counts[label_id]

    def restore_label(self, info: LabelInfo) -> None:
        """Restore metadata during redo without scanning the full label volume.

        Used by AddLabelCommand.undo()/redo() and from_array(). Creates the
        metadata entry without touching the voxel array — the voxels are
        handled separately.
        """
        # Prevent duplicate IDs
        if info.id in self._metadata:
            raise ValueError(f"label ID {info.id} already exists")

        # Prevent duplicate names (case-insensitive)
        if any(item.name.casefold() == info.name.casefold() for item in self._metadata.values()):
            raise ValueError(f"a label named {info.name!r} already exists")

        # Prevent duplicate display orders
        if any(item.order == info.order for item in self._metadata.values()):
            raise ValueError(f"label display order {info.order} is already in use")

        # Create a fresh LabelInfo (to re-validate through __post_init__)
        self._metadata[info.id] = LabelInfo(
            info.id, info.name, info.color, info.visible, info.locked, info.order
        )
        self._voxel_counts[info.id] = 0

        # Keep _next_id ahead of all known IDs to prevent collisions
        self._next_id = max(self._next_id, info.id + 1)
        self._notify()

    def get_label_mask(self, label_id: int) -> np.ndarray:
        """Return a read-only boolean mask of all voxels labeled with this ID."""
        self.get_label(label_id)  # validate existence
        mask = self._labels == label_id
        mask.setflags(write=False)
        return mask

    def set_label_properties(
        self,
        label_id: int,
        *,
        name: str | None = None,
        color: tuple[int, int, int] | None = None,
        visible: bool | None = None,
        locked: bool | None = None,
        order: int | None = None,
    ) -> LabelInfo:
        """Update metadata after validating uniqueness and color range.

        Only changes the properties that are explicitly provided (not None).
        Validates that the new name and order don't conflict with other labels.
        """
        current = self.get_label(label_id)

        # Build the updated LabelInfo, keeping current values for None parameters
        updated = LabelInfo(
            id=label_id,
            name=name if name is not None else current.name,
            color=color if color is not None else current.color,
            visible=visible if visible is not None else current.visible,
            locked=locked if locked is not None else current.locked,
            order=order if order is not None else current.order,
        )

        # Check name uniqueness (exclude self)
        if any(
            info.id != label_id and info.name.casefold() == updated.name.casefold()
            for info in self._metadata.values()
        ):
            raise ValueError(f"a label named {updated.name!r} already exists")

       
        if any(
            info.id != label_id and info.order == updated.order for info in self._metadata.values()
        ):
            raise ValueError(f"label display order {updated.order} is already in use")

        self._metadata[label_id] = updated
        self._notify()
        return updated

    def set_label_orders(self, orders: dict[int, int]) -> None:
        """Update several display positions together so observers never see duplicates.

        This is an atomic operation — all orders are validated and applied at once.
        Used by ReorderLabelsCommand.

        Args:
            orders: A dict mapping every current label_id to its new order value.
                    Must include ALL current labels, and orders must be unique non-negative.
        """
        # Must include exactly all current labels
        if set(orders) != set(self._metadata):
            raise ValueError("an order update must include every current label exactly once")

       
        values = tuple(orders.values())
        if any(order < 0 for order in values) or len(set(values)) != len(values):
            raise ValueError("label display orders must be unique non-negative integers")

        # Apply all order changes atomically
        self._metadata = {
            label_id: LabelInfo(
                info.id,
                info.name,
                info.color,
                info.visible,
                info.locked,
                orders[label_id],
            )
            for label_id, info in self._metadata.items()
        }
        self._notify()

    def remove_label(self, label_id: int) -> None:
        """Remove empty metadata only; voxel deletion must use an undoable command.

        Only labels with zero voxels can be removed this way. To remove a label
        that contains voxels, first clear its voxels using an undoable edit,
        then remove the metadata.
        """
        self.get_label(label_id)

        #  refuse to remove labels that still have voxels
        if self._voxel_counts[label_id]:
            raise PermissionError("non-empty labels must be cleared by an undoable edit first")

        del self._metadata[label_id]
        del self._voxel_counts[label_id]
        self._notify()

    def validate_edit(self, label_id: int, *, allow_locked: bool = False) -> LabelInfo:
        """Check whether a label can be edited. Raises PermissionError if locked.

        Args:
            label_id: The label to check. If 0, returns None (background).
            allow_locked: If True, bypasses the lock check (used by undo).
        """
        info = self.get_label(label_id) if label_id else None
        if info is not None and info.locked and not allow_locked:
            raise PermissionError(f"label {label_id} ({info.name}) is locked")
        return info

    def write_region(
        self,
        slices: tuple[slice, slice, slice],
        values: np.ndarray,
        *,
        allow_locked: bool = False,
    ) -> None:
        """Write a bounded region; callers must wrap this in a command.

        This is the central mutation method. ALL changes to the labelmap go
        through here. It:
        1. Validates the region shape matches
        2. Checks that all referenced label IDs are known
        3. Protects locked labels from being overwritten
        4. Updates voxel counts accurately
        5. Flushes to disk if memory-mapped
        6. Notifies all observers

        Args:
            slices: A 3-tuple of slice objects defining the region to write.
            values: New uint16 label values for the region (same shape as sliced region).
            allow_locked: If True, bypasses lock protection (used by undo to restore state).
        """
        region = self._labels[slices]
        candidate = np.asarray(values)

        # Shape must match exactly
        if candidate.shape != region.shape:
            raise ValueError(f"region shape {region.shape} does not match values {candidate.shape}")

       
        if candidate.dtype.kind not in "ui":
            raise TypeError("label values must be integers")

      
        if candidate.size and (int(candidate.min()) < 0 or int(candidate.max()) > 65535):
            raise ValueError("label values must fit uint16")

        # Validate that all label IDs in the new values are known
        present_ids = np.unique(candidate)
        for label_id in present_ids:
            if label_id and np.any((candidate == label_id) & (region != label_id)):
                self.validate_edit(int(label_id), allow_locked=allow_locked)

        
        if not allow_locked:
            current_ids = np.unique(region)
            for label_id in current_ids:
                if label_id and self.get_label(int(label_id)).locked:
                    # Existing locked segments remain protected from broad region writes.
                    # Only flag an error if the locked voxels would actually be changed.
                    changed = (region == label_id) & (candidate != label_id)
                    if np.any(changed):
                        raise PermissionError(f"edit would modify locked label {label_id}")

        # Update voxel counts
        # Subtract counts for what was in this region before
        previous_ids, previous_counts = np.unique(region, return_counts=True)
        new_ids, new_counts = np.unique(candidate, return_counts=True)

        for label_id, count in zip(previous_ids, previous_counts, strict=True):
            if label_id:
                self._voxel_counts[int(label_id)] -= int(count)

        # Add counts for what will be in this region after
        for label_id, count in zip(new_ids, new_counts, strict=True):
            if label_id:
                if int(label_id) not in self._voxel_counts:
                    raise ValueError(f"voxel data references unknown label ID {int(label_id)}")
                self._voxel_counts[int(label_id)] += int(count)

        # Write the new values into the labelmap
        self._labels[slices] = candidate.astype(np.uint16, copy=False)

        # Mark flush pending if memory-mapped; OS page cache writes dirty pages asynchronously
        if isinstance(self._labels, np.memmap):
            self._flush_pending = True

        # Notify all observers that the labelmap has changed
        self._notify()

    def flush(self) -> None:
        #Flush any pending memmap writes to physical storage.
        if isinstance(self._labels, np.memmap) and self._flush_pending:
            self._labels.flush()
            self._flush_pending = False

    def register_change_callback(self, callback: Callable[[int], None]) -> None:
        #Register a function to be called whenever the labelmap changes.
#
        #The callback receives the new version number as its argument.
        #Used by SliceView and VolumeView3D to trigger display refresh.
        #
        if callback not in self._callbacks:
            self._callbacks.append(callback)

    def unregister_change_callback(self, callback: Callable[[int], None]) -> None:
        #Remove a previously registered change callback.
        if callback in self._callbacks:
            self._callbacks.remove(callback)

    def close(self) -> None:
        #Release this label buffer while keeping its reference CT volume alive.
        if isinstance(self._labels, np.memmap):
            if self._flush_pending:
                try:
                    self._labels.flush()
                except Exception:
                    pass
                self._flush_pending = False
            self.reference.release_cache_array(self._labels)

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    @contextmanager
    def batch_updates(self, *, notify: bool = True):
        #Notify renderers once after a multi-slice command has fully applied.
        self._notification_depth += 1
        try:
            yield
        finally:
            self._notification_depth -= 1
            if self._notification_depth == 0:
                if self._flush_pending and isinstance(self._labels, np.memmap):
                    self._labels.flush()
                    self._flush_pending = False
                if self._notification_pending and notify:
                    self._notification_pending = False
                    self._notify()

    def publish_change(self) -> None:
        #Publish a previously silent bulk write from the owning UI thread.#
        if self._notification_depth:
            raise RuntimeError("cannot publish label changes inside a batch update")
        if self._notification_pending:
            self._notification_pending = False
            self._notify()

    def _notify(self) -> None:
        """Increment version and notify all observers.

        Observer failures are logged but never propagated — a failed renderer
        refresh must not undo a valid data commit. This is critical for
        preventing data loss when a UI widget throws an exception.
        """
        if self._notification_depth:
            self._notification_pending = True
            return
        self.version += 1
        for callback in tuple(self._callbacks):  # tuple() to allow removal during iteration
            try:
                callback(self.version)
            except Exception:
                # A failed renderer refresh must be visible, but must not undo a valid data commit.
                logger.exception(
                    "label-volume observer failed while processing version %d", self.version
                )
