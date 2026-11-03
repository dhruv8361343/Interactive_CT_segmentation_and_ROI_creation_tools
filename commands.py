"""
Transactional, bounding-box based undo/redo commands.

FILE PURPOSE:
    This file implements the Command pattern for undo/redo. Every edit in the
    toolkit (painting, morphology, label property changes) is wrapped in a
    Command object that knows how to `apply()` and `undo()` itself.

    The CommandStack manages a bounded history with configurable depth and
    memory limits. This prevents undo from consuming unbounded memory when
    working with large volumes.

    
created in: Phase 4 — Command / Undo System

"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Protocol

import numpy as np

from ctseg.core.label_volume import LabelInfo, LabelVolume


class Command(Protocol):
    """Structural command contract used by the history stack.

    Any object with these four members can be pushed onto the CommandStack.
    Python's Protocol class provides structural (duck) typing — no need to
    inherit from a base class.
    """

    description: str  # Human-readable description (shown in UI if needed)
    memory_bytes: int  # Estimated memory footprint (for budget enforcement)

    def apply(self) -> None: ...  # Execute the edit (or re-apply after undo)
    def undo(self) -> None: ...  # Reverse the edit


@dataclass(slots=True)
class VoxelDiffCommand:
    """Store only the old and new values touched by one committed operation.

    This is the workhorse command. It stores two snapshots of the same
    bounded region:
    - `before`: what the labelmap looked like before the edit
    - `after`: what it looks like after the edit

    Both snapshots are COPIED and made read-only in __post_init__ so that
    subsequent edits or preview overlays cannot corrupt the undo history.

    Memory cost = 2 × region_size × dtype_size (uint16 = 2 bytes per voxel).
    For a 100×100 single-slice edit, that's ~40 KB. For a full-volume
    morphology operation, it could be hundreds of MB.
    """

    label_volume: LabelVolume  # The labelmap this command operates on
    slices: tuple[slice, slice, slice]  # The bounded region (z, y, x slices)
    before: np.ndarray  # Snapshot before the edit
    after: np.ndarray  # Snapshot after the edit
    description: str = "Edit labels"  # Human-readable description
    storage_bytes: int = 0

    def __post_init__(self) -> None:
        """Own immutable snapshots so later previews cannot corrupt history.

        Makes independent copies of both arrays and makes them read-only.
        Also validates that shapes are consistent.
        """
        # Create independent copies so the command owns its data
        self.before = np.array(self.before, copy=True)
        self.after = np.array(self.after, copy=True)

        # Sanity checks: snapshots must have the same shape and match the actual region
        if self.before.shape != self.after.shape:
            raise ValueError("before and after snapshots must have identical shapes")
        if self.before.shape != self.label_volume.labelmap[self.slices].shape:
            raise ValueError("command snapshots must match the touched label region")

        # Make snapshots read-only to prevent accidental modification
        self.before.setflags(write=False)
        self.after.setflags(write=False)

    @property
    def memory_bytes(self) -> int:
        """Total memory used by both snapshots (for undo budget accounting)."""
        return int(self.before.nbytes + self.after.nbytes)

    def apply(self) -> None:
        """Write the 'after' snapshot into the labelmap (do or redo)."""
        self.label_volume.write_region(self.slices, self.after)

    def undo(self) -> None:
        """Write the 'before' snapshot back into the labelmap.

        Uses allow_locked=True because undo must always succeed, even if the
        user locked a label after the original edit was made.
        """
        self.label_volume.write_region(self.slices, self.before, allow_locked=True)


@dataclass(frozen=True, slots=True)
class _DiskPlaneDiff:
    """Cropped before/after values for a region of one axial slice in a diff file."""

    z_index: int
    y_start: int
    y_stop: int
    x_start: int
    x_stop: int
    path: Path
    offset: int
    value_bytes: int


@dataclass(slots=True)
class DiskBackedVoxelDiffCommand:
    """Stream 3D edit snapshots from disk for undo and redo."""

    label_volume: LabelVolume
    records: list[_DiskPlaneDiff]
    cache: TemporaryDirectory[str]
    description: str = "Apply 3D segmentation"
    storage_bytes: int = 0
    applied: bool = False

    @classmethod
    def from_volume_mask(
        cls,
        label_volume: LabelVolume,
        desired_mask: np.ndarray,
        label_id: int,
        *,
        overwrite: bool = False,
        replace_active: bool = False,
        description: str = "Apply 3D segmentation",
    ) -> DiskBackedVoxelDiffCommand | None:
        """Stage changed-plane snapshots into a single diff file."""
        label_volume.validate_edit(label_id)
        candidate = np.asanyarray(desired_mask)
        if candidate.shape != label_volume.shape_zyx or candidate.dtype != np.bool_:
            raise ValueError(f"volume mask must be boolean with shape {label_volume.shape_zyx}")
        cache = TemporaryDirectory(prefix="ctseg-undo-3d-", ignore_cleanup_errors=True)
        diff_path = Path(cache.name) / "plane_diffs.bin"
        records: list[_DiskPlaneDiff] = []
        stored_bytes = 0
        try:
            with open(diff_path, "w+b") as stream:
                for z_index in range(label_volume.shape_zyx[0]):
                    before = np.array(label_volume.labelmap[z_index], copy=True)
                    after = before.copy()
                    if replace_active:
                        after[before == label_id] = 0
                    eligible = (
                        np.ones(before.shape, dtype=bool)
                        if overwrite
                        else ((after == 0) | (after == label_id))
                    )
                    after[candidate[z_index] & eligible] = label_id
                    changed = before != after
                    rows = np.flatnonzero(np.any(changed, axis=1))
                    columns = np.flatnonzero(np.any(changed, axis=0))
                    if not rows.size or not columns.size:
                        continue
                    y_start, y_stop = int(rows[0]), int(rows[-1]) + 1
                    x_start, x_stop = int(columns[0]), int(columns[-1]) + 1
                    before_crop = np.ascontiguousarray(before[y_start:y_stop, x_start:x_stop])
                    after_crop = np.ascontiguousarray(after[y_start:y_stop, x_start:x_stop])
                    value_bytes = int(before_crop.nbytes)
                    offset = stream.tell()
                    stream.write(before_crop.tobytes(order="C"))
                    stream.write(after_crop.tobytes(order="C"))
                    stored_bytes += 2 * value_bytes
                    records.append(
                        _DiskPlaneDiff(
                            z_index,
                            y_start,
                            y_stop,
                            x_start,
                            x_stop,
                            diff_path,
                            offset,
                            value_bytes,
                        )
                    )
            if not records:
                cache.cleanup()
                return None
            return cls(label_volume, records, cache, description, stored_bytes, applied=False)
        except Exception:
            cache.cleanup()
            raise

    @classmethod
    def from_relabeled_volume(
        cls,
        label_volume: LabelVolume,
        new_labelmap_zyx: np.ndarray,
        *,
        description: str = "Split region into components",
    ) -> DiskBackedVoxelDiffCommand | None:
        """Stage changed-plane snapshots where multiple labels were created or modified."""
        cache = TemporaryDirectory(prefix="ctseg-undo-split-", ignore_cleanup_errors=True)
        diff_path = Path(cache.name) / "plane_diffs.bin"
        records: list[_DiskPlaneDiff] = []
        stored_bytes = 0
        try:
            with open(diff_path, "w+b") as stream:
                for z_index in range(label_volume.shape_zyx[0]):
                    before = np.array(label_volume.labelmap[z_index], copy=True)
                    after = np.array(new_labelmap_zyx[z_index], copy=True)
                    changed = before != after
                    rows = np.flatnonzero(np.any(changed, axis=1))
                    columns = np.flatnonzero(np.any(changed, axis=0))
                    if not rows.size or not columns.size:
                        continue
                    y_start, y_stop = int(rows[0]), int(rows[-1]) + 1
                    x_start, x_stop = int(columns[0]), int(columns[-1]) + 1
                    before_crop = np.ascontiguousarray(before[y_start:y_stop, x_start:x_stop])
                    after_crop = np.ascontiguousarray(after[y_start:y_stop, x_start:x_stop])
                    value_bytes = int(before_crop.nbytes)
                    offset = stream.tell()
                    stream.write(before_crop.tobytes(order="C"))
                    stream.write(after_crop.tobytes(order="C"))
                    stored_bytes += 2 * value_bytes
                    records.append(
                        _DiskPlaneDiff(
                            z_index,
                            y_start,
                            y_stop,
                            x_start,
                            x_stop,
                            diff_path,
                            offset,
                            value_bytes,
                        )
                    )
            if not records:
                cache.cleanup()
                return None
            return cls(label_volume, records, cache, description, stored_bytes, applied=False)
        except Exception:
            cache.cleanup()
            raise


    @property
    def memory_bytes(self) -> int:
        return 256 + len(self.records) * 96

    def apply(self) -> None:
        if self.applied:
            return
        self._write_snapshot("after", rollback_snapshot="before")
        self.applied = True

    def apply_silently(self) -> None:
        """Apply on a worker thread and defer view notifications until the UI thread."""
        if self.applied:
            return
        self._write_snapshot("after", rollback_snapshot="before", notify=False)
        self.applied = True

    def undo(self) -> None:
        if not self.applied:
            return
        self._write_snapshot("before", rollback_snapshot="after")
        self.applied = False

    def undo_silently(self) -> None:
        """Restore a partially accepted background command without notifying views."""
        if not self.applied:
            return
        self._write_snapshot("before", rollback_snapshot="after", notify=False)
        self.applied = False

    def dispose(self) -> None:
        """Remove temporary snapshots when history evicts the command."""
        try:
            self.cache.cleanup()
        except OSError:
            pass

    def _write_snapshot(
        self,
        snapshot: str,
        *,
        rollback_snapshot: str,
        notify: bool = True,
    ) -> None:
        completed: list[_DiskPlaneDiff] = []
        try:
            with self.label_volume.batch_updates(notify=notify):
                if self.records:
                    with open(self.records[0].path, "rb") as stream:
                        for record in self.records:
                            offset = (
                                record.offset
                                if snapshot == "before"
                                else record.offset + record.value_bytes
                            )
                            stream.seek(offset)
                            payload = stream.read(record.value_bytes)
                            if len(payload) != record.value_bytes:
                                raise OSError(f"corrupt undo record {record.path}")
                            values = np.frombuffer(payload, dtype=np.uint16).reshape(
                                record.y_stop - record.y_start, record.x_stop - record.x_start
                            )
                            slices = (
                                slice(record.z_index, record.z_index + 1),
                                slice(record.y_start, record.y_stop),
                                slice(record.x_start, record.x_stop),
                            )
                            self.label_volume.write_region(
                                slices,
                                values[np.newaxis, :, :],
                                allow_locked=snapshot == "before",
                            )
                            completed.append(record)
        except Exception:
            # Roll back partially written slices before reporting an I/O or data error.
            with self.label_volume.batch_updates(notify=notify):
                if completed:
                    with open(completed[0].path, "rb") as stream:
                        for record in reversed(completed):
                            offset = (
                                record.offset
                                if rollback_snapshot == "before"
                                else record.offset + record.value_bytes
                            )
                            stream.seek(offset)
                            payload = stream.read(record.value_bytes)
                            values = np.frombuffer(payload, dtype=np.uint16).reshape(
                                record.y_stop - record.y_start, record.x_stop - record.x_start
                            )
                            slices = (
                                slice(record.z_index, record.z_index + 1),
                                slice(record.y_start, record.y_stop),
                                slice(record.x_start, record.x_stop),
                            )
                            self.label_volume.write_region(
                                slices, values[np.newaxis, :, :], allow_locked=True
                            )
            raise

    @staticmethod
    def _read_snapshot(record: _DiskPlaneDiff, snapshot: str) -> np.ndarray:
        with open(record.path, "rb") as stream:
            offset = record.offset if snapshot == "before" else record.offset + record.value_bytes
            stream.seek(offset)
            payload = stream.read(record.value_bytes)
        if len(payload) != record.value_bytes:
            raise OSError(f"corrupt undo record {record.path}")
        return np.frombuffer(
            payload,
            dtype=np.uint16,
        ).reshape(record.y_stop - record.y_start, record.x_stop - record.x_start)


@dataclass(slots=True)
class LabelMetadataCommand:
    """Make one label-property change reversible without copying voxel data.
    """

    label_volume: LabelVolume  # The labelmap this command operates on
    label_id: int  # Which label to modify
    before: LabelInfo  # Properties before the change
    after: LabelInfo  # Properties after the change
    description: str = "Change label properties"

    @property
    def memory_bytes(self) -> int:
        """Metadata commands use negligible memory (string sizes + overhead)."""
        return len(self.before.name.encode("utf-8")) + len(self.after.name.encode("utf-8")) + 64

    def apply(self) -> None:
        """Apply the 'after' properties (do or redo)."""
        self._set(self.after)

    def undo(self) -> None:
        """Restore the 'before' properties."""
        self._set(self.before)

    def _set(self, info: LabelInfo) -> None:
        """Apply a LabelInfo's properties to the label volume."""
        self.label_volume.set_label_properties(
            self.label_id,
            name=info.name,
            color=info.color,
            visible=info.visible,
            locked=info.locked,
            order=info.order,
        )


@dataclass(slots=True)
class AddLabelCommand:
    """Make label creation reversible as a metadata-only operation.

    apply() creates the label, undo() removes it (only works if the label
    is empty — labels with voxels must be cleared first).
    """

    label_volume: LabelVolume
    info: LabelInfo  # The label to create/restore
    description: str = "Add label"

    @property
    def memory_bytes(self) -> int:
        return len(self.info.name.encode("utf-8")) + 64

    def apply(self) -> None:
        """Create the label by restoring its metadata."""
        self.label_volume.restore_label(self.info)

    def undo(self) -> None:
        """Remove the label (only works if it has zero voxels)."""
        self.label_volume.remove_label(self.info.id)


@dataclass(slots=True)
class RemoveEmptyLabelCommand:
    """Remove/restore label metadata only when the segment contains no voxels.

    The inverse of AddLabelCommand: apply() removes, undo() restores.
    """

    label_volume: LabelVolume
    info: LabelInfo  # The label to remove/restore
    description: str = "Remove empty label"

    @property
    def memory_bytes(self) -> int:
        return len(self.info.name.encode("utf-8")) + 64

    def apply(self) -> None:
        """Remove the empty label metadata."""
        self.label_volume.remove_label(self.info.id)

    def undo(self) -> None:
        """Restore the label metadata."""
        self.label_volume.restore_label(self.info)


@dataclass(slots=True)
class ReorderLabelsCommand:
    """Swap label display order as one reversible metadata command.

    Stores the complete before/after order maps so the reorder is atomic
    and fully reversible.
    """

    label_volume: LabelVolume
    before: dict[int, int]  # label_id → old order
    after: dict[int, int]  # label_id → new order
    description: str = "Reorder labels"

    @property
    def memory_bytes(self) -> int:
        """Estimated memory: 16 bytes per label_id/order pair + overhead."""
        return 32 + 16 * (len(self.before) + len(self.after))

    def apply(self) -> None:
        """Apply the new display order."""
        self.label_volume.set_label_orders(self.after)

    def undo(self) -> None:
        """Restore the previous display order."""
        self.label_volume.set_label_orders(self.before)


class CommandStack:
    """Bounded edit history with atomic commit and conflict-safe redo behavior.

    """

    def __init__(
        self,
        max_depth: int = 200,
        max_memory_bytes: int = 512 * 1024**2,
        max_disk_bytes: int = 8 * 1024**3,
    ) -> None:
        """Initialize an empty command stack with bounded limits.

        Args:
            max_depth: Maximum number of undo steps.
            max_memory_bytes: Maximum total memory for undo snapshots.
        """
        if max_depth <= 0 or max_memory_bytes <= 0 or max_disk_bytes <= 0:
            raise ValueError("history depth and memory/disk budgets must be positive")

        self.max_depth = max_depth
        self.max_memory_bytes = max_memory_bytes
        self.max_disk_bytes = max_disk_bytes
        self._undo: list[Command] = []  # Stack of undoable commands (newest at end)
        self._redo: list[Command] = []  # Stack of redoable commands (newest at end)
        self._undo_bytes = 0  # Total memory used by undo stack
        self._disk_bytes = 0

    @property
    def undo_depth(self) -> int:
        #Number of steps that can be undone.
        return len(self._undo)

    @property
    def last_undo_command(self) -> Command | None:
        #Return the command most recently applied, for associated annotation metadata.
        return self._undo[-1] if self._undo else None

    @property
    def redo_depth(self) -> int:
        #Number of steps that can be redone.
        return len(self._redo)

    @property
    def last_redo_command(self) -> Command | None:
        #Return the command that will be redone next.
        return self._redo[-1] if self._redo else None

    @property
    def memory_bytes(self) -> int:
        #Total memory used by both undo and redo stacks.
        return self._undo_bytes + sum(command.memory_bytes for command in self._redo)

    def execute(self, command: Command) -> None:
        #Apply first, then publish to history; failed edits never enter the stack.
        self.validate_command(command)
        disk_bytes = _storage_bytes(command)
        try:
            command.apply()
        except Exception:
            _dispose_command(command)
            raise

        self._record_applied(command, disk_bytes)

    def validate_command(self, command: Command) -> None:
        #Check undo-memory and diff-disk budgets before command application.
        disk_bytes = _storage_bytes(command)
        if command.memory_bytes > self.max_memory_bytes:
            raise MemoryError(
                f"{command.description} needs {command.memory_bytes} bytes, exceeding the "
                f"undo budget of {self.max_memory_bytes}; no edit was applied"
            )
        redo_disk_bytes = sum(_storage_bytes(item) for item in self._redo)
        retained_disk_bytes = self._disk_bytes - redo_disk_bytes
        if retained_disk_bytes + disk_bytes > self.max_disk_bytes:
            raise MemoryError(
                f"{command.description} needs {disk_bytes} disk bytes, exceeding the "
                f"undo-diff disk budget of {self.max_disk_bytes}; no edit was applied"
            )

    def accept_applied(self, command: Command) -> None:
        #Record a worker-applied command and publish its changes on the UI thread.
        self.validate_command(command)
        self._record_applied(command, _storage_bytes(command))
        label_volume = getattr(command, "label_volume", None)
        if label_volume is not None:
            label_volume.publish_change()

    def discard_applied(self, command: Command) -> None:
        #Roll back and dispose a command applied by a worker but not accepted.
        undo_silently = getattr(command, "undo_silently", None)
        if undo_silently is not None:
            undo_silently()
        else:
            command.undo()
        label_volume = getattr(command, "label_volume", None)
        if label_volume is not None:
            label_volume.publish_change()
        _dispose_command(command)

    def _record_applied(self, command: Command, disk_bytes: int) -> None:
        # A new edit invalidates the redo stack
        self._clear_redo()

        # Push to undo stack and update memory accounting
        self._undo.append(command)
        self._undo_bytes += command.memory_bytes
        self._disk_bytes += disk_bytes

        # Evict oldest entries if limits are exceeded
        self._trim_history()

    def ensure_capacity(self, estimated_command_bytes: int, description: str) -> None:
        """Fail before allocating a diff that cannot fit in the undo budget.
        allocating a huge numpy array only to discover it can't fit in history.
        """
        if estimated_command_bytes > self.max_memory_bytes:
            raise MemoryError(
                f"{description} needs about {estimated_command_bytes} bytes, exceeding the "
                f"undo budget of {self.max_memory_bytes}; no edit was applied"
            )

    def commit_region(
        self,
        label_volume: LabelVolume,
        slices: tuple[slice, slice, slice],
        new_values: np.ndarray,
        *,
        description: str = "Edit labels",
    ) -> VoxelDiffCommand | None:
        """Create and execute one bounded edit, skipping an exact no-op.
        Returns None if the edit was a no-op (identical values), otherwise
        returns the created command.
        """
        # Read current state of the target region
        current = label_volume.labelmap[slices]
        after = np.asarray(new_values, dtype=np.uint16)

        if after.shape != current.shape:
            raise ValueError(f"new region shape {after.shape} does not match {current.shape}")

        # Skip exact no-ops (avoids cluttering undo history)
        if np.array_equal(current, after):
            return None

        # Check budget before creating the command (each voxel needs 4 bytes: 2 before + 2 after)
        self.ensure_capacity(int(current.size) * 4, description)

        # Create and execute the command
        command = VoxelDiffCommand(label_volume, slices, current, after, description)
        self.execute(command)
        return command

    def undo(self) -> bool:
        #Undo the most recent edit. Returns True if an edit was undone.
#
        #Moves the command from the undo stack to the redo stack.
        #
        if not self._undo:
            return False

        command = self._undo[-1]
        command.undo()  # Restore previous state
        self._undo.pop()
        self._undo_bytes -= command.memory_bytes
        self._redo.append(command)  # Move to redo stack
        return True

    def redo(self) -> bool:
        """Redo the most recently undone edit. Returns True if an edit was redone.

        Moves the command from the redo stack back to the undo stack.
        """
        if not self._redo:
            return False

        command = self._redo[-1]
        command.apply()  # Re-apply the edit
        self._redo.pop()
        self._undo.append(command)
        self._undo_bytes += command.memory_bytes
        self._trim_history()  # May need to evict old entries
        return True

    def can_undo(self) -> bool:
        #Whether there are any edits that can be undone.
        return bool(self._undo)

    def can_redo(self) -> bool:
        #Whether there are any edits that can be redone.
        return bool(self._redo)

    def clear(self) -> None:
        #Discard all undo and redo history.
        for command in self._undo + self._redo:
            _dispose_command(command)
        self._undo.clear()
        self._redo.clear()
        self._undo_bytes = 0
        self._disk_bytes = 0

    def _clear_redo(self) -> None:
        #Clear the redo stack. Called when a new edit is made.
        for command in self._redo:
            self._disk_bytes -= _storage_bytes(command)
            _dispose_command(command)
        self._redo.clear()

    def _trim_history(self) -> None:
        #Evict the oldest undo entries when depth or memory limits are exceeded.
#
        #Uses FIFO eviction (pop from the front) to keep the most recent edits.
        #
        while self._undo and (
            len(self._undo) > self.max_depth
            or self._undo_bytes > self.max_memory_bytes
            or self._disk_bytes > self.max_disk_bytes
        ):
            oldest = self._undo.pop(0)
            self._undo_bytes -= oldest.memory_bytes
            self._disk_bytes -= _storage_bytes(oldest)
            _dispose_command(oldest)

    def commit_volume_mask(
        self,
        label_volume: LabelVolume,
        desired_mask: np.ndarray,
        label_id: int,
        *,
        overwrite: bool = False,
        replace_active: bool = False,
        description: str = "Apply 3D segmentation",
    ) -> DiskBackedVoxelDiffCommand | None:
       # Commit a 3D mask with disk-backed undo snapshots and one notification.
        command = DiskBackedVoxelDiffCommand.from_volume_mask(
            label_volume,
            desired_mask,
            label_id,
            overwrite=overwrite,
            replace_active=replace_active,
            description=description,
        )
        if command is None:
            return None
        self.execute(command)
        return command


def _storage_bytes(command: Command) -> int:
    return int(getattr(command, "storage_bytes", 0))


def _dispose_command(command: Command) -> None:
    dispose = getattr(command, "dispose", None)
    if dispose is not None:
        try:
            dispose()
        except OSError:
            pass
