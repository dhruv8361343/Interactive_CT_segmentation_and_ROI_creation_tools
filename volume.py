"""
Geometry-aware image volumes with optional disk-backed pixel storage.

FILE PURPOSE:
    This file defines `CTVolume`, the fundamental 3D grayscale image container
    used throughout the toolkit. It wraps:

    The array is made READ-ONLY after creation — all mutations happen through
    the `LabelVolume` class, which manages its own writable labelmap array.

    For large TIFF stacks (e.g., 847 slices × 1000×1000 pixels × 16 bits = ~1.7GB),
    the array can be a numpy memmap backed by a temp file. The cache directory
    system lets `LabelVolume` allocate additional memmapped arrays alongside
    the image data. A weakref finalizer ensures temp files are cleaned up even
    if `close()` isn't called explicitly.

Created in: Phase 1 — Core Data Model

"""

from __future__ import annotations

import tempfile
import weakref
from pathlib import Path
from typing import Any

import numpy as np

from ctseg.core.geometry import VolumeGeometry


class CTVolume:
    """Read-only 3D grayscale image plus source and calibration metadata.

    Arrays use `[z, y, x]` order to match TIFF stacks and SimpleITK's NumPy
    bridge. A disk-backed array keeps large reconstruction stacks from forcing
    an additional full-volume copy into resident memory.

    main Woek don - 
    - The pixel array is made read-only via setflags(write=False) to prevent
      accidental mutations that would bypass the undo system.
    - Geometry is stored as a separate VolumeGeometry object rather than inline
      fields, because geometry confirmation state must travel as a unit.
    - Intensity calibration (scale, offset, unit) is stored but NOT applied
      to the raw pixels. The raw scanner values are preserved so that
      calibration can be applied selectively by analysis tools.
    """

    def __init__(
        self,
        array_zyx: np.ndarray,
        geometry: VolumeGeometry | None = None,
        *,
        source: str | None = None,
        intensity_unit: str = "unknown",
        intensity_scale: float = 1.0,
        intensity_offset: float = 0.0,
        cache_directory: tempfile.TemporaryDirectory[str] | None = None,
    ) -> None:
        """Validate a grayscale volume and freeze the public pixel view.

        Argumntss:
            array_zyx: 3D numpy array in [z, y, x] order (depth, height, width).
            geometry: Physical-space metadata. Defaults to identity geometry.
            source: File path or description of where the data came from.
            intensity_unit: What the pixel values represent ("HU", "raw", "unknown").
            intensity_scale: Multiplier for raw-to-calibrated conversion (not applied).
            intensity_offset: Additive offset for raw-to-calibrated conversion (not applied).
            cache_directory: Optional pre-existing temp directory for memory-mapped data.
        """
        
        array = np.asanyarray(array_zyx)

        # Must be exactly 3D: [z, y, x]
        if array.ndim != 3:
            raise ValueError(f"CTVolume requires a 3D [z,y,x] array; received shape {array.shape}")

        # No empty dimensions allowed
        if any(length <= 0 for length in array.shape):
            raise ValueError("volume dimensions must all be non-zero")

        # Only numeric types: unsigned int (u), signed int (i), or float (f)
        if array.dtype.kind not in "uif":
            raise TypeError(f"unsupported image dtype {array.dtype}; expected integer or float")

        # Calibration values must be finite (no NaN or Inf)
        if not np.isfinite(intensity_scale) or not np.isfinite(intensity_offset):
            raise ValueError("intensity scale and offset must be finite")

       
        self._array = array.view()
        self._array.setflags(write=False)

        # Use provided geometry or fall back to default identity geometry
        self.geometry = geometry or VolumeGeometry()

        # Metadata about where the data came from and how to interpret intensities
        self.source = source
        self.intensity_unit = intensity_unit
        self.intensity_scale = float(intensity_scale)
        self.intensity_offset = float(intensity_offset)

        # Cache directory management 
        # The cache directory holds memory-mapped .npy files 
        self._cache_directory = cache_directory
       
        self._cache_owner = [cache_directory]
        # Track all memory-mapped file handles so we can close them before
        # deleting the temp directory (required on Windows)
        self._cache_mmaps: list[Any] = []
        current_array: Any = self._array
        seen_bases: set[int] = set()
        while current_array is not None and id(current_array) not in seen_bases:
            seen_bases.add(id(current_array))
            if isinstance(current_array, np.memmap):
                self._cache_mmaps.append(current_array._mmap)
            current_array = getattr(current_array, "base", None)

        # Register a weak-reference-based finalizer that will close mapped files
        # and clean up the temp directory when this CTVolume is garbage collected.
        # This is more reliable than __del__ because it handles reference cycles.
        self._cache_finalizer = weakref.finalize(
            self, _release_volume_cache, self._cache_owner, self._cache_mmaps
        )

        self._validate_calibration()

    def _validate_calibration(self) -> None:
        #Keep metadata from accidentally turning raw scanner values into HU.
#
        #The intensity_unit must be explicitly set — using an empty string could
        #cause downstream code to silently treat raw scanner counts as Hounsfield
        #Units, producing incorrect measurements.
        #
        if not self.intensity_unit:
            raise ValueError("intensity_unit must be explicit (use 'unknown' when unavailable)")

    @property
    def array(self) -> np.ndarray:
        #Return the read-only `[z,y,x]` pixel view.
#
        #The returned array cannot be modified. To edit the image, work through
        #LabelVolume (which manages its own writable labelmap).
        #
        return self._array

    @property
    def shape_zyx(self) -> tuple[int, int, int]:
        #Return dimensions in NumPy order: (depth, height, width).
        return tuple(int(value) for value in self._array.shape)

    @property
    def shape_xyz(self) -> tuple[int, int, int]:
        
        #Return dimensions in image-index order: (width, height, depth).
        #
        #This is the more intuitive order for 2D display (x=columns, y=rows),
        #and it matches the VolumeGeometry spacing_xyz convention.
        #
        z, y, x = self.shape_zyx
        return x, y, z

    @property
    def spacing_xyz(self) -> tuple[float, float, float]:
        #Shortcut to geometry.spacing_xyz — voxel size in mm.
        return self.geometry.spacing_xyz

    @property
    def origin_xyz(self) -> tuple[float, float, float]:
        #Shortcut to geometry.origin_xyz — physical position of voxel [0,0,0].
        return self.geometry.origin_xyz

    @property
    def direction(self) -> tuple[float, ...]:
        #Shortcut to geometry.direction — 3x3 direction cosine matrix (flat).
        return self.geometry.direction

    @property
    def depth(self) -> int:
        #Number of slices along the Z axis (axis 0 in numpy order).
        return self.shape_zyx[0]

    @property
    def height(self) -> int:
        #Number of rows along the Y axis (axis 1 in numpy order).
        return self.shape_zyx[1]

    @property
    def width(self) -> int:
        #Number of columns along the X axis (axis 2 in numpy order).
        return self.shape_zyx[2]

    def get_slice(self, axis: int, index: int) -> np.ndarray:
        """Return one 2D slice for image axis X=0, Y=1, or Z=2.

        The axis convention uses IMAGE-space (XYZ) rather than numpy-space (ZYX):
        - axis=0 (X): sagittal slice — picks one column plane → shape [z, y]
        - axis=1 (Y): coronal slice  — picks one row plane   → shape [z, x]
        - axis=2 (Z): axial slice    — picks one depth plane → shape [y, x]

        Args:
            axis: 0 for X (sagittal), 1 for Y (coronal), 2 for Z (axial).
            index: Which slice along that axis to extract.

        Returns:
            A 2D numpy array view (not a copy — modifications would be blocked
            by the read-only flag anyway).
        """
        if axis not in (0, 1, 2):
            raise ValueError("axis must be 0 (x), 1 (y), or 2 (z)")
        axis_length = self.shape_xyz[axis]
        if not 0 <= index < axis_length:
            raise IndexError(f"slice index {index} outside axis range [0, {axis_length})")

        # Map image-space axis to numpy axis:
        # axis=0 (X) → numpy axis 2 → _array[:, :, index]
        # axis=1 (Y) → numpy axis 1 → _array[:, index, :]
        # axis=2 (Z) → numpy axis 0 → _array[index, :, :]
        if axis == 0:
            return self._array[:, :, index]
        if axis == 1:
            return self._array[:, index, :]
        return self._array[index, :, :]

    def get_intensity_slice(self, axis: int, index: int, *, calibrated: bool = True) -> np.ndarray:
        #Return raw pixels or values transformed by the recorded scale/offset.
        raw = self.get_slice(axis, index)
        if not calibrated or (self.intensity_scale == 1.0 and self.intensity_offset == 0.0):
            return raw
        result = np.asarray(raw, dtype=np.float32)
        result *= self.intensity_scale
        result += self.intensity_offset
        return result

    def allocate_cache_array(self, name: str, dtype: np.dtype[Any] | type) -> np.memmap:
        """Allocate a zero-initialized array in the volume's owned cache directory.

        Memory-mapping the label buffer matters for the supplied 847-slice,
        1000x1000 16-bit stack: a dense labelmap alone occupies about 1.7 GB.
        By memory-mapping, the OS can page unused slices to disk automatically.

        Argumntss:
            name: Filename for the .npy file (sanitized to prevent path traversal).
            dtype: NumPy dtype for the new array (e.g., np.uint16 for labels).

        Returns:
            A writable memory-mapped numpy array with the same shape as this volume.
        """
        # Create a temp directory on first use (lazy initialization)
        if self._cache_directory is None:
            try:
                self._cache_directory = tempfile.TemporaryDirectory(
                    prefix="ctseg-volume-", ignore_cleanup_errors=True
                )
            except TypeError:
                self._cache_directory = tempfile.TemporaryDirectory(prefix="ctseg-volume-")
            self._cache_owner[0] = self._cache_directory

        if hasattr(self._cache_directory, "name") and not isinstance(
            self._cache_directory, (str, Path)
        ):
            dir_path = Path(self._cache_directory.name)
        else:
            dir_path = Path(self._cache_directory)
        dir_path.mkdir(parents=True, exist_ok=True)

        # Sanitize the filename to prevent path traversal attacks
        safe_name = Path(name).name
        if not safe_name:
            raise ValueError("cache array name must not be empty")

        path = dir_path / safe_name

        
        array = np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=self.shape_zyx)

        # Track the mmap handle so we can close it during cleanup
        self._cache_mmaps.append(array._mmap)

        # Flush ensures the zero-initialization is written to disk
        array.flush()
        return array

    def release_cache_array(self, array: np.ndarray) -> None:
        """Close and remove one temporary array once its preview/operation is done."""
        if not isinstance(array, np.memmap):
            return
        mapping = array._mmap
        path = Path(array.filename)
        if not mapping.closed:
            mapping.close()
        self._cache_mmaps[:] = [
            candidate for candidate in self._cache_mmaps if candidate is not mapping
        ]
        path.unlink(missing_ok=True)

    def close(self) -> None:
        #Release owned memory maps before deleting their temporary cache.
    #
        #On Windows, files cannot be deleted while they are memory-mapped.
        #This method closes all mapped file handles first, then cleans up
        #the temp directory.
#
        #This is also called automatically by the weakref finalizer when
        #the CTVolume is garbage collected.
        #
        self._cache_finalizer()

    @classmethod
    def from_array(
        cls,
        array_zyx: np.ndarray,
        *,
        spacing_xyz: tuple[float, float, float] | None = None,
        origin_xyz: tuple[float, float, float] | None = None,
        direction: tuple[float, ...] | None = None,
        source: str | None = None,
        intensity_unit: str = "unknown",
        cache_directory: Any = None,
    ) -> CTVolume:
        """Construct a volume while marking omitted geometry as assumed.

        This is the preferred constructor when you have a raw array and only
        partial metadata. Any metadata you DON'T provide will use default
        values AND be marked as "not confirmed", so downstream measurement
        tools won't silently produce incorrect results.

        Args:
            array_zyx: 3D pixel data in [z, y, x] order.
            spacing_xyz: Voxel size in mm. If None, defaults to (1,1,1) and marked unconfirmed.
            origin_xyz: Physical origin. If None, defaults to (0,0,0) and marked unconfirmed.
            direction: Direction cosines. If None, defaults to identity and marked unconfirmed.
            source: Description of data origin.
            intensity_unit: What pixel values represent.
            cache_directory: Optional temp directory or path for cache storage.
        """
        geometry = VolumeGeometry(
            spacing_xyz=spacing_xyz if spacing_xyz is not None else (1.0, 1.0, 1.0),
            origin_xyz=origin_xyz if origin_xyz is not None else (0.0, 0.0, 0.0),
            direction=(
                direction
                if direction is not None
                else (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
            ),
            # Only mark as confirmed if the caller explicitly provided the value
            spacing_confirmed=spacing_xyz is not None,
            origin_confirmed=origin_xyz is not None,
            direction_confirmed=direction is not None,
        )
        return cls(
            array_zyx,
            geometry,
            source=source,
            intensity_unit=intensity_unit,
            cache_directory=cache_directory,
        )

    def to_sitk(self) -> Any:
        #Create a SimpleITK image on demand without making it a core dependency.
#
        #SimpleITK is an optional dependency. This method converts the CTVolume
        #to a SimpleITK image with full geometry metadata, useful for interop
        #with other medical imaging tools.
        #
        try:
            import SimpleITK as sitk
        except ImportError as error:
            raise RuntimeError(
                "Install ctseg-toolkit[medical-io] to use the SimpleITK bridge"
            ) from error

        # sitk.GetImageFromArray expects a numpy array in [z,y,x] order
        image = sitk.GetImageFromArray(np.asarray(self._array))
        image.SetSpacing(self.geometry.spacing_xyz)
        image.SetOrigin(self.geometry.origin_xyz)
        image.SetDirection(self.geometry.direction)
        return image


def _release_volume_cache(
    owner_holder: list[tempfile.TemporaryDirectory[str] | None],
    mappings: list[Any],
) -> None:
    #Close mapped files before TemporaryDirectory removes them on Windows.
#
    #This function is called by the weakref finalizer. It receives copies
    #of the cache directory reference and mmap handles, not `self` (because
    #weak references can't see `self` after garbage collection starts).
#
    #On Windows, deleting a file that is still memory-mapped raises
    #PermissionError. This function closes all mmaps first, then lets
    #TemporaryDirectory.cleanup() safely delete the directory.
    #
    # Track which mmaps we've already closed to avoid double-close errors
    seen: set[int] = set()
    for mapping in mappings:
        try:
            if id(mapping) not in seen and not mapping.closed:
                mapping.close()
                seen.add(id(mapping))
        except Exception:
            pass

    # Now safely clean up the temp directory
    owner = owner_holder[0]
    if owner is not None:
        try:
            owner.cleanup()
        except Exception:
            pass
