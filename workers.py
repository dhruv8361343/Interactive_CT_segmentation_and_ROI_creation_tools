
#Qt worker helpers for potentially long whole-volume operations.

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

try:
    from PyQt6.QtCore import QObject, QRunnable, pyqtSignal
except ImportError as error:
    raise RuntimeError("Install ctseg-toolkit[ui] to run background image operations") from error


class _WorkerSignals(QObject):
    finished = pyqtSignal(int, object)
    failed = pyqtSignal(int, str)


class VolumeOperationWorker(QRunnable):
    #Run one immutable-snapshot operation without freezing the Qt event loop.

    def __init__(self, generation: int, operation: Callable[[], object]) -> None:
        super().__init__()
        self.generation = generation
        self.operation = operation
        self.signals = _WorkerSignals()

    def run(self) -> None:
        try:
            result = self.operation()
        except Exception as error:
            self.signals.failed.emit(self.generation, str(error))
        else:
            self.signals.finished.emit(self.generation, result)


class _VolumeLoadSignals(QObject):
    finished = pyqtSignal(int, object)
    failed = pyqtSignal(int, str)


class VolumeLoadWorker(QRunnable):
    #Loading  a large TIFF stack or medical volume without blocking the UI thread.

    def __init__(self, generation: int, path: str | Path, channel: int = 0) -> None:
        super().__init__()
        self.generation = generation
        self.path = Path(path)
        self.channel = channel
        self.signals = _VolumeLoadSignals()

    def run(self) -> None:
        try:
            volume = _load_volume(self.path, channel=self.channel)
        except Exception as error:
            self.signals.failed.emit(self.generation, f"{type(error).__name__}: {error}")
        else:
            self.signals.finished.emit(self.generation, volume)


def _load_volume(path: Path, channel: int = 0):
    #Select and execute the reader in the worker thread.
    from ctseg.io.readers import read_dicom_series, read_nifti, read_nrrd
    from ctseg.io.tiff import load_tiff_volume

    if path.is_dir():
        if any(item.suffix.casefold() in {".tif", ".tiff"} for item in path.iterdir()):
            return load_tiff_volume(path, channel=channel)
        return read_dicom_series(path)
    if path.name.casefold().endswith((".tif", ".tiff")):
        return load_tiff_volume(path, channel=channel)
    if path.name.casefold().endswith((".nii", ".nii.gz")):
        return read_nifti(path)
    if path.suffix.casefold() == ".nrrd":
        return read_nrrd(path)
    raise ValueError(f"unsupported input path {path}")


class LabelmapImportWorker(QRunnable):
    #Load and geometry-check a labelmap without copying it on the UI thread.

    def __init__(self, generation: int, path: str | Path, reference) -> None:
        super().__init__()
        self.generation = generation
        self.path = Path(path)
        self.reference = reference
        self.signals = _VolumeLoadSignals()

    def run(self) -> None:
        source_volume = None
        source_labels = None
        try:
            if self.path.is_dir() or self.path.suffix.casefold() in {".tif", ".tiff"}:
                from ctseg.io.tiff import load_tiff_labelmap

                source_volume, source_labels = load_tiff_labelmap(self.path)
                source_array = source_labels.labelmap
                label_info = source_labels.labels
            elif self.path.name.casefold().endswith((".nii", ".nii.gz")):
                from ctseg.io.readers import read_nifti

                source_volume = read_nifti(self.path)
                source_array = source_volume.array
                label_info = _label_info_from_array(source_array)
            elif self.path.suffix.casefold() == ".nrrd":
                from ctseg.io.readers import read_nrrd

                source_volume = read_nrrd(self.path)
                source_array = source_volume.array
                label_info = _label_info_from_array(source_array)
            else:
                raise ValueError("labelmap import supports TIFF, NRRD, and NIfTI files")
            _validate_labelmap_geometry(source_volume, self.reference)
            from ctseg.core.label_volume import LabelVolume

            labels = LabelVolume.from_array(
                self.reference,
                source_array,
                tuple(label_info),
                memory_map=True,
            )
        except Exception as error:
            if source_volume is not None:
                source_volume.close()
            self.signals.failed.emit(self.generation, f"{type(error).__name__}: {error}")
        else:
            if source_volume is not None:
                source_volume.close()
            self.signals.finished.emit(self.generation, labels)


def _label_info_from_array(array):
    #Build deterministic display metadata for labelmaps without a sidecar.
    import numpy as np

    from ctseg.core.label_volume import LabelInfo

    data = np.asanyarray(array)
    if data.dtype.kind not in "ui":
        if data.dtype.kind == "f":
            data = np.rint(data).astype(np.uint16)
        else:
            raise TypeError("labelmap must contain integer label IDs")
    palette = (
        (230, 70, 70),
        (70, 160, 240),
        (80, 200, 120),
        (230, 180, 60),
        (180, 100, 220),
        (50, 200, 200),
    )
    present: set[int] = set()
    for z_index in range(data.shape[0]):
        ids = np.unique(data[z_index])
        if ids.size and (int(ids[0]) < 0 or int(ids[-1]) > 65535):
            raise ValueError("label IDs must fit uint16")
        present.update(int(value) for value in ids if value)
    return tuple(
        LabelInfo(
            label_id,
            f"Label {label_id}",
            palette[(label_id - 1) % len(palette)],
            order=index,
        )
        for index, label_id in enumerate(sorted(present))
    )


def _validate_labelmap_geometry(labelmap, reference) -> None:
    #Require voxel-for-voxel alignment before using a mask as an overlay.
    import numpy as np

    if labelmap.shape_zyx != reference.shape_zyx:
        raise ValueError(
            f"labelmap shape {labelmap.shape_zyx} does not match image shape {reference.shape_zyx}"
        )
    if labelmap.geometry.is_confirmed and reference.geometry.is_confirmed:
        if (
            not np.allclose(labelmap.spacing_xyz, reference.spacing_xyz, rtol=1e-5, atol=1e-6)
            or not np.allclose(labelmap.origin_xyz, reference.origin_xyz, rtol=1e-5, atol=1e-6)
            or not np.allclose(labelmap.direction, reference.direction, rtol=1e-5, atol=1e-6)
        ):
            raise ValueError("labelmap physical geometry does not match the current image")
