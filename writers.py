#Writers for labelmaps, editable ROIs, and resumable project sessions.

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from ctseg.core.label_volume import LabelVolume
from ctseg.io.tiff import write_labelmap_tiff


def write_labelmap(
    label_volume: LabelVolume, path: str | Path, image_format: str | None = None
) -> Path:
    #Write a geometry-aware NRRD/NIfTI mask or TIFF plus JSON sidecar.
    destination = Path(path)
    destination_name = destination.name.casefold()
    inferred_format = (
        "nii.gz" if destination_name.endswith(".nii.gz") else destination.suffix.lstrip(".")
    )
    selected = (image_format or inferred_format).casefold()
    if selected in {"tif", "tiff"}:
        return write_labelmap_tiff(label_volume, destination)[0]
    if selected not in {"nrrd", "nii", "nii.gz", "nifti"}:
        raise ValueError(f"unsupported labelmap format {selected!r}")
    try:
        import SimpleITK as sitk
    except ImportError as error:
        raise RuntimeError(
            "Install ctseg-toolkit[medical-io] to write NIfTI or NRRD labelmaps"
        ) from error
    image = sitk.GetImageFromArray(np.asarray(label_volume.labelmap))
    image.SetSpacing(label_volume.reference.spacing_xyz)
    image.SetOrigin(label_volume.reference.origin_xyz)
    image.SetDirection(label_volume.reference.direction)
    destination.parent.mkdir(parents=True, exist_ok=True)
    output_suffix = ".nii.gz" if destination_name.endswith(".nii.gz") else destination.suffix
    output_stem = destination.name[: -len(output_suffix)]
    temporary = destination.with_name(output_stem + ".tmp" + output_suffix)
    try:
        sitk.WriteImage(image, str(temporary))
        os.replace(temporary, destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return destination


def write_roi_sidecar(rois: list[dict[str, Any]], path: str | Path) -> Path:
    #Persist editable contours separately from their rasterized labelmap.
    return _write_json({"schema_version": 1, "rois": rois}, path)


def write_session(state: dict[str, Any], path: str | Path) -> Path:
    #Persist portable label metadata and view choices as a versioned JSON file.
    payload = {"schema_version": 1, "session": state}
    return _write_json(payload, path)


def _write_json(payload: dict[str, Any], path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=destination.parent,
            prefix=destination.name + ".",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            json.dump(payload, stream, indent=2, allow_nan=False)
            stream.write("\n")
        os.replace(temporary, destination)
    except Exception:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise
    return destination
