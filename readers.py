#Readers for medical images and explicit raw-volume sidecars.

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from ctseg.core.geometry import VolumeGeometry
from ctseg.core.volume import CTVolume


def read_dicom_series(directory: str | Path) -> CTVolume:
    #Read one ordered DICOM series using SimpleITK's series-aware reader.
    sitk = _simpleitk()
    path = Path(directory)
    series_ids = sitk.ImageSeriesReader.GetGDCMSeriesIDs(str(path))
    if not series_ids:
        raise ValueError(f"no DICOM image series found in {path}")
    if len(series_ids) != 1:
        raise ValueError(
            f"{path} contains {len(series_ids)} DICOM series; select a series explicitly"
        )
    names = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(str(path), series_ids[0])
    first_slice = sitk.ReadImage(names[0])
    rescale_type = (
        first_slice.GetMetaData("0028|1054").strip()
        if first_slice.HasMetaDataKey("0028|1054")
        else "unknown"
    )
    intensity_unit = rescale_type if rescale_type.casefold() == "hu" else "unknown"
    reader = sitk.ImageSeriesReader()
    reader.SetFileNames(names)
    image = reader.Execute()
    return _from_sitk_image(
        image,
        source=str(path.resolve()),
        intensity_unit=intensity_unit,
    )


def read_nifti(path: str | Path) -> CTVolume:
    
    #Read a NIfTI scalar or 4D multichannel image and preserve spacing, origin, and direction.
    source = Path(path)
    sitk = _simpleitk()
    image = sitk.ReadImage(str(source))
    if image.GetDimension() == 4:
        size = list(image.GetSize())
        dim4 = size[3]
        if dim4 in (3, 4):
            sub_volumes = []
            for c in range(min(3, dim4)):
                extract_size = [size[0], size[1], size[2], 0]
                extract_index = [0, 0, 0, c]
                sub_volumes.append(sitk.Extract(image, extract_size, extract_index))
            image = sub_volumes[0] * 0.299 + sub_volumes[1] * 0.587 + sub_volumes[2] * 0.114
        else:
            extract_size = [size[0], size[1], size[2], 0]
            extract_index = [0, 0, 0, 0]
            image = sitk.Extract(image, extract_size, extract_index)
    elif image.GetDimension() != 3:
        raise ValueError(
            f"NIfTI input must be 3D or 4D; received {image.GetDimension()} dimensions"
        )
    return _from_sitk_image(image, source=str(source.resolve()), intensity_unit="unknown")


def read_nrrd(path: str | Path) -> CTVolume:
    
    #Read a NRRD scalar or 4D multichannel image and preserve geometry metadata.
    source = Path(path)
    sitk = _simpleitk()
    image = sitk.ReadImage(str(source))
    if image.GetDimension() == 4:
        size = list(image.GetSize())
        dim4 = size[3]
        if dim4 in (3, 4):
            sub_volumes = []
            for c in range(min(3, dim4)):
                extract_size = [size[0], size[1], size[2], 0]
                extract_index = [0, 0, 0, c]
                sub_volumes.append(sitk.Extract(image, extract_size, extract_index))
            image = sub_volumes[0] * 0.299 + sub_volumes[1] * 0.587 + sub_volumes[2] * 0.114
        else:
            extract_size = [size[0], size[1], size[2], 0]
            extract_index = [0, 0, 0, 0]
            image = sitk.Extract(image, extract_size, extract_index)
    elif image.GetDimension() != 3:
        raise ValueError(
            f"NRRD input must be 3D or 4D; received {image.GetDimension()} dimensions"
        )
    return _from_sitk_image(image, source=str(source.resolve()), intensity_unit="unknown")


def read_raw_with_sidecar(path: str | Path, sidecar_json: str | Path) -> CTVolume:
    #Memory-map a raw ZYX array using mandatory, explicit scanner metadata.
    source = Path(path)
    metadata_path = Path(sidecar_json)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    required = {
        "shape_zyx",
        "dtype",
        "spacing_mm_xyz",
        "origin_mm_lps_xyz",
        "direction_lps_row_major",
    }
    missing = sorted(required - metadata.keys())
    if missing:
        raise ValueError(f"raw-volume sidecar is missing required fields: {', '.join(missing)}")
    shape = tuple(int(value) for value in metadata["shape_zyx"])
    if len(shape) != 3 or any(value <= 0 for value in shape):
        raise ValueError("shape_zyx must contain three positive dimensions")
    dtype = np.dtype(metadata["dtype"])
    if dtype.kind not in "uif":
        raise TypeError(f"unsupported raw intensity dtype {dtype}")
    expected_size = int(np.prod(shape, dtype=np.int64)) * dtype.itemsize
    actual_size = source.stat().st_size
    if actual_size != expected_size:
        raise ValueError(
            f"raw file size is {actual_size} bytes, but metadata requires {expected_size}"
        )
    byte_order = metadata.get("byte_order")
    if byte_order in {"little", "big"}:
        dtype = dtype.newbyteorder("<" if byte_order == "little" else ">")
    array = np.memmap(source, mode="r", dtype=dtype, shape=shape)
    geometry = VolumeGeometry(
        spacing_xyz=tuple(metadata["spacing_mm_xyz"]),
        origin_xyz=tuple(metadata["origin_mm_lps_xyz"]),
        direction=tuple(metadata["direction_lps_row_major"]),
        spacing_confirmed=True,
        origin_confirmed=True,
        direction_confirmed=True,
    )
    return CTVolume(
        array,
        geometry,
        source=str(source.resolve()),
        intensity_unit=str(metadata.get("intensity_unit", "unknown")),
        intensity_scale=float(metadata.get("intensity_scale", 1.0)),
        intensity_offset=float(metadata.get("intensity_offset", 0.0)),
    )


def read_roi_sidecar(path: str | Path) -> list[dict[str, Any]]:
    #Read and validate the versioned editable ROI sidecar.
    value = _read_versioned_json(path, "rois")
    if not isinstance(value, list) or any(not isinstance(item, dict) for item in value):
        raise ValueError(f"ROI sidecar {path} must contain a list of ROI objects")
    return value


def read_session(path: str | Path) -> dict[str, Any]:
    #Read and validate a versioned session metadata file.
    return _read_versioned_json(path, "session")


def _read_versioned_json(path: str | Path, key: str) -> Any:
    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"could not read {source}: {error}") from error
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError(f"{source} must use schema_version 1")
    if key not in payload:
        raise ValueError(f"{source} does not contain {key!r}")
    value = payload[key]
    if not isinstance(value, (dict, list)):
        raise ValueError(f"{key!r} in {source} must be a JSON object or array")
    return value


def _from_sitk_image(image: Any, *, source: str, intensity_unit: str) -> CTVolume:
    sitk = _simpleitk()
    if image.GetDimension() != 3:
        raise ValueError("only 3D images are supported")
    components = image.GetNumberOfComponentsPerPixel()
    if components != 1:
        # Multichannel image: convert RGB to luminance, else take first component
        # VectorUInt8, VectorFloat32 etc. — check if 3 or 4 components for RGB(A)
        if components in (3, 4):
            # Extract first three channels and compute luminance
            channels = [sitk.VectorIndexSelectionCast(image, i, sitk.sitkFloat64) for i in range(3)]
            image = channels[0] * 0.299 + channels[1] * 0.587 + channels[2] * 0.114
        else:
            image = sitk.VectorIndexSelectionCast(image, 0)

    array = sitk.GetArrayFromImage(image)
    geometry = VolumeGeometry(
        spacing_xyz=tuple(float(value) for value in image.GetSpacing()),
        origin_xyz=tuple(float(value) for value in image.GetOrigin()),
        direction=tuple(float(value) for value in image.GetDirection()),
        spacing_confirmed=True,
        origin_confirmed=True,
        direction_confirmed=True,
    )
    return CTVolume(array, geometry, source=source, intensity_unit=intensity_unit)



def _simpleitk() -> Any:
    try:
        import SimpleITK as sitk
    except ImportError as error:
        raise RuntimeError(
            "Install ctseg-toolkit[medical-io] to read DICOM, NIfTI, NRRD, or SimpleITK images"
        ) from error
    return sitk
