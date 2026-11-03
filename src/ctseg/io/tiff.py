#TIFF, OME-TIFF, and TIFF-slice-directory import/export.

from __future__ import annotations

import concurrent.futures
import json
import os
import re
import tempfile
import uuid
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import tifffile

from ctseg.core.geometry import IDENTITY_DIRECTION, VolumeGeometry
from ctseg.core.label_volume import LabelInfo, LabelVolume
from ctseg.core.volume import CTVolume


@dataclass(frozen=True, slots=True)
class TiffInspection:
    #Metadata-only summary used by the import dialog and CLI.

    path: str
    layout: str
    shape: tuple[int, ...]
    axes: str
    dtype: str
    pages: int
    description: str | None
    num_channels: int = 1


def inspect_tiff(path: str | Path) -> TiffInspection:
    #Inspect a TIFF file or slice directory without loading the full stack.
    source = Path(path)
    if source.is_dir():
        files = _tiff_files(source)
        if not files:
            raise FileNotFoundError(f"no TIFF slices found in {source}")
        with tifffile.TiffFile(files[0]) as handle:
            pages_per_file = len(handle.pages)
            if not handle.pages:
                raise ValueError(f"no pages found in slice {files[0].name}")
            page = handle.pages[0]
            sample_count = int(page.samplesperpixel)
            num_channels = 1
            if pages_per_file > 1:
                axes = "ZCYX"
                page_shape = (pages_per_file, *page.shape)
                num_channels = pages_per_file
            elif sample_count == 1:
                axes = "ZYX"
                page_shape = page.shape
                num_channels = 1
            elif len(page.shape) == 3 and page.shape[-1] == sample_count:
                axes = "ZYXS"
                page_shape = page.shape
                num_channels = sample_count
            elif len(page.shape) == 3 and page.shape[0] == sample_count:
                axes = "ZSYX"
                page_shape = page.shape
                num_channels = sample_count
            else:
                axes = "ZYXS"
                page_shape = page.shape
                num_channels = sample_count
            return TiffInspection(
                path=str(source),
                layout="slice_directory",
                shape=(len(files), *page_shape),
                axes=axes,
                dtype=str(page.dtype),
                pages=len(files) * pages_per_file,
                description=page.description,
                num_channels=num_channels,
            )
    if not source.is_file():
        raise FileNotFoundError(source)
    with tifffile.TiffFile(source) as handle:
        if not handle.series:
            raise ValueError(f"no readable image series in {source}")
        series = handle.series[0]
        axes = str(series.axes)
        shape = tuple(int(value) for value in series.shape)
        layout = (
            "ome_tiff"
            if handle.ome_metadata
            else ("single_image" if len(handle.pages) == 1 else "multipage")
        )
        num_channels = 1
        if "C" in axes:
            num_channels = max(1, shape[axes.index("C")])
        elif "S" in axes:
            num_channels = max(1, shape[axes.index("S")])
        elif handle.pages and int(handle.pages[0].samplesperpixel) > 1:
            num_channels = int(handle.pages[0].samplesperpixel)
        return TiffInspection(
            path=str(source),
            layout=layout,
            shape=shape,
            axes=axes,
            dtype=str(series.dtype),
            pages=len(handle.pages),
            description=handle.pages[0].description,
            num_channels=num_channels,
        )


def load_tiff_volume(
    path: str | Path,
    *,
    sidecar_path: str | Path | None = None,
    series_index: int = 0,
    channel_index: int | None = None,
    channel: int | None = None,
    timepoint_index: int | None = None,
    axes: str | None = None,
    cache_directory: str | Path | None = None,
) -> CTVolume:
    #Read a TIFF image/stack while preserving intensities and geometry metadata.
#
    #A directory of individual slices is copied one plane at a time into an
    #owned `.npy` memory map. This avoids simultaneously holding every 2 MB TIFF
    #decode buffer and a second full 1.7 GB stack in RAM.
    #
    if channel_index is None and channel is not None:
        channel_index = channel
    source = Path(path)
    sidecar = _read_sidecar(source, sidecar_path)
    if source.is_dir():
        return _load_slice_directory(
            source, sidecar, cache_directory, channel_index=channel_index
        )
    if not source.is_file():
        raise FileNotFoundError(source)
    with tifffile.TiffFile(source) as handle:
        if not 0 <= series_index < len(handle.series):
            raise IndexError(f"series_index {series_index} outside available series")
        series = handle.series[series_index]
        source_axes = (axes or str(series.axes)).upper()
        # tifffile uses Q for an unclassified page axis and I for generic
        # image sequences.  Both are unambiguous depth dimensions when shape
        # and dtype are uniform across pages.
        source_axes = _remap_sequence_axes(source_axes, series.shape)
        # For multichannel files opened without an explicit channel, default to channel 0 so the import succeeds instead of raising.
         
        effective_channel = channel_index
        if effective_channel is None and _has_multichannel_axis(source_axes, series.shape):
            effective_channel = 0
        array = None
        # Try direct file memory-mapping first (zero-RAM path for contiguous TIFFs/OME-TIFFs)
        try:
            array = tifffile.memmap(source, series=series_index)
        except (ValueError, tifffile.TiffFileError, OSError):
            array = None

        if array is not None:
            try:
                array_zyx = _normalize_axes(
                    array,
                    source_axes,
                    channel_index=effective_channel,
                    timepoint_index=timepoint_index,
                    sample_photometric=int(handle.pages[0].photometric) if handle.pages else 0,
                )
                metadata = _ome_geometry(handle.ome_metadata)
                metadata.update(
                    _calibration_from_description(
                        handle.pages[0].description if handle.pages else None
                    )
                )
                metadata.update(sidecar)
                geometry, intensity = _make_geometry(metadata)
                return CTVolume(
                    array_zyx,
                    geometry,
                    source=str(source.resolve()),
                    intensity_unit=str(intensity.get("intensity_unit", "unknown")),
                    intensity_scale=float(intensity.get("intensity_scale", 1.0)),
                    intensity_offset=float(intensity.get("intensity_offset", 0.0)),
                )
            except Exception as map_err:
                logger.debug("Memmap normalization failed, will try page streaming: %s", map_err)
                array = None

        # Page-by-page streaming memmap path: for multipage, multichannel, and OME-TIFF stacks
        if len(series.pages) > 1:
            try:
                return _load_tiff_pages_to_memmap(
                    source,
                    series,
                    handle.pages[0].description if handle.pages else None,
                    sidecar,
                    cache_directory,
                    channel_index=effective_channel,
                    source_axes=source_axes,
                    ome_metadata=handle.ome_metadata,
                )
            except Exception as stream_err:
                logger.debug("Page streaming failed, falling back to asarray: %s", stream_err)

        if array is None:
            array = series.asarray()
        array_zyx = _normalize_axes(
            array,
            source_axes,
            channel_index=effective_channel,
            timepoint_index=timepoint_index,
            sample_photometric=int(handle.pages[0].photometric),
        )
        metadata = _ome_geometry(handle.ome_metadata)
        metadata.update(_calibration_from_description(handle.pages[0].description))
        metadata.update(sidecar)
        geometry, intensity = _make_geometry(metadata)
        return CTVolume(
            np.asarray(array_zyx),
            geometry,
            source=str(source.resolve()),
            intensity_unit=str(intensity.get("intensity_unit", "unknown")),
            intensity_scale=float(intensity.get("intensity_scale", 1.0)),
            intensity_offset=float(intensity.get("intensity_offset", 0.0)),
        )


def _load_tiff_pages_to_memmap(
    source: Path,
    series,
    description: str | None,
    sidecar: dict[str, Any],
    cache_directory: str | Path | None,
    *,
    channel_index: int | None = None,
    source_axes: str = "ZYX",
    ome_metadata: str | None = None,
) -> CTVolume:
    #Stream a multipage ZYX/multichannel/OME TIFF into disk-backed storage one page at a time.
    #
    #Multichannel and multi-sample data are extracted slice-by-slice so the result
    #is always a scalar ZYX volume without resident RAM spikes.
    #
    shape = tuple(int(value) for value in series.shape)
    dtype = np.dtype(series.dtype)
    if dtype.kind not in "uif":
        raise TypeError(f"unsupported TIFF intensity dtype {dtype}")

    first_page = series.pages[0]
    first_samples = int(first_page.samplesperpixel)

    # Determine layout, output shape, and page-to-slice mapping:
    if first_samples > 1:
        probe = first_page.asarray()
        axis = _sample_axis(probe.shape, first_samples, source.name)
        scalar_probe = _scalarize_samples(
            probe,
            axis,
            int(first_page.photometric),
            channel_index=channel_index,
            source_name=source.name,
        )
        expected_plane = scalar_probe.shape
        out_dtype = scalar_probe.dtype
        out_shape = (shape[0], *expected_plane)
        page_indices = list(range(len(series.pages)))
    elif len(shape) == 4 and "C" in source_axes and "Z" in source_axes:
        c_axis = source_axes.index("C")
        z_axis = source_axes.index("Z")
        c_count = shape[c_axis]
        z_count = shape[z_axis]
        eff_c = min(max(0, channel_index or 0), c_count - 1)
        expected_plane = (shape[source_axes.index("Y")], shape[source_axes.index("X")])
        out_dtype = dtype
        out_shape = (z_count, *expected_plane)
        if c_axis == 0 and z_axis == 1 and len(series.pages) == c_count * z_count:
            page_indices = [eff_c * z_count + z for z in range(z_count)]
        elif z_axis == 0 and c_axis == 1 and len(series.pages) == z_count * c_count:
            page_indices = [z * c_count + eff_c for z in range(z_count)]
        else:
            page_indices = list(range(min(z_count, len(series.pages))))
            out_shape = (len(page_indices), *expected_plane)
    else:
        expected_plane = shape[1:] if len(shape) >= 3 else shape
        out_dtype = dtype
        out_shape = (len(series.pages), *expected_plane)
        page_indices = list(range(len(series.pages)))

    if cache_directory is None:
        owner: tempfile.TemporaryDirectory[str] | None = tempfile.TemporaryDirectory(
            prefix="ctseg-tiff-pages-", ignore_cleanup_errors=True
        )
        root = Path(owner.name)
    else:
        owner = None
        root = Path(cache_directory)
        root.mkdir(parents=True, exist_ok=True)
    cache_path = root / f"volume-{uuid.uuid4().hex}.npy"
    stack = np.lib.format.open_memmap(cache_path, mode="w+", dtype=out_dtype, shape=out_shape)
    try:
        for z_out, p_idx in enumerate(page_indices):
            page = series.pages[p_idx]
            plane = page.asarray()
            if int(page.samplesperpixel) > 1:
                axis = _sample_axis(plane.shape, int(page.samplesperpixel), source.name)
                plane = _scalarize_samples(
                    plane,
                    axis,
                    int(page.photometric),
                    channel_index=channel_index,
                    source_name=source.name,
                )
            if plane.shape != expected_plane:
                raise ValueError(
                    f"TIFF page {p_idx} has shape {plane.shape}; expected {expected_plane}"
                )
            stack[z_out] = plane
        stack.flush()
        metadata = _ome_geometry(ome_metadata)
        metadata.update(_calibration_from_description(description))
        metadata.update(sidecar)
        geometry, intensity = _make_geometry(metadata)
        return CTVolume(
            stack,
            geometry,
            source=str(source.resolve()),
            intensity_unit=str(intensity.get("intensity_unit", "unknown")),
            intensity_scale=float(intensity.get("intensity_scale", 1.0)),
            intensity_offset=float(intensity.get("intensity_offset", 0.0)),
            cache_directory=owner,
        )
    except Exception:
        stack._mmap.close()
        cache_path.unlink(missing_ok=True)
        if owner is not None:
            owner.cleanup()
        raise


def _load_slice_directory(
    directory: Path,
    metadata: dict[str, Any],
    cache_directory: str | Path | None,
    *,
    channel_index: int | None = None,
) -> CTVolume:
    files = _tiff_files(directory)
    if not files:
        raise FileNotFoundError(f"no TIFF slices found in {directory}")
    first, first_description = _read_single_page(files[0], channel_index=channel_index)
    if first.ndim != 2:
        raise ValueError(f"slice {files[0].name} must be grayscale 2D; found shape {first.shape}")
    if first.dtype.kind not in "uif":
        raise TypeError(f"unsupported TIFF intensity type: {first.dtype}")
    shape = (len(files), *first.shape)
    calibration = _calibration_from_description(first_description)
    effective_metadata = dict(calibration)
    effective_metadata.update(metadata)
    if cache_directory is None:
        owner: tempfile.TemporaryDirectory[str] | None = tempfile.TemporaryDirectory(
            prefix="ctseg-tiff-stack-", ignore_cleanup_errors=True
        )
        cache_root = Path(owner.name) if owner else Path(tempfile.gettempdir())
    else:
        owner = None
        cache_root = Path(cache_directory)
        cache_root.mkdir(parents=True, exist_ok=True)
    cache_root.mkdir(parents=True, exist_ok=True)
    cache_path = cache_root / f"volume-{uuid.uuid4().hex}.npy"
    stack = np.lib.format.open_memmap(cache_path, mode="w+", dtype=first.dtype, shape=shape)
    success = False
    try:
        stack[0] = first
        if len(files) > 1:
            # Bound both active decoders and completed-but-not-yet-written results.
            # Submitting every TIFF at once retains every decoded plane in its Future,
            # which can consume several GB for a large RGB/RGBA slice directory.
            max_workers = min(4, max(1, os.cpu_count() or 4))
            max_pending = max_workers * 2
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                pending: dict[concurrent.futures.Future, int] = {}
                next_index = 1
                while next_index < len(files) or pending:
                    while next_index < len(files) and len(pending) < max_pending:
                        future = executor.submit(
                            _read_single_page,
                            files[next_index],
                            channel_index=channel_index,
                        )
                        pending[future] = next_index
                        next_index += 1
                    completed, _ = concurrent.futures.wait(
                        pending,
                        return_when=concurrent.futures.FIRST_COMPLETED,
                    )
                    for future in completed:
                        index = pending.pop(future)
                        plane, description = future.result()
                        if plane.shape != first.shape:
                            name = files[index].name
                            raise ValueError(
                                f"slice {name} has shape {plane.shape}; expected {first.shape}"
                            )
                        if plane.dtype != first.dtype:
                            name = files[index].name
                            raise TypeError(
                                f"slice {name} has dtype {plane.dtype}; expected {first.dtype}"
                            )
                        if _calibration_from_description(description) != calibration:
                            name = files[index].name
                            raise ValueError(
                                f"slice {name} has different intensity calibration metadata"
                            )
                        stack[index] = plane
                    # Drop references to completed Future results before submitting
                    # the next bounded batch of decoders.
                    del completed, future, plane, description
        stack.flush()
        geometry, intensity = _make_geometry(effective_metadata)
        volume = CTVolume(
            stack,
            geometry,
            source=str(directory.resolve()),
            intensity_unit=str(intensity.get("intensity_unit", "unknown")),
            intensity_scale=float(intensity.get("intensity_scale", 1.0)),
            intensity_offset=float(intensity.get("intensity_offset", 0.0)),
            cache_directory=owner,
        )
        success = True
        return volume
    finally:
        if not success:
            stack._mmap.close()
            cache_path.unlink(missing_ok=True)
            if owner is not None:
                owner.cleanup()


def _read_single_page(
    path: Path, *, channel_index: int | None = None
) -> tuple[np.ndarray, str | None]:
    # Fast path: PIL/Pillow decodes compressed TIFF slices (like PackBits or Deflate) using native C libtiff, which is orders of magnitude faster than pure-Python decoders.
    try:
        from PIL import Image

        with Image.open(path) as img:
            if getattr(img, "n_frames", 1) == 1:
                desc = None
                if hasattr(img, "tag_v2"):
                    raw_desc = img.tag_v2.get(270)
                    if isinstance(raw_desc, bytes):
                        desc = raw_desc.decode("utf-8", errors="ignore")
                    elif isinstance(raw_desc, str):
                        desc = raw_desc
                mode = img.mode
                if channel_index is not None:
                    if mode in ("RGB", "RGBA"):
                        bands = img.split()
                        if 0 <= channel_index < len(bands):
                            return np.array(bands[channel_index]), desc
                        raise IndexError(
                            f"channel_index {channel_index} outside available channels "
                            f"[0, {len(bands)}) in {path.name}"
                        )
                    elif mode in ("L", "I;16", "I", "F"):
                        if channel_index == 0:
                            return np.array(img), desc
                        raise IndexError(
                            f"channel_index {channel_index} outside available channels "
                            f"[0, 1) in {path.name}"
                        )
                else:
                    if mode in ("RGB", "RGBA"):
                        return np.array(img.convert("L")), desc
                    elif mode in ("L", "I;16", "I", "F"):
                        return np.array(img), desc
    except Exception:
        pass  # Fall back to tifffile below

    with tifffile.TiffFile(path) as handle:
        if not handle.pages:
            raise ValueError(f"{path.name} contains no TIFF pages")
        # In a slice directory, multi-page files represent separate channels;
        # a single-page color TIFF stores channels as per-pixel samples.
        if len(handle.pages) > 1:
            page_index = 0 if channel_index is None else channel_index
            sample_index = None
        else:
            page_index = 0
            sample_index = channel_index
        if not 0 <= page_index < len(handle.pages):
            raise IndexError(
                f"channel_index {page_index} outside available TIFF pages "
                f"[0, {len(handle.pages)}) in {path.name}"
            )
        page = handle.pages[page_index]
        samples = int(page.samplesperpixel)
        image = page.asarray()
        if samples == 1:
            if image.ndim != 2:
                raise ValueError(
                    f"{path.name} has a single-sample page with unsupported shape "
                    f"{image.shape}; expected a 2D slice"
                )
            return image, page.description
        sample_axis = _sample_axis(image.shape, samples, path.name)
        scalar = _scalarize_samples(
            image,
            sample_axis,
            int(page.photometric),
            channel_index=sample_index,
            source_name=path.name,
        )
        return scalar, page.description


def _tiff_files(directory: Path) -> list[Path]:
    files = [
        child
        for child in directory.iterdir()
        if child.is_file() and child.suffix.casefold() in {".tif", ".tiff"}
    ]
    return sorted(files, key=_natural_sort_key)


def _natural_sort_key(path: Path) -> tuple[Any, ...]:
    return tuple(
        int(part) if part.isdigit() else part.casefold() for part in re.split(r"(\d+)", path.name)
    )


def _sample_axis(shape: tuple[int, ...], sample_count: int, source_name: str) -> int:
    #Locate the sample dimension in either interleaved or planar TIFF data.
    if len(shape) >= 3 and shape[-1] == sample_count:
        return -1
    if len(shape) >= 3 and shape[0] == sample_count:
        return 0
    if sample_count in shape:
        return shape.index(sample_count)
    if len(shape) == 3 and shape[-1] in (3, 4):
        return -1
    if len(shape) == 3 and shape[0] in (3, 4):
        return 0
    raise ValueError(
        f"{source_name} declares {sample_count} samples per pixel but has "
        f"unsupported array shape {shape}"
    )


def _scalarize_samples(
    array: np.ndarray,
    sample_axis: int,
    photometric: int,
    *,
    channel_index: int | None = None,
    source_name: str = "TIFF",
) -> np.ndarray:
    #Convert pixel samples to a scalar image for the existing CTVolume pipeline.
    #
    #RGB/RGBA data is converted to luminance (alpha is ignored). Other multisample
    #TIFFs use their first sample by default. An explicit index selects one sample.
    #
    samples = np.moveaxis(array, sample_axis, -1)
    sample_count = samples.shape[-1]
    if channel_index is not None:
        if not 0 <= channel_index < sample_count:
            raise IndexError(
                f"channel_index {channel_index} outside available samples [0, {sample_count}) "
                f"for {source_name}"
            )
        return samples[..., channel_index]

    if sample_count in (3, 4):
        rgb = samples[..., :3].astype(np.float64, copy=False)
        luminance = rgb @ np.array((0.299, 0.587, 0.114), dtype=np.float64)
        if array.dtype.kind in "ui":
            luminance = np.rint(luminance)
        return luminance.astype(array.dtype, copy=False)
    return samples[..., 0]


def _remap_sequence_axes(axes: str, shape: tuple[int, ...] | None = None) -> str:
    #Normalize tifffile axes string to standard dimensions (T, C, Z, Y, X, S).
    #
    #Replaces ambiguous sequence axes (Q, I, _, ?) with appropriate spatial or
    #channel dimensions so downstream processing can handle them cleanly without
    #creating duplicate axes.
    #
    axes_list = list(axes)
    seq_indices = [i for i, c in enumerate(axes_list) if c in ("I", "Q", "?", "_")]
    if seq_indices:
        if len(seq_indices) == 1:
            idx = seq_indices[0]
            axes_list[idx] = "Z" if "Z" not in axes_list else ("C" if "C" not in axes_list else "T")
        elif len(seq_indices) == 2 and shape is not None and len(shape) == len(axes_list):
            i0, i1 = seq_indices
            s0, s1 = shape[i0], shape[i1]
            if "Z" in axes_list:
                axes_list[i0] = "C"
                axes_list[i1] = "T"
            elif s0 <= 16 and s1 > s0:
                axes_list[i0] = "C"
                axes_list[i1] = "Z"
            else:
                axes_list[i0] = "Z"
                axes_list[i1] = "C"
        elif len(seq_indices) >= 3 and shape is not None:
            names = ["T", "C", "Z"]
            for idx, name in zip(seq_indices, names):
                axes_list[idx] = name
        else:
            for idx in seq_indices:
                for cand in ("Z", "C", "T", "S"):
                    if cand not in axes_list:
                        axes_list[idx] = cand
                        break

    # Guard against duplicate dimension letters
    seen: set[str] = set()
    available = [c for c in ("C", "T", "S", "Z", "Y", "X") if c not in axes_list]
    for i, char in enumerate(axes_list):
        if char in seen:
            new_char = available.pop(0) if available else "C"
            axes_list[i] = new_char
            seen.add(new_char)
        else:
            seen.add(char)

    return "".join(axes_list)


def _has_multichannel_axis(axes: str, shape: tuple[int, ...]) -> bool:
    #Return True when the axes contain a multi-valued C dimension.
    axis_list = list(axes)
    if "C" in axis_list:
        idx = axis_list.index("C")
        if idx < len(shape) and shape[idx] > 1:
            return True
    return False


def _normalize_axes(
    array: np.ndarray,
    axes: str,
    *,
    channel_index: int | None,
    timepoint_index: int | None,
    sample_photometric: int | None = None,
) -> np.ndarray:
    #Select non-spatial dimensions and normalize to `[z,y,x]`.
    result = np.asarray(array)
    axis_names = list(axes)
    if len(axis_names) != result.ndim:
        # Infer default axes from array shape
        if result.ndim == 2:
            axis_names = ["Y", "X"]
        elif result.ndim == 3:
            if result.shape[-1] in (3, 4):
                axis_names = ["Y", "X", "S"]
            elif result.shape[0] in (3, 4) and result.shape[1] > 4:
                axis_names = ["C", "Y", "X"]
            else:
                axis_names = ["Z", "Y", "X"]
        elif result.ndim == 4:
            if result.shape[-1] in (3, 4):
                axis_names = ["Z", "Y", "X", "S"]
            elif result.shape[1] in (3, 4):
                axis_names = ["Z", "C", "Y", "X"]
            else:
                axis_names = ["C", "Z", "Y", "X"]
        else:
            raise ValueError(
                f"TIFF axes {axes!r} do not match array shape {result.shape}; "
                "provide axes explicitly"
            )

    # Disambiguate duplicate dimensions if any
    seen: set[str] = set()
    available = [c for c in ("C", "T", "S", "Z", "Y", "X") if c not in axis_names]
    for i, char in enumerate(axis_names):
        if char in seen:
            new_char = available.pop(0) if available else "C"
            axis_names[i] = new_char
            seen.add(new_char)
        else:
            seen.add(char)

    sample_index = channel_index if "C" not in axis_names else None
    for axis_name, requested_index in (("T", timepoint_index), ("C", channel_index)):
        if axis_name in axis_names:
            axis_number = axis_names.index(axis_name)
            axis_size = result.shape[axis_number]
            if requested_index is None:
                requested_index = 0
            if not 0 <= requested_index < axis_size:
                raise IndexError(f"{axis_name} selection outside [0, {axis_size})")
            slc = [slice(None)] * result.ndim
            slc[axis_number] = requested_index
            result = result[tuple(slc)]
            axis_names.pop(axis_number)

    for sample_axis in ("S",):
        if sample_axis in axis_names:
            axis_number = axis_names.index(sample_axis)
            result = _scalarize_samples(
                result,
                axis_number,
                sample_photometric or 0,
                channel_index=sample_index,
            )
            axis_names.pop(axis_number)

    # Squeeze any remaining non-spatial axes that have size 1
    i = 0
    while i < len(axis_names):
        if axis_names[i] not in "ZYX" and result.shape[i] == 1:
            slc = [slice(None)] * result.ndim
            slc[i] = 0
            result = result[tuple(slc)]
            axis_names.pop(i)
        else:
            i += 1

    # If any non-ZYX axes remain, collapse them by taking index 0
    i = 0
    while i < len(axis_names):
        if axis_names[i] not in "ZYX":
            slc = [slice(None)] * result.ndim
            slc[i] = 0
            result = result[tuple(slc)]
            axis_names.pop(i)
        else:
            i += 1

    if "Y" not in axis_names or "X" not in axis_names:
        raise ValueError(f"TIFF must contain Y and X axes; found {axis_names}")
    if "Z" not in axis_names:
        result = np.expand_dims(result, axis=0)
        axis_names.insert(0, "Z")
    if set(axis_names) != {"Z", "Y", "X"}:
        raise ValueError(f"expected Z/Y/X axes after selection; found {axis_names}")
    result = np.transpose(result, tuple(axis_names.index(axis) for axis in "ZYX"))
    if result.dtype.kind not in "uif":
        raise TypeError(f"unsupported TIFF intensity dtype {result.dtype}")
    return result


def _read_sidecar(source: Path, sidecar_path: str | Path | None) -> dict[str, Any]:
    candidate: Path | None = Path(sidecar_path) if sidecar_path is not None else None
    if candidate is None:
        candidates = (
            [source / "geometry.json", source / f"{source.name}.geometry.json"]
            if source.is_dir()
            else [
                source.with_suffix(".geometry.json"),
                source.with_suffix(".json"),
                source.parent / "geometry.json",
            ]
        )
        candidate = next((path for path in candidates if path.is_file()), None)
    if candidate is None:
        return {}
    try:
        metadata = json.loads(candidate.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"could not read TIFF geometry sidecar {candidate}: {error}") from error
    if not isinstance(metadata, dict):
        raise ValueError("TIFF sidecar root must be a JSON object")
    version = metadata.get("schema_version", 1)
    if version != 1:
        raise ValueError(f"unsupported TIFF sidecar schema version {version!r}")
    return metadata


def _ome_geometry(xml: str | None) -> dict[str, Any]:
    if not xml:
        return {}
    try:
        root = ET.fromstring(xml)
    except ET.ParseError as error:
        raise ValueError(f"invalid OME-TIFF XML metadata: {error}") from error
    pixels = next((node for node in root.iter() if node.tag.endswith("Pixels")), None)
    if pixels is None:
        return {}
    result: dict[str, Any] = {}
    spacing: list[float] = []
    for axis in "XYZ":
        value = pixels.attrib.get(f"PhysicalSize{axis}")
        unit = pixels.attrib.get(f"PhysicalSize{axis}Unit", "µm")
        if value is None:
            break
        multiplier = _unit_to_mm(unit)
        spacing.append(float(value) * multiplier)
    if len(spacing) == 3:
        result["spacing_mm_xyz"] = spacing
        result["spacing_confirmed"] = True
    return result


def _unit_to_mm(unit: str) -> float:
    normalized = unit.casefold().replace("μ", "µ")
    if normalized in {"µm", "um", "micrometer", "micrometre"}:
        return 0.001
    if normalized in {"mm", "millimeter", "millimetre"}:
        return 1.0
    if normalized in {"cm", "centimeter", "centimetre"}:
        return 10.0
    if normalized in {"m", "meter", "metre"}:
        return 1000.0
    raise ValueError(f"unsupported OME physical-size unit {unit!r}")


def _calibration_from_description(description: str | None) -> dict[str, Any]:
    #Preserve explicit scanner scale/offset without altering source pixels.
    if not description:
        return {}
    values: dict[str, Any] = {}
    for key in ("slope", "offset"):
        match = re.search(
            rf"\b{key}\s*=\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][-+]?\d+)?)",
            description,
            flags=re.IGNORECASE,
        )
        if match:
            values["intensity_scale" if key == "slope" else "intensity_offset"] = float(
                match.group(1)
            )
    if values:
        values["intensity_unit"] = "unknown"
    return values


def _make_geometry(metadata: dict[str, Any]) -> tuple[VolumeGeometry, dict[str, Any]]:
    spacing = metadata.get("spacing_mm_xyz")
    origin = metadata.get("origin_mm_lps_xyz")
    direction = metadata.get("direction_lps_row_major")
    spacing_confirmed = bool(metadata.get("spacing_confirmed", spacing is not None))
    origin_confirmed = bool(metadata.get("origin_confirmed", origin is not None))
    direction_confirmed = bool(metadata.get("direction_confirmed", direction is not None))
    geometry = VolumeGeometry(
        spacing_xyz=tuple(spacing) if spacing is not None else (1.0, 1.0, 1.0),
        origin_xyz=tuple(origin) if origin is not None else (0.0, 0.0, 0.0),
        direction=tuple(direction) if direction is not None else IDENTITY_DIRECTION,
        spacing_confirmed=spacing_confirmed,
        origin_confirmed=origin_confirmed,
        direction_confirmed=direction_confirmed,
    )
    return geometry, metadata


def write_labelmap_tiff(
    label_volume: LabelVolume,
    path: str | Path,
    *,
    compression: str | None = None,
) -> tuple[Path, Path]:
    #Write a uint16 ZYX label stack plus its explicit geometry sidecar.
    destination = Path(path)
    if destination.suffix.casefold() not in {".tif", ".tiff"}:
        raise ValueError("labelmap TIFF output must use .tif or .tiff")
    destination.parent.mkdir(parents=True, exist_ok=True)
    sidecar = destination.with_suffix(".geometry.json")
    temporary = destination.with_name(destination.stem + ".tmp" + destination.suffix)
    metadata = {
        "axes": "ZYX",
        "schema_version": 1,
        "spacing_mm_xyz": list(label_volume.reference.spacing_xyz),
        "origin_mm_lps_xyz": list(label_volume.reference.origin_xyz),
        "direction_lps_row_major": list(label_volume.reference.direction),
        "spacing_confirmed": label_volume.reference.geometry.spacing_confirmed,
        "origin_confirmed": label_volume.reference.geometry.origin_confirmed,
        "direction_confirmed": label_volume.reference.geometry.direction_confirmed,
        "intensity_unit": "label_id",
        "labels": [
            {
                "id": info.id,
                "name": info.name,
                "color": list(info.color),
                "visible": info.visible,
                "locked": info.locked,
                "order": info.order,
            }
            for info in label_volume.labels
        ],
    }
    try:
        tifffile.imwrite(
            temporary,
            label_volume.labelmap,
            photometric="minisblack",
            compression=compression,
            metadata={"axes": "ZYX"},
        )
        temporary.replace(destination)
        sidecar.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return destination, sidecar


def load_tiff_labelmap(
    path: str | Path, *, sidecar_path: str | Path | None = None
) -> tuple[CTVolume, LabelVolume]:
    #Read a TIFF label stack and restore sidecar colors/names when available.
    source = Path(path)
    metadata = _read_sidecar(source, sidecar_path)
    volume = load_tiff_volume(source, sidecar_path=sidecar_path)
    sidecar_infos = metadata.get("labels", [])
    infos = tuple(
        LabelInfo(
            id=int(item["id"]),
            name=str(item["name"]),
            color=tuple(int(value) for value in item["color"]),
            visible=bool(item.get("visible", True)),
            locked=bool(item.get("locked", False)),
            order=int(item.get("order", index)),
        )
        for index, item in enumerate(sidecar_infos)
    )
    if not infos:
        present_ids: set[int] = set()
        for z_index in range(volume.depth):
            present_ids.update(int(value) for value in np.unique(volume.array[z_index]) if value)
        infos = tuple(
            LabelInfo(label_id, f"Label {label_id}", (255, 64, 64))
            for label_id in sorted(present_ids)
        )
    labels = LabelVolume.from_array(volume, volume.array, infos, memory_map=True)
    return volume, labels
