"""Seeded intensity-constrained region growing for 2D slices or 3D volumes."""

from __future__ import annotations

import numpy as np

from ctseg.core.ops.flood_fill import flood_fill
from ctseg.core.volume import CTVolume


def grow_from_seed(
    image: CTVolume | np.ndarray,
    seed: tuple[int, ...],
    lower: float,
    upper: float,
    *,
    method: str = "connected_threshold",
    connectivity: int | None = None,
    multiplier: float = 2.5,
    iterations: int = 3,
) -> np.ndarray:
    """Grow a connected region; confidence mode updates limits from the region.

    Passing an image or slice array keeps dimension choice explicit. Phase 3 UI
    uses the active 2D slice; the same deterministic operation supports 3D.
    """
    array = image.array if isinstance(image, CTVolume) else np.asarray(image)
    if lower > upper:
        raise ValueError("lower threshold must not exceed upper threshold")
    if method == "connected_threshold":
        return flood_fill(array, seed, connectivity=connectivity, lower=lower, upper=upper)
    if method != "confidence_connected":
        raise ValueError(f"unsupported region-growing method {method!r}")
    if multiplier <= 0 or iterations <= 0:
        raise ValueError("multiplier and iteration count must be positive")
    region = flood_fill(array, seed, connectivity=connectivity, lower=lower, upper=upper)
    seed_value = float(array[seed])
    for _ in range(iterations):
        values = array[region]
        if values.size == 0:
            break
        mean = float(values.mean())
        standard_deviation = float(values.std())
        lo = max(lower, mean - multiplier * standard_deviation)
        hi = min(upper, mean + multiplier * standard_deviation)
        region = flood_fill(array, seed, connectivity=connectivity, lower=lo, upper=hi)
        if not lo <= seed_value <= hi:
            return np.zeros(array.shape, dtype=bool)
    return region
