"""
UI-independent paint, ROI, morphology, and propagation operations.


"""

from ctseg.core.ops.boolean import combine_add, combine_intersect, combine_subtract
from ctseg.core.ops.components import (
    keep_largest_n,
    label_components,
    remove_small_islands,
    select_component,
    split_components_2d,
)
from ctseg.core.ops.flood_fill import flood_fill
from ctseg.core.ops.interpolation import propagate_morphological, propagate_nearest
from ctseg.core.ops.morphology import closing, dilate, erode, opening
from ctseg.core.ops.paint import apply_stamp, apply_stamp_2d, apply_stroke_3d, make_stamp
from ctseg.core.ops.region_growing import grow_from_seed
from ctseg.core.ops.roi_shapes import ellipse_mask, rasterize_contour, rectangle_mask
from ctseg.core.ops.volume_segmentation import (
    boolean_volume_labels,
    filter_components_chunked,
    generate_distinct_label_colors,
    grow_connected_3d,
    keep_largest_components_volume,
    label_mask_volume,
    morphology_volume_chunked,
    otsu_threshold,
    remove_small_islands_volume,
    select_label_component_3d,
    slice_seed_to_zyx,
    split_label_components_volume,
    threshold_volume,
)

__all__ = [
    "apply_stamp",
    "apply_stamp_2d",
    "apply_stroke_3d",
    "boolean_volume_labels",
    "closing",
    "combine_add",
    "combine_intersect",
    "combine_subtract",
    "dilate",
    "ellipse_mask",
    "erode",
    "filter_components_chunked",
    "flood_fill",
    "grow_from_seed",
    "grow_connected_3d",
    "keep_largest_components_volume",
    "keep_largest_n",
    "label_mask_volume",
    "label_components",
    "make_stamp",
    "morphology_volume_chunked",
    "opening",
    "otsu_threshold",
    "propagate_morphological",
    "propagate_nearest",
    "rasterize_contour",
    "rectangle_mask",
    "remove_small_islands",
    "remove_small_islands_volume",
    "select_component",
    "select_label_component_3d",
    "slice_seed_to_zyx",
    "threshold_volume",
]
