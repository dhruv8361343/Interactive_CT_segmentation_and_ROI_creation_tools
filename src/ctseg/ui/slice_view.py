#Orthogonal slice viewer and interactive 2D paint/ROI canvas.

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np

try:
    from PyQt6.QtCore import QPoint, QPointF, Qt, pyqtSignal
    from PyQt6.QtGui import QImage, QPainter, QPainterPath, QPen, QPixmap
    from PyQt6.QtWidgets import (
        QGraphicsPathItem,
        QGraphicsPixmapItem,
        QGraphicsScene,
        QGraphicsView,
    )
except ImportError as error:
    raise RuntimeError("Install ctseg-toolkit[ui] to use the Qt slice viewer") from error

from ctseg.core.commands import CommandStack
from ctseg.core.label_volume import LabelVolume
from ctseg.core.ops.paint import (
    apply_mask_2d,
    make_stamp,
)
from ctseg.core.ops.roi_shapes import ellipse_mask, rasterize_contour, rectangle_mask
from ctseg.core.volume import CTVolume


class SliceView(QGraphicsView):
    #Display one image axis and route gestures through undoable core actions.

    slice_changed = pyqtSignal(int, int)
    seed_selected = pyqtSignal(int, int, int, int)
    view_focused = pyqtSignal(int)
    edit_committed = pyqtSignal()
    edit_failed = pyqtSignal(str)
    voxel_edited = pyqtSignal(int, int, int)
    roi_committed = pyqtSignal(str, int, int, object)

    def __init__(self, axis: int, parent=None) -> None:
        #Construct an empty orthogonal viewer before a source is loaded.
        if axis not in (0, 1, 2):
            raise ValueError("slice axis must be 0 (x), 1 (y), or 2 (z)")
        super().__init__(parent)
        self.axis = axis
        self.volume: CTVolume | None = None
        self.label_volume: LabelVolume | None = None
        self.command_stack = CommandStack()
        self.slice_index = 0
        self.window_center = 0.0
        self.window_width = 1.0
        self.active_label_id = 1
        self.tool = "navigate"
        self.brush_radius = 2.0
        self.brush_shape = "circle"
        self.overwrite = False
        self.mask_constraint_label_id: int | None = None
        self._preview_mask: np.ndarray | None = None
        self._drag_points: list[tuple[int, int]] = []
        self._polygon_points: list[tuple[float, float]] = []
        self._preview_volume_mask: np.ndarray | None = None
        self._preview_item: QGraphicsPathItem | None = None
        self._preview_is_replacement = False
        self._preview_erase_label_id: int | None = None
        self._scene = QGraphicsScene(self)
        self._image_item: QGraphicsPixmapItem = self._scene.addPixmap(QPixmap())
        self.setScene(self._scene)
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, False)
        self.setBackgroundBrush(Qt.GlobalColor.black)
        self.setDragMode(QGraphicsView.DragMode.NoDrag)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setMinimumSize(160, 130)
        self._fit_needed = True
        self._is_zoomed = False
        self._is_panning = False
        self._pan_start: QPoint | None = None
        self._labels_callback = self._labels_changed
        self._cached_grayscale: np.ndarray | None = None
        self._cached_grayscale_key: tuple[int, float, float, int] | None = None

    @property
    def maximum_slice(self) -> int:
        if self.volume is None:
            return 0
        return self.volume.shape_xyz[self.axis] - 1

    @property
    def plane_title(self) -> str:
        return ("Sagittal (X)", "Coronal (Y)", "Axial (Z)")[self.axis]

    def set_data(
        self,
        volume: CTVolume,
        label_volume: LabelVolume,
        command_stack: CommandStack,
    ) -> None:
        """Bind the shared volume and label model to this view."""
        if self.label_volume is not None:
            self.label_volume.unregister_change_callback(self._labels_callback)
        self.volume = volume
        self.label_volume = label_volume
        self.command_stack = command_stack
        self._cached_grayscale = None
        self._cached_grayscale_key = None
        self.slice_index = self.maximum_slice // 2
        self.resetTransform()
        self._fit_needed = True
        self._is_zoomed = False
        self.label_volume.register_change_callback(self._labels_callback)
        image_slice = volume.get_slice(self.axis, self.slice_index)
        low, high = np.percentile(image_slice, (1, 99))
        self.window_center = float((low + high) / 2)
        self.window_width = max(1.0, float(high - low))
        self.refresh()
        self.slice_changed.emit(self.axis, self.slice_index)

    def set_slice_index(self, index: int, *, emit: bool = True) -> None:
        if not 0 <= index <= self.maximum_slice:
            raise IndexError(f"slice index must be in [0, {self.maximum_slice}]")
        if self.slice_index == index:
            return
        self.slice_index = index
        self.refresh()
        if emit:
            self.slice_changed.emit(self.axis, index)

    def set_window_level(self, width: float, center: float) -> None:
        if width <= 0 or not np.isfinite(width) or not np.isfinite(center):
            raise ValueError("window width must be positive and center must be finite")
        self.window_width = float(width)
        self.window_center = float(center)
        self.refresh()

    def set_tool(self, tool: str) -> None:
        valid_tools = {
            "navigate",
            "brush",
            "erase",
            "rectangle",
            "ellipse",
            "freehand",
            "polygon",
            "seed",
        }
        if tool not in valid_tools:
            raise ValueError(f"unsupported slice tool {tool!r}")
        self.tool = tool
        self._polygon_points.clear()
        self._clear_preview()
        self.setCursor(
            Qt.CursorShape.CrossCursor if tool != "navigate" else Qt.CursorShape.ArrowCursor
        )

    def set_brush(self, radius_mm: float, shape: str = "circle") -> None:
        if radius_mm <= 0 or not np.isfinite(radius_mm):
            raise ValueError("brush radius must be a positive finite value")
        self.brush_radius = float(radius_mm)
        self.brush_shape = shape

    def set_mask_constraint(self, label_id: int | None) -> None:
        """Constrain paint/ROI edits strictly to voxels with this label ID (or None for unconstrained)."""
        self.mask_constraint_label_id = label_id

    def set_preview_mask(
        self,
        mask: np.ndarray | None,
        *,
        replace_label: bool = False,
        erase_label_id: int | None = None,
        slice_index: int | None = None,
    ) -> None:
        #Display a candidate in yellow without changing the label model.
        self._preview_volume_mask = None
        if mask is not None:
            if self.volume is None:
                raise RuntimeError("load a volume before adding an operation preview")
            expected = self.volume.get_slice(self.axis, self.slice_index).shape
            if np.shape(mask) != expected:
                raise ValueError(f"preview mask shape must be {expected}")
        self._preview_mask = None if mask is None else np.asarray(mask, dtype=bool).copy()
        self._preview_slice_index = self.slice_index if slice_index is None else slice_index
        self._preview_is_replacement = replace_label
        self._preview_erase_label_id = erase_label_id
        self.refresh()

    def set_preview_volume_mask(
        self, mask: np.ndarray | None, *, replace_label: bool = False
    ) -> None:
        #Render the matching plane from a whole-volume candidate mask.
        if mask is not None:
            if self.volume is None or np.shape(mask) != self.volume.shape_zyx:
                raise ValueError("whole-volume preview must match the loaded [z,y,x] image")
        self._preview_volume_mask = mask
        self._preview_mask = None
        self._preview_is_replacement = replace_label
        self._preview_erase_label_id = None
        self.refresh()

    def refresh(self) -> None:
        #Render the current slice and just its corresponding label plane.
        if self.volume is None or self.label_volume is None:
            return

        cache_key = (self.slice_index, self.window_width, self.window_center, id(self.volume))
        if self._cached_grayscale is not None and self._cached_grayscale_key == cache_key:
            grayscale = self._cached_grayscale
        else:
            image = np.asarray(self.volume.get_slice(self.axis, self.slice_index), dtype=np.float32)
            window_width = max(1e-5, float(self.window_width))
            lower = float(self.window_center) - window_width / 2.0
            grayscale = np.clip((image - lower) * (255.0 / window_width), 0, 255).astype(np.uint8)
            self._cached_grayscale = grayscale
            self._cached_grayscale_key = cache_key

        labels = _label_slice(self.label_volume, self.axis, self.slice_index)

        preview_mask = None
        if (
            self._preview_mask is not None
            and getattr(self, "_preview_slice_index", self.slice_index) == self.slice_index
        ):
            preview_mask = self._preview_mask
        elif self._preview_volume_mask is not None:
            if self.axis == 0:
                preview_mask = self._preview_volume_mask[:, :, self.slice_index]
            elif self.axis == 1:
                preview_mask = self._preview_volume_mask[:, self.slice_index, :]
            else:
                preview_mask = self._preview_volume_mask[self.slice_index]

        has_labels = np.any(labels != 0)
        has_preview = preview_mask is not None and np.any(preview_mask)

        if has_labels or has_preview:
            rgb = np.empty((grayscale.shape[0], grayscale.shape[1], 3), dtype=np.uint8)
            rgb[..., 0] = grayscale
            rgb[..., 1] = grayscale
            rgb[..., 2] = grayscale
            for info in self.label_volume.labels:
                if not info.visible:
                    continue
                region = labels == info.id
                if np.any(region):
                    color = np.asarray(info.color, dtype=np.float32)
                    blended = rgb[region] * 0.45 + color * 0.55
                    rgb[region] = np.clip(blended, 0, 255).astype(np.uint8)
            if preview_mask is not None and preview_mask.shape == labels.shape:
                added = preview_mask & (labels == 0)
                rgb[added] = np.array([255, 220, 20], dtype=np.uint8)
                if self._preview_is_replacement:
                    removed = (labels == self.active_label_id) & ~preview_mask
                    rgb[removed] = np.array([40, 210, 255], dtype=np.uint8)
                if self._preview_erase_label_id is not None:
                    removed = (labels == self._preview_erase_label_id) & preview_mask
                    rgb[removed] = np.array([40, 210, 255], dtype=np.uint8)
            height, width = rgb.shape[:2]
            qimage = QImage(
                np.ascontiguousarray(rgb).data,
                width,
                height,
                int(rgb.strides[0]),
                QImage.Format.Format_RGB888,
            ).copy()
        else:
            height, width = grayscale.shape[:2]
            qimage = QImage(
                np.ascontiguousarray(grayscale).data,
                width,
                height,
                int(grayscale.strides[0]),
                QImage.Format.Format_Grayscale8,
            ).copy()
        self._image_item.setPixmap(QPixmap.fromImage(qimage))
        self._scene.setSceneRect(0, 0, width, height)
        if getattr(self, "_fit_needed", True):
            self.fitInView(self._scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)
            self._fit_needed = False
        self.setToolTip(
            f"{self.plane_title} — slice {self.slice_index + 1}/{self.maximum_slice + 1}"
        )

    def reset_zoom(self) -> None:
        #Reset scale and center the slice in the viewport.
        self.resetTransform()
        if self._scene is not None and not self._scene.sceneRect().isEmpty():
            self.fitInView(self._scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)
        self._is_zoomed = False

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if self.volume is not None and not getattr(self, "_is_zoomed", False):
            self.fitInView(self._scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)

    def keyPressEvent(self, event) -> None:
        if event.key() == Qt.Key.Key_Escape:
            if self._polygon_points or self._drag_points:
                self._polygon_points.clear()
                self._drag_points.clear()
                self._clear_preview()
                event.accept()
                return
        if event.key() == Qt.Key.Key_R:
            self.reset_zoom()
            event.accept()
            return
        super().keyPressEvent(event)

    def wheelEvent(self, event) -> None:
        if self.volume is None:
            super().wheelEvent(event)
            return

        # Ctrl + Wheel zooms in/out anchored at mouse cursor without changing slices
        modifiers = event.modifiers()
        if modifiers & Qt.KeyboardModifier.ControlModifier:
            delta = event.angleDelta().y()
            if delta == 0:
                delta = event.angleDelta().x()
            if delta != 0:
                factor = 1.15 if delta > 0 else (1.0 / 1.15)
                self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
                self.scale(factor, factor)
                self._is_zoomed = True
            event.accept()
            return

        # Wheel without Ctrl scrolls slices
        step = 1 if event.angleDelta().y() > 0 else -1
        self.slice_changed.emit(self.axis, min(self.maximum_slice, max(0, self.slice_index + step)))
        event.accept()

    def mousePressEvent(self, event) -> None:
        point = self._event_point(event)
        self.view_focused.emit(self.axis)
        if self.volume is None:
            super().mousePressEvent(event)
            return

        # Panning with Middle-click, Right-click, or Navigate tool with Left-click
        if event.button() in (Qt.MouseButton.MiddleButton, Qt.MouseButton.RightButton) or (
            event.button() == Qt.MouseButton.LeftButton and self.tool == "navigate"
        ):
            self._is_panning = True
            self._pan_start = event.position().toPoint()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            event.accept()
            return

        if event.button() != Qt.MouseButton.LeftButton:
            super().mousePressEvent(event)
            return
        if self.tool == "seed":
            self.seed_selected.emit(
                self.axis,
                self.slice_index,
                int(point.y()),
                int(point.x()),
            )
            event.accept()
            return
        if self.tool == "polygon":
            self._polygon_points.append((point.x(), point.y()))
            self._update_polygon_preview()
            event.accept()
            return
        if self.tool != "navigate":
            self._drag_points = [(int(point.y()), int(point.x()))]
            if self.tool in {"rectangle", "ellipse"}:
                self._update_drag_preview(point)
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        if getattr(self, "_is_panning", False) and getattr(self, "_pan_start", None) is not None:
            current_pos = event.position().toPoint()
            delta = current_pos - self._pan_start
            self._pan_start = current_pos
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - delta.x())
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - delta.y())
            event.accept()
            return
        point = self._event_point(event)
        if self._drag_points and self.tool in {"brush", "erase", "freehand"}:
            pt_yx = (int(point.y()), int(point.x()))
            if pt_yx != self._drag_points[-1]:
                self._drag_points.append(pt_yx)
                self._update_path_preview()
            event.accept()
            return
        if self._drag_points and self.tool in {"rectangle", "ellipse"}:
            self._update_drag_preview(point)
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        if getattr(self, "_is_panning", False) and event.button() in (
            Qt.MouseButton.MiddleButton,
            Qt.MouseButton.RightButton,
            Qt.MouseButton.LeftButton,
        ):
            self._is_panning = False
            self._pan_start = None
            self.unsetCursor()
            if self.tool == "navigate":
                self.setCursor(Qt.CursorShape.OpenHandCursor)
            elif self.tool in {"brush", "erase", "seed"}:
                self.setCursor(Qt.CursorShape.CrossCursor)
            event.accept()
            return
        if not self._drag_points or event.button() != Qt.MouseButton.LeftButton:
            super().mouseReleaseEvent(event)
            return
        point = self._event_point(event)
        end_yx = (int(point.y()), int(point.x()))
        if self.tool in {"brush", "erase"}:
            centers = self._interpolate(self._drag_points + [end_yx])
            try:
                stamp = make_stamp(self.brush_shape, self._brush_radii_yx())
            except ValueError as error:
                self.edit_failed.emit(str(error))
                self._drag_points.clear()
                self._clear_preview()
                event.accept()
                return
            from ctseg.core.ops.paint import apply_stroke_2d

            self._run_edit(
                lambda: apply_stroke_2d(
                    self.label_volume,
                    self.axis,
                    self.slice_index,
                    centers,
                    stamp,
                    self.active_label_id if self.tool == "brush" else 0,
                    self.command_stack,
                    active_label_id=self.active_label_id,
                    mask_label_id=self.mask_constraint_label_id,
                ),
                self._plane_to_index_xyz(self.slice_index, *end_yx),
            )
        elif self.tool == "freehand":
            points = self._drag_points + [end_yx]
            points_xy = [(column + 0.5, row + 0.5) for row, column in points]
            if len(points_xy) >= 3:
                mask = rasterize_contour(
                    points_xy, self.volume.get_slice(self.axis, self.slice_index).shape
                )
                self._commit_roi(
                    mask,
                    end_yx,
                    roi_shape="freehand",
                    vertices_xy=points_xy,
                )
        elif self.tool in {"rectangle", "ellipse"}:
            first_row, first_column = self._drag_points[0]
            first_xy = (first_column, first_row)
            second_xy = (end_yx[1], end_yx[0])
            shape = self.volume.get_slice(self.axis, self.slice_index).shape
            mask = (
                rectangle_mask(first_xy, second_xy, shape)
                if self.tool == "rectangle"
                else ellipse_mask(first_xy, second_xy, shape)
            )
            self._commit_roi(
                mask,
                end_yx,
                roi_shape=self.tool,
                vertices_xy=[first_xy, second_xy],
            )
        self._drag_points.clear()
        self._clear_preview()
        event.accept()

    def mouseDoubleClickEvent(self, event) -> None:
        if self.tool == "polygon" and self._polygon_points and self.volume is not None:
            point = self._event_point(event)
            self._polygon_points.append((point.x(), point.y()))
            mask = rasterize_contour(
                self._polygon_points,
                self.volume.get_slice(self.axis, self.slice_index).shape,
            )
            self._commit_roi(
                mask,
                roi_shape="polygon",
                vertices_xy=list(self._polygon_points),
            )
            self._polygon_points.clear()
            self._clear_preview()
            event.accept()
            return
        if event.button() in (Qt.MouseButton.MiddleButton, Qt.MouseButton.RightButton) or (
            event.button() == Qt.MouseButton.LeftButton and self.tool == "navigate"
        ):
            self.reset_zoom()
            event.accept()
            return
        super().mouseDoubleClickEvent(event)

    def _commit_roi(
        self,
        mask: np.ndarray,
        point_yx: tuple[int, int] | None = None,
        *,
        roi_shape: str | None = None,
        vertices_xy: Sequence[tuple[float, float]] | None = None,
    ) -> bool:
        if point_yx is None:
            rows, columns = np.nonzero(mask)
            if rows.size:
                point_yx = (int(np.median(rows)), int(np.median(columns)))
            else:
                point_yx = (0, 0)
        committed = self._run_edit(
            lambda: apply_mask_2d(
                self.label_volume,
                self.axis,
                self.slice_index,
                mask,
                self.active_label_id,
                self.command_stack,
                description=f"Commit {self.tool} ROI",
                mask_label_id=self.mask_constraint_label_id,
            ),
            self._plane_to_index_xyz(self.slice_index, *point_yx),
        )
        if committed and roi_shape is not None and vertices_xy is not None:
            self.roi_committed.emit(
                roi_shape,
                self.axis,
                self.slice_index,
                [(float(x), float(y)) for x, y in vertices_xy],
            )
        return committed

    def _run_edit(
        self,
        operation: Callable[[], bool],
        edited_index_xyz: tuple[int, int, int] | None = None,
    ) -> bool:
        #Surface expected edit failures in the application instead of escaping Qt handlers.
        try:
            if operation():
                self.edit_committed.emit()
                if edited_index_xyz is not None:
                    self.voxel_edited.emit(*edited_index_xyz)
                return True
        except (IndexError, MemoryError, PermissionError, ValueError) as error:
            self.edit_failed.emit(str(error))
        return False

    def _event_point(self, event) -> QPointF:
        return self.mapToScene(event.position().toPoint())

    def _update_drag_preview(self, point: QPointF) -> None:
        start_row, start_column = self._drag_points[0]
        path = QPainterPath()
        path.addRect(
            min(start_column, point.x()),
            min(start_row, point.y()),
            abs(point.x() - start_column),
            abs(point.y() - start_row),
        )
        self._set_preview(path)

    def _update_path_preview(self) -> None:
        if not self._drag_points:
            return
        path = QPainterPath()
        path.moveTo(self._drag_points[0][1] + 0.5, self._drag_points[0][0] + 0.5)
        for row, column in self._drag_points[1:]:
            path.lineTo(column + 0.5, row + 0.5)
        self._set_preview(path)

    def _update_polygon_preview(self) -> None:
        path = QPainterPath()
        for index, (column, row) in enumerate(self._polygon_points):
            if index == 0:
                path.moveTo(column, row)
            else:
                path.lineTo(column, row)
        self._set_preview(path)

    def _set_preview(self, path: QPainterPath) -> None:
        if self._preview_item is None:
            pen = QPen(Qt.GlobalColor.yellow)
            pen.setWidth(1)
            pen.setStyle(Qt.PenStyle.DashLine)
            self._preview_item = self._scene.addPath(path, pen)
            self._preview_item.setZValue(2)
        else:
            self._preview_item.setPath(path)

    def _clear_preview(self) -> None:
        if self._preview_item is not None:
            self._scene.removeItem(self._preview_item)
            self._preview_item = None

    def _interpolate(self, points: Sequence[tuple[int, int]]) -> list[tuple[int, int]]:
        output: list[tuple[int, int]] = []
        for first, second in zip(points[:-1], points[1:], strict=False):
            distance = max(abs(second[0] - first[0]), abs(second[1] - first[1]), 1)
            for step in range(distance):
                fraction = step / distance
                output.append(
                    (
                        round(first[0] + (second[0] - first[0]) * fraction),
                        round(first[1] + (second[1] - first[1]) * fraction),
                    )
                )
        if points:
            output.append(points[-1])
        return output

    def _labels_changed(self, version: int) -> None:
        self.refresh()

    def _brush_radii_yx(self) -> tuple[float, float]:
        if self.volume is None or not self.volume.geometry.spacing_confirmed:
            return self.brush_radius, self.brush_radius
        row_axis, column_axis = ((2, 1), (2, 0), (1, 0))[self.axis]
        spacing = self.volume.spacing_xyz
        return (
            self.brush_radius / spacing[row_axis],
            self.brush_radius / spacing[column_axis],
        )

    def _plane_to_index_xyz(self, slice_index: int, row: int, column: int) -> tuple[int, int, int]:
        #Convert a 2D plane click to the shared image index convention.
        if self.axis == 0:
            return slice_index, column, row
        if self.axis == 1:
            return column, slice_index, row
        return column, row, slice_index


def _label_slice(label_volume: LabelVolume, axis: int, index: int) -> np.ndarray:
    if axis == 0:
        return label_volume.labelmap[:, :, index]
    if axis == 1:
        return label_volume.labelmap[:, index, :]
    return label_volume.labelmap[index]
