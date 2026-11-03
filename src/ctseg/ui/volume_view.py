#Optional VTK volume renderer with downsampled data for large source stacks.

from __future__ import annotations

import logging
import math
import os
from collections.abc import Sequence

import numpy as np

try:
    from PyQt6.QtCore import (
        QEvent,
        QObject,
        QRunnable,
        Qt,
        QThreadPool,
        QTimer,
        pyqtSignal,
    )
    from PyQt6.QtWidgets import (
        QCheckBox,
        QComboBox,
        QDoubleSpinBox,
        QHBoxLayout,
        QLabel,
        QPushButton,
        QSlider,
        QVBoxLayout,
        QWidget,
    )
    from vtkmodules import vtkRenderingVolumeOpenGL2 as _vtk_volume_backend
    from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
    from vtkmodules.util.numpy_support import numpy_to_vtk
    from vtkmodules.vtkCommonDataModel import vtkImageData, vtkPiecewiseFunction, vtkPolyData
    from vtkmodules.vtkFiltersCore import vtkFlyingEdges3D
    from vtkmodules.vtkInteractionStyle import vtkInteractorStyleTrackballCamera
    from vtkmodules.vtkInteractionWidgets import vtkOrientationMarkerWidget
    from vtkmodules.vtkRenderingAnnotation import vtkAxesActor
    from vtkmodules.vtkRenderingCore import (
        vtkActor,
        vtkCellPicker,
        vtkColorTransferFunction,
        vtkPolyDataMapper,
        vtkRenderer,
        vtkVolume,
        vtkVolumeProperty,
    )
    from vtkmodules.vtkRenderingVolume import vtkVolumePicker
    from vtkmodules.vtkRenderingVolumeOpenGL2 import vtkSmartVolumeMapper
except ImportError as error:
    raise RuntimeError(
        "Install ctseg-toolkit[ui] to use the PyQt6/VTK 3D volume renderer"
    ) from error

from ctseg.core.gpu import classify_gpu, opengl_renderer_name
from ctseg.core.label_volume import LabelVolume
from ctseg.core.volume import CTVolume

logger = logging.getLogger(__name__)


class _SurfaceBuildSignals(QObject):
    ready = pyqtSignal(int, int, object)
    failed = pyqtSignal(int, int, str)


class _LabelSurfaceWorker(QRunnable):
    #Build coarse VTK polydata away from the UI/render thread.

    def __init__(
        self,
        request_id: int,
        label_version: int,
        labelmap: np.ndarray,
        labels,
        spacing_xyz: tuple[float, float, float],
        origin_xyz: tuple[float, float, float],
        direction: tuple[float, ...],
        shape_zyx: tuple[int, int, int],
        cache_owner: object,
        preview_mask: np.ndarray | None,
        preview_label_id: int,
        preview_color: tuple[int, int, int],
    ) -> None:
        super().__init__()
        self.request_id = request_id
        self.label_version = label_version
        self.labelmap = labelmap
        self.labels = labels
        self.spacing_xyz = spacing_xyz
        self.origin_xyz = origin_xyz
        self.direction = direction
        self.shape_zyx = shape_zyx
        self.cache_owner = cache_owner
        self.preview_mask = preview_mask
        self.preview_label_id = preview_label_id
        self.preview_color = preview_color
        self.signals = _SurfaceBuildSignals()

    def run(self) -> None:
        try:
            surfaces = _build_label_surfaces(
                self.labelmap,
                self.labels,
                self.spacing_xyz,
                self.origin_xyz,
                self.direction,
                self.shape_zyx,
                self.preview_mask,
                self.preview_label_id,
                self.preview_color,
            )
        except Exception as error:
            logger.exception("3D label surface build failed")
            self.signals.failed.emit(
                self.request_id, self.label_version, f"{type(error).__name__}: {error}"
            )
        else:
            self.signals.ready.emit(self.request_id, self.label_version, surfaces)


class VolumeView3D(QWidget):
    #Render a reduced-resolution preview without duplicating the full CT stack.

    label_changed = pyqtSignal(int)
    paint_requested = pyqtSignal(int, int, int, bool)
    paint_stroke_requested = pyqtSignal(object, bool)
    paint_rejected = pyqtSignal(str)
    brush_radius_changed = pyqtSignal(float)
    brush_shape_changed = pyqtSignal(str)
    _surfaces_ready = pyqtSignal(int, int, object)
    _surfaces_failed = pyqtSignal(int, int, str)

    def __init__(self, parent=None, *, thread_pool: QThreadPool | None = None) -> None:
        super().__init__(parent)
        self.brush_radius = 2.0
        self.paint_toggle = QCheckBox("Enable 3D brush editing")
        self.paint_toggle.setToolTip(
            "Click or drag on the rendered volume to paint or erase the active label."
        )
        self.paint_toggle.setEnabled(False)
        self.paint_mode = QComboBox()
        self.paint_mode.addItem("Paint", False)
        self.paint_mode.addItem("Erase", True)
        self.paint_mode.setEnabled(False)
        self.brush_shape = QComboBox()
        self.brush_shape.addItem("Round", "ball")
        self.brush_shape.addItem("Square", "box")
        self.brush_shape.setEnabled(False)
        self.brush_shape.currentIndexChanged.connect(self._on_brush_shape_changed)

        self.brush_radius_spin = QDoubleSpinBox()
        self.brush_radius_spin.setRange(0.1, 100.0)
        self.brush_radius_spin.setDecimals(1)
        self.brush_radius_spin.setSingleStep(0.5)
        self.brush_radius_spin.setValue(self.brush_radius)
        self.brush_radius_spin.setEnabled(False)
        self.brush_radius_spin.valueChanged.connect(self._on_brush_radius_changed)

        self._preview_mask: np.ndarray | None = None
        self._preview_label_id = 1
        self._preview_actor: vtkActor | None = None
        self._painting = False
        self._last_painted_voxel: tuple[int, int, int] | None = None
        self._stroke_points: list[tuple[int, int, int]] = []

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Three short rows (camera, segmentation display, brush) instead of one long row
        # so nothing is pushed beyond the edge of a narrow 3D pane.
        camera_row = QHBoxLayout()
        camera_row.setContentsMargins(6, 4, 6, 2)
        camera_row.setSpacing(6)
        self.camera_preset = QComboBox()
        self.camera_preset.addItems(["Perspective", "Axial", "Coronal", "Sagittal"])
        self.camera_preset.currentTextChanged.connect(self.set_camera_preset)
        camera_row.addWidget(QLabel("View"))
        camera_row.addWidget(self.camera_preset)
        self.reset_cam_btn = QPushButton("Reset")
        self.reset_cam_btn.setToolTip("Reset 3D camera to fit the volume")
        self.reset_cam_btn.clicked.connect(self.reset_camera)
        camera_row.addWidget(self.reset_cam_btn)
        camera_row.addSpacing(8)
        self.ct_mode = QComboBox()
        self.ct_mode.addItems(["Auto", "Full", "Ghost", "Hidden"])
        self.ct_mode.setToolTip(
            "CT volume display. Auto fades the CT to a transparent 'ghost' while segmentation "
            "surfaces exist, so labels buried inside the sample stay visible."
        )
        self.ct_mode.currentTextChanged.connect(self._apply_ct_display)
        camera_row.addWidget(QLabel("CT"))
        camera_row.addWidget(self.ct_mode)
        self.volume_opacity_slider = QSlider(Qt.Orientation.Horizontal)
        self.volume_opacity_slider.setRange(0, 100)
        self.volume_opacity_slider.setValue(35)
        self.volume_opacity_slider.setMinimumWidth(50)
        self.volume_opacity_slider.setToolTip("CT volume opacity")
        self.volume_opacity_slider.valueChanged.connect(self._apply_ct_display)
        camera_row.addWidget(self.volume_opacity_slider, 1)
        layout.addLayout(camera_row)

        surface_row = QHBoxLayout()
        surface_row.setContentsMargins(6, 2, 6, 2)
        surface_row.setSpacing(6)
        self.show_surfaces_box = QCheckBox("Segmentation")
        self.show_surfaces_box.setToolTip("Show segmentation surfaces for the whole 3D volume")
        self.show_surfaces_box.setChecked(True)
        self.show_surfaces_box.toggled.connect(self._update_surfaces_visibility)
        surface_row.addWidget(self.show_surfaces_box)
        self.refresh_surfaces_btn = QPushButton("Refresh")
        self.refresh_surfaces_btn.setToolTip(
            "Rebuild 3D surfaces from all committed labels in the full volume."
        )
        self.refresh_surfaces_btn.setEnabled(False)
        self.refresh_surfaces_btn.clicked.connect(self.refresh_surfaces)
        surface_row.addWidget(self.refresh_surfaces_btn)
        surface_row.addWidget(QLabel("Opacity"))
        self.surface_opacity_slider = QSlider(Qt.Orientation.Horizontal)
        self.surface_opacity_slider.setRange(0, 100)
        self.surface_opacity_slider.setValue(90)
        self.surface_opacity_slider.setMinimumWidth(50)
        self.surface_opacity_slider.setToolTip("Segmentation surface opacity")
        self.surface_opacity_slider.valueChanged.connect(self._update_surfaces_opacity)
        surface_row.addWidget(self.surface_opacity_slider, 1)
        layout.addLayout(surface_row)

        paint_row = QHBoxLayout()
        paint_row.setContentsMargins(6, 2, 6, 4)
        paint_row.setSpacing(6)
        self.paint_toggle.setText("3D brush")
        paint_row.addWidget(self.paint_toggle)
        paint_row.addWidget(self.paint_mode)
        paint_row.addWidget(self.brush_shape)
        paint_row.addWidget(QLabel("Radius"))
        paint_row.addWidget(self.brush_radius_spin)
        paint_row.addStretch(1)
        layout.addLayout(paint_row)

        # Status strip kept below the render window; labels are elided rather than overflowing.
        self.surface_status_label = QLabel("No segmentation surfaces")
        self.surface_status_label.setToolTip(
            "Status of the 3D surfaces generated from the committed whole-volume labelmap."
        )
        self.render_mode_label = QLabel("Renderer: starting")
        self.render_mode_label.setToolTip(
            "VTK SmartVolumeMapper selects GPU or software volume rendering; the GPU name "
            "is the OpenGL device actually used by this window."
        )
        self.spacing_label = QLabel("Voxel spacing: not loaded")
        self.spacing_label.setToolTip(
            "Physical voxel size. TIFF pixel-resolution tags may describe page resolution, "
            "not sample spacing; use Image > Set TIFF spacing when acquisition metadata is known."
        )
        self._status_strip = QWidget()
        self._status_strip.setObjectName("viewStatusStrip")
        strip_layout = QHBoxLayout(self._status_strip)
        strip_layout.setContentsMargins(8, 3, 8, 3)
        strip_layout.setSpacing(14)
        for status_label in (self.render_mode_label, self.surface_status_label, self.spacing_label):
            status_label.setMinimumWidth(0)
            strip_layout.addWidget(status_label)
        strip_layout.addStretch(1)

        self.widget = None
        self._offscreen_disabled = (
            os.name == "nt" and os.environ.get("QT_QPA_PLATFORM", "").casefold() == "offscreen"
        )
        if self._offscreen_disabled:
            self._fallback = QWidget(self)
            fallback_layout = QVBoxLayout(self._fallback)
            from PyQt6.QtWidgets import QLabel as QtLabel

            message = QtLabel(
                "3D rendering is disabled in Windows Qt off-screen mode. "
                "Open the desktop app in a graphical session to use 3D."
            )
            message.setWordWrap(True)
            fallback_layout.addWidget(message)
            layout.addWidget(self._fallback)
            layout.addWidget(self._status_strip)
            self.renderer = None
            self._volume_actor = None
            self._source = None
            self._label_volume = None
            self._reduced_shape = None
            self.brush_radius = 2.0
            self.brush_shape.setEnabled(False)
            self.paint_mode.setEnabled(False)
            self._picker = None
            return
        self.widget = QVTKRenderWindowInteractor(self)
        self.widget.setMinimumSize(200, 160)
        layout.addWidget(self.widget, 1)
        layout.addWidget(self._status_strip)
        self.renderer = vtkRenderer()
        self.widget.GetRenderWindow().AddRenderer(self.renderer)
        self.renderer.SetBackground(0.08, 0.09, 0.11)
        self._volume_actor = vtkVolume()
        self._source: CTVolume | None = None
        self._label_volume: LabelVolume | None = None
        self._reduced_shape: tuple[int, int, int] | None = None
        self._surface_actors: list[vtkActor] = []
        self._retired_label_volumes: list[LabelVolume] = []
        self._deferred_preview_releases: list[tuple[CTVolume, np.ndarray]] = []
        self._vtk_initialized = False
        self._image_ready = False
        self._surface_request_id = 0
        self._requested_label_version = -1
        self._rendered_label_version = -1
        self._surface_worker_active = False
        self._surface_refresh_pending = False
        self._current_surface_worker: _LabelSurfaceWorker | None = None
        self._surface_pool = thread_pool or QThreadPool(self)
        try:
            from ctseg.core.resources import query_system_resources

            self._surface_pool.setMaxThreadCount(query_system_resources().recommended_threads)
        except Exception:
            self._surface_pool.setMaxThreadCount(4)
        self._surface_timer = QTimer(self)
        self._surface_timer.setSingleShot(True)
        self._surface_timer.setInterval(450)
        self._surface_timer.timeout.connect(self._start_surface_refresh)
        self._surfaces_ready.connect(self._apply_surface_result)
        self._surfaces_failed.connect(self._surface_build_failed)
        self._picker = vtkVolumePicker()
        self._picker.SetTolerance(0.01)
        self._picker.SetVolumeOpacityIsovalue(0.01)
        self._surface_picker = vtkCellPicker()
        self._surface_picker.SetTolerance(0.005)
        self._orientation_marker = None
        self.widget.installEventFilter(self)
        self.label_changed.connect(self._queue_surface_refresh)
        self.paint_toggle.toggled.connect(self._paint_mode_changed)

    def set_data(self, volume: CTVolume, label_volume: LabelVolume) -> None:
        self._update_spacing_label(volume)
        if self._offscreen_disabled:
            if self._label_volume is not None and self._label_volume is not label_volume:
                self._label_volume.close()
            self._source = volume
            self._label_volume = label_volume
            self._rendered_label_version = -1
            return
        previous_preview = self._preview_mask
        previous_source = self._source
        self._preview_mask = None
        if isinstance(previous_preview, np.memmap) and previous_source is not None:
            self.release_preview_mask(previous_preview, previous_source)
        if self._label_volume is not None:
            self._label_volume.unregister_change_callback(self._on_label_change)
            if self._label_volume is not label_volume:
                if self._surface_worker_active:
                    self._retired_label_volumes.append(self._label_volume)
                else:
                    self._label_volume.close()
        self._surface_timer.stop()
        self._surface_refresh_pending = False
        self._source = volume
        self._label_volume = label_volume
        self._image_ready = False
        self._rendered_label_version = -1
        self.refresh_surfaces_btn.setEnabled(True)
        self.paint_toggle.setEnabled(False)
        self.paint_mode.setEnabled(False)
        self.brush_shape.setEnabled(False)
        self.brush_radius_spin.setEnabled(False)
        label_volume.register_change_callback(self._on_label_change)
        if self.isVisible():
            self._prepare_data()
            self._initialize_renderer()
        self._queue_surface_refresh(label_volume.version)

    def refresh_surfaces(self) -> None:
        #Force a rebuild of visible 3D surfaces from the full committed labelmap.
        if self._offscreen_disabled:
            return
        if self._source is None or self._label_volume is None:
            self.paint_rejected.emit("Load a volume before building segmentation surfaces.")
            return
        if not self.show_surfaces_box.isChecked():
            self.show_surfaces_box.setChecked(True)
        self.surface_status_label.setText("Building surfaces…")
        self._queue_surface_refresh(self._label_volume.version)

    def _update_spacing_label(self, volume: CTVolume) -> None:
        spacing = " x ".join(f"{value:g}" for value in volume.spacing_xyz)
        if volume.geometry.spacing_confirmed:
            self.spacing_label.setText(f"Voxel spacing: {spacing} mm")
            self.spacing_label.setStyleSheet("")
        else:
            self.spacing_label.setText(f"Voxel spacing assumed: {spacing} mm")
            self.spacing_label.setStyleSheet("color: #e7b65c;")

    def _prepare_data(self) -> None:
        if self._image_ready or self.widget is None or self._source is None:
            return
        self._set_image(self._source)
        self._surface_actors.clear()
        self._preview_actor = None
        self._picker.InitializePickList()
        self._picker.PickFromListOn()
        self._picker.AddPickList(self._volume_actor)
        self._surface_picker.InitializePickList()
        self._surface_picker.PickFromListOn()
        self.paint_toggle.setEnabled(True)
        self.paint_mode.setEnabled(True)
        self.brush_shape.setEnabled(True)
        self.brush_radius_spin.setEnabled(True)
        self.renderer.ResetCamera()
        self._image_ready = True

    def _on_brush_radius_changed(self, value: float) -> None:
        self.brush_radius = float(value)
        self.brush_radius_changed.emit(self.brush_radius)

    def _on_brush_shape_changed(self, index: int) -> None:
        data = str(self.brush_shape.itemData(index))
        shape_name = "circle" if data == "ball" else "square"
        self.brush_shape_changed.emit(shape_name)

    def set_brush_radius(self, radius_mm: float) -> None:
        if radius_mm <= 0 or not np.isfinite(radius_mm):
            raise ValueError("brush radius must be positive and finite")
        self.brush_radius = float(radius_mm)
        self.brush_radius_spin.blockSignals(True)
        self.brush_radius_spin.setValue(self.brush_radius)
        self.brush_radius_spin.blockSignals(False)

    def set_brush_shape(self, shape: str) -> None:
        selected = "ball" if shape in {"circle", "ball", "round"} else "box"
        index = self.brush_shape.findData(selected)
        if index < 0:
            raise ValueError(f"unsupported 3D brush shape {shape!r}")
        self.brush_shape.blockSignals(True)
        self.brush_shape.setCurrentIndex(index)
        self.brush_shape.blockSignals(False)

    def set_preview_volume_mask(
        self,
        mask: np.ndarray | None,
        label_id: int = 1,
        color_rgb: tuple[int, int, int] | None = None,
    ) -> None:
        #Queue a temporary 3D surface without blocking the Qt/render thread.
        if self._offscreen_disabled:
            return
        if mask is None and self._preview_mask is None:
            return
        if mask is not None:
            if self._source is None or np.shape(mask) != self._source.shape_zyx:
                raise ValueError("3D preview mask must match the loaded volume")
        previous_mask = self._preview_mask
        self._preview_mask = mask
        self._preview_label_id = label_id
        if (
            previous_mask is not None
            and previous_mask is not mask
            and isinstance(previous_mask, np.memmap)
            and self._source is not None
        ):
            self.release_preview_mask(previous_mask, self._source)
        if self._label_volume is not None:
            self._queue_surface_refresh(self._label_volume.version)

    def release_preview_mask(self, mask: np.ndarray, owner: CTVolume) -> None:
        #Keep a memmap alive until the worker building its surface has finished.
        if (
            self._current_surface_worker is not None
            and self._current_surface_worker.preview_mask is mask
        ):
            self._deferred_preview_releases.append((owner, mask))
        else:
            owner.release_cache_array(mask)

    def clear_preview(self) -> None:
        #Clear any rendered 3D preview surface.
        self.set_preview_volume_mask(None)

    def _coords_from_event_pos(self, position) -> tuple[int, int, int] | None:
        if self.widget is None or self._source is None:
            return None
        render_width, render_height = self.widget.GetRenderWindow().GetSize()
        scale_x = render_width / max(1, self.widget.width())
        scale_y = render_height / max(1, self.widget.height())
        display_x = int(round(position.x() * scale_x))
        display_y = int(round((self.widget.height() - position.y()) * scale_y))
        picked = self._surface_picker.Pick(display_x, display_y, 0, self.renderer)
        picker = self._surface_picker if picked else self._picker
        if picked or picker.Pick(display_x, display_y, 0, self.renderer):
            world = picker.GetPickPosition()
            index_xyz = self._source.geometry.physical_to_index(world, allow_assumed=True)
            x, y, z = (int(round(value)) for value in index_xyz)
            shape_x, shape_y, shape_z = self._source.shape_xyz
            if 0 <= x < shape_x and 0 <= y < shape_y and 0 <= z < shape_z:
                return z, y, x
        return None

    def eventFilter(self, watched, event) -> bool:
        #Consume clicks and drags in paint mode so camera gestures remain unchanged.
        if self._offscreen_disabled:
            return False
        if watched is self.widget and self.paint_toggle.isChecked():
            event_type = event.type()
            if (
                event_type == QEvent.Type.MouseButtonPress
                and event.button() == Qt.MouseButton.LeftButton
            ):
                self._painting = True
                self._stroke_points = []
                coords = self._coords_from_event_pos(event.position())
                if coords is not None:
                    self._last_painted_voxel = coords
                    self._stroke_points.append(coords)
                else:
                    self._painting = False
                    self.paint_rejected.emit(
                        "Could not pick the volume at that position. Rotate/zoom the 3D view and "
                        "click an opaque part of the rendered volume."
                    )
                return True
            elif (
                event_type == QEvent.Type.MouseMove
                and self._painting
                and (event.buttons() & Qt.MouseButton.LeftButton)
            ):
                coords = self._coords_from_event_pos(event.position())
                if coords is not None and coords != self._last_painted_voxel:
                    self._last_painted_voxel = coords
                    self._stroke_points.append(coords)
                return True
            elif (
                event_type == QEvent.Type.MouseButtonRelease
                and event.button() == Qt.MouseButton.LeftButton
            ):
                if self._painting:
                    coords = self._coords_from_event_pos(event.position())
                    if coords is not None and coords != self._last_painted_voxel:
                        self._stroke_points.append(coords)
                    if self._stroke_points:
                        self.paint_stroke_requested.emit(
                            tuple(self._stroke_points),
                            bool(self.paint_mode.currentData()),
                        )
                    self._painting = False
                    self._last_painted_voxel = None
                    self._stroke_points = []
                    return True
        return super().eventFilter(watched, event)

    def _paint_mode_changed(self, enabled: bool) -> None:
        if self.widget is None:
            return
        if enabled:
            self.widget.setCursor(Qt.CursorShape.CrossCursor)
        else:
            self.widget.unsetCursor()

    def closeEvent(self, event) -> None:
        #Stop pending UI timers and detach from the shared data model on close.
        self._surface_timer.stop()
        if self._label_volume is not None:
            self._label_volume.unregister_change_callback(self._on_label_change)
        super().closeEvent(event)

    def showEvent(self, event) -> None:
        super().showEvent(event)
        self.on_tab_shown()

    def on_tab_shown(self) -> None:
        #Ensure surfaces and preview actors are refreshed when the 3D tab is opened.
        if self._offscreen_disabled or self._source is None:
            return
        self._prepare_data()
        self._initialize_renderer()
        if self._label_volume is not None and (
            getattr(self, "_surface_refresh_pending", False)
            or self._rendered_label_version != self._label_volume.version
        ):
            self._surface_refresh_pending = False
            self._queue_surface_refresh(self._label_volume.version)
        if self._preview_actor is not None:
            if hasattr(self, "show_surfaces_box") and self.show_surfaces_box.isChecked():
                self._preview_actor.SetVisibility(True)
        if self.widget is not None and self._vtk_initialized:
            self.widget.GetRenderWindow().Render()

    def _update_render_mode_label(self) -> None:
        mapper = self._volume_actor.GetMapper() if self._volume_actor is not None else None
        if mapper is None or not hasattr(mapper, "GetLastUsedRenderMode"):
            self.render_mode_label.setText("Renderer: unavailable")
            return
        modes = {
            vtkSmartVolumeMapper.GPURenderMode: "GPU ray-cast",
            vtkSmartVolumeMapper.RayCastRenderMode: "CPU ray-cast",
            vtkSmartVolumeMapper.OSPRayRenderMode: "OSPRay",
            vtkSmartVolumeMapper.AnariRenderMode: "ANARI",
        }
        mode_text = modes.get(mapper.GetLastUsedRenderMode(), "automatic")
        device = ""
        if self.widget is not None and self._vtk_initialized:
            device = opengl_renderer_name(self.widget.GetRenderWindow().ReportCapabilities())
        if not device:
            self.render_mode_label.setText(f"Renderer: {mode_text}")
            self.render_mode_label.setStyleSheet("")
            return
        vendor = classify_gpu(device)
        short_name = device.split("/")[0].replace("(R)", "").replace("(TM)", "").strip()
        self.render_mode_label.setText(f"{mode_text} · {short_name}")
        if vendor == "nvidia":
            self.render_mode_label.setStyleSheet("color: #76e08a;")
        else:
            self.render_mode_label.setStyleSheet("color: #e7b65c;")
            self.render_mode_label.setToolTip(
                f"OpenGL is running on {short_name}, not the NVIDIA GPU. Run "
                "'ctseg --prefer-nvidia-gpu' once and restart the app to use the RTX GPU."
            )

    def _initialize_renderer(self) -> None:
        if self.widget is None or not self._image_ready:
            return
        if not self._vtk_initialized:
            self.widget.Initialize()
            interactor = self.widget.GetRenderWindow().GetInteractor()
            self._camera_style = vtkInteractorStyleTrackballCamera()
            self._camera_style.SetDefaultRenderer(self.renderer)
            interactor.SetInteractorStyle(self._camera_style)
            interactor.SetDesiredUpdateRate(15.0)
            interactor.SetStillUpdateRate(5.0)
            axes = vtkAxesActor()
            self._orientation_marker = vtkOrientationMarkerWidget()
            self._orientation_marker.SetOrientationMarker(axes)
            self._orientation_marker.SetInteractor(interactor)
            self._orientation_marker.SetViewport(0.0, 0.0, 0.16, 0.16)
            self._orientation_marker.SetEnabled(1)
            self._orientation_marker.InteractiveOff()
            self._vtk_initialized = True
            self.renderer.ResetCamera()
        self.widget.GetRenderWindow().Render()
        self._update_render_mode_label()

    def _set_image(self, volume: CTVolume) -> None:
        if not hasattr(_vtk_volume_backend, "vtkOpenGLGPUVolumeRayCastMapper"):
            raise RuntimeError(
                "VTK OpenGL volume-rendering backend did not register its GPU mapper"
            )
        z, y, x = volume.shape_zyx
        stride = tuple(max(1, math.ceil(size / 192)) for size in (z, y, x))
        z_indices = list(range(0, z, stride[0]))
        y_step = stride[1]
        x_step = stride[2]
        reduced = np.empty(
            (len(z_indices), math.ceil(y / y_step), math.ceil(x / x_step)),
            dtype=volume.array.dtype,
        )
        for out_idx, z_idx in enumerate(z_indices):
            # Read single contiguous 2D plane from disk memmap
            reduced[out_idx] = volume.array[z_idx, ::y_step, ::x_step]
        reduced = np.ascontiguousarray(reduced)
        self._reduced_shape = reduced.shape
        image = vtkImageData()
        image.SetDimensions(reduced.shape[2], reduced.shape[1], reduced.shape[0])
        image.SetSpacing(
            volume.spacing_xyz[0] * stride[2],
            volume.spacing_xyz[1] * stride[1],
            volume.spacing_xyz[2] * stride[0],
        )
        image.SetOrigin(*volume.origin_xyz)
        direction = image.GetDirectionMatrix()
        for row in range(3):
            for column in range(3):
                direction.SetElement(row, column, volume.direction[row * 3 + column])
        image.SetDirectionMatrix(direction)
        scalars = numpy_to_vtk(reduced.ravel(order="C"), deep=True)
        scalars.SetName("CT intensity")
        image.GetPointData().SetScalars(scalars)
        mapper = vtkSmartVolumeMapper()
        mapper.SetRequestedRenderModeToDefault()
        mapper.SetInputData(image)
        properties = vtkVolumeProperty()
        properties.SetInterpolationTypeToLinear()
        properties.ShadeOn()
        colors = vtkColorTransferFunction()
        low, high = np.percentile(reduced[::4, ::4, ::4], (1, 99))
        if high <= low:
            high = low + 1
        colors.AddRGBPoint(float(low), 0.0, 0.0, 0.0)
        colors.AddRGBPoint(float((low + high) / 2), 0.75, 0.72, 0.68)
        colors.AddRGBPoint(float(high), 1.0, 1.0, 1.0)
        self._volume_low_high = (float(low), float(high))
        properties.SetColor(colors)
        # Opacity is expressed per displayed voxel, so the CT looks the same for a 1000-voxel
        # stack that is reduced for display and for a small one.
        properties.SetScalarOpacityUnitDistance(float(np.mean(volume.spacing_xyz) * max(stride)))
        self._volume_actor.SetMapper(mapper)
        self._volume_actor.SetProperty(properties)
        self._apply_ct_display(render=False)
        self.renderer.RemoveAllViewProps()
        self.renderer.AddVolume(self._volume_actor)

    def reset_camera(self) -> None:
        if self.renderer is not None:
            self.renderer.ResetCamera()
            if self.widget is not None and self._vtk_initialized:
                self.widget.GetRenderWindow().Render()

    def set_camera_preset(self, preset: str) -> None:
        #Provide Slicer-style axial, coronal, sagittal and perspective camera views.
        if self.renderer is None or self._volume_actor is None:
            return
        if preset == "Perspective":
            self.reset_camera()
            return
        bounds = self._volume_actor.GetBounds()
        if not bounds or any(not np.isfinite(value) for value in bounds):
            self.paint_rejected.emit("The volume has no valid display bounds.")
            return
        center = np.array(
            [
                (bounds[0] + bounds[1]) / 2,
                (bounds[2] + bounds[3]) / 2,
                (bounds[4] + bounds[5]) / 2,
            ]
        )
        camera = self.renderer.GetActiveCamera()
        distance = max(
            float(
                np.linalg.norm(
                    np.array(
                        [
                            bounds[1] - bounds[0],
                            bounds[3] - bounds[2],
                            bounds[5] - bounds[4],
                        ]
                    )
                )
            ),
            1.0,
        )
        directions = {
            "Axial": ((0.0, 0.0, 1.0), (0.0, 1.0, 0.0)),
            "Coronal": ((0.0, -1.0, 0.0), (0.0, 0.0, 1.0)),
            "Sagittal": ((1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
        }
        direction, view_up = directions[preset]
        position = center + np.asarray(direction) * distance * 1.5
        camera.SetFocalPoint(*center)
        camera.SetPosition(*position)
        camera.SetViewUp(*view_up)
        camera.ParallelProjectionOn()
        self.renderer.ResetCameraClippingRange()
        if self.widget is not None and self._vtk_initialized:
            self.widget.GetRenderWindow().Render()

    def _effective_ct_mode(self) -> str:
        #Resolve Auto: ghost the CT whenever segmentation surfaces would be hidden inside it.
        mode = self.ct_mode.currentText()
        if mode != "Auto":
            return mode
        segmentation_shown = self.show_surfaces_box.isChecked() and (
            bool(self._surface_actors) or self._preview_actor is not None
        )
        return "Ghost" if segmentation_shown else "Full"

    def _apply_ct_display(self, *_args, render: bool = True) -> None:
        #Set CT visibility/opacity for the current mode without rebuilding the volume.
        if self._volume_actor is None or not hasattr(self, "_volume_low_high"):
            return
        mode = self._effective_ct_mode()
        self._volume_actor.SetVisibility(mode != "Hidden")
        prop = self._volume_actor.GetProperty()
        if prop is not None and mode != "Hidden":
            slider = self.volume_opacity_slider.value() / 100.0
            # Ghost keeps the optical depth of the whole sample low so inner labels show through.
            peak = slider * (0.05 if mode == "Ghost" else 0.5)
            low, high = self._volume_low_high
            opacity = vtkPiecewiseFunction()
            opacity.AddPoint(float(low), 0.0)
            opacity.AddPoint(float(low + (high - low) * 0.45), peak * 0.12)
            opacity.AddPoint(float(high), peak)
            prop.SetScalarOpacity(opacity)
        if render and self.widget is not None and self._vtk_initialized:
            self.widget.GetRenderWindow().Render()

    def _update_surfaces_visibility(self, visible: bool) -> None:
        for actor in self._surface_actors:
            actor.SetVisibility(visible)
        if self._preview_actor is not None:
            self._preview_actor.SetVisibility(visible)
        if not visible:
            self.surface_status_label.setText("Surface display off")
        elif self._surface_actors:
            count = len(self._surface_actors)
            self.surface_status_label.setText(f"{count} segment surface(s)")
        elif self._label_volume is not None:
            self.surface_status_label.setText("No segment surfaces")
            if self._rendered_label_version != self._label_volume.version:
                self._queue_surface_refresh(self._label_volume.version)
        self._apply_ct_display()

    def _update_surfaces_opacity(self, value: int) -> None:
        op = value / 100.0
        for actor in self._surface_actors:
            actor.GetProperty().SetOpacity(op)
        if self._preview_actor is not None:
            self._preview_actor.GetProperty().SetOpacity(op)
        if self.widget is not None and self._vtk_initialized:
            self.widget.GetRenderWindow().Render()

    def _on_label_change(self, version: int) -> None:
        self.label_changed.emit(version)

    def _queue_surface_refresh(self, version: int) -> None:
        #Debounce edits so a brush drag creates at most one expensive surface build.
        if self._offscreen_disabled or self._source is None or self._label_volume is None:
            return
        self._surface_request_id += 1
        self._requested_label_version = int(version)
        if not self.isVisible():
            self._surface_refresh_pending = True
            return
        if self._surface_worker_active:
            self._surface_refresh_pending = True
            return
        self._surface_timer.start()

    def _start_surface_refresh(self) -> None:
        if self._surface_worker_active or self._source is None or self._label_volume is None:
            self._surface_refresh_pending = True
            return
        if not self.isVisible():
            self._surface_refresh_pending = True
            return
        self._surface_worker_active = True
        self._surface_refresh_pending = False
        self.surface_status_label.setText("Building surfaces…")
        request_id = self._surface_request_id
        label_version = self._requested_label_version
        preview_color = (255, 200, 60)
        if self._preview_mask is not None and self._label_volume is not None:
            preview_info = self._label_volume.get_label(self._preview_label_id)
            preview_color = preview_info.color
        worker = _LabelSurfaceWorker(
            request_id,
            label_version,
            self._label_volume.labelmap,
            tuple(
                info
                for info in self._label_volume.labels
                if self._label_volume.voxel_count(info.id) > 0
            ),
            self._source.spacing_xyz,
            self._source.origin_xyz,
            self._source.direction,
            self._source.shape_zyx,
            self._source,
            self._preview_mask,
            self._preview_label_id,
            preview_color,
        )
        worker.signals.ready.connect(self._apply_surface_result)
        worker.signals.failed.connect(self._surface_build_failed)
        self._current_surface_worker = worker
        self._surface_pool.start(worker)

    def _apply_surface_result(self, request_id: int, label_version: int, surfaces: object) -> None:
        self._current_surface_worker = None
        self._surface_worker_active = False
        if (
            self._source is None
            or self._label_volume is None
            or request_id != self._surface_request_id
            or label_version != self._label_volume.version
        ):
            self._release_retired_label_volumes()
            self._release_deferred_preview_masks()
            if self._surface_refresh_pending or (
                self._label_volume is not None
                and self._requested_label_version != self._rendered_label_version
            ):
                self._surface_refresh_pending = False
                self._surface_timer.start()
            return

        for actor in self._surface_actors:
            self.renderer.RemoveActor(actor)
        self._surface_actors.clear()
        if self._preview_actor is not None:
            self.renderer.RemoveActor(self._preview_actor)
            self._preview_actor = None
        self._surface_picker.InitializePickList()
        self._surface_picker.PickFromListOn()
        surfaces_visible = (
            self.show_surfaces_box.isChecked() if hasattr(self, "show_surfaces_box") else True
        )
        surface_opacity = (
            self.surface_opacity_slider.value() / 100.0
            if hasattr(self, "surface_opacity_slider")
            else 0.9
        )
        segment_actors, preview_actor = install_surface_actors(
            self.renderer, surfaces, surface_opacity, visible=surfaces_visible
        )
        self._surface_actors.extend(segment_actors)
        self._preview_actor = preview_actor
        for actor in (*segment_actors, *([preview_actor] if preview_actor is not None else [])):
            self._surface_picker.AddPickList(actor)
        segment_surface_count = len(segment_actors)
        preview_surface_present = preview_actor is not None
        self._rendered_label_version = label_version
        if not surfaces_visible:
            self.surface_status_label.setText("Surface display off")
        elif segment_surface_count:
            self.surface_status_label.setText(f"{segment_surface_count} segment surface(s)")
        elif preview_surface_present:
            self.surface_status_label.setText("Preview surface shown")
        else:
            self.surface_status_label.setText("No labeled voxels to render")
        self._apply_ct_display(render=False)
        if self.isVisible() and self._vtk_initialized:
            self.widget.GetRenderWindow().Render()
        self._release_retired_label_volumes()
        self._release_deferred_preview_masks()

    def _surface_build_failed(self, request_id: int, label_version: int, message: str) -> None:
        self._current_surface_worker = None
        self._surface_worker_active = False
        self.surface_status_label.setText("Surface build failed")
        self._release_retired_label_volumes()
        self._release_deferred_preview_masks()
        self.paint_rejected.emit(f"Could not refresh the 3D segmentation surface: {message}")
        if self._surface_refresh_pending or (
            self._label_volume is not None
            and self._requested_label_version != self._rendered_label_version
        ):
            self._surface_refresh_pending = False
            self._surface_timer.start()

    def _release_retired_label_volumes(self) -> None:
        retired, self._retired_label_volumes = self._retired_label_volumes, []
        for label_volume in retired:
            label_volume.close()

    def _release_deferred_preview_masks(self) -> None:
        deferred, self._deferred_preview_releases = self._deferred_preview_releases, []
        for owner, mask in deferred:
            owner.release_cache_array(mask)


def install_surface_actors(
    renderer,
    surfaces,
    opacity: float,
    *,
    visible: bool = True,
) -> tuple[list[vtkActor], vtkActor | None]:
    #Create an actor per built surface and add each to the renderer.
    #
    #Returns the segment actors and the (optional) preview actor. Adding to the renderer is
    #the step that makes segmentation visible, so it is kept here where it can be tested
    #without an OpenGL window.
    
    segment_actors: list[vtkActor] = []
    preview_actor: vtkActor | None = None
    for _label_id, color, surface_data, is_preview in surfaces:
        if surface_data.GetNumberOfPoints() == 0:
            continue
        actor = make_surface_actor(surface_data, color, opacity, is_preview=is_preview)
        actor.SetVisibility(visible)
        renderer.AddActor(actor)
        if is_preview:
            preview_actor = actor
        else:
            segment_actors.append(actor)
    return segment_actors, preview_actor


def make_surface_actor(
    surface_data: vtkPolyData,
    color_rgb: tuple[int, int, int],
    opacity: float,
    *,
    is_preview: bool = False,
) -> vtkActor:
   # Build a lit, flat-coloured actor for one segment surface.
    #
    #The contour filter attaches scalars to the mesh; if the mapper is allowed to colour by
    #them the surface renders black, so scalar colouring is disabled and the label colour is
    #applied explicitly.
    #
    mapper = vtkPolyDataMapper()
    mapper.SetInputData(surface_data)
    mapper.ScalarVisibilityOff()
    actor = vtkActor()
    actor.SetMapper(mapper)
    display_color = (255, 205, 70) if is_preview else color_rgb
    prop = actor.GetProperty()
    prop.SetColor(*(component / 255 for component in display_color))
    prop.SetOpacity(opacity * 0.72 if is_preview else opacity)
    prop.SetAmbient(0.35)
    prop.SetDiffuse(0.7)
    return actor


def _downsample_multi_label_blocks(
    labelmap_zyx: np.ndarray,
    label_ids: Sequence[int],
    stride_zyx: tuple[int, int, int],
) -> dict[int, np.ndarray]:
    #Extract coarse segment masks for multiple labels in a single pass over slabs.
    source = np.asanyarray(labelmap_zyx)
    if source.ndim != 3:
        raise ValueError("3D label reduction requires a ZYX labelmap")
    sz, sy, sx = stride_zyx
    out_z = math.ceil(source.shape[0] / sz)
    out_y = math.ceil(source.shape[1] / sy)
    out_x = math.ceil(source.shape[2] / sx)

    reduced_dict = {
        lid: np.zeros((out_z, out_y, out_x), dtype=np.uint8)
        for lid in label_ids
    }
    if not label_ids:
        return reduced_dict

    target_set = set(label_ids)

    for z_out in range(out_z):
        z_start = z_out * sz
        z_stop = min(source.shape[0], z_start + sz)
        slab = source[z_start:z_stop]
        if not np.any(slab != 0):
            continue
        present = set(np.unique(slab)) & target_set
        if not present:
            continue
        for lid in present:
            plane = np.any(slab == lid, axis=0)
            padded = np.zeros((out_y * sy, out_x * sx), dtype=bool)
            padded[: plane.shape[0], : plane.shape[1]] = plane
            reduced_dict[lid][z_out] = np.any(padded.reshape(out_y, sy, out_x, sx), axis=(1, 3))

    return reduced_dict


def _build_label_surfaces(
    labelmap_zyx: np.ndarray,
    labels,
    spacing_xyz: tuple[float, float, float],
    origin_xyz: tuple[float, float, float],
    direction: tuple[float, ...],
    shape_zyx: tuple[int, int, int],
    preview_mask: np.ndarray | None = None,
    preview_label_id: int = 1,
    preview_color: tuple[int, int, int] = (255, 200, 60),
) -> list[tuple[int, tuple[int, int, int], vtkPolyData, bool]]:
    #Build each visible segment surface without blocking the Qt/VTK render thread.
    stride_zyx = tuple(max(1, math.ceil(size / 192)) for size in shape_zyx)
    stride_xyz = (stride_zyx[2], stride_zyx[1], stride_zyx[0])
    display_spacing = tuple(spacing_xyz[axis] * stride_xyz[axis] for axis in range(3))
    output: list[tuple[int, tuple[int, int, int], vtkPolyData, bool]] = []

    visible_labels = [info for info in labels if info.visible]
    if visible_labels:
        visible_ids = [info.id for info in visible_labels]
        reduced_masks = _downsample_multi_label_blocks(labelmap_zyx, visible_ids, stride_zyx)
        for info in visible_labels:
            reduced_mask = reduced_masks.get(info.id)
            if reduced_mask is None or not np.any(reduced_mask):
                continue
            surface = _extract_binary_surface(
                reduced_mask,
                spacing_xyz=display_spacing,
                origin_xyz=origin_xyz,
                direction=direction,
                block_stride_xyz=stride_xyz,
            )
            if surface.GetNumberOfPoints():
                output.append((info.id, info.color, surface, False))

    if preview_mask is not None:
        preview = _downsample_binary_blocks(preview_mask, stride_zyx)
        if np.any(preview):
            surface = _extract_binary_surface(
                preview,
                spacing_xyz=display_spacing,
                origin_xyz=origin_xyz,
                direction=direction,
                block_stride_xyz=stride_xyz,
            )
            if surface.GetNumberOfPoints():
                output.append((preview_label_id, preview_color, surface, True))
    return output


def _downsample_binary_blocks(mask: np.ndarray, stride_zyx: tuple[int, int, int]) -> np.ndarray:
    #Preserve any foreground voxel in each display block instead of point-sampling.
#
    #Point-sampling a sparse brush label every 3–4 voxels can erase the entire
    #display surface even though the full-resolution labelmap is correct. Block
    #reduction deliberately favors visibility in the coarse 3D preview.
    #
    source = np.asanyarray(mask)
    if source.ndim != 3:
        raise ValueError("3D display reduction requires a ZYX mask")
    sz, sy, sx = stride_zyx
    out_z = math.ceil(source.shape[0] / sz)
    out_y = math.ceil(source.shape[1] / sy)
    out_x = math.ceil(source.shape[2] / sx)
    reduced = np.zeros((out_z, out_y, out_x), dtype=np.uint8)
    for z_out in range(out_z):
        z_start = z_out * sz
        z_stop = min(source.shape[0], z_start + sz)
        slab = source[z_start:z_stop]
        if not np.any(slab != 0):
            continue
        plane = np.any(slab != 0, axis=0)
        padded = np.zeros((out_y * sy, out_x * sx), dtype=bool)
        padded[: plane.shape[0], : plane.shape[1]] = plane
        blocks = padded.reshape(out_y, sy, out_x, sx)
        reduced[z_out] = np.any(blocks, axis=(1, 3))
    return reduced


def _downsample_label_blocks(
    labelmap_zyx: np.ndarray,
    label_id: int,
    stride_zyx: tuple[int, int, int],
) -> np.ndarray:
    #Create a coarse segment mask without allocating a full-volume boolean copy.
    if label_id <= 0:
        raise ValueError("foreground label IDs must be positive")
    source = np.asanyarray(labelmap_zyx)
    if source.ndim != 3:
        raise ValueError("3D label reduction requires a ZYX labelmap")
    sz, sy, sx = stride_zyx
    out_z = math.ceil(source.shape[0] / sz)
    out_y = math.ceil(source.shape[1] / sy)
    out_x = math.ceil(source.shape[2] / sx)
    reduced = np.zeros((out_z, out_y, out_x), dtype=np.uint8)
    for z_out in range(out_z):
        z_start = z_out * sz
        z_stop = min(source.shape[0], z_start + sz)
        slab = source[z_start:z_stop]
        if not np.any(slab == label_id):
            continue
        # The comparison is limited to one reduction slab (not the 1.7 GB full
        # reconstruction) to keep the asynchronous surface worker memory-safe.
        plane = np.any(slab == label_id, axis=0)
        padded = np.zeros((out_y * sy, out_x * sx), dtype=bool)
        padded[: plane.shape[0], : plane.shape[1]] = plane
        reduced[z_out] = np.any(padded.reshape(out_y, sy, out_x, sx), axis=(1, 3))
    return reduced


def _extract_binary_surface(
    mask_zyx: np.ndarray,
    *,
    spacing_xyz: tuple[float, float, float],
    origin_xyz: tuple[float, float, float],
    direction: tuple[float, ...],
    block_stride_xyz: tuple[int, int, int],
):
    #Extract a closed binary surface and preserve block-center coordinates.
    binary = np.pad(np.asarray(mask_zyx, dtype=np.uint8), 1)
    image = vtkImageData()
    image.SetDimensions(binary.shape[2], binary.shape[1], binary.shape[0])
    image.SetSpacing(*spacing_xyz)
    direction_matrix = np.asarray(direction, dtype=np.float64).reshape(3, 3)
    block_center_xyz = (np.asarray(block_stride_xyz, dtype=np.float64) - 1.0) / 2.0
    padded_origin = (
        np.asarray(origin_xyz)
        + direction_matrix
        @ (block_center_xyz * np.asarray(spacing_xyz) / np.asarray(block_stride_xyz))
        - direction_matrix @ np.asarray(spacing_xyz)
    )
    image.SetOrigin(*tuple(float(value) for value in padded_origin))
    matrix = image.GetDirectionMatrix()
    for row in range(3):
        for column in range(3):
            matrix.SetElement(row, column, float(direction_matrix[row, column]))
    image.SetDirectionMatrix(matrix)
    scalars = numpy_to_vtk(binary.ravel(order="C"), deep=True)
    scalars.SetName("binary segment")
    image.GetPointData().SetScalars(scalars)
    surface = vtkFlyingEdges3D()
    surface.SetInputData(image)
    surface.SetNumberOfContours(1)
    surface.SetValue(0, 0.5)
    surface.ComputeNormalsOn()
    surface.Update()
    output = vtkPolyData()
    output.ShallowCopy(surface.GetOutput())
    return output
