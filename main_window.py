#Standalone development shell for the embeddable CT segmentation widgets.

from __future__ import annotations

import logging
from dataclasses import replace
from pathlib import Path

import numpy as np

try:
    from PyQt6.QtCore import QSize, Qt, QThreadPool
    from PyQt6.QtGui import QAction, QActionGroup, QGuiApplication
    from PyQt6.QtWidgets import (
        QAbstractSpinBox,
        QComboBox,
        QDockWidget,
        QFileDialog,
        QFormLayout,
        QFrame,
        QInputDialog,
        QLabel,
        QMainWindow,
        QMessageBox,
        QScrollArea,
        QSizePolicy,
        QSplitter,
        QStyle,
        QToolBar,
        QVBoxLayout,
        QWidget,
    )
except ImportError as error:
    raise RuntimeError("Install ctseg-toolkit[ui] to launch the desktop application") from error

from ctseg.core.commands import (
    AddLabelCommand,
    CommandStack,
    DiskBackedVoxelDiffCommand,
    LabelMetadataCommand,
    RemoveEmptyLabelCommand,
    ReorderLabelsCommand,
)
from ctseg.core.label_volume import LabelInfo, LabelVolume
from ctseg.core.ops.boolean import combine_add
from ctseg.core.ops.components import (
    keep_largest_n,
    label_components,
    remove_small_islands,
    select_component,
)
from ctseg.core.ops.flood_fill import flood_fill_2d
from ctseg.core.ops.interpolation import propagate_morphological, propagate_nearest
from ctseg.core.ops.morphology import closing, dilate, erode, opening
from ctseg.core.ops.paint import (
    apply_mask_2d,
    apply_stamp,
    apply_stroke_3d,
    apply_volume_mask,
    make_stamp,
    replace_label_mask_2d,
)
from ctseg.core.ops.region_growing import grow_from_seed
from ctseg.core.ops.roi_shapes import ellipse_mask, rasterize_contour, rectangle_mask
from ctseg.core.ops.volume_segmentation import (
    boolean_volume_labels,
    grow_connected_3d,
    keep_largest_components_volume,
    label_mask_volume,
    morphology_volume_chunked,
    remove_small_islands_volume,
    select_label_component_3d,
    slice_seed_to_zyx,
    threshold_volume,
)
from ctseg.io.readers import read_roi_sidecar
from ctseg.io.writers import write_labelmap, write_roi_sidecar, write_session
from ctseg.ui.histogram_panel import HistogramPanel
from ctseg.ui.label_manager_panel import LabelManagerPanel
from ctseg.ui.propagation_panel import PropagationPanel
from ctseg.ui.roi_manager_panel import ROIManagerPanel
from ctseg.ui.slice_view import SliceView
from ctseg.ui.toolbox_panel import ToolboxPanel
from ctseg.ui.workers import LabelmapImportWorker, VolumeLoadWorker, VolumeOperationWorker

logger = logging.getLogger(__name__)


class MainWindow(QMainWindow):
    #Developer shell; host applications should embed the individual widgets.

    def __init__(self, initial_path: str | Path | None = None) -> None:
        super().__init__()
        self.setWindowTitle("CT Segmentation Toolkit")
        self.volume = None
        self.label_volume: LabelVolume | None = None
        self.command_stack = CommandStack()
        self._views: dict[int, SliceView] = {}
        self._active_axis = 2
        self._crosshair_xyz = [0, 0, 0]
        self._synchronizing = False
        self._preview: dict[str, object] | None = None
        self._roi_records: dict[int, dict[str, object]] = {}
        self._roi_redo_records: dict[int, dict[str, object]] = {}
        self._pending_seed_operation: tuple[str, float, float, str, int] | None = None
        self._volume_operation_generation = 0
        self._volume_worker: VolumeOperationWorker | None = None
        self._volume_operation_contexts: dict[int, dict[str, object]] = {}
        self._volume_load_generation = 0
        self._volume_load_worker: VolumeLoadWorker | None = None
        self._labelmap_import_generation = 0
        self._labelmap_import_worker: LabelmapImportWorker | None = None
        self._busy_operation = False
        self._thread_pool = QThreadPool(self)
        try:
            from ctseg.core.resources import query_system_resources

            self._resources = query_system_resources()
            self._thread_pool.setMaxThreadCount(self._resources.recommended_threads)
        except Exception:
            self._resources = None
            self._thread_pool.setMaxThreadCount(4)
        self._current_channel = 0
        self._solo_label_id: int | None = None
        self._saved_visibilities: dict[int, bool] = {}
        self._manual_tool_actions: dict[str, QAction] = {}
        self._label_callback = self._labels_changed
        self._build_window()
        self._fit_to_screen()
        if initial_path is not None:
            self.load_path(initial_path)

    def _fit_to_screen(self) -> None:
        #Open maximized and never demand more space than the screen provides.
        screen = QGuiApplication.primaryScreen()
        if screen is not None:
            available = screen.availableGeometry()
            self.setMinimumSize(min(960, available.width()), min(600, available.height()))
            self.resize(int(available.width() * 0.96), int(available.height() * 0.94))
        self.showMaximized()

    def _build_window(self) -> None:
        self._build_menus()
        workbench = QSplitter(Qt.Orientation.Horizontal, self)
        workbench.setChildrenCollapsible(False)
        self._main_tabs = workbench
        self.setCentralWidget(workbench)

        views_tab = QWidget()
        views_layout = QVBoxLayout(views_tab)
        views_layout.setContentsMargins(4, 4, 4, 4)
        views_layout.setSpacing(4)
        self._empty_view = QLabel(
            "Open a TIFF stack or image directory to begin.\n"
            "The supplied reconstruction stack contains individual TIFF slices."
        )
        self._empty_view.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._empty_view.setWordWrap(True)
        views_layout.addWidget(self._empty_view)

        # Axial on top (largest), coronal and sagittal side by side below; every divider is
        # draggable so the layout adapts to any screen without clipping.
        slice_panels: dict[int, QWidget] = {}
        for axis in (2, 1, 0):
            panel = QWidget()
            layout = QVBoxLayout(panel)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.setSpacing(2)
            label = QLabel(("Sagittal (X)", "Coronal (Y)", "Axial (Z)")[axis])
            label.setObjectName("paneTitle")
            view = SliceView(axis)
            view.slice_changed.connect(self._slice_changed)
            view.seed_selected.connect(self._seed_selected)
            view.view_focused.connect(self._view_focused)
            view.edit_committed.connect(self._slice_edit_committed)
            view.edit_failed.connect(self._edit_failed)
            view.voxel_edited.connect(self._edit_location_xyz)
            view.roi_committed.connect(self._record_roi)
            layout.addWidget(label)
            layout.addWidget(view, 1)
            slice_panels[axis] = panel
            self._views[axis] = view
        lower_views = QSplitter(Qt.Orientation.Horizontal)
        lower_views.setChildrenCollapsible(False)
        lower_views.addWidget(slice_panels[1])
        lower_views.addWidget(slice_panels[0])
        slice_splitter = QSplitter(Qt.Orientation.Vertical)
        slice_splitter.setChildrenCollapsible(False)
        slice_splitter.addWidget(slice_panels[2])
        slice_splitter.addWidget(lower_views)
        slice_splitter.setStretchFactor(0, 3)
        slice_splitter.setStretchFactor(1, 2)
        views_layout.addWidget(slice_splitter, 1)
        workbench.addWidget(views_tab)

        try:
            from ctseg.ui.volume_view import VolumeView3D

            self.volume_view = VolumeView3D()
            self.volume_view.paint_stroke_requested.connect(self._paint_3d_stroke)
            self.volume_view.paint_rejected.connect(self._edit_failed)
            self.volume_view.brush_radius_changed.connect(self._set_brush_from_3d)
            self.volume_view.brush_shape_changed.connect(self._set_brush_shape_from_3d)
            workbench.addWidget(self.volume_view)
            self._volume_view_error = None
        except (ImportError, RuntimeError) as error:
            self.volume_view = None
            self._volume_view_error = str(error)
            fallback = QLabel(f"3D rendering is unavailable.\n{error}")
            fallback.setAlignment(Qt.AlignmentFlag.AlignCenter)
            workbench.addWidget(fallback)
        workbench.setStretchFactor(0, 3)
        workbench.setStretchFactor(1, 2)
        workbench.setSizes([900, 650])

        dock_content = QWidget()
        dock_layout = QVBoxLayout(dock_content)
        dock_layout.setContentsMargins(6, 6, 6, 6)
        dock_layout.setSpacing(8)
        self.histogram = HistogramPanel()
        self.toolbox = ToolboxPanel()
        self.label_manager = LabelManagerPanel()
        self.roi_manager = ROIManagerPanel()
        self.propagation_panel = PropagationPanel()
        dock_layout.addWidget(self.histogram)
        dock_layout.addWidget(self.toolbox)
        dock_layout.addWidget(self.label_manager)
        dock_layout.addWidget(self.roi_manager)
        dock_layout.addWidget(self.propagation_panel)
        dock_layout.addStretch(1)
        # Let long labels wrap above their field and long combo items elide, so the panels
        # always fit the dock width instead of clipping.
        for form in dock_content.findChildren(QFormLayout):
            form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
            form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)
        for combo in dock_content.findChildren(QComboBox):
            combo.setSizeAdjustPolicy(
                QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
            )
            combo.setMinimumContentsLength(10)
        for field in (
            *dock_content.findChildren(QComboBox),
            *dock_content.findChildren(QAbstractSpinBox),
        ):
            field.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        # The panels are taller than a laptop screen; scroll instead of growing the window.
        dock_scroll = QScrollArea()
        dock_scroll.setWidgetResizable(True)
        dock_scroll.setFrameShape(QFrame.Shape.NoFrame)
        dock_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        dock_scroll.setMinimumWidth(320)
        dock_scroll.setWidget(dock_content)
        dock = QDockWidget("Tools and labels", self)
        dock.setWidget(dock_scroll)
        self._tools_dock = dock
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, dock)
        self.resizeDocks([dock], [400], Qt.Orientation.Horizontal)

        self.toolbox.tool_changed.connect(self._set_tool)
        self.toolbox.tool_combo.currentTextChanged.connect(self._sync_manual_tool_actions)
        self.toolbox.brush_radius_changed.connect(self._set_brush)
        self.toolbox.brush_shape_changed.connect(self._set_brush_shape)
        self.toolbox.window_level_changed.connect(self._set_window_level)
        self.toolbox.operation_requested.connect(self._run_operation)
        self.toolbox.apply_direct_requested.connect(self._run_operation_apply)
        self.toolbox.accept_preview_requested.connect(self._accept_preview)
        self.toolbox.cancel_preview_requested.connect(self._clear_preview)
        self.toolbox.undo_requested.connect(self._undo)
        self.toolbox.redo_requested.connect(self._redo)
        self.toolbox.otsu_requested.connect(self._auto_otsu_threshold)
        self.toolbox.intensity_range_changed.connect(self.histogram.set_selected_range)
        self.label_manager.active_label_changed.connect(self._set_active_label)
        self.label_manager.add_label_requested.connect(self._add_label)
        self.label_manager.remove_label_requested.connect(self._remove_label)
        self.label_manager.rename_label_requested.connect(self._rename_label)
        self.label_manager.reorder_label_requested.connect(self._reorder_label)
        self.label_manager.visibility_changed.connect(self._set_visibility)
        self.label_manager.lock_changed.connect(self._set_locked)
        self.label_manager.color_change_requested.connect(self._set_color)
        self.roi_manager.selected.connect(self._select_roi_definition)
        self.roi_manager.rename_requested.connect(self._rename_roi_definition)
        self.roi_manager.remove_requested.connect(self._remove_roi_definition)
        self.histogram.range_selected.connect(self._histogram_range_selected)
        self.histogram.label_selected.connect(self.label_manager.select_label)
        self.histogram.label_color_requested.connect(self._set_color)
        self.propagation_panel.propagation_requested.connect(self._run_propagation)
        self.toolbox.mask_constraint_changed.connect(self._mask_constraint_changed)
        self.label_manager.split_requested.connect(self._split_active_label)
        self.label_manager.isolate_requested.connect(self._isolate_label)
        self._build_task_toolbars()
        self._apply_professional_theme()
        if getattr(self, "_resources", None) is not None:
            self._hw_status_label = QLabel(self._resources.summary_string())
            self._hw_status_label.setStyleSheet(
                "color: #7ec8f0; font-size: 11px; padding-right: 8px; font-weight: bold;"
            )
            self.statusBar().addPermanentWidget(self._hw_status_label)
        self.statusBar().showMessage("Ready — open a TIFF stack to begin")
        self._sync_history_controls()

    def _build_menus(self) -> None:
        file_menu = self.menuBar().addMenu("&File")
        open_file = file_menu.addAction("Open TIFF/image file…")
        open_file.triggered.connect(self._open_file_dialog)
        open_directory = file_menu.addAction("Open TIFF slice directory…")
        open_directory.triggered.connect(self._open_directory_dialog)
        file_menu.addSeparator()
        export_action = file_menu.addAction("Export labelmap…")
        export_action.triggered.connect(self._export_labelmap)
        import_action = file_menu.addAction("Import labelmap into current image…")
        import_action.triggered.connect(self._import_labelmap)
        session_action = file_menu.addAction("Save session metadata…")
        session_action.triggered.connect(self._save_session)
        export_rois = file_menu.addAction("Export ROI definitions…")
        export_rois.triggered.connect(self._export_roi_definitions)
        import_rois = file_menu.addAction("Import ROI definitions…")
        import_rois.triggered.connect(self._import_roi_definitions)
        file_menu.addSeparator()
        close_action = file_menu.addAction("Exit")
        close_action.triggered.connect(self.close)

        edit_menu = self.menuBar().addMenu("&Edit")
        undo_action = edit_menu.addAction("Undo")
        undo_action.setShortcut("Ctrl+Z")
        undo_action.triggered.connect(self._undo)
        redo_action = edit_menu.addAction("Redo")
        redo_action.setShortcut("Ctrl+Y")
        redo_action.triggered.connect(self._redo)
        self._undo_action = undo_action
        self._redo_action = redo_action
        geometry_menu = self.menuBar().addMenu("&Image")
        spacing_action = geometry_menu.addAction("Set TIFF spacing…")
        spacing_action.triggered.connect(self._set_spacing_dialog)

    def _build_task_toolbars(self) -> None:
        #Expose frequently used tools as visible top-level task buttons.
        manual = QToolBar("Manual segmentation", self)
        manual.setObjectName("manualSegmentationToolbar")
        manual.setMovable(False)
        manual.setFloatable(False)
        manual.setIconSize(QSize(16, 16))
        manual.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextUnderIcon)
        manual.setStyleSheet("QToolButton { min-width: 50px; padding: 3px 4px; }")
        self.addToolBar(Qt.ToolBarArea.TopToolBarArea, manual)
        self._manual_action_group = QActionGroup(self)
        self._manual_action_group.setExclusive(True)
        manual_specs = (
            ("Navigate", "Navigate", QStyle.StandardPixmap.SP_ArrowBack),
            ("Brush", "Paint", QStyle.StandardPixmap.SP_DialogApplyButton),
            ("Erase", "Erase", QStyle.StandardPixmap.SP_TrashIcon),
            ("Rectangle ROI", "Rectangle", QStyle.StandardPixmap.SP_FileDialogDetailedView),
            ("Ellipse ROI", "Ellipse", QStyle.StandardPixmap.SP_DialogYesButton),
            ("Freehand ROI", "Freehand", QStyle.StandardPixmap.SP_BrowserReload),
            ("Polygon ROI", "Polygon", QStyle.StandardPixmap.SP_FileDialogListView),
        )
        for tool, title, icon in manual_specs:
            action = QAction(self.style().standardIcon(icon), title, self)
            action.setCheckable(True)
            action.setToolTip(f"Select {title} tool")
            action.triggered.connect(
                lambda checked=False, selected=tool: self._toolbar_select_tool(selected)
            )
            self._manual_action_group.addAction(action)
            manual.addAction(action)
            self._manual_tool_actions[tool] = action
        self._manual_tool_actions["Navigate"].setChecked(True)

        segmentation = QToolBar("Segmentation tasks", self)
        segmentation.setObjectName("segmentationTaskToolbar")
        segmentation.setMovable(False)
        segmentation.setFloatable(False)
        segmentation.setIconSize(QSize(16, 16))
        segmentation.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextUnderIcon)
        segmentation.setStyleSheet("QToolButton { min-width: 50px; padding: 3px 4px; }")
        self.addToolBar(Qt.ToolBarArea.TopToolBarArea, segmentation)

        operation_specs = (
            ("Threshold", "Threshold preview", QStyle.StandardPixmap.SP_FileDialogContentsView),
            ("Seed grow", "Seeded region grow", QStyle.StandardPixmap.SP_DialogOpenButton),
            ("Flood fill", "Flood fill", QStyle.StandardPixmap.SP_DialogResetButton),
            ("Component", "Select connected component", QStyle.StandardPixmap.SP_DirIcon),
            ("Dilate", "Dilate", QStyle.StandardPixmap.SP_ArrowUp),
            ("Erode", "Erode", QStyle.StandardPixmap.SP_ArrowDown),
            ("Close gaps", "Closing", QStyle.StandardPixmap.SP_DialogApplyButton),
            ("Open", "Opening", QStyle.StandardPixmap.SP_DialogCancelButton),
            ("Islands", "Remove small islands", QStyle.StandardPixmap.SP_TrashIcon),
            ("Largest", "Keep largest component", QStyle.StandardPixmap.SP_ComputerIcon),
            ("Split", "Split region into components", QStyle.StandardPixmap.SP_FileDialogToParent),
            ("Merge label", "Add another label", QStyle.StandardPixmap.SP_ArrowForward),
            ("Erase label", "Erase another label", QStyle.StandardPixmap.SP_DialogCancelButton),
        )
        for title, operation, icon in operation_specs:
            action = QAction(self.style().standardIcon(icon), title, self)
            action.setToolTip(f"Preview {operation} using the current parameter and scope controls")
            action.triggered.connect(
                lambda _checked=False, selected=operation: self._toolbar_preview_operation(selected)
            )
            segmentation.addAction(action)

        segmentation.addSeparator()
        otsu_action = QAction(
            self.style().standardIcon(QStyle.StandardPixmap.SP_BrowserReload), "Auto Otsu", self
        )
        otsu_action.setToolTip("Estimate the current slice threshold automatically")
        otsu_action.triggered.connect(self._auto_otsu_threshold)
        segmentation.addAction(otsu_action)
        preview_action = QAction(
            self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPlay), "Preview", self
        )
        preview_action.triggered.connect(self._toolbar_run_current_operation)
        segmentation.addAction(preview_action)
        apply_action = QAction(
            self.style().standardIcon(QStyle.StandardPixmap.SP_DialogApplyButton), "Apply", self
        )
        apply_action.triggered.connect(self._toolbar_apply_current_operation)
        segmentation.addAction(apply_action)
        cancel_action = QAction(
            self.style().standardIcon(QStyle.StandardPixmap.SP_DialogCancelButton), "Cancel", self
        )
        cancel_action.triggered.connect(self._clear_preview)
        segmentation.addAction(cancel_action)

        segmentation.addSeparator()
        self.channel_combo = QComboBox()
        self.channel_combo.addItem("Ch 0")
        self.channel_combo.setEnabled(False)
        self.channel_combo.setToolTip("Multichannel image: select active channel to view/segment")
        self.channel_combo.currentIndexChanged.connect(self._channel_selection_changed)
        segmentation.addWidget(QLabel(" Ch: "))
        segmentation.addWidget(self.channel_combo)

        views = QToolBar("View layout", self)
        views.setObjectName("viewLayoutToolbar")
        views.setMovable(False)
        views.setFloatable(False)
        views.setIconSize(QSize(16, 16))
        views.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextUnderIcon)
        views.setStyleSheet("QToolButton { min-width: 50px; padding: 3px 4px; }")
        self.addToolBar(Qt.ToolBarArea.TopToolBarArea, views)
        # Put the view buttons beside the manual tools (row 1) and keep the long
        # segmentation operations on their own row (row 2).
        self.insertToolBar(segmentation, views)
        self.insertToolBarBreak(segmentation)
        show_both = QAction(
            self.style().standardIcon(QStyle.StandardPixmap.SP_DesktopIcon),
            "2D + 3D",
            self,
        )
        show_both.setToolTip("Show orthogonal slices and the 3D scene side by side")
        show_both.triggered.connect(self._show_both_views)
        views.addAction(show_both)
        focus_2d = QAction(
            self.style().standardIcon(QStyle.StandardPixmap.SP_FileDialogDetailedView),
            "Focus 2D",
            self,
        )
        focus_2d.triggered.connect(lambda: self._set_workbench_sizes(1, 0))
        views.addAction(focus_2d)
        focus_3d = QAction(
            self.style().standardIcon(QStyle.StandardPixmap.SP_ComputerIcon),
            "Focus 3D",
            self,
        )
        focus_3d.triggered.connect(lambda: self._set_workbench_sizes(0, 1))
        views.addAction(focus_3d)
        reset_3d = QAction("Reset 3D camera", self)
        reset_3d.triggered.connect(
            lambda: self.volume_view.reset_camera() if self.volume_view is not None else None
        )
        views.addAction(reset_3d)

    def _apply_professional_theme(self) -> None:
        #Use a premium futuristic dark theme with glassmorphism and glow accents.
        self.setStyleSheet(
            """
            /* === Global Base === */
            QMainWindow, QWidget {
                font-family: "Segoe UI", "Inter", "Roboto", sans-serif;
                font-size: 13px;
            }
            QMainWindow { background-color: #0c1017; }
            QWidget { background-color: #0f1520; color: #dce4ef; }

            /* === Menu Bar === */
            QMenuBar {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #141a24, stop:1 #0d1219);
                color: #c8d6e5; padding: 5px 8px;
                border-bottom: 1px solid #1e2a3a;
                font-size: 13px;
            }
            QMenuBar::item { padding: 5px 12px; border-radius: 4px; }
            QMenuBar::item:selected { background: #1a3a5c; color: #7ec8f0; }
            QMenu {
                background: #131b27; border: 1px solid #243447;
                border-radius: 6px; padding: 4px;
            }
            QMenu::item { padding: 6px 28px 6px 16px; border-radius: 3px; }
            QMenu::item:selected { background: #1a3a5c; color: #7ec8f0; }
            QMenu::separator { height: 1px; background: #1e2a3a; margin: 4px 8px; }

            /* === Toolbars === */
            QToolBar {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #111821, stop:1 #0c1219);
                border: 0; spacing: 4px; padding: 5px 6px;
                border-bottom: 1px solid rgba(56, 189, 248, 0.08);
            }
            QToolBar::separator {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 transparent, stop:0.5 #2a5f8a, stop:1 transparent);
                width: 1px; margin: 6px 4px;
            }
            QToolButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #1a2332, stop:1 #141c28);
                border: 1px solid #273545; border-radius: 5px;
                color: #bccbe0; padding: 4px 8px;
                font-size: 11px;
            }
            QToolButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #1d3050, stop:1 #162540);
                border-color: #3b8ec9;
                color: #e0ecf8;
            }
            QToolButton:checked {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #1a4a75, stop:1 #133a5e);
                border-color: #4da3dc;
                color: #fff;
            }
            QToolButton:pressed, QPushButton:pressed {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #0e6eab, stop:1 #0a5a8e);
            }

            /* === Dock Widget === */
            QDockWidget {
                titlebar-close-icon: none; titlebar-normal-icon: none;
            }
            QDockWidget::title {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #111821, stop:1 #141e2c);
                padding: 8px 12px; font-weight: 700;
                font-size: 13px; color: #7ec8f0;
                border-bottom: 1px solid rgba(56, 189, 248, 0.12);
            }

            /* === Tabs === */
            QTabWidget::pane { border: 1px solid #1e2a3a; }
            QTabBar::tab {
                background: #151d28; padding: 8px 16px; margin-right: 2px;
                border-top-left-radius: 5px; border-top-right-radius: 5px;
                color: #8fa0b8;
            }
            QTabBar::tab:selected {
                background: #1a3a5c; color: #7ec8f0;
                border-bottom: 2px solid #38bdf8;
            }

            /* === Group Boxes === */
            QGroupBox {
                border: 1px solid #1e2a3a;
                border-radius: 8px;
                margin-top: 14px; padding: 12px 8px 8px 8px;
                font-weight: 700; font-size: 12px;
                background: rgba(15, 21, 32, 0.7);
            }
            QGroupBox::title {
                subcontrol-origin: margin; left: 12px;
                padding: 0 6px; color: #7ec8f0;
            }

            /* === Buttons === */
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #1a2738, stop:1 #141e2c);
                border: 1px solid #2a3e55;
                border-radius: 6px; padding: 6px 12px;
                color: #c0d0e4; font-size: 12px;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #1d3555, stop:1 #162b45);
                border-color: #3b8ec9; color: #e0ecf8;
            }
            QPushButton:disabled {
                color: #4a5568; background: #111821;
                border-color: #1e2a3a;
            }

            /* === Inputs === */
            QComboBox, QDoubleSpinBox, QSpinBox, QLineEdit {
                background: #0c1219;
                border: 1px solid #243447; border-radius: 5px;
                padding: 4px 8px; color: #c0d0e4;
                selection-background-color: #1a3a5c;
            }
            QComboBox:hover, QDoubleSpinBox:hover, QSpinBox:hover, QLineEdit:hover {
                border-color: #3b8ec9;
            }
            QComboBox:focus, QDoubleSpinBox:focus, QSpinBox:focus, QLineEdit:focus {
                border-color: #4da3dc;
            }
            QComboBox:disabled, QDoubleSpinBox:disabled, QSpinBox:disabled {
                color: #4a5568; background: #0a0f16; border-color: #1a2230;
            }
            QComboBox::drop-down {
                border: none; width: 22px;
            }
            QComboBox QAbstractItemView {
                background: #131b27; border: 1px solid #243447;
                selection-background-color: #1a3a5c;
                color: #c0d0e4;
            }

            /* === Sliders === */
            QSlider::groove:horizontal {
                height: 4px; background: #1e2a3a; border-radius: 2px;
            }
            QSlider::handle:horizontal {
                width: 14px; height: 14px; margin: -5px 0;
                background: #3b8ec9; border-radius: 7px;
                border: 1px solid #4da3dc;
            }
            QSlider::handle:horizontal:hover {
                background: #4da3dc;
            }
            QSlider::sub-page:horizontal {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #1a4a75, stop:1 #3b8ec9);
                border-radius: 2px;
            }

            /* === Check Boxes === */
            QCheckBox { spacing: 6px; color: #bccbe0; }
            QCheckBox::indicator {
                width: 16px; height: 16px;
                border: 1px solid #3b4f66; border-radius: 3px;
                background: #0c1219;
            }
            QCheckBox::indicator:checked {
                background: #38bdf8;
                border-color: #7ec8f0;
            }
            QCheckBox::indicator:hover {
                border-color: #3b8ec9;
            }

            /* === List Widgets === */
            QListWidget {
                background: #0c1219;
                border: 1px solid #1e2a3a; border-radius: 6px;
                padding: 4px;
            }
            QListWidget::item {
                padding: 5px 8px; border-radius: 4px;
                color: #bccbe0;
            }
            QListWidget::item:selected {
                background: #1a3a5c; color: #7ec8f0;
            }
            QListWidget::item:hover {
                background: #141e2c;
            }

            /* === Scroll Bars === */
            QScrollBar:vertical {
                background: #0c1017; width: 10px; border: none;
            }
            QScrollBar::handle:vertical {
                background: #243447; border-radius: 5px; min-height: 28px;
            }
            QScrollBar::handle:vertical:hover { background: #2a5f8a; }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
            QScrollBar:horizontal {
                background: #0c1017; height: 10px; border: none;
            }
            QScrollBar::handle:horizontal {
                background: #243447; border-radius: 5px; min-width: 28px;
            }
            QScrollBar::handle:horizontal:hover { background: #2a5f8a; }
            QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal { width: 0; }

            /* === Status Bar === */
            QStatusBar {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #0c1219, stop:1 #111821);
                color: #7ec8f0; font-size: 12px;
                border-top: 1px solid rgba(56, 189, 248, 0.08);
                padding: 4px 8px;
            }

            /* === Splitter === */
            QSplitter::handle {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #0c1219, stop:0.5 #1a3a5c, stop:1 #0c1219);
                width: 4px;
            }
            QSplitter::handle:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #0c1219, stop:0.5 #3b8ec9, stop:1 #0c1219);
            }

            /* === Labels === */
            QLabel {
                color: #8fa0b8; background: transparent;
                font-size: 12px;
            }

            /* === Form Layout === */
            QFormLayout { spacing: 6px; }

            QSplitter::handle:horizontal { width: 4px; }
            QSplitter::handle:vertical {
                height: 4px;
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #0c1219, stop:0.5 #1a3a5c, stop:1 #0c1219);
            }

            /* === Pane chrome === */
            QLabel#paneTitle {
                color: #7ec8f0; font-size: 11px; font-weight: 700;
                letter-spacing: 1px; padding: 2px 6px;
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 rgba(26, 58, 92, 0.65), stop:1 transparent);
                border-left: 2px solid #38bdf8;
            }
            QWidget#viewStatusStrip {
                background: #0c1219; border-top: 1px solid #1e2a3a;
            }
            QWidget#viewStatusStrip QLabel { font-size: 11px; }
            QScrollArea { background: transparent; border: none; }
            QScrollArea > QWidget > QWidget { background: #0f1520; }

            /* === Tooltip === */
            QToolTip {
                background: #1a2738; color: #c0d0e4;
                border: 1px solid #2a5f8a;
                border-radius: 4px; padding: 4px 8px;
                font-size: 11px;
            }
            """
        )
        # Apply accent styling to specific UI elements
        self._empty_view.setStyleSheet(
            "color: #4a7099; font-size: 16px; font-weight: 300; background: transparent;"
        )

    def _toolbar_select_tool(self, tool: str) -> None:
        names = {
            "Navigate": "Navigate",
            "Brush": "Brush",
            "Erase": "Erase",
            "Rectangle ROI": "Rectangle ROI",
            "Ellipse ROI": "Ellipse ROI",
            "Freehand ROI": "Freehand ROI",
            "Polygon ROI": "Polygon ROI",
        }
        self.toolbox.tool_combo.setCurrentText(names[tool])

    def _sync_manual_tool_actions(self, selected_name: str) -> None:
        for tool, action in self._manual_tool_actions.items():
            action.setChecked(tool == selected_name)

    def _toolbar_preview_operation(self, operation: str) -> None:
        self.toolbox.operation_combo.setCurrentText(operation)
        self._toolbar_run_current_operation()

    def _toolbar_run_current_operation(self) -> None:
        self._run_operation(
            self.toolbox.operation_combo.currentText(),
            self.toolbox.lower_value.value(),
            self.toolbox.upper_value.value(),
            self.toolbox.scope_combo.currentText(),
            int(self.toolbox.connectivity_combo.currentData() or 3),
        )

    def _toolbar_apply_current_operation(self) -> None:
        if self._preview is not None:
            self._accept_preview()
            return
        self._run_operation_apply(
            self.toolbox.operation_combo.currentText(),
            self.toolbox.lower_value.value(),
            self.toolbox.upper_value.value(),
            self.toolbox.scope_combo.currentText(),
            int(self.toolbox.connectivity_combo.currentData() or 3),
        )

    def _show_both_views(self) -> None:
        self._set_workbench_sizes(3, 2)

    def _set_workbench_sizes(self, left_weight: int, right_weight: int) -> None:
        total = max(1, self._main_tabs.width())
        if left_weight == 0:
            sizes = [0, total]
        elif right_weight == 0:
            sizes = [total, 0]
        else:
            ratio = left_weight / (left_weight + right_weight)
            sizes = [int(total * ratio), total - int(total * ratio)]
        self._main_tabs.setSizes(sizes)
        if right_weight and self.volume_view is not None:
            self.volume_view.on_tab_shown()

    def _apply_tool_settings(self) -> None:
        #Apply toolbox defaults after a source has established window/level values.
        tool_name = {
            "Navigate": "navigate",
            "Brush": "brush",
            "Erase": "erase",
            "Rectangle ROI": "rectangle",
            "Ellipse ROI": "ellipse",
            "Freehand ROI": "freehand",
            "Polygon ROI": "polygon",
        }[self.toolbox.tool_combo.currentText()]
        for view in self._views.values():
            view.set_tool(tool_name)
            view.set_brush(self.toolbox.brush_radius.value())
            view.brush_shape = str(self.toolbox.brush_shape.currentData())
        active_view = self._views[2]
        self.toolbox.set_window_level_values(
            active_view.window_width,
            active_view.window_center,
        )
        sample = np.asarray(
            self.volume.get_intensity_slice(2, self.volume.depth // 2, calibrated=True)
        )
        lower, upper = np.percentile(sample, (10, 90))
        self.toolbox.set_intensity_range(float(lower), float(upper))

    def load_path(self, path: str | Path, channel: int = 0) -> None:
        #Load a selected image asynchronously so large TIFF stacks do not freeze Qt.
        if self._busy_operation:
            self.statusBar().showMessage(
                "A whole-volume edit is being committed. Wait for it to finish "
                "before loading another image.",
                8000,
            )
            return
        if self._volume_load_worker is not None:
            self.statusBar().showMessage("An image load is already running.", 5000)
            return
        if self._preview is not None:
            self._clear_preview()
        self._volume_operation_generation += 1
        self._volume_worker = None
        self.toolbox.set_operation_enabled(True, "Preview / Apply")
        source = Path(path)
        self._current_channel = channel
        self._volume_load_generation += 1
        generation = self._volume_load_generation
        worker = VolumeLoadWorker(generation, source, channel=channel)
        worker.signals.finished.connect(self._volume_load_finished)
        worker.signals.failed.connect(self._volume_load_failed)
        self._volume_load_worker = worker
        self.statusBar().showMessage(
            f"Loading {source.name} (ch {channel}) in background. Interface remains responsive."
        )
        self._thread_pool.start(worker)

    def _volume_load_finished(self, generation: int, volume: object) -> None:
        if generation != self._volume_load_generation:
            volume.close()
            return
        self._volume_load_worker = None
        self._install_volume(volume)

    def _volume_load_failed(self, generation: int, message: str) -> None:
        if generation != self._volume_load_generation:
            return
        self._volume_load_worker = None
        self.statusBar().showMessage(f"Open failed: {message}", 12000)
        from PyQt6.QtWidgets import QApplication

        app = QApplication.instance()
        if app is None or app.platformName() != "offscreen":
            QMessageBox.critical(self, "Could not open image", message)

    def _install_volume(self, volume) -> None:
        #Replace the current project only after the background load succeeds.
        source = Path(volume.source) if volume.source else Path("image")
        if self.label_volume is not None:
            self.label_volume.unregister_change_callback(self._label_callback)
            try:
                self.label_volume.close()
            except Exception:
                pass
        if self.volume is not None and self.volume is not volume:
            try:
                self.volume.close()
            except Exception:
                pass
        self.volume = volume
        self.label_volume = LabelVolume(volume, memory_map=True)
        self.label_volume.register_change_callback(self._label_callback)
        self.command_stack.clear()
        self._preview = None
        self._pending_seed_operation = None
        self._roi_records.clear()
        self._roi_redo_records.clear()
        self.roi_manager.set_records(self._roi_records)
        self._empty_view.hide()
        self.label_manager.set_label_volume(self.label_volume)
        self.toolbox.set_labels(
            self.label_volume.labels,
            exclude_id=self.label_manager.current_label_id,
        )
        self._crosshair_xyz = [size // 2 for size in volume.shape_xyz]
        self._synchronizing = True
        try:
            for axis, view in self._views.items():
                view.set_data(volume, self.label_volume, self.command_stack)
                view.set_slice_index(self._crosshair_xyz[axis], emit=False)
                view.active_label_id = 1
            self.histogram.set_volume(volume)
            self.histogram.set_label_volume(self.label_volume)
            if self.volume_view is not None:
                try:
                    self.volume_view.set_data(volume, self.label_volume)
                    self._volume_view_error = (
                        "Qt off-screen mode disables hardware rendering"
                        if self.volume_view.widget is None
                        else None
                    )
                except Exception as error:
                    logger.exception("VTK 3D view failed to initialize")
                    self._volume_view_error = str(error)
                    self.statusBar().showMessage(f"3D view unavailable: {error}", 12000)
            self._apply_tool_settings()
            self.histogram.set_selected_range(
                self.toolbox.lower_value.value(),
                self.toolbox.upper_value.value(),
            )
            if self.volume_view is not None:
                self.volume_view.set_brush_radius(self.toolbox.brush_radius.value())
                self.volume_view.set_brush_shape(str(self.toolbox.brush_shape.currentData()))
        finally:
            self._synchronizing = False

        # Update multichannel channel selector
        try:
            from ctseg.io.tiff import inspect_tiff

            src_path = Path(volume.source)
            if src_path.is_file() and src_path.suffix.casefold() in {".tif", ".tiff"}:
                inspection = inspect_tiff(src_path)
                num_ch = max(1, getattr(inspection, "num_channels", 1))
            else:
                num_ch = 1
        except Exception:
            num_ch = 1

        self.channel_combo.blockSignals(True)
        self.channel_combo.clear()
        if num_ch > 1:
            for c in range(num_ch):
                self.channel_combo.addItem(f"Ch {c}", c)
            curr_ch = min(num_ch - 1, getattr(self, "_current_channel", 0))
            self.channel_combo.setCurrentIndex(curr_ch)
            self.channel_combo.setEnabled(True)
        else:
            self.channel_combo.addItem("Ch 0", 0)
            self.channel_combo.setEnabled(False)
        self.channel_combo.blockSignals(False)

        geometry_state = (
            "physical geometry confirmed"
            if volume.geometry.is_confirmed
            else "spacing/orientation incomplete — pixel-based editing only"
        )
        intensity_state = f"intensity unit: {volume.intensity_unit}"
        if volume.intensity_scale != 1.0 or volume.intensity_offset != 0.0:
            intensity_state += (
                f" (raw × {volume.intensity_scale:g} + {volume.intensity_offset:g}; "
                "used by assisted thresholds)"
            )
        self.statusBar().showMessage(
            f"Loaded {source.name} — {volume.width}×{volume.height}×{volume.depth}, "
            f"{volume.array.dtype}, {geometry_state}, {intensity_state}"
            + (f", 3D unavailable: {self._volume_view_error}" if self._volume_view_error else ""),
            0,
        )
        self._sync_history_controls()

    def _channel_selection_changed(self, index: int) -> None:
        if self.volume is None or index < 0:
            return
        channel = self.channel_combo.itemData(index)
        if channel is not None and channel != getattr(self, "_current_channel", 0):
            self.load_path(self.volume.source, channel=int(channel))

    def _get_active_mask_constraint(self) -> int | None:
        active_id = self.label_manager.current_label_id or 1
        return self.toolbox.get_mask_constraint(active_label_id=active_id)

    def _mask_constraint_changed(self, _constraint_data: object) -> None:
        mask_id = self._get_active_mask_constraint()
        for view in self._views.values():
            view.set_mask_constraint(mask_id)
        if mask_id is not None:
            name = f"Label {mask_id}"
            if self.label_volume is not None:
                try:
                    name = self.label_volume.get_label(mask_id).name
                except Exception:
                    pass
            self.statusBar().showMessage(f"Edits and assisted tools constrained inside region: '{name}'", 5000)
        else:
            self.statusBar().showMessage("Region constraint disabled (full scope)", 3000)

    def _split_active_label(self, label_id: int | None = None) -> None:
        #Trigger component splitting for the active or selected label.
        self.toolbox.operation_combo.setCurrentText("Split region into components")
        self._run_operation_apply(
            "Split region into components",
            0.0,
            self.toolbox.upper_value.value(),
            scope=self.toolbox.scope_combo.currentText(),
            connectivity=int(self.toolbox.connectivity_combo.currentData()),
        )

    def _isolate_label(self, label_id: int) -> None:
        #Toggle Solo mode: isolate this region by hiding other labels.
        if self.label_volume is None:
            return
        if getattr(self, "_solo_label_id", None) == label_id:
            for lid, vis in getattr(self, "_saved_visibilities", {}).items():
                if self.label_volume.has_label(lid):
                    self.label_volume.set_visibility(lid, vis)
            self._solo_label_id = None
            self._saved_visibilities = {}
            self.statusBar().showMessage("Restored visibility for all labels.", 4000)
        else:
            self._saved_visibilities = {
                info.id: info.visible for info in self.label_volume.labels
            }
            self._solo_label_id = label_id
            for info in self.label_volume.labels:
                self.label_volume.set_visibility(info.id, info.id == label_id)
            lbl_name = (
                self.label_volume.get_label(label_id).name
                if self.label_volume.has_label(label_id)
                else f"Label {label_id}"
            )
            self.statusBar().showMessage(
                f"Isolated region: '{lbl_name}'. Click Solo again to restore all labels.",
                6000,
            )
        self.label_manager.refresh()

    def _plane_representative_xyz(
        self, axis: int, slice_index: int, mask: np.ndarray
    ) -> tuple[int, int, int] | None:
        #Choose an edited voxel so other orthogonal views cross the same ROI.
        rows, columns = np.nonzero(mask)
        if not rows.size:
            return None
        row = int(np.median(rows))
        column = int(np.median(columns))
        if axis == 0:
            return slice_index, column, row
        if axis == 1:
            return column, slice_index, row
        return column, row, slice_index

    def _volume_representative_xyz(
        self,
        mask: np.ndarray,
        target_range: tuple[int, int, int] | None = None,
    ) -> tuple[int, int, int] | None:
        #Find a representative segmented voxel without allocating a 3D index list.
        if self.volume is None:
            return None
        if target_range is None:
            axis_xyz = 2
            first, last = 0, self.volume.depth - 1
        else:
            axis_xyz, first, last = target_range
        np_axis = 2 - axis_xyz
        length = self.volume.shape_xyz[axis_xyz]
        first = max(0, min(first, length - 1))
        last = max(first, min(last, length - 1))
        middle = (first + last) // 2
        indices = sorted(range(first, last + 1), key=lambda value: abs(value - middle))
        for plane_index in indices:
            plane = np.take(mask, plane_index, axis=np_axis)
            rows, columns = np.nonzero(plane)
            if rows.size:
                row = int(np.median(rows))
                column = int(np.median(columns))
                if axis_xyz == 0:
                    return plane_index, column, row
                if axis_xyz == 1:
                    return column, plane_index, row
                return column, row, plane_index
        return None

    def _open_file_dialog(self) -> None:
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "Open CT image",
            "",
            "Image volumes (*.tif *.tiff *.nii *.nii.gz *.nrrd);;All files (*.*)",
        )
        if filename:
            self.load_path(filename)

    def _open_directory_dialog(self) -> None:
        directory = QFileDialog.getExistingDirectory(self, "Open TIFF slice directory")
        if directory:
            self.load_path(directory)

    def _export_labelmap(self) -> None:
        if self.label_volume is None:
            return
        filename, selected_filter = QFileDialog.getSaveFileName(
            self,
            "Export labelmap",
            "",
            "TIFF stack (*.tif);;NRRD (*.nrrd);;NIfTI (*.nii.gz)",
        )
        if not filename:
            return
        path = Path(filename)
        if not path.suffix:
            path = path.with_suffix(
                ".tif"
                if "TIFF" in selected_filter
                else ".nrrd"
                if "NRRD" in selected_filter
                else ".nii.gz"
            )
        try:
            output = write_labelmap(self.label_volume, path)
        except (OSError, ValueError, RuntimeError) as error:
            QMessageBox.critical(self, "Export failed", str(error))
            return
        self.statusBar().showMessage(f"Labelmap exported to {output}", 10000)

    def _import_labelmap(self) -> None:
        #Replace the current segmentation overlay after an async geometry check.
        if self.volume is None or self.label_volume is None:
            QMessageBox.information(
                self, "No image", "Open a CT image before importing a labelmap."
            )
            return
        if self._busy_operation or self._labelmap_import_worker is not None:
            self.statusBar().showMessage("Another whole-volume operation is already running.", 6000)
            return
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "Import labelmap",
            "",
            "Labelmaps (*.tif *.tiff *.nrrd *.nii *.nii.gz)",
        )
        if not filename:
            return
        self._labelmap_import_generation += 1
        generation = self._labelmap_import_generation
        worker = LabelmapImportWorker(generation, filename, self.volume)
        worker.signals.finished.connect(self._labelmap_import_finished)
        worker.signals.failed.connect(self._labelmap_import_failed)
        self._labelmap_import_worker = worker
        self._labelmap_import_version = self.label_volume.version
        self._set_busy_operation(True)
        self.statusBar().showMessage(
            "Importing labelmap and checking image geometry in the background…"
        )
        self._thread_pool.start(worker)

    def _labelmap_import_finished(self, generation: int, labels: LabelVolume) -> None:
        if generation != self._labelmap_import_generation:
            return
        self._labelmap_import_worker = None
        if (
            self.volume is None
            or self.label_volume is None
            or self.label_volume.version != self._labelmap_import_version
        ):
            labels.close()
            self._set_busy_operation(False)
            self.statusBar().showMessage(
                "Labelmap import was discarded because the current segmentation changed.",
                10000,
            )
            return
        old_labels = self.label_volume
        old_labels.unregister_change_callback(self._label_callback)
        self.label_volume = labels
        labels.register_change_callback(self._label_callback)
        self.command_stack.clear()
        self._roi_records.clear()
        self._roi_redo_records.clear()
        self.label_manager.set_label_volume(labels)
        self.toolbox.set_labels(labels.labels, exclude_id=self.label_manager.current_label_id)
        for axis, view in self._views.items():
            view.set_data(self.volume, labels, self.command_stack)
            view.set_slice_index(self._crosshair_xyz[axis], emit=False)
        self.histogram.set_label_volume(labels)
        if self.volume_view is not None:
            self.volume_view.set_data(self.volume, labels)
        else:
            old_labels.close()
        self._set_busy_operation(False)
        self.statusBar().showMessage("Labelmap imported and aligned with the current image.", 8000)

    def _labelmap_import_failed(self, generation: int, message: str) -> None:
        if generation != self._labelmap_import_generation:
            return
        self._labelmap_import_worker = None
        self._set_busy_operation(False)
        self.statusBar().showMessage(f"Labelmap import failed: {message}", 12000)
        QMessageBox.critical(self, "Labelmap import failed", message)

    def _save_session(self) -> None:
        if self.volume is None or self.label_volume is None:
            return
        filename, _ = QFileDialog.getSaveFileName(
            self, "Save session metadata", "ctseg-session.json", "Session JSON (*.json)"
        )
        if not filename:
            return
        state = {
            "source": self.volume.source,
            "labelmap_path": Path(filename).stem + ".labels.tif",
            "roi_sidecar_path": Path(filename).stem + ".rois.json",
            "shape_zyx": list(self.volume.shape_zyx),
            "geometry": {
                "spacing_xyz": list(self.volume.spacing_xyz),
                "origin_xyz": list(self.volume.origin_xyz),
                "direction": list(self.volume.direction),
                "spacing_confirmed": self.volume.geometry.spacing_confirmed,
                "origin_confirmed": self.volume.geometry.origin_confirmed,
                "direction_confirmed": self.volume.geometry.direction_confirmed,
            },
            "intensity_unit": self.volume.intensity_unit,
            "labels": [
                {
                    "id": info.id,
                    "name": info.name,
                    "color": list(info.color),
                    "visible": info.visible,
                    "locked": info.locked,
                    "order": info.order,
                }
                for info in self.label_volume.labels
            ],
            "slice_crosshair_xyz": self._crosshair_xyz,
            "active_label_id": self.label_manager.current_label_id,
            "active_axis": self._active_axis,
            "window_level": {
                "width": self._views[2].window_width,
                "center": self._views[2].window_center,
            },
        }
        try:
            write_labelmap(
                self.label_volume,
                Path(filename).with_name(state["labelmap_path"]),
            )
            write_roi_sidecar(
                list(self._roi_records.values()),
                Path(filename).with_name(state["roi_sidecar_path"]),
            )
            write_session(state, filename)
        except (OSError, ValueError) as error:
            QMessageBox.critical(self, "Session save failed", str(error))
            return
        self.statusBar().showMessage(f"Session metadata saved to {filename}", 10000)

    def _record_roi(
        self,
        shape: str,
        axis: int,
        slice_index: int,
        vertices_xy: list[tuple[float, float]],
    ) -> None:
       # Associate editable ROI geometry with the voxel command that created it.
        if self.volume is None or self.label_volume is None:
            return
        command = self.command_stack.last_undo_command
        if command is None:
            return
        self._roi_redo_records.clear()
        self._roi_records[id(command)] = {
            "name": f"{shape.title()} ROI",
            "shape": shape,
            "axis": axis,
            "slice_index": slice_index,
            "vertices_xy": [[float(x), float(y)] for x, y in vertices_xy],
            "label_id": self.label_manager.current_label_id or 1,
            "label_name": self.label_volume.get_label(
                self.label_manager.current_label_id or 1
            ).name,
            "volume_shape_zyx": list(self.volume.shape_zyx),
            "spacing_xyz": list(self.volume.spacing_xyz),
            "origin_xyz": list(self.volume.origin_xyz),
            "direction": list(self.volume.direction),
        }
        self.roi_manager.set_records(self._roi_records)

    def _select_roi_definition(self, command_id: int) -> None:
        record = self._roi_records.get(command_id)
        if record is None:
            return
        try:
            normalized, mask = self._prepare_roi_record(record)
        except (KeyError, TypeError, ValueError, IndexError) as error:
            self.statusBar().showMessage(f"ROI cannot be selected: {error}", 8000)
            return
        index_xyz = self._plane_representative_xyz(
            int(normalized["axis"]),
            int(normalized["slice_index"]),
            mask,
        )
        if index_xyz is not None:
            self._edit_location_xyz(*index_xyz)

    def _rename_roi_definition(self, command_id: int) -> None:
        record = self._roi_records.get(command_id)
        if record is None:
            return
        name, accepted = QInputDialog.getText(
            self,
            "Rename ROI definition",
            "ROI name:",
            text=str(record.get("name", f"{record['shape']} ROI")),
        )
        if accepted and name.strip():
            record["name"] = name.strip()
            self.roi_manager.set_records(self._roi_records)

    def _remove_roi_definition(self, command_id: int) -> None:
        if self._roi_records.pop(command_id, None) is None:
            return
        self.roi_manager.set_records(self._roi_records)
        self.statusBar().showMessage(
            "Removed the editable ROI definition; the rasterized label voxels were preserved.",
            8000,
        )

    def _slice_edit_committed(self) -> None:
       # A new slice edit invalidates ROI redo metadata alongside command redo.
        self._roi_redo_records.clear()
        self._sync_history_controls()

    def _export_roi_definitions(self) -> None:
        if self.volume is None:
            return
        filename, _ = QFileDialog.getSaveFileName(
            self, "Export editable ROI definitions", "", "ROI definitions (*.json)"
        )
        if not filename:
            return
        try:
            write_roi_sidecar(list(self._roi_records.values()), filename)
        except (OSError, ValueError) as error:
            QMessageBox.critical(self, "ROI export failed", str(error))
            return
        self.statusBar().showMessage(
            f"Exported {len(self._roi_records)} editable ROI definitions.", 8000
        )

    def _import_roi_definitions(self) -> None:
        if self.volume is None or self.label_volume is None:
            QMessageBox.information(self, "No image", "Open an image before importing ROIs.")
            return
        filename, _ = QFileDialog.getOpenFileName(
            self, "Import editable ROI definitions", "", "ROI definitions (*.json)"
        )
        if not filename:
            return
        try:
            records = read_roi_sidecar(filename)
            prepared = [self._prepare_roi_record(record) for record in records]
            committed: list[dict[str, object]] = []
            for record, mask in prepared:
                label_id = int(record["label_id"])
                changed = apply_mask_2d(
                    self.label_volume,
                    int(record["axis"]),
                    int(record["slice_index"]),
                    mask,
                    label_id,
                    self.command_stack,
                    description=f"Import {record['shape']} ROI",
                )
                if changed:
                    command = self.command_stack.last_undo_command
                    self._roi_records[id(command)] = record
                    committed.append(record)
            if committed:
                self._roi_redo_records.clear()
                self.roi_manager.set_records(self._roi_records)
            self._labels_changed(self.label_volume.version)
            self.statusBar().showMessage(
                f"Imported {len(committed)} editable ROI definitions.", 8000
            )
        except (OSError, KeyError, TypeError, ValueError, IndexError, PermissionError) as error:
            QMessageBox.critical(self, "ROI import failed", str(error))

    def _prepare_roi_record(
        self, record: dict[str, object]
    ) -> tuple[dict[str, object], np.ndarray]:
        #Validate stored geometry and rasterize a supported contour for import.
        if self.volume is None or self.label_volume is None:
            raise RuntimeError("open a volume before importing ROI definitions")
        required = {"shape", "axis", "slice_index", "vertices_xy", "label_id"}
        missing = required - record.keys()
        if missing:
            raise ValueError(f"ROI record is missing fields: {', '.join(sorted(missing))}")
        source_shape = record.get("volume_shape_zyx")
        if source_shape is not None and tuple(source_shape) != self.volume.shape_zyx:
            raise ValueError("ROI definitions were created for a different volume shape")
        if record.get("spacing_xyz") is not None and not np.allclose(
            record["spacing_xyz"], self.volume.spacing_xyz, rtol=1e-5, atol=1e-6
        ):
            raise ValueError("ROI definitions use different voxel spacing")
        if record.get("origin_xyz") is not None and not np.allclose(
            record["origin_xyz"], self.volume.origin_xyz, rtol=1e-5, atol=1e-6
        ):
            raise ValueError("ROI definitions use a different image origin")
        if record.get("direction") is not None and not np.allclose(
            record["direction"], self.volume.direction, rtol=1e-5, atol=1e-6
        ):
            raise ValueError("ROI definitions use a different image orientation")
        axis = int(record["axis"])
        slice_index = int(record["slice_index"])
        label_id = int(record["label_id"])
        label_info = self.label_volume.get_label(label_id)
        self.label_volume.validate_edit(label_id)
        plane = self.volume.get_slice(axis, slice_index)
        vertices = [tuple(map(float, point)) for point in record["vertices_xy"]]
        shape = str(record["shape"])
        if shape in {"rectangle", "ellipse"}:
            if len(vertices) != 2:
                raise ValueError(f"{shape} ROI requires two corner points")
            mask = (
                rectangle_mask(vertices[0], vertices[1], plane.shape)
                if shape == "rectangle"
                else ellipse_mask(vertices[0], vertices[1], plane.shape)
            )
        elif shape in {"polygon", "freehand"}:
            mask = rasterize_contour(vertices, plane.shape)
        else:
            raise ValueError(f"unsupported ROI shape {shape!r}")
        normalized = dict(record)
        normalized["label_id"] = label_id
        normalized["label_name"] = label_info.name
        normalized["vertices_xy"] = [[x, y] for x, y in vertices]
        return normalized, mask

    def _set_spacing_dialog(self) -> None:
        if self.volume is None:
            return
        current = ",".join(str(value) for value in self.volume.spacing_xyz)
        text, accepted = QInputDialog.getText(
            self,
            "Set voxel spacing",
            "Enter voxel spacing in millimeters as X,Y,Z:",
            text=current,
        )
        if not accepted:
            return
        try:
            spacing = tuple(float(value.strip()) for value in text.split(","))
            if len(spacing) != 3 or any(value <= 0 or not np.isfinite(value) for value in spacing):
                raise ValueError("spacing requires three positive finite values")
        except ValueError as error:
            QMessageBox.warning(self, "Invalid spacing", str(error))
            return
        confirm_orientation = QMessageBox.question(
            self,
            "Confirm orientation",
            "Also confirm origin (0,0,0) and identity LPS direction? "
            "Only choose Yes if that matches the scan acquisition.",
        )
        geometry = replace(
            self.volume.geometry,
            spacing_xyz=spacing,
            spacing_confirmed=True,
            origin_confirmed=confirm_orientation == QMessageBox.StandardButton.Yes,
            direction_confirmed=confirm_orientation == QMessageBox.StandardButton.Yes,
        )
        self.volume.geometry = geometry
        self.statusBar().showMessage(
            "Voxel spacing updated. Origin/orientation are confirmed only if explicitly approved.",
            10000,
        )
        if self.volume_view is not None:
            try:
                self.volume_view.set_data(self.volume, self.label_volume)
            except Exception:
                logger.exception("Could not refresh the VTK view after geometry update")

    def _slice_changed(self, axis: int, index: int) -> None:
        if self._synchronizing or self.volume is None:
            return
        self._active_axis = axis
        self._crosshair_xyz[axis] = index
        self._synchronizing = True
        try:
            for view_axis, view in self._views.items():
                view.set_slice_index(self._crosshair_xyz[view_axis], emit=False)
        finally:
            self._synchronizing = False

    def _edit_location_xyz(self, x: int, y: int, z: int) -> None:
        #Center all orthogonal slice views on the voxel most recently edited.
        if self.volume is None:
            return
        self._crosshair_xyz = [
            int(np.clip(x, 0, self.volume.shape_xyz[0] - 1)),
            int(np.clip(y, 0, self.volume.shape_xyz[1] - 1)),
            int(np.clip(z, 0, self.volume.shape_xyz[2] - 1)),
        ]
        self._synchronizing = True
        try:
            for axis, view in self._views.items():
                target_idx = self._crosshair_xyz[axis]
                if view.slice_index != target_idx:
                    view.set_slice_index(target_idx, emit=False)
        finally:
            self._synchronizing = False

    def _histogram_range_selected(self, lower: float, upper: float) -> None:
        self.toolbox.set_intensity_range(lower, upper)
        self.histogram.set_selected_range(lower, upper)

    def _auto_otsu_threshold(self) -> None:
        if self.volume is None:
            QMessageBox.information(
                self, "No image", "Open an image before calculating Otsu threshold."
            )
            return
        from ctseg.core.ops.volume_segmentation import otsu_threshold

        view = self._views[self._active_axis]
        slice_data = self.volume.get_intensity_slice(view.axis, view.slice_index, calibrated=True)
        mask_id = self._get_active_mask_constraint()
        if mask_id is not None and self.label_volume is not None:
            label_slice = np.take(self.label_volume.labelmap, view.slice_index, axis=2 - view.axis)
            constrained_voxels = slice_data[label_slice == mask_id]
            if constrained_voxels.size > 0:
                threshold = otsu_threshold(constrained_voxels)
                max_val = float(np.nanmax(constrained_voxels))
            else:
                threshold = otsu_threshold(slice_data)
                max_val = float(np.nanmax(slice_data))
        else:
            threshold = otsu_threshold(slice_data)
            max_val = float(np.nanmax(slice_data))
        if threshold >= max_val:
            threshold = max_val - 1.0
        self.toolbox.set_intensity_range(threshold, max_val)
        self.histogram.set_selected_range(threshold, max_val)
        self.statusBar().showMessage(
            f"Auto Otsu threshold set: lower = {threshold:.2f}, upper = {max_val:.2f}"
            + (f" (constrained to region {mask_id})" if mask_id is not None else ""),
            6000,
        )

    def _view_focused(self, axis: int) -> None:
        self._active_axis = axis

    def _set_tool(self, tool: str) -> None:
        self._pending_seed_operation = None
        if self._volume_worker is not None:
            self._volume_operation_generation += 1
            self._volume_worker = None
            self.toolbox.set_operation_enabled(True, "Preview")
            self.toolbox._operation_selected(self.toolbox.operation_combo.currentText())
        self._clear_preview()
        for view in self._views.values():
            view.set_tool(tool)

    def _set_brush(self, radius: float) -> None:
        for view in self._views.values():
            view.set_brush(radius)
        if self.volume_view is not None:
            self.volume_view.set_brush_radius(radius)

    def _set_brush_from_3d(self, radius: float) -> None:
        self.toolbox.brush_radius.blockSignals(True)
        self.toolbox.brush_radius.setValue(radius)
        self.toolbox.brush_radius.blockSignals(False)
        for view in self._views.values():
            view.set_brush(radius)

    def _set_brush_shape(self, shape: str) -> None:
        for view in self._views.values():
            view.set_brush(view.brush_radius, shape)
        if self.volume_view is not None:
            self.volume_view.set_brush_shape(shape)

    def _set_brush_shape_from_3d(self, shape: str) -> None:
        shape_data = "circle" if shape in {"circle", "ball", "round"} else "square"
        for index in range(self.toolbox.brush_shape.count()):
            if self.toolbox.brush_shape.itemData(index) == shape_data:
                self.toolbox.brush_shape.blockSignals(True)
                self.toolbox.brush_shape.setCurrentIndex(index)
                self.toolbox.brush_shape.blockSignals(False)
                break
        for view in self._views.values():
            view.set_brush(view.brush_radius, shape_data)

    def _paint_3d(self, z: int, y: int, x: int, erase: bool = False) -> None:
        if self.label_volume is None or self.volume_view is None:
            return
        active_label = self.label_manager.current_label_id or 1
        radius = self.volume_view.brush_radius
        brush_shape = str(self.volume_view.brush_shape.currentData())
        if self.volume.geometry.spacing_confirmed:
            spacing_x, spacing_y, spacing_z = self.volume.spacing_xyz
            radius_zyx = (radius / spacing_z, radius / spacing_y, radius / spacing_x)
        else:
            radius_zyx = (radius, radius, radius)
        shape = "ball" if brush_shape in {"circle", "ball", "round"} else "box"
        try:
            stamp = make_stamp(shape, radius_zyx)
            changed = apply_stamp(
                self.label_volume,
                (z, y, x),
                stamp,
                0 if erase else active_label,
                self.command_stack,
                active_label_id=active_label if erase else None,
            )
            if changed:
                self._roi_redo_records.clear()
                self._edit_location_xyz(x, y, z)
                self.statusBar().showMessage(
                    f"{'Erased' if erase else f'Painted label {active_label}'} "
                    f"at voxel (x={x}, y={y}, z={z}).",
                    4000,
                )
            else:
                self.statusBar().showMessage(
                    "No new voxels painted. The brush may be outside the image or the target "
                    "voxels already belong to another label.",
                    6000,
                )
        except (ValueError, PermissionError, MemoryError) as error:
            QMessageBox.warning(self, "3D paint failed", str(error))
        self._sync_history_controls()

    def _paint_3d_stroke(self, points_zyx, erase: bool = False) -> None:
        #Commit a complete 3D drag as one bounded and undoable brush stroke.
        if self.label_volume is None or self.volume_view is None or self.volume is None:
            return
        active_label = self.label_manager.current_label_id or 1
        radius_mm = self.volume_view.brush_radius
        if self.volume.geometry.spacing_confirmed:
            spacing_x, spacing_y, spacing_z = self.volume.spacing_xyz
            radii_zyx = (
                radius_mm / spacing_z,
                radius_mm / spacing_y,
                radius_mm / spacing_x,
            )
        else:
            radii_zyx = (radius_mm, radius_mm, radius_mm)
        shape = str(self.volume_view.brush_shape.currentData())
        value = 0 if erase else active_label
        try:
            stamp = make_stamp(shape, radii_zyx)
            changed = apply_stroke_3d(
                self.label_volume,
                points_zyx,
                stamp,
                value,
                self.command_stack,
                active_label_id=active_label if erase else None,
            )
            if changed:
                last_z, last_y, last_x = points_zyx[-1]
                self._roi_redo_records.clear()
                self._edit_location_xyz(last_x, last_y, last_z)
                action = "Erased" if erase else f"Painted label {active_label}"
                self.statusBar().showMessage(
                    f"{action} a 3D stroke across {len(points_zyx)} picked positions.",
                    5000,
                )
            else:
                self.statusBar().showMessage(
                    "The 3D stroke did not change any voxels; the target may be outside the "
                    "volume or protected by another label.",
                    7000,
                )
        except (ValueError, PermissionError, MemoryError) as error:
            self.statusBar().showMessage(f"3D stroke was not applied: {error}", 12000)
        self._sync_history_controls()

    def _set_window_level(self, width: float, center: float) -> None:
        for view in self._views.values():
            view.set_window_level(width, center)

    def _set_active_label(self, label_id: int) -> None:
        for view in self._views.values():
            view.active_label_id = label_id
        if self.label_volume is not None:
            self.toolbox.set_labels(self.label_volume.labels, exclude_id=label_id)
        mask_id = self._get_active_mask_constraint()
        for view in self._views.values():
            view.set_mask_constraint(mask_id)

    def _add_label(self) -> None:
        if self.label_volume is None:
            return
        name, accepted = QInputDialog.getText(self, "Add label", "Label name:")
        if not accepted or not name.strip():
            return
        color = (64, 192, 255)
        try:
            info = LabelInfo(
                self.label_volume.next_label_id,
                name.strip(),
                color,
                order=max((info.order for info in self.label_volume.labels), default=-1) + 1,
            )
            self.command_stack.execute(AddLabelCommand(self.label_volume, info))
            self._roi_redo_records.clear()
        except (ValueError, OverflowError) as error:
            QMessageBox.warning(self, "Could not add label", str(error))
            return
        self.label_manager.refresh()
        self._labels_changed(self.label_volume.version)

    def _remove_label(self, label_id: int) -> None:
        if self.label_volume is None:
            return
        if len(self.label_volume.labels) <= 1:
            QMessageBox.information(
                self,
                "Keep one label",
                "At least one label must remain available for editing.",
            )
            return
        if self.label_volume.voxel_count(label_id):
            QMessageBox.warning(
                self,
                "Label contains voxels",
                "Clear or reassign this label before removing it. This avoids an untracked "
                "full-volume mutation.",
            )
            return
        try:
            self.command_stack.execute(
                RemoveEmptyLabelCommand(self.label_volume, self.label_volume.get_label(label_id))
            )
            self._roi_redo_records.clear()
        except (KeyError, PermissionError, MemoryError) as error:
            QMessageBox.warning(self, "Could not remove label", str(error))
        self.label_manager.refresh()
        self._sync_history_controls()

    def _rename_label(self, label_id: int) -> None:
        if self.label_volume is None:
            return
        info = self.label_volume.get_label(label_id)
        name, accepted = QInputDialog.getText(self, "Rename label", "Label name:", text=info.name)
        if accepted and name.strip():
            self._execute_metadata(info, name=name.strip())
            self.label_manager.refresh()

    def _reorder_label(self, label_id: int, direction: int) -> None:
        if self.label_volume is None:
            return
        ordered = list(self.label_volume.labels)
        current_index = next(
            (index for index, info in enumerate(ordered) if info.id == label_id), None
        )
        if current_index is None:
            return
        target_index = current_index + direction
        if not 0 <= target_index < len(ordered):
            return
        ordered[current_index], ordered[target_index] = (
            ordered[target_index],
            ordered[current_index],
        )
        before = {info.id: info.order for info in self.label_volume.labels}
        after = {info.id: position for position, info in enumerate(ordered)}
        self.command_stack.execute(ReorderLabelsCommand(self.label_volume, before, after))
        self._roi_redo_records.clear()
        self.label_manager.refresh()
        self._sync_history_controls()

    def _set_visibility(self, label_id: int, visible: bool) -> None:
        info = self.label_volume.get_label(label_id)
        self._execute_metadata(info, visible=visible)

    def _set_locked(self, label_id: int, locked: bool) -> None:
        info = self.label_volume.get_label(label_id)
        self._execute_metadata(info, locked=locked)
        self.label_manager.refresh()

    def _set_color(self, label_id: int, color: tuple[int, int, int]) -> None:
        info = self.label_volume.get_label(label_id)
        self._execute_metadata(info, color=color)
        self.label_manager.refresh()

    def _execute_metadata(self, before: LabelInfo, **changes) -> None:
        after = LabelInfo(
            before.id,
            changes.get("name", before.name),
            changes.get("color", before.color),
            changes.get("visible", before.visible),
            changes.get("locked", before.locked),
            changes.get("order", before.order),
        )
        if after != before:
            try:
                self.command_stack.execute(
                    LabelMetadataCommand(
                        self.label_volume,
                        before.id,
                        before,
                        after,
                        description="Change label properties",
                    )
                )
                self._roi_redo_records.clear()
            except (MemoryError, PermissionError, ValueError) as error:
                QMessageBox.warning(self, "Label update failed", str(error))
        self._sync_history_controls()

    def _tab_changed(self, index: int) -> None:
        if index == 1 and getattr(self, "volume_view", None) is not None:
            self.volume_view.on_tab_shown()

    def _run_operation_apply(
        self,
        operation: str,
        first: float,
        second: float,
        scope: str = "Current slice",
        connectivity: int = 3,
    ) -> None:
        if self._preview is not None:
            self._accept_preview()
            return
        self._run_operation(
            operation,
            first,
            second,
            scope=scope,
            connectivity=connectivity,
            auto_commit=True,
        )

    def _run_edit(
        self,
        operation,
        edited_index_xyz: tuple[int, int, int] | None = None,
    ) -> bool:
        #Execute an edit, report failures, and sync history controls.
        try:
            if operation():
                self._roi_redo_records.clear()
                if edited_index_xyz is not None:
                    self._edit_location_xyz(*edited_index_xyz)
                self._sync_history_controls()
                return True
        except (IndexError, MemoryError, PermissionError, ValueError, OSError) as error:
            QMessageBox.warning(self, "Edit failed", str(error))
        self._sync_history_controls()
        return False

    def _commit_or_preview_slice(
        self,
        view: SliceView,
        candidate: np.ndarray,
        *,
        replace_label: bool,
        overwrite: bool = False,
        erase_label_id: int | None = None,
        auto_commit: bool = False,
    ) -> None:
        if auto_commit:
            active_id = self.label_manager.current_label_id or 1
            target_id = 0 if erase_label_id is not None else active_id
            active_for_erase = active_id if erase_label_id is None else erase_label_id

            def commit_operation() -> bool:
                if erase_label_id is not None:
                    return apply_mask_2d(
                        self.label_volume,
                        view.axis,
                        view.slice_index,
                        candidate,
                        0,
                        self.command_stack,
                        active_label_id=active_for_erase,
                        overwrite=overwrite,
                        description="Erase selected label",
                    )
                if replace_label:
                    return replace_label_mask_2d(
                        self.label_volume,
                        view.axis,
                        view.slice_index,
                        candidate,
                        target_id,
                        self.command_stack,
                        overwrite=overwrite,
                        description="Apply assisted edit",
                    )
                return apply_mask_2d(
                    self.label_volume,
                    view.axis,
                    view.slice_index,
                    candidate,
                    target_id,
                    self.command_stack,
                    overwrite=overwrite,
                    description="Apply assisted edit",
                )

            self._run_edit(
                commit_operation,
                self._plane_representative_xyz(view.axis, view.slice_index, candidate),
            )
            self._clear_preview()
        else:
            self._stage_preview(
                view,
                candidate,
                replace_label=replace_label,
                overwrite=overwrite,
                erase_label_id=erase_label_id,
            )

    def _run_operation(
        self,
        operation: str,
        first: float,
        second: float,
        scope: str = "Current slice",
        connectivity: int = 3,
        *,
        auto_commit: bool = False,
    ) -> None:
        if self.volume is None or self.label_volume is None:
            QMessageBox.information(self, "No image", "Open an image before running tools.")
            return
        self._pending_seed_operation = None
        if self._preview is not None:
            self._clear_preview()
        view = self._views[self._active_axis]
        whole_volume = scope == "Whole volume"
        label_slice = np.take(self.label_volume.labelmap, view.slice_index, axis=2 - view.axis)
        active_id = self.label_manager.current_label_id or 1
        mask_id = self._get_active_mask_constraint()
        try:
            if whole_volume and operation == "Remove small islands":
                min_size = max(0, int(second))
                labels = self.label_volume
                self._start_volume_operation(
                    "Whole-volume island removal",
                    lambda: remove_small_islands_volume(labels, active_id, min_size),
                    active_id,
                    replace_active=True,
                    auto_commit=auto_commit,
                )
                return

            if whole_volume and operation == "Split region into components":
                min_size = max(1, int(second))
                self._start_split_volume_operation(active_id, min_size, connectivity)
                return

            if operation == "Split region into components":
                min_size = max(1, int(second))
                conn_2d = 1 if connectivity == 1 else 2
                active_mask = label_slice == active_id
                from ctseg.core.ops.components import split_components_2d
                from ctseg.core.ops.volume_segmentation import generate_distinct_label_colors

                sub_masks = split_components_2d(
                    active_mask, min_size=min_size, max_components=256, connectivity=conn_2d
                )
                if len(sub_masks) <= 1:
                    raise ValueError(
                        f"Region has {len(sub_masks)} component(s) >= {min_size} voxels; nothing to split."
                    )
                if auto_commit:
                    orig_label = self.label_volume.get_label(active_id)
                    num_new = len(sub_masks) - 1
                    existing_colors = [l.color for l in self.label_volume.labels]
                    new_colors = generate_distinct_label_colors(
                        num_new, existing_colors=existing_colors
                    )
                    for i in range(num_new):
                        new_id = self.label_volume.next_available_id()
                        self.label_volume.metadata.add_label(
                            name=f"{orig_label.name}_part_{i+2}",
                            color=new_colors[i],
                            target_id=new_id,
                        )
                        apply_mask_2d(
                            self.label_volume,
                            view.axis,
                            view.slice_index,
                            sub_masks[i + 1],
                            new_id,
                            self.command_stack,
                            overwrite=True,
                            description=f"Split region into component {i+2}",
                        )
                    self.label_manager.refresh()
                    self.toolbox.set_labels(
                        self.label_volume.labels, exclude_id=self.label_manager.current_label_id
                    )
                    self.statusBar().showMessage(
                        f"Split region into {len(sub_masks)} sub-regions on current slice.", 6000
                    )
                else:
                    preview_mask = np.zeros_like(active_mask, dtype=bool)
                    for m in sub_masks[1:]:
                        preview_mask |= m
                    self._commit_or_preview_slice(
                        view, preview_mask, replace_label=False, auto_commit=False
                    )
                return

            if operation == "Threshold preview":
                if first > second:
                    raise ValueError("lower threshold must not exceed upper threshold")
                if whole_volume:
                    volume = self.volume
                    labels = self.label_volume
                    self._start_volume_operation(
                        "Whole-volume threshold"
                        + (f" (inside region {mask_id})" if mask_id is not None else ""),
                        lambda: threshold_volume(
                            volume,
                            first,
                            second,
                            calibrated=True,
                            mask_volume=labels if mask_id is not None else None,
                            mask_label_id=mask_id,
                        ),
                        active_id,
                        replace_active=True,
                        auto_commit=auto_commit,
                    )
                else:
                    image_slice = self.volume.get_intensity_slice(
                        view.axis, view.slice_index, calibrated=True
                    )
                    candidate = (image_slice >= first) & (image_slice <= second)
                    if mask_id is not None:
                        candidate = candidate & (label_slice == mask_id)
                    if not np.any(candidate):
                        raise ValueError(
                            "No voxels match this intensity range within the active scope/constraint."
                        )
                    self._commit_or_preview_slice(
                        view, candidate, replace_label=True, auto_commit=auto_commit
                    )
            elif operation in {"Add another label", "Erase another label"}:
                other_id = self.toolbox.other_label_combo.currentData()
                if other_id is None or int(other_id) == active_id:
                    raise ValueError("select a different second label")
                if whole_volume:
                    bool_op = "add" if operation == "Add another label" else "erase"
                    labels = self.label_volume
                    self._start_volume_operation(
                        f"Whole-volume {bool_op} label",
                        lambda: boolean_volume_labels(labels, active_id, int(other_id), bool_op),
                        active_id,
                        replace_active=True,
                        auto_commit=auto_commit,
                    )
                    return
                other_mask = label_slice == int(other_id)
                if operation == "Add another label":
                    candidate = combine_add(label_slice == active_id, other_mask)
                    self._commit_or_preview_slice(
                        view,
                        candidate,
                        replace_label=True,
                        overwrite=True,
                        auto_commit=auto_commit,
                    )
                else:
                    if not np.any(other_mask):
                        raise ValueError("the selected second label is not present on this slice")
                    self._commit_or_preview_slice(
                        view,
                        other_mask,
                        replace_label=False,
                        erase_label_id=int(other_id),
                        auto_commit=auto_commit,
                    )
            elif operation == "Select connected component":
                self._pending_seed_operation = (
                    operation,
                    first,
                    second,
                    scope,
                    connectivity,
                    auto_commit,
                )
                for slice_view in self._views.values():
                    slice_view.set_tool("seed")
                self.statusBar().showMessage(
                    "Click a seed in the active label. "
                    "Whole-volume scope selects its 3D connected component."
                    if whole_volume
                    else "Click a voxel inside the active label."
                )
            elif operation in {"Seeded region grow", "Flood fill"}:
                self._pending_seed_operation = (
                    operation,
                    first,
                    second,
                    scope,
                    connectivity,
                    auto_commit,
                )
                for slice_view in self._views.values():
                    slice_view.set_tool("seed")
                self.statusBar().showMessage(
                    "Click a seed in any orthogonal view. Whole-volume scope grows through "
                    "connected voxels across the 3D stack."
                    if whole_volume
                    else "Click a seed voxel in the selected slice."
                )
            elif operation in {"Dilate", "Erode", "Opening", "Closing"}:
                function = {
                    "Dilate": dilate,
                    "Erode": erode,
                    "Opening": opening,
                    "Closing": closing,
                }[operation]
                if whole_volume:
                    radius = self._radius_zyx(first)
                    volume = self.volume
                    labels = self.label_volume
                    self._start_volume_operation(
                        f"Whole-volume {operation.lower()}",
                        lambda: self._morphology_active_label(
                            labels,
                            active_id,
                            radius,
                            operation.casefold(),
                            volume,
                        ),
                        active_id,
                        replace_active=True,
                        auto_commit=auto_commit,
                    )
                else:
                    radius = self._radius_yx(view.axis, first)
                    candidate = function(label_slice == active_id, radius)
                    if mask_id is not None:
                        candidate = candidate & (label_slice == mask_id)
                    self._commit_or_preview_slice(
                        view, candidate, replace_label=True, auto_commit=auto_commit
                    )
            elif operation == "Remove small islands":
                min_voxels = max(0, int(second))
                if whole_volume:
                    labels = self.label_volume
                    self._start_volume_operation(
                        "Whole-volume island removal",
                        lambda: remove_small_islands_volume(labels, active_id, min_voxels),
                        active_id,
                        replace_active=True,
                        auto_commit=auto_commit,
                    )
                else:
                    candidate = remove_small_islands(label_slice == active_id, min_voxels)
                    self._commit_or_preview_slice(
                        view, candidate, replace_label=True, auto_commit=auto_commit
                    )
            elif operation == "Keep largest component":
                num_to_keep = max(1, int(second))
                if whole_volume:
                    labels = self.label_volume
                    self._start_volume_operation(
                        f"Whole-volume keep largest {num_to_keep} component(s)",
                        lambda: keep_largest_components_volume(labels, active_id, num_to_keep),
                        active_id,
                        replace_active=True,
                        auto_commit=auto_commit,
                    )
                else:
                    active_mask = label_slice == active_id
                    labeled = label_components(active_mask)
                    candidate = keep_largest_n(labeled, num_to_keep)
                    self._commit_or_preview_slice(
                        view, candidate, replace_label=True, auto_commit=auto_commit
                    )
            else:
                raise ValueError(f"unsupported operation {operation!r}")
        except (ValueError, IndexError, PermissionError, RuntimeError, MemoryError) as error:
            QMessageBox.warning(self, "Operation failed", str(error))

    def _seed_selected(self, axis: int, slice_index: int, row: int, column: int) -> None:
        if self._pending_seed_operation is None or self.volume is None:
            return
        seed_data = self._pending_seed_operation
        operation, lower, upper, scope, connectivity = seed_data[:5]
        auto_commit = seed_data[5] if len(seed_data) > 5 else False
        view = self._views[axis]
        self._pending_seed_operation = None
        for slice_view in self._views.values():
            slice_view.set_tool("navigate")
        try:
            active_id = self.label_manager.current_label_id or 1
            mask_id = self._get_active_mask_constraint()
            if scope == "Whole volume" and operation == "Select connected component":
                seed_zyx = slice_seed_to_zyx(axis, slice_index, row, column)
                labels = self.label_volume
                self._start_volume_operation(
                    "Whole-volume connected component",
                    lambda: select_label_component_3d(
                        labels,
                        active_id,
                        seed_zyx,
                        connectivity=connectivity,
                    ),
                    active_id,
                    replace_active=True,
                    auto_commit=auto_commit,
                )
            elif scope == "Whole volume":
                seed_zyx = slice_seed_to_zyx(axis, slice_index, row, column)
                volume = self.volume
                seed_plane = volume.get_slice(2, seed_zyx[0])
                if volume.intensity_scale != 1.0 or volume.intensity_offset != 0.0:
                    seed_val = (
                        float(seed_plane[seed_zyx[1], seed_zyx[2]]) * volume.intensity_scale
                        + volume.intensity_offset
                    )
                else:
                    seed_val = float(seed_plane[seed_zyx[1], seed_zyx[2]])

                eff_lower = lower
                eff_upper = upper
                if operation in {"Flood fill", "Seeded region grow"} and (
                    seed_val < lower or seed_val > upper
                ):
                    half_span = abs(upper - lower) / 2.0
                    if half_span <= 0:
                        half_span = 50.0
                    eff_lower = seed_val - half_span
                    eff_upper = seed_val + half_span
                    self.toolbox.set_intensity_range(eff_lower, eff_upper)

                labels = self.label_volume
                self._start_volume_operation(
                    "Whole-volume seeded region"
                    + (f" (inside region {mask_id})" if mask_id is not None else ""),
                    lambda: grow_connected_3d(
                        volume,
                        seed_zyx,
                        eff_lower,
                        eff_upper,
                        connectivity=connectivity,
                        calibrated=True,
                        mask_volume=labels if mask_id is not None else None,
                        mask_label_id=mask_id,
                    ),
                    self.label_manager.current_label_id or 1,
                    replace_active=False,
                    auto_commit=auto_commit,
                )
            else:
                image = np.asarray(
                    self.volume.get_intensity_slice(axis, slice_index, calibrated=True)
                )
                seed = (row, column)
                plane_connectivity = 1 if connectivity == 1 else 2
                if operation == "Select connected component":
                    active_mask = (
                        np.take(
                            self.label_volume.labelmap,
                            slice_index,
                            axis=2 - axis,
                        )
                        == active_id
                    )
                    candidate = select_component(
                        active_mask,
                        seed,
                        connectivity=plane_connectivity,
                    )
                    if not np.any(candidate):
                        raise ValueError("the selected seed is not inside the active label")
                    self._commit_or_preview_slice(
                        view, candidate, replace_label=True, auto_commit=auto_commit
                    )
                    return
                seed_value = float(image[seed])
                eff_lower = lower
                eff_upper = upper
                if operation in {"Flood fill", "Seeded region grow"} and (
                    seed_value < lower or seed_value > upper
                ):
                    half_span = abs(upper - lower) / 2.0
                    if half_span <= 0:
                        half_span = 50.0
                    eff_lower = seed_value - half_span
                    eff_upper = seed_value + half_span
                    self.toolbox.set_intensity_range(eff_lower, eff_upper)

                if operation == "Seeded region grow":
                    candidate = grow_from_seed(
                        image,
                        seed,
                        eff_lower,
                        eff_upper,
                        connectivity=plane_connectivity,
                    )
                else:
                    connectivity_2d = 4 if plane_connectivity == 1 else 8
                    candidate = flood_fill_2d(
                        image,
                        seed,
                        connectivity=connectivity_2d,
                        lower=eff_lower,
                        upper=eff_upper,
                    )
                if mask_id is not None:
                    slice_label = np.take(self.label_volume.labelmap, slice_index, axis=2 - axis)
                    candidate = candidate & (slice_label == mask_id)
                if not np.any(candidate):
                    raise ValueError(
                        f"No voxels matched from this seed (intensity {seed_value:g}) within the region constraint; "
                        "expand the intensity range or adjust the seed."
                    )
                self._commit_or_preview_slice(
                    view, candidate, replace_label=False, auto_commit=auto_commit
                )
            if auto_commit:
                self._pending_seed_operation = seed_data
                for slice_view in self._views.values():
                    slice_view.set_tool("seed")
        except (ValueError, IndexError, RuntimeError, MemoryError) as error:
            self._pending_seed_operation = seed_data
            for slice_view in self._views.values():
                slice_view.set_tool("seed")
            QMessageBox.warning(self, "Seed operation failed", str(error))

    def _radius_zyx(self, radius: float) -> tuple[float, float, float]:
        if not self.volume.geometry.spacing_confirmed:
            return (radius, radius, radius)
        spacing_x, spacing_y, spacing_z = self.volume.spacing_xyz
        return (radius / spacing_z, radius / spacing_y, radius / spacing_x)

    @staticmethod
    def _morphology_active_label(
        labels: LabelVolume,
        label_id: int,
        radius: tuple[float, float, float],
        operation: str,
        volume,
    ):
        #Run chunked morphology and release the temporary input mask.
        source = label_mask_volume(labels, label_id)
        try:
            return morphology_volume_chunked(source, radius, operation, volume)
        finally:
            volume.release_cache_array(source)

    def _radius_yx(self, axis: int, radius: float) -> tuple[float, float]:
        if not self.volume.geometry.spacing_confirmed:
            return (radius, radius)
        row_axis, column_axis = ((2, 1), (2, 0), (1, 0))[axis]
        spacing = self.volume.spacing_xyz
        return (radius / spacing[row_axis], radius / spacing[column_axis])

    def _start_volume_operation(
        self,
        description: str,
        operation,
        label_id: int,
        *,
        replace_active: bool,
        auto_commit: bool = False,
    ) -> None:
        #Run volume-scale work in a worker and reject stale results on return.
        self._volume_operation_generation += 1
        generation = self._volume_operation_generation

        def compute_mask_and_count():
            mask = operation()
            count = sum(int(np.count_nonzero(mask[z])) for z in range(mask.shape[0]))
            return mask, count

        worker = VolumeOperationWorker(generation, compute_mask_and_count)
        worker.signals.finished.connect(self._volume_operation_finished)
        worker.signals.failed.connect(self._volume_operation_failed)
        self._volume_worker = worker
        self._volume_operation_contexts[generation] = {
            "generation": generation,
            "purpose": "calculate_mask",
            "auto_commit": auto_commit,
            "label_version": self.label_volume.version,
            "volume": self.volume,
            "label_volume": self.label_volume,
            "label_id": label_id,
            "replace_active": replace_active,
            "description": description,
        }
        self.toolbox.set_operation_enabled(False, f"Processing {description}…")
        self.statusBar().showMessage(
            f"{description}: computing over the full volume. The interface remains responsive."
        )
        self._thread_pool.start(worker)

    def _start_volume_commit(
        self,
        mask: np.ndarray,
        label_id: int,
        *,
        replace_active: bool,
        description: str,
    ) -> None:
        """Prepare and apply a volume-sized edit off the UI thread."""
        self._volume_operation_generation += 1
        generation = self._volume_operation_generation
        label_volume = self.label_volume
        command_stack = self.command_stack

        def apply_commit():
            command = DiskBackedVoxelDiffCommand.from_volume_mask(
                label_volume,
                mask,
                label_id,
                replace_active=replace_active,
                description=description,
            )
            if command is None:
                return None
            try:
                command_stack.validate_command(command)
                command.apply_silently()
            except Exception:
                label_volume.publish_change()
                command.dispose()
                raise
            return command

        worker = VolumeOperationWorker(generation, apply_commit)
        worker.signals.finished.connect(self._volume_operation_finished)
        worker.signals.failed.connect(self._volume_operation_failed)
        self._volume_worker = worker
        self._volume_operation_contexts[generation] = {
            "generation": generation,
            "purpose": "commit",
            "label_version": label_volume.version,
            "volume": self.volume,
            "label_volume": label_volume,
            "label_id": label_id,
            "replace_active": replace_active,
            "description": description,
        }
        self._set_busy_operation(True)
        self.statusBar().showMessage(
            f"Applying {description} in the background. The application remains responsive; "
            "please wait before closing."
        )
        self._thread_pool.start(worker)

    def _start_split_volume_operation(
        self, active_id: int, min_size: int, connectivity: int
    ) -> None:
        #Run memory-bounded 3D component splitting and relabeling across the whole volume.
        self._volume_operation_generation += 1
        generation = self._volume_operation_generation
        label_volume = self.label_volume
        command_stack = self.command_stack
        orig_name = label_volume.get_label(active_id).name

        def split_and_diff():
            from ctseg.core.ops.volume_segmentation import (
                split_label_components_volume,
                generate_distinct_label_colors,
            )

            relabeled_memmap, new_label_ids, comp_sizes = split_label_components_volume(
                label_volume,
                active_id,
                min_size=min_size,
                max_components=256,
                connectivity=connectivity,
            )
            if not new_label_ids:
                label_volume.reference.release_cache_array(relabeled_memmap)
                return None, 0, []

            try:
                diff_cmd = DiskBackedVoxelDiffCommand.from_relabeled_volume(
                    label_volume,
                    relabeled_memmap,
                    description=f"Split region '{orig_name}' into {len(new_label_ids) + 1} components",
                )
            finally:
                label_volume.reference.release_cache_array(relabeled_memmap)

            if diff_cmd is None:
                return None, 0, []

            existing_colors = [l.color for l in label_volume.labels]
            new_colors = generate_distinct_label_colors(
                len(new_label_ids), existing_colors=existing_colors
            )
            new_label_tuples = [
                (nid, f"{orig_name}_part_{idx + 2}", new_colors[idx])
                for idx, nid in enumerate(new_label_ids)
            ]

            return diff_cmd, len(new_label_ids) + 1, new_label_tuples

        worker = VolumeOperationWorker(generation, split_and_diff)
        worker.signals.finished.connect(self._split_volume_operation_finished)
        worker.signals.failed.connect(self._volume_operation_failed)
        self._volume_worker = worker
        self._volume_operation_contexts[generation] = {
            "generation": generation,
            "label_version": label_volume.version,
            "label_volume": label_volume,
            "active_id": active_id,
        }
        self._set_busy_operation(True)
        self.statusBar().showMessage(
            f"Dividing 3D region '{orig_name}' into components... The interface remains responsive."
        )
        self._thread_pool.start(worker)

    def _split_volume_operation_finished(self, generation: int, result: object) -> None:
        context = self._volume_operation_contexts.pop(generation, None)
        self._set_busy_operation(False)
        self.toolbox.set_operation_enabled(True, "Preview")
        if context is None or generation != self._volume_operation_generation:
            return
        self._volume_worker = None
        if not isinstance(result, tuple) or len(result) != 3:
            return
        diff_cmd, total_comps, new_label_tuples = result
        if diff_cmd is None or total_comps <= 1:
            self.statusBar().showMessage(
                "Region contains only 1 component (or none above the minimum size threshold).",
                6000,
            )
            return

        for nid, name, color in new_label_tuples:
            self.label_volume.add_label(name=name, color=color, label_id=nid)

        self.command_stack.validate_command(diff_cmd)
        diff_cmd.apply_silently()
        self.command_stack.accept_applied(diff_cmd)
        self.label_volume._notify()
        self.label_volume.publish_change()
        self.label_manager.refresh()
        self.toolbox.set_labels(
            self.label_volume.labels, exclude_id=self.label_manager.current_label_id
        )
        self.statusBar().showMessage(
            f"Successfully divided region into {total_comps} distinct 3D sub-region labels.",
            8000,
        )

    def _set_busy_operation(self, busy: bool) -> None:
        self._busy_operation = busy
        self._main_tabs.setEnabled(not busy)
        self._tools_dock.setEnabled(not busy)
        self.menuBar().setEnabled(not busy)
        if not busy:
            self._sync_history_controls()

    def _volume_operation_finished(self, generation: int, result: object) -> None:
        context = self._volume_operation_contexts.pop(generation, None)
        if context is None or generation != self._volume_operation_generation:
            if isinstance(result, DiskBackedVoxelDiffCommand):
                self.command_stack.discard_applied(result)
            if isinstance(result, tuple) and len(result) == 2:
                stale_mask = result[0]
                if isinstance(stale_mask, np.memmap):
                    context_volume = context.get("volume") if context is not None else None
                    if context_volume is not None:
                        context_volume.release_cache_array(stale_mask)
            return
        self._volume_worker = None
        if context.get("purpose") == "commit":
            command = result
            label_volume = context["label_volume"]
            try:
                if (
                    label_volume is not self.label_volume
                    or label_volume.version != context["label_version"]
                ):
                    if isinstance(command, DiskBackedVoxelDiffCommand):
                        self.command_stack.discard_applied(command)
                    self.statusBar().showMessage(
                        "The segmentation was rolled back because the active volume changed.",
                        10000,
                    )
                elif command is not None:
                    self.command_stack.accept_applied(command)
                    self.label_volume.publish_change()
                    self._clear_preview()
                    self.statusBar().showMessage(
                        f"{context['description']} completed across the full volume.",
                        8000,
                    )
                else:
                    self._clear_preview()
                    self.statusBar().showMessage("No new voxels were changed.", 6000)
            except (OSError, ValueError, PermissionError, MemoryError, RuntimeError) as error:
                logger.exception("Could not accept completed whole-volume edit")
                if isinstance(command, DiskBackedVoxelDiffCommand):
                    self.command_stack.discard_applied(command)
                QMessageBox.critical(self, "Could not apply segmentation", str(error))
            finally:
                self._set_busy_operation(False)
            return
        mask, selected_count = result
        if self.label_volume is None or self.label_volume.version != context["label_version"]:
            self.toolbox.set_operation_enabled(True, "Preview")
            self.toolbox._operation_selected(self.toolbox.operation_combo.currentText())
            self._release_volume_mask(mask, context["volume"])
            self.statusBar().showMessage(
                "The 3D result was discarded because labels changed while it was computing.",
                10000,
            )
            return
        if np.shape(mask) != self.volume.shape_zyx:
            self.toolbox.set_operation_enabled(True, "Preview")
            self.toolbox._operation_selected(self.toolbox.operation_combo.currentText())
            self._release_volume_mask(mask, context["volume"])
            QMessageBox.warning(
                self,
                "Invalid 3D result",
                "The assisted operation did not return a mask matching the loaded volume.",
            )
            return
        if not selected_count:
            self.toolbox.set_operation_enabled(True, "Preview")
            self.toolbox._operation_selected(self.toolbox.operation_combo.currentText())
            self._release_volume_mask(mask, context["volume"])
            QMessageBox.information(
                self,
                "No voxels selected",
                "No voxels matched. Adjust the thresholds or choose a different seed.",
            )
            return

        if context.get("auto_commit", False):
            self.toolbox.set_operation_enabled(True, "Preview")
            self.toolbox._operation_selected(self.toolbox.operation_combo.currentText())
            self._start_volume_commit(
                mask,
                int(context["label_id"]),
                replace_active=bool(context["replace_active"]),
                description=str(context["description"]),
            )
            return

        self.toolbox.set_operation_enabled(True, "Preview")
        self.toolbox._operation_selected(self.toolbox.operation_combo.currentText())
        self._stage_volume_preview(
            mask,
            int(context["label_id"]),
            replace_active=bool(context["replace_active"]),
            description=str(context["description"]),
        )

    @staticmethod
    def _release_volume_mask(mask: object, volume) -> None:
        if isinstance(mask, np.memmap):
            volume.release_cache_array(mask)

    def _volume_operation_failed(self, generation: int, message: str) -> None:
        context = self._volume_operation_contexts.pop(generation, None)
        if context is None or generation != self._volume_operation_generation:
            return
        self._volume_worker = None
        if context.get("purpose") == "commit":
            self._set_busy_operation(False)
            self.statusBar().showMessage(f"3D edit failed: {message}", 12000)
            QMessageBox.warning(self, "3D edit failed", message)
            return
        self.toolbox.set_operation_enabled(True, "Preview")
        self.toolbox._operation_selected(self.toolbox.operation_combo.currentText())
        self.statusBar().showMessage(f"3D operation failed: {message}", 12000)
        QMessageBox.warning(self, "3D operation failed", message)

    def _stage_preview(
        self,
        view: SliceView,
        mask: np.ndarray,
        *,
        replace_label: bool,
        overwrite: bool = False,
        erase_label_id: int | None = None,
    ) -> None:
        self._clear_preview()
        edited_index_xyz = self._plane_representative_xyz(view.axis, view.slice_index, mask)
        if edited_index_xyz is not None:
            self._edit_location_xyz(*edited_index_xyz)
        view.set_preview_mask(
            mask,
            replace_label=replace_label,
            erase_label_id=erase_label_id,
        )
        self._preview = {
            "axis": view.axis,
            "slice": view.slice_index,
            "label": self.label_manager.current_label_id or 1,
            "mask": np.asarray(mask, dtype=bool).copy(),
            "replace": replace_label,
            "overwrite": overwrite,
            "erase_label_id": erase_label_id,
            "version": self.label_volume.version,
            "volume_mask": None,
            "range": None,
        }
        self.toolbox.set_operation_enabled(False, "Accept preview")
        self.toolbox.run_operation.setEnabled(True)
        self.statusBar().showMessage(
            "Preview ready — accept with the button or select another tool to cancel."
        )

    def _stage_volume_preview(
        self,
        mask: np.ndarray,
        label_id: int,
        *,
        replace_active: bool,
        description: str,
    ) -> None:
        #Show per-view slices of one shared 3D candidate without copying the volume.
        self._clear_preview()
        edited_index_xyz = self._volume_representative_xyz(mask)
        if edited_index_xyz is not None:
            self._edit_location_xyz(*edited_index_xyz)
        view = self._views[self._active_axis]
        self._preview = {
            "axis": view.axis,
            "slice": view.slice_index,
            "label": label_id,
            "mask": np.take(mask, view.slice_index, axis=2 - view.axis),
            "replace": replace_active,
            "version": self.label_volume.version,
            "volume_mask": mask,
            "range": None,
            "description": description,
        }
        for slice_view in self._views.values():
            slice_view.set_preview_volume_mask(mask, replace_label=replace_active)
        if self.volume_view is not None:
            self.volume_view.set_preview_volume_mask(mask, label_id)
        self.toolbox.set_operation_enabled(False, "Accept preview")
        self.toolbox.run_operation.setEnabled(True)
        self.statusBar().showMessage(
            f"{description} preview is shown on all orthogonal views. "
            "Accept to commit the complete 3D mask as one undoable edit."
        )

    def _accept_preview(self) -> None:
        if self._preview is None or self.label_volume is None:
            return
        if self.label_volume.version != self._preview["version"]:
            QMessageBox.information(
                self, "Preview expired", "The labels changed after this preview; compute it again."
            )
            self._clear_preview()
            return
        axis = int(self._preview["axis"])
        slice_index = int(self._preview["slice"])
        label_id = int(self._preview["label"])
        mask = np.asarray(self._preview["mask"], dtype=bool)
        edited_index_xyz = None
        changed = False
        try:
            volume_mask = self._preview["volume_mask"]
            if volume_mask is not None:
                target_range = self._preview["range"]
                if target_range is None:
                    self._start_volume_commit(
                        volume_mask,
                        label_id,
                        replace_active=bool(self._preview["replace"]),
                        description=str(self._preview.get("description", "Apply 3D segmentation")),
                    )
                    return
                else:
                    axis, first, last = target_range
                    changed = apply_volume_mask(
                        self.label_volume,
                        volume_mask,
                        label_id,
                        self.command_stack,
                        replace_range=(axis, first, last),
                        replace_active=True,
                        description="Propagate segmentation",
                    )
                    edited_index_xyz = self._volume_representative_xyz(
                        volume_mask, (axis, first, last)
                    )
            elif self._preview.get("erase_label_id") is not None:
                erase_label_id = int(self._preview["erase_label_id"])
                changed = apply_mask_2d(
                    self.label_volume,
                    axis,
                    slice_index,
                    mask,
                    0,
                    self.command_stack,
                    active_label_id=erase_label_id,
                    description="Erase selected label",
                )
                edited_index_xyz = self._plane_representative_xyz(axis, slice_index, mask)
            elif bool(self._preview["replace"]):
                changed = replace_label_mask_2d(
                    self.label_volume,
                    axis,
                    slice_index,
                    mask,
                    label_id,
                    self.command_stack,
                    overwrite=bool(self._preview.get("overwrite", False)),
                    description="Apply assisted edit",
                )
                edited_index_xyz = self._plane_representative_xyz(axis, slice_index, mask)
            else:
                changed = apply_mask_2d(
                    self.label_volume,
                    axis,
                    slice_index,
                    mask,
                    label_id,
                    self.command_stack,
                    description="Apply assisted edit",
                )
                edited_index_xyz = self._plane_representative_xyz(axis, slice_index, mask)
        except (OSError, ValueError, PermissionError, MemoryError) as error:
            QMessageBox.warning(self, "Could not apply preview", str(error))
        self._clear_preview()
        if changed:
            self._roi_redo_records.clear()
        if changed and edited_index_xyz is not None:
            self._edit_location_xyz(*edited_index_xyz)
        self._sync_history_controls()

    def _run_propagation(self, method: str, keyframe_text: str, target_text: str) -> None:
        if self.label_volume is None:
            return
        try:
            keyframes = sorted(
                {int(value.strip()) - 1 for value in keyframe_text.split(",") if value.strip()}
            )
            target_values = [
                int(value.strip()) - 1 for value in target_text.split(",") if value.strip()
            ]
            if not keyframes:
                raise ValueError("Enter at least one keyframe slice number (1-based, e.g. 10, 20).")
            if method == "Contour interpolation":
                if not target_values:
                    if len(keyframes) < 2:
                        raise ValueError(
                            "Contour interpolation requires at least two keyframes (e.g. 10, 20)."
                        )
                    first, last = keyframes[0], keyframes[-1]
                elif len(target_values) == 2:
                    first, last = target_values
                else:
                    raise ValueError("Target range must have two values (e.g. 10, 20).")
            else:
                if len(target_values) != 2:
                    raise ValueError(
                        "Nearest slice copy requires a two-value target range (e.g. 11, 20)."
                    )
                first, last = target_values
            if first > last:
                raise ValueError("Target range start must not exceed its end.")
            if method == "Nearest slice copy":
                candidate = propagate_nearest(
                    self.label_volume,
                    self.label_manager.current_label_id or 1,
                    keyframes[0],
                    (first, last),
                    self._active_axis,
                )
            else:
                candidate = propagate_morphological(
                    self.label_volume,
                    self.label_manager.current_label_id or 1,
                    keyframes,
                    self._active_axis,
                )
        except (ValueError, IndexError, MemoryError) as error:
            QMessageBox.warning(self, "Propagation failed", str(error))
            return
        active_view = self._views[self._active_axis]
        preview_slice = (first + last) // 2
        active_view.set_slice_index(preview_slice, emit=False)
        self._slice_changed(self._active_axis, preview_slice)
        self._stage_volume_preview(
            candidate,
            self.label_manager.current_label_id or 1,
            replace_active=True,
            description="Propagate segmentation",
        )
        self._preview["range"] = (self._active_axis, first, last)

    def _clear_preview(self) -> None:
        had_preview = self._preview is not None
        preview_mask = self._preview.get("volume_mask") if self._preview is not None else None
        if self._preview is not None:
            self._preview = None
        for view in self._views.values():
            view.set_preview_mask(None)
        if self.volume_view is not None and had_preview:
            self.volume_view.set_preview_volume_mask(
                None,
                self.label_manager.current_label_id or 1,
            )
        if isinstance(preview_mask, np.memmap):
            preview_volume = self.volume
            if preview_volume is not None:
                if self.volume_view is not None:
                    self.volume_view.release_preview_mask(preview_mask, preview_volume)
                else:
                    preview_volume.release_cache_array(preview_mask)
        self.toolbox.set_operation_enabled(True, "Preview")
        self.toolbox._operation_selected(self.toolbox.operation_combo.currentText())

    def _undo(self) -> None:
        command = self.command_stack.last_undo_command
        if self.command_stack.undo() and command is not None:
            record = self._roi_records.pop(id(command), None)
            if record is not None:
                self._roi_redo_records[id(command)] = record
                self.roi_manager.set_records(self._roi_records)
        self._sync_history_controls()
        self.label_manager.refresh()

    def _redo(self) -> None:
        command = self.command_stack.last_redo_command
        if self.command_stack.redo() and command is not None:
            record = self._roi_redo_records.pop(id(command), None)
            if record is not None:
                self._roi_records[id(command)] = record
                self.roi_manager.set_records(self._roi_records)
        self._sync_history_controls()
        self.label_manager.refresh()

    def _sync_history_controls(self) -> None:
        self.toolbox.set_history_state(self.command_stack.can_undo(), self.command_stack.can_redo())
        if hasattr(self, "_undo_action"):
            self._undo_action.setEnabled(self.command_stack.can_undo())
            self._redo_action.setEnabled(self.command_stack.can_redo())

    def closeEvent(self, event) -> None:
        #Do not destroy the model underneath an active disk-backed 3D commit.
        if (
            self._busy_operation
            or self._volume_load_worker is not None
            or self._thread_pool.activeThreadCount() > 0
        ):
            self.statusBar().showMessage(
                "A background image/3D operation is still running. Wait for it to finish "
                "before closing.",
                8000,
            )
            event.ignore()
            return
        self._volume_load_generation += 1
        if self.label_volume is not None:
            self.label_volume.unregister_change_callback(self._label_callback)
        super().closeEvent(event)

    def _edit_failed(self, message: str) -> None:
        self.statusBar().showMessage(f"Edit was not applied: {message}", 10000)

    def _labels_changed(self, _version: int) -> None:
        self.label_manager.refresh()
        if self.label_volume is not None:
            for record in self._roi_records.values():
                label_id = int(record["label_id"])
                try:
                    record["label_name"] = self.label_volume.get_label(label_id).name
                except KeyError:
                    logger.warning("ROI definition references removed label ID %d", label_id)
            self.toolbox.set_labels(
                self.label_volume.labels,
                exclude_id=self.label_manager.current_label_id,
            )
            self.roi_manager.set_records(self._roi_records)
        self.histogram.refresh_labels()
        self._sync_history_controls()

    def keyPressEvent(self, event) -> None:
        if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            if self._preview is not None:
                self._accept_preview()
                event.accept()
                return
        elif event.key() == Qt.Key.Key_Escape:
            if self._preview is not None:
                self._clear_preview()
                event.accept()
                return
            if self._pending_seed_operation is not None:
                self._pending_seed_operation = None
                for slice_view in self._views.values():
                    slice_view.set_tool("navigate")
                self.statusBar().showMessage("Seed operation cancelled.", 3000)
                event.accept()
                return
        super().keyPressEvent(event)

