#Qt controls for manual and semi-automatic editing operations.

from __future__ import annotations

try:
    from PyQt6.QtCore import pyqtSignal
    from PyQt6.QtWidgets import (
        QComboBox,
        QDoubleSpinBox,
        QFormLayout,
        QGroupBox,
        QHBoxLayout,
        QPushButton,
        QVBoxLayout,
        QWidget,
    )
except ImportError as error:
    raise RuntimeError("Install ctseg-toolkit[ui] to use the Qt toolbox") from error


class ToolboxPanel(QWidget):
    #Expose simple, explicit controls while keeping operation logic in core.

    tool_changed = pyqtSignal(str)
    brush_radius_changed = pyqtSignal(float)
    brush_shape_changed = pyqtSignal(str)
    window_level_changed = pyqtSignal(float, float)
    operation_requested = pyqtSignal(str, float, float, str, int)
    apply_direct_requested = pyqtSignal(str, float, float, str, int)
    accept_preview_requested = pyqtSignal()
    cancel_preview_requested = pyqtSignal()
    undo_requested = pyqtSignal()
    redo_requested = pyqtSignal()
    otsu_requested = pyqtSignal()
    intensity_range_changed = pyqtSignal(float, float)
    mask_constraint_changed = pyqtSignal(object)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._last_intensity_lower = 0.0
        self._last_intensity_upper = 65535.0
        self._last_morphology_radius = 2.0
        self._last_island_min_size = 50.0
        self._last_largest_n = 1.0
        self._current_op_type = "threshold"

        layout = QVBoxLayout(self)
        edit_group = QGroupBox("2D editing")
        edit_form = QFormLayout(edit_group)
        self.tool_combo = QComboBox()
        self.tool_combo.addItems(
            [
                "Navigate",
                "Brush",
                "Erase",
                "Rectangle ROI",
                "Ellipse ROI",
                "Freehand ROI",
                "Polygon ROI",
            ]
        )
        self.tool_combo.currentTextChanged.connect(self._tool_selected)
        edit_form.addRow("Tool", self.tool_combo)
        self.brush_radius = QDoubleSpinBox()
        self.brush_radius.setRange(0.1, 100)
        self.brush_radius.setDecimals(1)
        self.brush_radius.setValue(2.0)
        self.brush_radius.setToolTip(
            "Radius in millimeters when spacing is confirmed; otherwise treated as voxel radii."
        )
        self.brush_radius.valueChanged.connect(self.brush_radius_changed)
        edit_form.addRow("Brush radius (mm / voxels*)", self.brush_radius)
        self.brush_shape = QComboBox()
        self.brush_shape.addItem("Round", "circle")
        self.brush_shape.addItem("Square", "square")
        self.brush_shape.currentIndexChanged.connect(
            lambda _index: self.brush_shape_changed.emit(str(self.brush_shape.currentData()))
        )
        edit_form.addRow("Brush shape", self.brush_shape)
        history_row = QHBoxLayout()
        self.undo_button = QPushButton("Undo")
        self.redo_button = QPushButton("Redo")
        self.undo_button.clicked.connect(self.undo_requested)
        self.redo_button.clicked.connect(self.redo_requested)
        history_row.addWidget(self.undo_button)
        history_row.addWidget(self.redo_button)
        edit_form.addRow("History", history_row)
        layout.addWidget(edit_group)

        display_group = QGroupBox("Window / level")
        display_form = QFormLayout(display_group)
        self.window_width = QDoubleSpinBox()
        self.window_width.setRange(1, 131072)
        self.window_width.setDecimals(1)
        self.window_width.setValue(4096)
        self.window_center = QDoubleSpinBox()
        self.window_center.setRange(-65536, 65536)
        self.window_center.setDecimals(1)
        self.window_width.valueChanged.connect(self._window_level_changed)
        self.window_center.valueChanged.connect(self._window_level_changed)
        display_form.addRow("Width", self.window_width)
        display_form.addRow("Center", self.window_center)
        layout.addWidget(display_group)

        algorithm_group = QGroupBox("Assisted tools")
        algorithm_form = QFormLayout(algorithm_group)
        self.operation_combo = QComboBox()
        self.operation_combo.addItems(
            [
                "Threshold preview",
                "Seeded region grow",
                "Flood fill",
                "Select connected component",
                "Split region into components",
                "Dilate",
                "Erode",
                "Opening",
                "Closing",
                "Remove small islands",
                "Keep largest component",
                "Add another label",
                "Erase another label",
            ]
        )
        self.operation_combo.currentTextChanged.connect(self._operation_selected)
        self.scope_combo = QComboBox()
        self.scope_combo.addItems(["Current slice", "Whole volume"])
        self.mask_combo = QComboBox()
        self.mask_combo.addItem("None (Full scope)", None)
        self.mask_combo.addItem("Inside active label", -1)
        self.mask_combo.currentIndexChanged.connect(
            lambda: self.mask_constraint_changed.emit(self.mask_combo.currentData())
        )
        self.connectivity_combo = QComboBox()
        self.connectivity_combo.addItem("6-connected (3D) / 4-connected (2D)", 1)
        self.connectivity_combo.addItem("18-connected (3D) / 8-connected (2D)", 2)
        self.connectivity_combo.addItem("26-connected (3D)", 3)
        self.other_label_combo = QComboBox()
        self.lower_value = QDoubleSpinBox()
        self.lower_value.setRange(-1e12, 1e12)
        self.lower_value.setDecimals(2)
        self.lower_value.setValue(0)
        self.upper_value = QDoubleSpinBox()
        self.upper_value.setRange(-1e12, 1e12)
        self.upper_value.setDecimals(2)
        self.upper_value.setValue(65535)
        self.lower_value.valueChanged.connect(self._on_threshold_spinbox_changed)
        self.upper_value.valueChanged.connect(self._on_threshold_spinbox_changed)
        self.lower_label = "Lower threshold / radius"
        self.upper_label = "Upper threshold / minimum size"
        algorithm_form.addRow("Operation", self.operation_combo)
        algorithm_form.addRow("Scope", self.scope_combo)
        algorithm_form.addRow("Region constraint", self.mask_combo)
        algorithm_form.addRow(self.lower_label, self.lower_value)
        algorithm_form.addRow(self.upper_label, self.upper_value)

        self.auto_otsu_button = QPushButton("Auto Otsu Threshold")
        self.auto_otsu_button.setToolTip(
            "Automatically compute optimal intensity threshold using Otsu's method"
        )
        self.auto_otsu_button.clicked.connect(self.otsu_requested)
        algorithm_form.addRow("", self.auto_otsu_button)

        algorithm_form.addRow("Connectivity", self.connectivity_combo)
        algorithm_form.addRow("Other label", self.other_label_combo)
        self.run_operation = QPushButton("Preview")
        self.run_operation.setToolTip(
            "Compute and display a preview highlight across the chosen scope"
        )
        self.run_operation.clicked.connect(self._run_button_clicked)

        self.apply_button = QPushButton("Apply")
        self.apply_button.setToolTip(
            "Compute and apply the segmentation directly to the volume/slice"
        )
        self.apply_button.clicked.connect(self._apply_button_clicked)

        self.cancel_preview_button = QPushButton("Cancel")
        self.cancel_preview_button.setToolTip("Discard the current preview")
        self.cancel_preview_button.clicked.connect(self.cancel_preview_requested)
        self.cancel_preview_button.setVisible(False)

        button_row = QHBoxLayout()
        button_row.addWidget(self.run_operation)
        button_row.addWidget(self.apply_button)
        button_row.addWidget(self.cancel_preview_button)
        algorithm_form.addRow(button_row)
        layout.addWidget(algorithm_group)
        layout.addStretch(1)
        self._operation_selected(self.operation_combo.currentText())

    def set_history_state(self, can_undo: bool, can_redo: bool) -> None:
        self.undo_button.setEnabled(can_undo)
        self.redo_button.setEnabled(can_redo)

    def set_window_level_values(self, width: float, center: float) -> None:
        self.window_width.blockSignals(True)
        self.window_center.blockSignals(True)
        self.window_width.setValue(max(1.0, width))
        self.window_center.setValue(center)
        self.window_width.blockSignals(False)
        self.window_center.blockSignals(False)

    def set_intensity_range(self, lower: float, upper: float) -> None:
        #Set useful initial thresholds from an image sample, not datatype limits.
        if not lower < upper:
            upper = lower + 1.0
        self._last_intensity_lower = lower
        self._last_intensity_upper = upper
        if getattr(self, "_current_op_type", "threshold") == "threshold":
            self.lower_value.blockSignals(True)
            self.upper_value.blockSignals(True)
            self.lower_value.setValue(lower)
            self.upper_value.setValue(upper)
            self.lower_value.blockSignals(False)
            self.upper_value.blockSignals(False)
        self.lower_value.setToolTip(f"Calibrated intensity lower bound: {lower:g}")
        self.upper_value.setToolTip(f"Calibrated intensity upper bound: {upper:g}")

    def set_operation_enabled(self, enabled: bool, button_text: str | None = None) -> None:
        self.operation_combo.setEnabled(enabled)
        self.scope_combo.setEnabled(enabled)
        self.connectivity_combo.setEnabled(enabled)
        self.other_label_combo.setEnabled(enabled)
        self.lower_value.setEnabled(enabled)
        self.upper_value.setEnabled(enabled)
        self.auto_otsu_button.setEnabled(
            enabled and self.operation_combo.currentText() == "Threshold preview"
        )
        if button_text is not None:
            text_lower = button_text.casefold()
            if "accept" in text_lower:
                self.run_operation.setText("✓ Accept Preview")
                self.run_operation.setStyleSheet(
                    "background-color: #2e7d32; color: white; font-weight: bold;"
                )
                self.cancel_preview_button.setVisible(True)
                self.apply_button.setVisible(False)
            elif text_lower.startswith("processing") or text_lower.startswith("applying"):
                self.run_operation.setText(button_text)
                self.run_operation.setStyleSheet("")
                self.cancel_preview_button.setVisible(False)
                self.apply_button.setVisible(False)
            else:
                self.run_operation.setText("Preview")
                self.run_operation.setStyleSheet("")
                self.cancel_preview_button.setVisible(False)
                self.apply_button.setVisible(True)
        else:
            self.run_operation.setText("Preview")
            self.run_operation.setStyleSheet("")
            self.cancel_preview_button.setVisible(False)
            self.apply_button.setVisible(True)

        if button_text is not None and "accept" in button_text.casefold():
            self.run_operation.setEnabled(True)
            self.cancel_preview_button.setEnabled(True)
            self.apply_button.setEnabled(False)
        else:
            self.run_operation.setEnabled(enabled)
            self.apply_button.setEnabled(enabled)
            self.cancel_preview_button.setEnabled(enabled)

    def set_scope_enabled(self, enabled: bool) -> None:
        self.scope_combo.setEnabled(enabled)

    def set_labels(self, labels, *, exclude_id: int | None = None) -> None:
        selected_id = self.other_label_combo.currentData()
        self.other_label_combo.blockSignals(True)
        self.other_label_combo.clear()
        for info in labels:
            if info.id != exclude_id:
                self.other_label_combo.addItem(info.name, info.id)
        index = self.other_label_combo.findData(selected_id)
        if index < 0 and self.other_label_combo.count():
            index = 0
        if index >= 0:
            self.other_label_combo.setCurrentIndex(index)
        self.other_label_combo.blockSignals(False)

        # Update region constraint dropdown
        selected_mask = self.mask_combo.currentData()
        self.mask_combo.blockSignals(True)
        self.mask_combo.clear()
        self.mask_combo.addItem("None (Full scope)", None)
        self.mask_combo.addItem("Inside active label", -1)
        for info in labels:
            self.mask_combo.addItem(f"Inside: {info.name}", info.id)
        mask_idx = self.mask_combo.findData(selected_mask)
        if mask_idx >= 0:
            self.mask_combo.setCurrentIndex(mask_idx)
        else:
            self.mask_combo.setCurrentIndex(0)
        self.mask_combo.blockSignals(False)

    def get_mask_constraint(self, active_label_id: int | None = None) -> int | None:
        #Return the label ID to constrain operations to, or None if unconstrained.
        data = self.mask_combo.currentData()
        if data is None:
            return None
        if data == -1:
            return active_label_id
        return int(data)

    def _run_button_clicked(self) -> None:
        if "accept" in self.run_operation.text().casefold():
            self.accept_preview_requested.emit()
            return
        self.operation_requested.emit(
            self.operation_combo.currentText(),
            self.lower_value.value(),
            self.upper_value.value(),
            self.scope_combo.currentText(),
            int(self.connectivity_combo.currentData()),
        )

    def _apply_button_clicked(self) -> None:
        self.apply_direct_requested.emit(
            self.operation_combo.currentText(),
            self.lower_value.value(),
            self.upper_value.value(),
            self.scope_combo.currentText(),
            int(self.connectivity_combo.currentData()),
        )

    def _operation_selected(self, operation: str) -> None:
        # Save previous tool state before configuring new tool
        if hasattr(self, "_current_op_type"):
            if self._current_op_type == "threshold":
                self._last_intensity_lower = self.lower_value.value()
                self._last_intensity_upper = self.upper_value.value()
            elif self._current_op_type == "morphology":
                self._last_morphology_radius = self.lower_value.value()
            elif self._current_op_type == "islands":
                self._last_island_min_size = self.upper_value.value()
            elif self._current_op_type == "largest":
                self._last_largest_n = self.upper_value.value()

        self.lower_value.blockSignals(True)
        self.upper_value.blockSignals(True)

        if operation in {"Threshold preview", "Seeded region grow", "Flood fill"}:
            self._current_op_type = "threshold"
            self.lower_label = "Lower intensity"
            self.upper_label = "Upper intensity"
            self.lower_value.setRange(-1e12, 1e12)
            self.upper_value.setRange(-1e12, 1e12)
            self.lower_value.setDecimals(2)
            self.upper_value.setDecimals(2)
            self.lower_value.setSingleStep(0.05)
            self.upper_value.setSingleStep(0.05)
            self.lower_value.setValue(self._last_intensity_lower)
            self.upper_value.setValue(self._last_intensity_upper)
            self.lower_value.setEnabled(True)
            self.upper_value.setEnabled(True)
            self.scope_combo.setEnabled(True)
            self.auto_otsu_button.setVisible(operation == "Threshold preview")
        elif operation == "Select connected component":
            self._current_op_type = "other"
            self.lower_label = "Not used"
            self.upper_label = "Not used"
            self.lower_value.setEnabled(False)
            self.upper_value.setEnabled(False)
            self.scope_combo.setEnabled(True)
            self.auto_otsu_button.setVisible(False)
        elif operation == "Split region into components":
            self._current_op_type = "split"
            self.lower_label = "Not used"
            self.upper_label = "Minimum component size (voxels)"
            self.lower_value.setEnabled(False)
            self.upper_value.setRange(1, 100_000_000)
            self.upper_value.setDecimals(0)
            self.upper_value.setSingleStep(10)
            self.upper_value.setValue(50)
            self.upper_value.setEnabled(True)
            self.scope_combo.setEnabled(True)
            self.auto_otsu_button.setVisible(False)
        elif operation in {"Dilate", "Erode", "Opening", "Closing"}:
            self._current_op_type = "morphology"
            self.lower_label = "Radius (mm if spacing known)"
            self.upper_label = "Not used"
            self.lower_value.setRange(0.1, 100.0)
            self.lower_value.setDecimals(1)
            self.lower_value.setSingleStep(0.5)
            self.lower_value.setValue(self._last_morphology_radius)
            self.lower_value.setEnabled(True)
            self.upper_value.setEnabled(False)
            self.scope_combo.setEnabled(True)
            self.auto_otsu_button.setVisible(False)
        elif operation == "Remove small islands":
            self._current_op_type = "islands"
            self.lower_label = "Not used"
            self.upper_label = "Minimum component size (voxels)"
            self.lower_value.setEnabled(False)
            self.upper_value.setRange(1, 100_000_000)
            self.upper_value.setDecimals(0)
            self.upper_value.setSingleStep(10)
            self.upper_value.setValue(self._last_island_min_size)
            self.upper_value.setEnabled(True)
            self.scope_combo.setEnabled(True)
            self.auto_otsu_button.setVisible(False)
        elif operation == "Keep largest component":
            self._current_op_type = "largest"
            self.lower_label = "Not used"
            self.upper_label = "Number of components to keep"
            self.lower_value.setEnabled(False)
            self.upper_value.setRange(1, 1000)
            self.upper_value.setDecimals(0)
            self.upper_value.setSingleStep(1)
            self.upper_value.setValue(self._last_largest_n)
            self.upper_value.setEnabled(True)
            self.scope_combo.setEnabled(True)
            self.auto_otsu_button.setVisible(False)
        elif operation in {"Add another label", "Erase another label"}:
            self._current_op_type = "other"
            self.lower_label = "Not used"
            self.upper_label = "Not used"
            self.lower_value.setEnabled(False)
            self.upper_value.setEnabled(False)
            self.scope_combo.setEnabled(True)
            self.auto_otsu_button.setVisible(False)
        else:
            self._current_op_type = "other"
            self.lower_label = "Not used"
            self.upper_label = "Not used"
            self.lower_value.setEnabled(False)
            self.upper_value.setEnabled(False)
            self.scope_combo.setEnabled(True)
            self.auto_otsu_button.setVisible(False)

        self.lower_value.blockSignals(False)
        self.upper_value.blockSignals(False)
        if self._current_op_type == "threshold":
            self.intensity_range_changed.emit(
                self._last_intensity_lower, self._last_intensity_upper
            )

        form = self.run_operation.parentWidget().layout()
        for row in range(form.rowCount()):
            label_item = form.itemAt(row, form.ItemRole.LabelRole)
            field_item = form.itemAt(row, form.ItemRole.FieldRole)
            if field_item is not None and field_item.widget() is self.lower_value:
                if label_item is not None:
                    label_item.widget().setText(self.lower_label)
            if field_item is not None and field_item.widget() is self.upper_value:
                if label_item is not None:
                    label_item.widget().setText(self.upper_label)
        self.other_label_combo.setEnabled(operation in {"Add another label", "Erase another label"})

    def _tool_selected(self, visible_name: str) -> None:
        tool_name = {
            "Navigate": "navigate",
            "Brush": "brush",
            "Erase": "erase",
            "Rectangle ROI": "rectangle",
            "Ellipse ROI": "ellipse",
            "Freehand ROI": "freehand",
            "Polygon ROI": "polygon",
        }[visible_name]
        self.tool_changed.emit(tool_name)

    def _window_level_changed(self, _value: float) -> None:
        self.window_level_changed.emit(self.window_width.value(), self.window_center.value())

    def _on_threshold_spinbox_changed(self) -> None:
        if getattr(self, "_current_op_type", "") == "threshold":
            lower = float(self.lower_value.value())
            upper = float(self.upper_value.value())
            self._last_intensity_lower = lower
            self._last_intensity_upper = upper
            self.intensity_range_changed.emit(lower, upper)
