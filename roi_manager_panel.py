#Manage saved editable ROI contours separately from their rasterized labels.

from __future__ import annotations

try:
    from PyQt6.QtCore import Qt, pyqtSignal
    from PyQt6.QtWidgets import (
        QHBoxLayout,
        QListWidget,
        QListWidgetItem,
        QPushButton,
        QVBoxLayout,
        QWidget,
    )
except ImportError as error:
    raise RuntimeError("Install ctseg-toolkit[ui] to use the ROI manager") from error


class ROIManagerPanel(QWidget):
    #List persisted ROI definitions and select, rename, or remove a definition.

    selected = pyqtSignal(int)
    rename_requested = pyqtSignal(int)
    remove_requested = pyqtSignal(int)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        self.items = QListWidget()
        self.items.currentItemChanged.connect(self._selection_changed)
        layout.addWidget(self.items)
        buttons = QHBoxLayout()
        self.rename_button = QPushButton("Rename")
        self.remove_button = QPushButton("Remove definition")
        self.rename_button.clicked.connect(self._rename_current)
        self.remove_button.clicked.connect(self._remove_current)
        buttons.addWidget(self.rename_button)
        buttons.addWidget(self.remove_button)
        layout.addLayout(buttons)
        self._records: dict[int, dict[str, object]] = {}

    def set_records(self, records: dict[int, dict[str, object]]) -> None:
        selected_id = self.current_roi_id
        self._records = records
        self.items.blockSignals(True)
        self.items.clear()
        for command_id, record in records.items():
            item = QListWidgetItem(self._display_name(record))
            item.setData(Qt.ItemDataRole.UserRole, command_id)
            self.items.addItem(item)
            if command_id == selected_id:
                self.items.setCurrentItem(item)
        self.items.blockSignals(False)

    @property
    def current_roi_id(self) -> int | None:
        item = self.items.currentItem()
        return int(item.data(Qt.ItemDataRole.UserRole)) if item is not None else None

    @staticmethod
    def _display_name(record: dict[str, object]) -> str:
        name = str(record.get("name", f"{record['shape']} ROI"))
        label = str(record.get("label_name", f"Label {record['label_id']}"))
        axis_name = ("Sagittal", "Coronal", "Axial")[int(record["axis"])]
        return f"{name} — {label} — {axis_name} slice {int(record['slice_index']) + 1}"

    def _selection_changed(self, current, _previous) -> None:
        if current is not None:
            self.selected.emit(int(current.data(Qt.ItemDataRole.UserRole)))

    def _rename_current(self) -> None:
        roi_id = self.current_roi_id
        if roi_id is not None:
            self.rename_requested.emit(roi_id)

    def _remove_current(self) -> None:
        roi_id = self.current_roi_id
        if roi_id is not None:
            self.remove_requested.emit(roi_id)
