#Small label list with metadata editing and active-label selection.

from __future__ import annotations

try:
    from PyQt6.QtCore import Qt, pyqtSignal
    from PyQt6.QtGui import QColor
    from PyQt6.QtWidgets import (
        QColorDialog,
        QGridLayout,
        QListWidget,
        QListWidgetItem,
        QPushButton,
        QVBoxLayout,
        QWidget,
    )
except ImportError as error:
    raise RuntimeError("Install ctseg-toolkit[ui] to use the label manager") from error

from ctseg.core.label_volume import LabelVolume


class LabelManagerPanel(QWidget):
    #Present labels and emit requested changes to the host controller.

    active_label_changed = pyqtSignal(int)
    add_label_requested = pyqtSignal()
    remove_label_requested = pyqtSignal(int)
    rename_label_requested = pyqtSignal(int)
    reorder_label_requested = pyqtSignal(int, int)
    visibility_changed = pyqtSignal(int, bool)
    lock_changed = pyqtSignal(int, bool)
    color_change_requested = pyqtSignal(int, tuple)
    split_requested = pyqtSignal(int)
    isolate_requested = pyqtSignal(int)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        self.items = QListWidget()
        self.items.currentItemChanged.connect(self._selection_changed)
        self.items.itemChanged.connect(self._item_changed)
        layout.addWidget(self.items)
        row = QGridLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(4)
        self.add_button = QPushButton("Add")
        self.remove_button = QPushButton("Remove")
        self.color_button = QPushButton("Color")
        self.lock_button = QPushButton("Lock")
        self.lock_button.setToolTip("Toggle protection of the selected label from edits")
        self.rename_button = QPushButton("Rename")
        self.isolate_button = QPushButton("Solo")
        self.isolate_button.setToolTip("Isolate: show only this region (hides other labels)")
        self.split_button = QPushButton("Split")
        self.split_button.setToolTip("Divide this region into disconnected sub-region labels")
        self.up_button = QPushButton("Up")
        self.down_button = QPushButton("Down")
        self.add_button.clicked.connect(self.add_label_requested)
        self.remove_button.clicked.connect(self._remove_selected)
        self.color_button.clicked.connect(self._choose_color)
        self.lock_button.clicked.connect(self._toggle_lock)
        self.rename_button.clicked.connect(self._rename_selected)
        self.isolate_button.clicked.connect(self._isolate_selected)
        self.split_button.clicked.connect(self._split_selected)
        self.up_button.clicked.connect(lambda: self._reorder_selected(-1))
        self.down_button.clicked.connect(lambda: self._reorder_selected(1))
        # Three compact rows of three buttons so the panel fits a narrow dock.
        for index, button in enumerate(
            (
                self.add_button,
                self.remove_button,
                self.rename_button,
                self.color_button,
                self.lock_button,
                self.isolate_button,
                self.split_button,
                self.up_button,
                self.down_button,
            )
        ):
            row.addWidget(button, index // 3, index % 3)
        layout.addLayout(row)
        self._volume: LabelVolume | None = None

    def set_label_volume(self, label_volume: LabelVolume) -> None:
        self._volume = label_volume
        self.refresh()

    def refresh(self) -> None:
        if self._volume is None:
            return
        selected = self.current_label_id
        self.items.blockSignals(True)
        self.items.clear()
        for info in self._volume.labels:
            item = QListWidgetItem(info.name)
            item.setData(256, info.id)
            item.setFlags(
                item.flags() | Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsSelectable
            )
            item.setCheckState(Qt.CheckState.Checked if info.visible else Qt.CheckState.Unchecked)
            item.setForeground(QColor(*info.color))
            item.setToolTip("Locked" if info.locked else "Editable")
            self.items.addItem(item)
            if info.id == selected:
                self.items.setCurrentItem(item)
        if self.items.count() and self.items.currentItem() is None:
            self.items.setCurrentRow(0)
        self.items.blockSignals(False)
        current = self.items.currentItem()
        if current is not None:
            self.active_label_changed.emit(int(current.data(256)))

    @property
    def current_label_id(self) -> int | None:
        item = self.items.currentItem()
        return int(item.data(256)) if item is not None else None

    def select_label(self, label_id: int) -> None:
        #Select a label by ID, e.g. from the histogram's color palette.
        for row in range(self.items.count()):
            item = self.items.item(row)
            if int(item.data(256)) == label_id:
                self.items.setCurrentItem(item)
                return
        raise KeyError(f"label {label_id} is not present in the label list")

    def _selection_changed(self, current, _previous) -> None:
        if current is not None:
            self.active_label_changed.emit(int(current.data(256)))

    def _item_changed(self, item: QListWidgetItem) -> None:
        if self._volume is None:
            return
        label_id = int(item.data(256))
        visible = item.checkState() == Qt.CheckState.Checked
        self.visibility_changed.emit(label_id, visible)

    def _toggle_lock(self) -> None:
        if self._volume is None or self.current_label_id is None:
            return
        info = self._volume.get_label(self.current_label_id)
        self.lock_changed.emit(info.id, not info.locked)

    def _rename_selected(self) -> None:
        label_id = self.current_label_id
        if label_id is not None:
            self.rename_label_requested.emit(label_id)

    def _reorder_selected(self, direction: int) -> None:
        label_id = self.current_label_id
        if label_id is not None:
            self.reorder_label_requested.emit(label_id, direction)

    def _remove_selected(self) -> None:
        label_id = self.current_label_id
        if label_id is not None:
            self.remove_label_requested.emit(label_id)

    def _choose_color(self) -> None:
        if self._volume is None or self.current_label_id is None:
            return
        info = self._volume.get_label(self.current_label_id)
        selected = QColorDialog.getColor(QColor(*info.color), self, "Choose label color")
        if selected.isValid():
            self.color_change_requested.emit(
                info.id, (selected.red(), selected.green(), selected.blue())
            )

    def _split_selected(self) -> None:
        label_id = self.current_label_id
        if label_id is not None:
            self.split_requested.emit(label_id)

    def _isolate_selected(self) -> None:
        label_id = self.current_label_id
        if label_id is not None:
            self.isolate_requested.emit(label_id)
