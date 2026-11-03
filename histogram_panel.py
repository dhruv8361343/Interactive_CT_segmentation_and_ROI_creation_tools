#Compact intensity histogram, range selector, and segment color palette.

from __future__ import annotations

import numpy as np

try:
    from PyQt6.QtCore import QPointF, Qt, pyqtSignal
    from PyQt6.QtGui import QColor, QPainter, QPainterPath, QPen
    from PyQt6.QtWidgets import (
        QColorDialog,
        QHBoxLayout,
        QLabel,
        QPushButton,
        QVBoxLayout,
        QWidget,
    )
except ImportError as error:
    raise RuntimeError("Install ctseg-toolkit[ui] to use the histogram panel") from error

from ctseg.core.label_volume import LabelInfo, LabelVolume
from ctseg.core.volume import CTVolume


class _RangeHistogram(QWidget):
    #Paint a sampled intensity distribution and select a threshold interval.

    range_selected = pyqtSignal(float, float)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setMinimumHeight(112)
        self.setMouseTracking(True)
        self._counts = np.zeros(128, dtype=np.float64)
        self._minimum = 0.0
        self._maximum = 1.0
        self._low = 0.0
        self._high = 1.0
        self._drag_start: float | None = None
        self._hover: float | None = None
        self.setToolTip("Drag across the histogram to set assisted-tool intensity bounds.")

    def set_histogram(
        self,
        counts: np.ndarray,
        minimum: float,
        maximum: float,
        low: float,
        high: float,
    ) -> None:
        self._counts = np.asarray(counts, dtype=np.float64)
        self._minimum = float(minimum)
        self._maximum = max(float(maximum), float(minimum) + np.finfo(float).eps)
        self._low = float(np.clip(low, self._minimum, self._maximum))
        self._high = float(np.clip(high, self._low, self._maximum))
        self.update()

    def set_selected_range(self, low: float, high: float) -> None:
        self._low = float(np.clip(low, self._minimum, self._maximum))
        self._high = float(np.clip(high, self._low, self._maximum))
        self.update()

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor("#17202b"))
        plot = self.rect().adjusted(30, 8, -10, -24)
        if plot.width() <= 0 or plot.height() <= 0 or not self._counts.size:
            return
        painter.setPen(QPen(QColor("#526172"), 1))
        painter.drawRect(plot)
        maximum_count = max(float(np.max(self._counts)), 1.0)
        selected_left = self._value_to_x(self._low, plot)
        selected_right = self._value_to_x(self._high, plot)
        painter.fillRect(
            int(selected_left),
            plot.top() + 1,
            max(1, int(selected_right - selected_left)),
            plot.height() - 1,
            QColor(40, 150, 220, 42),
        )
        path = QPainterPath()
        bin_width = plot.width() / max(1, self._counts.size)
        for index, count in enumerate(self._counts):
            x = plot.left() + index * bin_width
            bar_height = plot.height() * (count / maximum_count) ** 0.55
            path.moveTo(x, plot.bottom())
            path.lineTo(x, plot.bottom() - bar_height)
        painter.setPen(QPen(QColor("#76c9ff"), max(1.0, bin_width * 0.7)))
        painter.drawPath(path)
        painter.setPen(QPen(QColor("#f3c969"), 1.5))
        painter.drawLine(QPointF(selected_left, plot.top()), QPointF(selected_left, plot.bottom()))
        painter.drawLine(
            QPointF(selected_right, plot.top()), QPointF(selected_right, plot.bottom())
        )
        painter.setPen(QColor("#bdc7d2"))
        painter.drawText(2, plot.top() + 10, f"{maximum_count:.0f}")
        painter.drawText(plot.left(), self.height() - 5, f"{self._minimum:.3g}")
        painter.drawText(plot.right() - 48, self.height() - 5, f"{self._maximum:.3g}")
        if self._hover is not None:
            painter.setPen(QColor("#ffffff"))
            painter.drawText(plot.left() + 4, plot.top() + 13, f"I = {self._hover:.4g}")

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            plot = self.rect().adjusted(30, 8, -10, -24)
            self._drag_start = self._x_to_value(event.position().x(), plot)
            self._low = self._drag_start
            self._high = self._drag_start
            self.update()
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        plot = self.rect().adjusted(30, 8, -10, -24)
        self._hover = self._x_to_value(event.position().x(), plot)
        if self._drag_start is not None:
            current = self._hover
            self._low = min(self._drag_start, current)
            self._high = max(self._drag_start, current)
        self.update()
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        if self._drag_start is not None and event.button() == Qt.MouseButton.LeftButton:
            self._drag_start = None
            if self._high > self._low:
                self.range_selected.emit(self._low, self._high)
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def leaveEvent(self, event) -> None:
        self._hover = None
        self.update()
        super().leaveEvent(event)

    def _x_to_value(self, x: float, plot) -> float:
        fraction = np.clip((x - plot.left()) / max(1, plot.width()), 0.0, 1.0)
        return float(self._minimum + fraction * (self._maximum - self._minimum))

    def _value_to_x(self, value: float, plot) -> float:
        fraction = (value - self._minimum) / (self._maximum - self._minimum)
        return float(plot.left() + fraction * plot.width())


class _LabelSwatch(QPushButton):
    #Select a segment with one click and change its color with a double click.

    double_clicked = pyqtSignal(int)

    def __init__(self, info: LabelInfo, parent=None) -> None:
        super().__init__(parent)
        self.label_id = info.id
        self.setMinimumWidth(62)
        self.setMaximumWidth(110)
        self.setToolTip(f"Select {info.name}; double-click to change color")
        self.update_info(info)

    def update_info(self, info: LabelInfo) -> None:
        self.setText(info.name)
        self.setStyleSheet(
            "QPushButton {"
            f"background-color: rgb({info.color[0]}, {info.color[1]}, {info.color[2]});"
            "color: #101820; border: 1px solid #91a0af; border-radius: 4px;"
            "padding: 3px 6px;}"
        )

    def mouseDoubleClickEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self.double_clicked.emit(self.label_id)
            event.accept()
            return
        super().mouseDoubleClickEvent(event)


class HistogramPanel(QWidget):
    #Sampled image histogram with draggable threshold range and label swatches.

    range_selected = pyqtSignal(float, float)
    label_selected = pyqtSignal(int)
    label_color_requested = pyqtSignal(int, tuple)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        title = QLabel("Intensity histogram · drag to select threshold")
        title.setStyleSheet("font-weight: 600;")
        title.setWordWrap(True)
        layout.addWidget(title)
        self.canvas = _RangeHistogram()
        self.canvas.range_selected.connect(self.range_selected)
        layout.addWidget(self.canvas)
        self._swatch_row = QHBoxLayout()
        self._swatch_row.setContentsMargins(0, 0, 0, 0)
        layout.addLayout(self._swatch_row)
        self._swatches: dict[int, _LabelSwatch] = {}
        self._label_volume: LabelVolume | None = None

    def set_volume(self, volume: CTVolume) -> None:
        #Sample evenly spaced slices so a histogram never copies the full stack.
        sample_parts = []
        slice_count = min(volume.depth, 32)
        z_indices = np.unique(np.linspace(0, volume.depth - 1, slice_count, dtype=int))
        for z_index in z_indices:
            plane = volume.get_intensity_slice(2, int(z_index), calibrated=True)
            sample_parts.append(np.asarray(plane[::8, ::8], dtype=np.float32).ravel())
        samples = np.concatenate(sample_parts)
        samples = samples[np.isfinite(samples)]
        if not samples.size:
            raise ValueError("cannot build an intensity histogram from an empty image")
        low, high = np.percentile(samples, (0.5, 99.5))
        if high <= low:
            high = low + 1.0
        counts, edges = np.histogram(samples, bins=128, range=(float(low), float(high)))
        current_low, current_high = np.percentile(samples, (10, 90))
        self.canvas.set_histogram(
            counts, float(edges[0]), float(edges[-1]), current_low, current_high
        )

    def set_selected_range(self, low: float, high: float) -> None:
        self.canvas.set_selected_range(low, high)

    def set_label_volume(self, label_volume: LabelVolume) -> None:
        self._label_volume = label_volume
        self._rebuild_swatches()

    def _rebuild_swatches(self) -> None:
        while self._swatch_row.count():
            item = self._swatch_row.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self._swatches.clear()
        if self._label_volume is None:
            return
        for info in self._label_volume.labels:
            swatch = _LabelSwatch(info)
            swatch.clicked.connect(
                lambda _checked=False, label_id=info.id: self.label_selected.emit(label_id)
            )
            swatch.double_clicked.connect(self._choose_color)
            self._swatches[info.id] = swatch
            self._swatch_row.addWidget(swatch)
        self._swatch_row.addStretch(1)

    def refresh_labels(self) -> None:
        if self._label_volume is None:
            return
        for info in self._label_volume.labels:
            if info.id in self._swatches:
                self._swatches[info.id].update_info(info)
        current_ids = {info.id for info in self._label_volume.labels}
        if current_ids != set(self._swatches):
            self._rebuild_swatches()

    def _choose_color(self, label_id: int) -> None:
        if self._label_volume is None:
            return
        info = self._label_volume.get_label(label_id)
        selected = QColorDialog.getColor(QColor(*info.color), self, f"Color: {info.name}")
        if selected.isValid():
            self.label_color_requested.emit(
                label_id, (selected.red(), selected.green(), selected.blue())
            )
