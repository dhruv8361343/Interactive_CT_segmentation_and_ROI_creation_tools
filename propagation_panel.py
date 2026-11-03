#Controls for keyframe-based slice propagation.

from __future__ import annotations

try:
    from PyQt6.QtCore import pyqtSignal
    from PyQt6.QtWidgets import (
        QComboBox,
        QFormLayout,
        QGroupBox,
        QLineEdit,
        QPushButton,
        QVBoxLayout,
        QWidget,
    )
except ImportError as error:
    raise RuntimeError("Install ctseg-toolkit[ui] to use slice propagation controls") from error


class PropagationPanel(QWidget):
    #Collect keyframe and target-range choices; computation stays in the controller.

    propagation_requested = pyqtSignal(str, str, str)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        group = QGroupBox("Slice propagation")
        form = QFormLayout(group)
        self.method = QComboBox()
        self.method.addItems(["Nearest slice copy", "Contour interpolation"])
        self.keyframes = QLineEdit()
        self.keyframes.setPlaceholderText("e.g. 10, 20")
        self.target_range = QLineEdit()
        self.target_range.setPlaceholderText("e.g. 11, 19")
        form.addRow("Method", self.method)
        form.addRow("Keyframe slices (1-based)", self.keyframes)
        form.addRow("Target range (1-based)", self.target_range)
        self.run_button = QPushButton("Preview propagation")
        self.run_button.clicked.connect(
            lambda: self.propagation_requested.emit(
                self.method.currentText(), self.keyframes.text(), self.target_range.text()
            )
        )
        form.addRow(self.run_button)
        layout.addWidget(group)
