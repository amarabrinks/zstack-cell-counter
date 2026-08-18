#!/usr/bin/env python3
"""Manual four-marker cell counting for microscopy Z-stacks in napari.

The application loads CZYX TIFF/OME-TIFF images, applies the original sampled
percentile contrast behavior, optionally average-pools only the XY dimensions,
and records point annotations classified as marker 1, 2, 3, or 4.

Examples
--------
    python vgat_cell_counter.py D:/images
    python vgat_cell_counter.py D:/images --downsample 2 --grid 4 4

If the image folder is omitted, a folder picker is shown.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import re
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import tifffile

try:
    from aicsimageio import AICSImage
except ImportError:
    AICSImage = None

try:
    import napari
    from qtpy.QtCore import Qt, QTimer
    from qtpy.QtGui import QKeySequence
    from qtpy.QtWidgets import (
        QApplication,
        QCheckBox,
        QComboBox,
        QFileDialog,
        QGridLayout,
        QGroupBox,
        QHBoxLayout,
        QHeaderView,
        QLabel,
        QMessageBox,
        QPushButton,
        QShortcut,
        QTableWidget,
        QTableWidgetItem,
        QVBoxLayout,
        QWidget,
    )
except ImportError as exc:  # permit helper functions to be imported without napari
    napari = None
    QApplication = None
    _GUI_IMPORT_ERROR = exc
else:
    _GUI_IMPORT_ERROR = None


LOGGER = logging.getLogger("vgat_cell_counter")
IMAGE_SUFFIXES = (".tif", ".tiff")
DEFAULT_CHANNEL_NAMES = ("PVALB", "SYN", "VGAT", "NeuN")
DEFAULT_CHANNEL_COLORS = ("blue", "red", "green", "magenta")
MARKER_COLORS = {
    1: (0.95, 0.25, 0.25, 1.0),
    2: (0.25, 0.65, 1.00, 1.0),
    3: (0.25, 0.90, 0.40, 1.0),
    4: (1.00, 0.75, 0.20, 1.0),
}
CSV_FIELDS = (
    "timestamp",
    "image",
    "marker",
    "square",
    "z",
    "y",
    "x",
    "downsample",
)


@dataclass
class AppConfig:
    """Runtime settings independent of any particular study."""

    input_folder: Path
    output_folder: Path
    percentile_max: float = 99.0
    floor_high: float = 255.0
    downsample: int = 1
    grid_rows: int = 4
    grid_cols: int = 4
    point_size: float = 12.0
    target_size: float = 12.0
    annotations_all_z: bool = True
    initial_visible_channel: int = 0
    metadata_profile: "MetadataProfile | None" = None
    channel_names: tuple[str, ...] = DEFAULT_CHANNEL_NAMES
    channel_colors: tuple[str, ...] = DEFAULT_CHANNEL_COLORS

    def __post_init__(self) -> None:
        self.input_folder = self.input_folder.expanduser().resolve()
        self.output_folder = self.output_folder.expanduser().resolve()
        if self.downsample not in (1, 2, 4):
            raise ValueError("downsample must be 1, 2, or 4")
        if self.grid_rows < 1 or self.grid_cols < 1:
            raise ValueError("grid dimensions must be positive")
        if not 0.0 < self.percentile_max <= 100.0:
            raise ValueError("percentile_max must be in (0, 100]")
        if not 0 <= self.initial_visible_channel < len(self.channel_names):
            raise ValueError("initial_visible_channel is outside the configured channel list")


@dataclass
class CellAnnotation:
    """One cell coordinate in the currently displayed (possibly downsampled) data."""

    timestamp: str
    image: str
    marker: int
    square: int
    z: int
    y: int
    x: int
    downsample: int

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "CellAnnotation":
        marker = int(value["marker"])
        if marker not in MARKER_COLORS:
            raise ValueError(f"Invalid marker class: {marker}")
        return cls(
            timestamp=str(value.get("timestamp", "")),
            image=str(value["image"]),
            marker=marker,
            square=int(value["square"]),
            z=int(value["z"]),
            y=int(value["y"]),
            x=int(value["x"]),
            downsample=int(value.get("downsample", 1)),
        )


@dataclass(frozen=True)
class MetadataProfile:
    """Optional, data-driven filename parsing and dataset expectations."""

    name: str
    filename_regex: str
    subject_field: str
    expected_images: dict[str, dict[str, int]]
    display_labels: dict[str, dict[str, str]]

    @classmethod
    def from_json(cls, path: Path) -> "MetadataProfile":
        data = json.loads(path.expanduser().read_text(encoding="utf-8"))
        required = ("name", "filename_regex", "subject_field")
        missing = [key for key in required if key not in data]
        if missing:
            raise ValueError(f"Profile is missing required keys: {', '.join(missing)}")
        re.compile(str(data["filename_regex"]))
        return cls(
            name=str(data["name"]),
            filename_regex=str(data["filename_regex"]),
            subject_field=str(data["subject_field"]),
            expected_images={
                str(region).lower(): {
                    str(layer): int(count) for layer, count in layers.items()
                }
                for region, layers in data.get("expected_images", {}).items()
            },
            display_labels={
                str(field): {str(value): str(label) for value, label in labels.items()}
                for field, labels in data.get("display_labels", {}).items()
            },
        )

    def parse(self, filename: str) -> dict[str, str] | None:
        match = re.fullmatch(self.filename_regex, strip_image_suffix(filename), re.IGNORECASE)
        if match is None:
            return None
        return {key: str(value) for key, value in match.groupdict().items()}

    def label(self, field: str, value: str) -> str:
        return self.display_labels.get(field, {}).get(value, value)


def strip_image_suffix(filename: str) -> str:
    """Remove common TIFF and compound OME-TIFF suffixes."""

    lowered = filename.casefold()
    for suffix in (".ome.tiff", ".ome.tif", ".tiff", ".tif"):
        if lowered.endswith(suffix):
            return filename[: -len(suffix)]
    return Path(filename).stem


def percentile_limits(
    volume: np.ndarray, percentile_max: float = 99.0, floor_high: float = 255.0
) -> tuple[float, float]:
    """Return the original lightly sampled percentile display limits."""

    z_step = max(1, volume.shape[0] // 8)
    sample = volume[::z_step, ::8, ::8]
    low_percentile = max(0.0, 100.0 - percentile_max)
    low = float(np.percentile(sample, low_percentile))
    high = max(float(np.percentile(sample, percentile_max)), float(floor_high))
    if high <= low:
        high = low + 1.0
    return low, high


def percentile_norm(
    volume: np.ndarray, percentile_max: float = 99.0, floor_high: float = 3000.0
) -> np.ndarray:
    """Preserved normalization helper from the original workflow."""

    low = float(volume.min())
    z_step = max(1, volume.shape[0] // 8)
    sample = volume[::z_step, ::8, ::8]
    high = max(float(np.percentile(sample, percentile_max)), float(floor_high))
    if high <= low:
        high = low + 1e-6
    normalized = (volume.astype(np.float32) - low) / (high - low)
    return np.clip(normalized, 0.0, 1.0)


def load_czyx(path: Path) -> np.ndarray:
    """Load an image as float32 CZYX."""

    if AICSImage is not None:
        array = AICSImage(str(path)).get_image_data("CZYX")
    else:
        array = tifffile.imread(path)
        if array.ndim == 4 and array.shape[0] <= 10:
            pass
        elif array.ndim == 4 and array.shape[-1] <= 10:
            array = np.transpose(array, (3, 0, 1, 2))
        elif array.ndim == 3:
            array = array[np.newaxis, ...]
        else:
            raise ValueError(
                f"Unsupported shape {array.shape} in {path.name}. "
                "Install aicsimageio for additional microscopy formats."
            )
    if array.ndim != 4:
        raise ValueError(f"Expected CZYX data, received shape {array.shape}")
    return np.asarray(array, dtype=np.float32)


def downsample_czyx(array: np.ndarray, factor: int) -> np.ndarray:
    """Average-pool Y and X only; preserve C and Z exactly."""

    if factor == 1:
        return array
    if factor not in (2, 4):
        raise ValueError("factor must be 1, 2, or 4")
    channels, z_planes, height, width = array.shape
    trim_y = height - height % factor
    trim_x = width - width % factor
    if trim_y == 0 or trim_x == 0:
        raise ValueError(f"Image XY dimensions {height}x{width} are too small for {factor}x")
    trimmed = array[:, :, :trim_y, :trim_x]
    return trimmed.reshape(
        channels,
        z_planes,
        trim_y // factor,
        factor,
        trim_x // factor,
        factor,
    ).mean(axis=(3, 5), dtype=np.float32)


def make_grid_labels(shape_zyx: Sequence[int], rows: int, cols: int) -> np.ndarray:
    """Create ZYX integer labels for a regular XY counting grid."""

    z_planes, height, width = map(int, shape_zyx)
    labels_2d = np.zeros((height, width), dtype=np.int32)
    y_edges = np.linspace(0, height, rows + 1, dtype=int)
    x_edges = np.linspace(0, width, cols + 1, dtype=int)
    label = 1
    for row in range(rows):
        for col in range(cols):
            labels_2d[y_edges[row] : y_edges[row + 1], x_edges[col] : x_edges[col + 1]] = label
            label += 1
    return np.repeat(labels_2d[np.newaxis, :, :], z_planes, axis=0)


def grid_outline(labels: np.ndarray) -> np.ndarray:
    """Return a binary array containing boundaries between grid squares."""

    outline = np.zeros_like(labels, dtype=np.uint8)
    outline[:, 1:, :][labels[:, 1:, :] != labels[:, :-1, :]] = 1
    outline[:, :, 1:][labels[:, :, 1:] != labels[:, :, :-1]] = 1
    return outline


def discover_images(folder: Path) -> list[Path]:
    """Return TIFF-family files in deterministic, case-insensitive name order."""

    if not folder.is_dir():
        raise FileNotFoundError(f"Image folder does not exist: {folder}")
    return sorted(
        (path for path in folder.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES),
        key=lambda path: path.name.casefold(),
    )


if napari is not None:

    class CellCounterDock(QWidget):
        """Napari dock widget coordinating image navigation and annotation state."""

        def __init__(self, viewer: Any, files: Sequence[Path], config: AppConfig) -> None:
            super().__init__()
            self.viewer = viewer
            self.files = list(files)
            self.config = config
            self.index = 0
            self.marker = 1
            self.z_planes = 1
            self.grid_labels: np.ndarray | None = None
            self.current_square = 0
            self.target: tuple[int, int, int] | None = None
            self.last_hover: tuple[int, int, int] | None = None
            self.erase_mode = False
            self.annotations: list[CellAnnotation] = []
            self.image_status: dict[str, dict[str, Any]] = {}
            self.load_errors: dict[str, str] = {}
            self.metadata = {
                path.name: config.metadata_profile.parse(path.name)
                if config.metadata_profile is not None
                else None
                for path in self.files
            }
            self.image_layers: dict[str, Any] = {}
            self.grid_layer = None
            self.target_layer = None
            self.cells_layer = None
            self.display_shape: tuple[int, int, int, int] | None = None

            self.session_json = config.output_folder / "cell_counter_session.json"
            self.session_csv = config.output_folder / "cell_counter_autosave.csv"
            self.activity_csv = config.output_folder / "activity_log.csv"
            self._build_ui()
            self._bind_events()
            self._resume_session()
            startup_index = self._first_unreviewed_index()
            self.load_image(startup_index)
            self._select_current_subject()
            if startup_index == 0 and all(
                self.image_status.get(path.name, {}).get("done", False) for path in self.files
            ):
                self._set_status("All images are marked complete; opened the first image")
            self._log_event("session_started", details=f"images={len(self.files)}")

        @property
        def current_image_name(self) -> str:
            return self.files[self.index].name

        def _build_ui(self) -> None:
            self.setMinimumWidth(360)
            self.setStyleSheet(
                """
                QPushButton[role="primary"] {
                    background: #3f6fa8;
                    border: 1px solid #6f9bd0;
                    color: white;
                    font-weight: 600;
                    min-height: 34px;
                }
                QPushButton[role="primary"]:hover { background: #4b7fbb; }
                QPushButton[role="next"] {
                    background: #2f7d68;
                    border: 1px solid #59a891;
                    color: white;
                    font-weight: 700;
                    min-height: 36px;
                }
                QPushButton[role="next"]:hover { background: #388f78; }
                QPushButton[role="marker"] {
                    min-height: 34px;
                    font-weight: 700;
                }
                QPushButton[role="marker"]:checked {
                    background: #d99b32;
                    border: 2px solid #ffd27a;
                    color: #171717;
                }
                QCheckBox#imageComplete {
                    background: #7a3035;
                    border: 2px solid #d36b72;
                    border-radius: 6px;
                    padding: 10px;
                    color: white;
                    font-size: 15px;
                    font-weight: 700;
                }
                QCheckBox#imageComplete:hover {
                    background: #8d383e;
                    border-color: #ef858c;
                }
                QCheckBox#imageComplete:checked {
                    background: #245f4e;
                    border-color: #65c5a5;
                    color: white;
                }
                QCheckBox#imageComplete::indicator {
                    width: 22px;
                    height: 22px;
                }
                """
            )
            root = QVBoxLayout(self)

            annotate = QWidget()
            layout = QVBoxLayout(annotate)
            root.addWidget(annotate)
            image_box = QGroupBox("Image")
            image_grid = QGridLayout(image_box)
            image_grid.setContentsMargins(8, 8, 8, 8)
            image_grid.setVerticalSpacing(3)
            self.image_value = QLabel("-")
            self.image_value.setWordWrap(True)
            self.image_summary_value = QLabel("0 / 0")
            self.annotation_value = QLabel("0 in image / 0 session")
            self.output_value = QLabel(self.config.output_folder.name)
            self.output_value.setToolTip(str(self.config.output_folder))
            image_grid.addWidget(QLabel("Image:"), 0, 0)
            image_grid.addWidget(self.image_value, 0, 1)
            image_grid.addWidget(QLabel("View:"), 1, 0)
            image_grid.addWidget(self.image_summary_value, 1, 1)
            image_grid.addWidget(QLabel("Cells:"), 2, 0)
            image_grid.addWidget(self.annotation_value, 2, 1)
            image_grid.addWidget(QLabel("Output:"), 3, 0)
            image_grid.addWidget(self.output_value, 3, 1)
            layout.addWidget(image_box)

            self.metadata_box = QGroupBox("Dataset metadata")
            metadata_grid = QGridLayout(self.metadata_box)
            metadata_grid.setContentsMargins(8, 8, 8, 8)
            metadata_grid.setVerticalSpacing(3)
            self.subject_value = QLabel("Generic mode")
            self.batch_value = QLabel("-")
            self.region_value = QLabel("-")
            self.layer_value = QLabel("-")
            self.repetition_value = QLabel("-")
            self.subject_progress_value = QLabel("Profile not selected")
            self.subject_progress_value.setWordWrap(True)
            metadata_grid.addWidget(QLabel("Subject:"), 0, 0)
            metadata_grid.addWidget(self.subject_value, 0, 1)
            metadata_grid.addWidget(QLabel("Batch:"), 0, 2)
            metadata_grid.addWidget(self.batch_value, 0, 3)
            metadata_grid.addWidget(QLabel("Region:"), 1, 0)
            metadata_grid.addWidget(self.region_value, 1, 1)
            metadata_grid.addWidget(QLabel("Layer:"), 1, 2)
            metadata_grid.addWidget(self.layer_value, 1, 3)
            metadata_grid.addWidget(QLabel("Rep:"), 1, 4)
            metadata_grid.addWidget(self.repetition_value, 1, 5)
            metadata_grid.addWidget(QLabel("Progress:"), 2, 0)
            metadata_grid.addWidget(self.subject_progress_value, 2, 1, 1, 5)
            self.subject_combo = QComboBox()
            self.subject_combo.addItem("All subjects", None)
            if self.config.metadata_profile is not None:
                subjects = sorted(
                    {
                        item[self.config.metadata_profile.subject_field]
                        for item in self.metadata.values()
                        if item is not None
                    }
                )
                for subject in subjects:
                    self.subject_combo.addItem(subject, subject)
            self.subject_combo.currentIndexChanged.connect(self._subject_filter_changed)
            metadata_grid.addWidget(QLabel("Subject:"), 3, 0)
            metadata_grid.addWidget(self.subject_combo, 3, 1, 1, 5)
            self.image_done = QCheckBox("Image reviewed and complete")
            self.image_done.setObjectName("imageComplete")
            self.image_done.toggled.connect(self._image_done_toggled)
            metadata_grid.addWidget(self.image_done, 4, 0, 1, 6)
            layout.addWidget(self.metadata_box)

            controls = QGroupBox("Annotation")
            controls_layout = QVBoxLayout(controls)
            marker_row = QHBoxLayout()
            marker_row.addWidget(QLabel("Marker class:"))
            self.marker_buttons: dict[int, QPushButton] = {}
            for marker in MARKER_COLORS:
                button = QPushButton(str(marker))
                button.setCheckable(True)
                button.setAutoExclusive(True)
                button.setProperty("role", "marker")
                button.clicked.connect(lambda _checked=False, value=marker: self.set_marker(value))
                marker_row.addWidget(button)
                self.marker_buttons[marker] = button
            controls_layout.addLayout(marker_row)
            self.marker_buttons[1].setChecked(True)

            target_row = QHBoxLayout()
            self.target_button = QPushButton("Set target (Space)")
            self.add_button = QPushButton("Add cell (Enter)")
            self.undo_button = QPushButton("Undo")
            self.target_button.clicked.connect(self.set_target)
            self.add_button.clicked.connect(self.add_cell)
            self.undo_button.clicked.connect(self.undo)
            self.target_button.setProperty("role", "primary")
            self.add_button.setProperty("role", "primary")
            for button in (self.target_button, self.add_button, self.undo_button):
                target_row.addWidget(button)
            controls_layout.addLayout(target_row)

            navigation_row = QHBoxLayout()
            self.previous_button = QPushButton("Previous")
            self.next_button = QPushButton("Next")
            self.unreviewed_button = QPushButton("Next unreviewed")
            self.previous_button.clicked.connect(lambda: self.navigate_image(-1))
            self.next_button.clicked.connect(lambda: self.navigate_image(1))
            self.unreviewed_button.clicked.connect(self.goto_next_unreviewed)
            self.next_button.setProperty("role", "next")
            self.unreviewed_button.setProperty("role", "next")
            for button in (
                self.previous_button,
                self.next_button,
                self.unreviewed_button,
            ):
                navigation_row.addWidget(button)
            controls_layout.addLayout(navigation_row)

            utility_row = QHBoxLayout()
            self.downsample_combo = QComboBox()
            self.downsample_combo.addItems(["1x", "2x", "4x"])
            self.downsample_combo.setCurrentText(f"{self.config.downsample}x")
            self.reload_button = QPushButton("Reload")
            self.erase_button = QPushButton("Erase mode")
            self.erase_button.setCheckable(True)
            self.fit_button = QPushButton("Fit view")
            self.reload_button.clicked.connect(self.reload_downsample)
            self.erase_button.toggled.connect(self.set_erase_mode)
            self.fit_button.clicked.connect(self.viewer.reset_view)
            utility_row.addWidget(QLabel("Downsample:"))
            for widget in (self.downsample_combo, self.reload_button, self.erase_button, self.fit_button):
                utility_row.addWidget(widget)
            controls_layout.addLayout(utility_row)

            export_row = QHBoxLayout()
            self.csv_button = QPushButton("Export CSV")
            self.json_button = QPushButton("Export JSON")
            self.snapshot_button = QPushButton("Snapshot")
            self.csv_button.clicked.connect(self.export_csv)
            self.json_button.clicked.connect(self.export_json)
            self.snapshot_button.clicked.connect(self.snapshot)
            for button in (self.csv_button, self.json_button, self.snapshot_button):
                export_row.addWidget(button)
            controls_layout.addLayout(export_row)

            self.status = QLabel("")
            self.status.setWordWrap(True)
            controls_layout.addWidget(self.status)
            layout.addWidget(controls)

            table_box = QGroupBox("Cells in current image")
            table_layout = QVBoxLayout(table_box)
            self.table = QTableWidget(0, 7)
            self.table.setHorizontalHeaderLabels(
                ["#", "Marker", "Grid", "Z", "Y", "X", "DS"]
            )
            self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
            self.table.horizontalHeader().setStretchLastSection(True)
            self.table.setMinimumHeight(150)
            table_layout.addWidget(self.table)
            layout.addWidget(table_box)

        def _bind_events(self) -> None:
            bindings = {
                "1": lambda _viewer: self.set_marker(1),
                "2": lambda _viewer: self.set_marker(2),
                "3": lambda _viewer: self.set_marker(3),
                "4": lambda _viewer: self.set_marker(4),
                "Space": lambda _viewer: self.set_target(),
                "Enter": lambda _viewer: self.add_cell(),
                "a": lambda _viewer: self.add_cell(at_cursor=True),
                "u": lambda _viewer: self.undo(),
                "[": lambda _viewer: self.step_z(-1),
                "]": lambda _viewer: self.step_z(1),
                "Left": lambda _viewer: self.navigate_image(-1),
                "Right": lambda _viewer: self.navigate_image(1),
            }
            for key, callback in bindings.items():
                self.viewer.bind_key(key, callback, overwrite=True)
            self.viewer.mouse_move_callbacks.append(self._on_mouse_move)
            self.viewer.mouse_drag_callbacks.append(self._on_mouse_click)
            shortcut = QShortcut(QKeySequence("Ctrl+S"), self)
            shortcut.setContext(Qt.ApplicationShortcut)
            shortcut.activated.connect(self.autosave)

        def _selected_subject(self) -> str | None:
            return self.subject_combo.currentData()

        def _first_unreviewed_index(self) -> int:
            """Return the first dataset image not explicitly marked complete."""

            incomplete = [
                index
                for index, path in enumerate(self.files)
                if not self.image_status.get(path.name, {}).get("done", False)
            ]
            return next(
                (
                    index
                    for index in incomplete
                    if not self.image_status.get(self.files[index].name, {}).get(
                        "flagged_for_review", False
                    )
                ),
                incomplete[0] if incomplete else 0,
            )

        def _subject_for_image(self, image: str) -> str | None:
            profile = self.config.metadata_profile
            metadata = self.metadata.get(image)
            if profile is None or metadata is None:
                return None
            return metadata.get(profile.subject_field)

        def _navigation_indices(self) -> list[int]:
            subject = self._selected_subject()
            if subject is None:
                return list(range(len(self.files)))
            return [
                index
                for index, path in enumerate(self.files)
                if self._subject_for_image(path.name) == subject
            ]

        def navigate_image(self, direction: int) -> None:
            indices = self._navigation_indices()
            if not indices:
                self._set_status("No images match the selected subject")
                return
            try:
                position = indices.index(self.index)
            except ValueError:
                position = -1 if direction > 0 else 0
            destination = position + direction
            if 0 <= destination < len(indices):
                if self._confirm_leave_current_image():
                    self.load_image(indices[destination])
            else:
                self._set_status("Reached the end of the selected subject")

        def goto_next_unreviewed(self) -> None:
            indices = self._navigation_indices()
            if not indices:
                return
            ordered = [index for index in indices if index > self.index] + [
                index for index in indices if index <= self.index
            ]
            for index in ordered:
                if not self.image_status.get(self.files[index].name, {}).get("done", False):
                    if self._confirm_leave_current_image():
                        self.load_image(index)
                    return
            self._set_status("All images in this selection are complete")

        def _confirm_leave_current_image(self) -> bool:
            """Require an explicit disposition before leaving an unfinished image."""

            status = self.image_status.get(self.current_image_name, {})
            if status.get("done", False) or status.get("flagged_for_review", False):
                return True
            dialog = QMessageBox(self)
            dialog.setWindowTitle("Image not complete")
            dialog.setText(
                "This image has not been marked complete. What would you like to do before moving on?"
            )
            complete_button = dialog.addButton("Mark complete", QMessageBox.AcceptRole)
            flag_button = dialog.addButton("Flag for review", QMessageBox.ActionRole)
            stay_button = dialog.addButton("Stay here", QMessageBox.RejectRole)
            dialog.setDefaultButton(stay_button)
            dialog.exec()
            clicked = dialog.clickedButton()
            if clicked is complete_button:
                self._set_image_completion(True)
                return True
            if clicked is flag_button:
                status = self.image_status.setdefault(self.current_image_name, {})
                status["flagged_for_review"] = True
                status["flagged_at"] = datetime.now().isoformat(timespec="seconds")
                self._log_event("image_flagged_for_review")
                self.autosave()
                self._update_done_label(False)
                return True
            return False

        def _subject_filter_changed(self, _index: int) -> None:
            indices = self._navigation_indices()
            if indices and self.index not in indices:
                if self._confirm_leave_current_image():
                    self.load_image(indices[0])
                else:
                    current_subject = self._subject_for_image(self.current_image_name)
                    previous_index = self.subject_combo.findData(current_subject)
                    self.subject_combo.blockSignals(True)
                    self.subject_combo.setCurrentIndex(max(0, previous_index))
                    self.subject_combo.blockSignals(False)
            else:
                self._refresh_image_summary()

        def _select_current_subject(self) -> None:
            subject = self._subject_for_image(self.current_image_name)
            if subject is None:
                return
            combo_index = self.subject_combo.findData(subject)
            if combo_index >= 0:
                self.subject_combo.setCurrentIndex(combo_index)

        def _cursor_indices(self) -> tuple[int, int, int] | None:
            if self.grid_labels is None or self.viewer.cursor.position is None:
                return None
            position = self.viewer.cursor.position
            try:
                position = self.grid_layer.world_to_data(position)
            except (AttributeError, ValueError):
                pass
            if len(position) < 3:
                return None
            z_max, y_max, x_max = np.asarray(self.grid_labels.shape) - 1
            return (
                int(np.clip(round(position[-3]), 0, z_max)),
                int(np.clip(round(position[-2]), 0, y_max)),
                int(np.clip(round(position[-1]), 0, x_max)),
            )

        def set_marker(self, marker: int) -> None:
            if marker not in MARKER_COLORS:
                raise ValueError(f"Unknown marker: {marker}")
            self.marker = marker
            self.marker_buttons[marker].setChecked(True)
            self._set_status(f"Marker {marker} selected")

        def set_target(self) -> None:
            indices = self._cursor_indices()
            if indices is None:
                self._set_status("Move the cursor over the image first")
                return
            self.target = indices
            self.target_layer.data = np.asarray([indices], dtype=float)
            self._set_status(f"Target z{indices[0]}, y{indices[1]}, x{indices[2]}")

        def add_cell(self, at_cursor: bool = False) -> bool:
            indices = self._cursor_indices() if at_cursor else self.target
            if indices is None:
                self._set_status("Set a target with Space, or press A at the cursor")
                return False
            z_coord, y_coord, x_coord = indices
            square = int(self.grid_labels[indices])
            annotation = CellAnnotation(
                timestamp=datetime.now().isoformat(timespec="seconds"),
                image=self.current_image_name,
                marker=self.marker,
                square=square,
                z=z_coord,
                y=y_coord,
                x=x_coord,
                downsample=self.config.downsample,
            )
            self.annotations.append(annotation)
            self._mark_image_modified("cell_added")
            self.target = None
            self.target_layer.data = np.empty((0, 3), dtype=float)
            self._refresh_annotations()
            self.autosave()
            self._set_status(f"Added marker {self.marker} cell")
            return True

        def undo(self) -> None:
            for index in range(len(self.annotations) - 1, -1, -1):
                if self.annotations[index].image == self.current_image_name:
                    removed = self.annotations.pop(index)
                    self._mark_image_modified("cell_removed")
                    self._refresh_annotations()
                    self.autosave()
                    self._set_status(f"Removed marker {removed.marker} cell")
                    return
            self._set_status("No annotation to undo in this image")

        def set_erase_mode(self, enabled: bool) -> None:
            self.erase_mode = enabled
            self.erase_button.setText("Erase mode ON" if enabled else "Erase mode")
            self._set_status("Click near a cell to erase it" if enabled else "Erase mode off")

        def _erase_nearest(self, indices: tuple[int, int, int], tolerance: float = 6.0) -> None:
            candidates = [
                (index, annotation)
                for index, annotation in enumerate(self.annotations)
                if annotation.image == self.current_image_name
            ]
            if not candidates:
                self._set_status("No cells in this image")
                return
            _, y_coord, x_coord = indices
            index, annotation = min(
                candidates,
                key=lambda item: (item[1].y - y_coord) ** 2 + (item[1].x - x_coord) ** 2,
            )
            distance = ((annotation.y - y_coord) ** 2 + (annotation.x - x_coord) ** 2) ** 0.5
            if distance > tolerance:
                self._set_status("No cell close enough to erase")
                return
            self.annotations.pop(index)
            self._mark_image_modified("cell_erased")
            self._refresh_annotations()
            self.autosave()
            self._set_status("Cell erased")

        def _mark_image_modified(self, event: str) -> None:
            status = self.image_status.setdefault(self.current_image_name, {})
            if status.get("done", False):
                status["done"] = False
                status["completed_at"] = ""
                status["modified_after_completion"] = True
                self.image_done.blockSignals(True)
                self.image_done.setChecked(False)
                self.image_done.blockSignals(False)
                self._update_done_label(False)
            self._log_event(event)

        def _image_done_toggled(self, checked: bool) -> None:
            if not hasattr(self, "image_done") or not self.files:
                return
            self._set_image_completion(checked)

        def _set_image_completion(self, checked: bool) -> None:
            """Set completion state from either the checkbox or navigation prompt."""

            status = self.image_status.setdefault(self.current_image_name, {})
            status["done"] = bool(checked)
            status["completed_at"] = (
                datetime.now().isoformat(timespec="seconds") if checked else ""
            )
            if checked:
                status["modified_after_completion"] = False
                status["flagged_for_review"] = False
                status["flagged_at"] = ""
            self.image_done.blockSignals(True)
            self.image_done.setChecked(checked)
            self.image_done.blockSignals(False)
            self._update_done_label(checked)
            self._log_event("image_completed" if checked else "image_reopened")
            self.autosave()
            self._refresh_hud()

        def _on_mouse_click(self, _viewer: Any, event: Any) -> None:
            if not self.erase_mode or getattr(event, "button", None) != 1:
                return
            if getattr(event, "type", None) != "mouse_press":
                return
            indices = self._cursor_indices()
            if indices is not None:
                self._erase_nearest(indices)

        def _on_mouse_move(self, _viewer: Any, _event: Any) -> None:
            indices = self._cursor_indices()
            if indices is None or self.grid_labels is None:
                return
            self.last_hover = indices
            square = int(self.grid_labels[indices])
            if square == self.current_square:
                return
            self.current_square = square
            self._refresh_hud()

        def step_z(self, delta: int) -> None:
            if self.z_planes <= 1:
                return
            current = int(self.viewer.dims.current_step[0])
            self.viewer.dims.set_point(0, int(np.clip(current + delta, 0, self.z_planes - 1)))

        def reload_downsample(self) -> None:
            factor = int(self.downsample_combo.currentText()[0])
            self.config.downsample = factor
            self.load_image(self.index)

        def load_image(self, index: int) -> None:
            if not 0 <= index < len(self.files):
                return
            self.index = index
            try:
                array = downsample_czyx(load_czyx(self.files[index]), self.config.downsample)
                self.load_errors.pop(self.current_image_name, None)
            except Exception as exc:
                self.load_errors[self.current_image_name] = str(exc)
                self._log_event("image_load_failed", details=str(exc))
                QMessageBox.warning(self, "Image could not be loaded", f"{self.current_image_name}\n\n{exc}")
                self._refresh_metadata_hud()
                return
            channels, self.z_planes, height, width = array.shape
            self.display_shape = (channels, self.z_planes, height, width)
            visible_channel = min(self.config.initial_visible_channel, channels - 1)
            for channel in range(min(channels, len(self.config.channel_names))):
                name = self.config.channel_names[channel]
                limits = percentile_limits(
                    array[channel], self.config.percentile_max, self.config.floor_high
                )
                layer = self.image_layers.get(name)
                if layer is None:
                    layer = self.viewer.add_image(
                        array[channel],
                        name=name,
                        colormap=self.config.channel_colors[channel],
                        blending="additive",
                        rendering="translucent",
                        contrast_limits=limits,
                        visible=channel == visible_channel,
                    )
                    self.image_layers[name] = layer
                else:
                    layer.data = array[channel]
                    layer.contrast_limits = limits
                    layer.visible = channel == visible_channel

            for channel in range(channels, len(self.config.channel_names)):
                layer = self.image_layers.get(self.config.channel_names[channel])
                if layer is not None:
                    layer.visible = False

            self.grid_labels = make_grid_labels(
                (self.z_planes, height, width), self.config.grid_rows, self.config.grid_cols
            )
            outline = grid_outline(self.grid_labels)
            self.grid_layer = self._update_or_add_labels(
                self.grid_layer, outline, "Grid reference", opacity=0.25
            )
            if self.target_layer is None:
                self.target_layer = self.viewer.add_points(
                    np.empty((0, 3)),
                    name="Target",
                    ndim=3,
                    size=self.config.target_size,
                    face_color="cyan",
                    border_color="black",
                )
            else:
                self.target_layer.data = np.empty((0, 3))
            if self.cells_layer is None:
                self.cells_layer = self.viewer.add_points(
                    np.empty((0, 3)),
                    name="Cells",
                    ndim=3,
                    size=self.config.point_size,
                    face_color="white",
                    border_color="black",
                )

            self.target = None
            self.last_hover = None
            self.current_square = 0
            if self.z_planes > 1:
                self.viewer.dims.set_point(0, self.z_planes // 2)
            self.image_value.setText(self.current_image_name)
            self._refresh_image_summary()
            self.image_done.blockSignals(True)
            self.image_done.setChecked(
                bool(self.image_status.get(self.current_image_name, {}).get("done", False))
            )
            self.image_done.blockSignals(False)
            self._update_done_label(self.image_done.isChecked())
            self._refresh_metadata_hud()
            self._refresh_annotations()
            self._refresh_hud()
            self._set_status(f"Loaded {self.current_image_name}")
            self._log_event("image_loaded")

        def _update_done_label(self, checked: bool) -> None:
            flagged = self.image_status.get(self.current_image_name, {}).get(
                "flagged_for_review", False
            )
            if checked:
                text = "IMAGE COMPLETE — reviewed"
            elif flagged:
                text = "FLAGGED FOR REVIEW — mark complete when resolved"
            else:
                text = "MARK IMAGE REVIEWED AND COMPLETE"
            self.image_done.setText(text)

        def _refresh_image_summary(self) -> None:
            if self.display_shape is None:
                return
            channels, z_planes, height, width = self.display_shape
            visible_channel = min(self.config.initial_visible_channel, channels - 1)
            indices = self._navigation_indices()
            if self.index in indices and self._selected_subject() is not None:
                position = f"{indices.index(self.index) + 1}/{len(indices)} subject"
            else:
                position = f"{self.index + 1}/{len(self.files)} dataset"
            self.image_summary_value.setText(
                f"{position} | C{channels} Z{z_planes} {height}x{width} | "
                f"{self.config.downsample}x | {self.config.channel_names[visible_channel]}"
            )

        def _update_or_add_labels(
            self, layer: Any, data: np.ndarray, name: str, opacity: float
        ) -> Any:
            if layer is None:
                return self.viewer.add_labels(data, name=name, opacity=opacity)
            layer.data = data
            layer.visible = True
            return layer

        def _annotation_display_position(self, annotation: CellAnnotation) -> tuple[float, float, float]:
            ratio = annotation.downsample / self.config.downsample
            return float(annotation.z), annotation.y * ratio, annotation.x * ratio

        def _refresh_annotations(self) -> None:
            current = [a for a in self.annotations if a.image == self.current_image_name]
            positions: list[tuple[float, float, float]] = []
            colors: list[tuple[float, float, float, float]] = []
            for annotation in current:
                z_coord, y_coord, x_coord = self._annotation_display_position(annotation)
                if self.config.annotations_all_z and self.z_planes > 1:
                    positions.extend((float(z), y_coord, x_coord) for z in range(self.z_planes))
                    colors.extend([MARKER_COLORS[annotation.marker]] * self.z_planes)
                else:
                    positions.append((z_coord, y_coord, x_coord))
                    colors.append(MARKER_COLORS[annotation.marker])
            self.cells_layer.data = np.asarray(positions, dtype=float).reshape((-1, 3))
            if colors:
                self.cells_layer.face_color = np.asarray(colors, dtype=np.float32)
            self._rebuild_table()
            self._refresh_hud()

        def _rebuild_table(self) -> None:
            self.table.setSortingEnabled(False)
            self.table.setRowCount(0)
            current = [
                annotation
                for annotation in self.annotations
                if annotation.image == self.current_image_name
            ]
            for number, annotation in enumerate(current, start=1):
                row = self.table.rowCount()
                self.table.insertRow(row)
                values = (
                    number,
                    annotation.marker,
                    annotation.square,
                    annotation.z,
                    annotation.y,
                    annotation.x,
                    annotation.downsample,
                )
                for column, value in enumerate(values):
                    self.table.setItem(row, column, QTableWidgetItem(str(value)))
            self.table.setSortingEnabled(True)

        def _refresh_hud(self) -> None:
            """Update study-independent image and workflow details."""

            image_count = sum(
                annotation.image == self.current_image_name for annotation in self.annotations
            )
            self.annotation_value.setText(
                f"{image_count} in image / {len(self.annotations)} session"
            )
            self._refresh_metadata_hud()

        def _refresh_metadata_hud(self) -> None:
            profile = self.config.metadata_profile
            metadata = self.metadata.get(self.current_image_name)
            if profile is None:
                self.subject_value.setText("Generic mode")
                self.subject_progress_value.setText("Use --profile to enable subject QC")
                self.image_done.setEnabled(True)
                return
            if metadata is None:
                for label in (
                    self.subject_value,
                    self.batch_value,
                    self.region_value,
                    self.layer_value,
                    self.repetition_value,
                ):
                    label.setText("Unrecognized")
                self.subject_progress_value.setText("Filename does not match the active profile")
                return
            subject = metadata.get(profile.subject_field, "-")
            self.subject_value.setText(subject)
            self.batch_value.setText(metadata.get("batch", "-"))
            region = metadata.get("region", "-").lower()
            layer = metadata.get("layer", "-")
            self.region_value.setText(profile.label("region", region))
            self.layer_value.setText(profile.label("layer", layer))
            self.repetition_value.setText(metadata.get("repetition", "-"))
            subject_images = [
                path.name for path in self.files if self._subject_for_image(path.name) == subject
            ]
            completed = sum(
                bool(self.image_status.get(image, {}).get("done", False))
                for image in subject_images
            )
            breakdown = []
            for expected_region, layers in profile.expected_images.items():
                for expected_layer, expected_count in layers.items():
                    found = [
                        image
                        for image in subject_images
                        if self.metadata[image].get("region", "").lower() == expected_region
                        and self.metadata[image].get("layer") == expected_layer
                    ]
                    done = sum(
                        bool(self.image_status.get(image, {}).get("done", False)) for image in found
                    )
                    breakdown.append(
                        f"{profile.label('region', expected_region)} "
                        f"{profile.label('layer', expected_layer)} {done}/{expected_count}"
                    )
            expected_total = sum(
                count for layers in profile.expected_images.values() for count in layers.values()
            )
            self.subject_progress_value.setText(
                f"{completed}/{expected_total} complete | " + " | ".join(breakdown)
            )

        def _session_payload(self) -> dict[str, Any]:
            return {
                "schema_version": 1,
                "input_folder": str(self.config.input_folder),
                "grid": {"rows": self.config.grid_rows, "cols": self.config.grid_cols},
                "image_status": self.image_status,
                "load_errors": self.load_errors,
                "annotations": [asdict(annotation) for annotation in self.annotations],
            }

        def autosave(self) -> None:
            try:
                self.config.output_folder.mkdir(parents=True, exist_ok=True)
                temporary = self.session_json.with_suffix(".json.tmp")
                temporary.write_text(
                    json.dumps(self._session_payload(), indent=2), encoding="utf-8"
                )
                temporary.replace(self.session_json)
                self._write_csv(self.session_csv)
                self._write_structured_outputs()
            except (OSError, TypeError, ValueError) as exc:
                LOGGER.exception("Autosave failed")
                self._set_status(f"Autosave failed: {exc}")

        def _resume_session(self) -> None:
            if not self.session_json.exists():
                return
            try:
                payload = json.loads(self.session_json.read_text(encoding="utf-8"))
                saved_folder = Path(payload.get("input_folder", "")).resolve()
                if saved_folder != self.config.input_folder:
                    LOGGER.warning("Ignoring session for a different input folder: %s", saved_folder)
                    return
                self.annotations = [
                    CellAnnotation.from_dict(item) for item in payload.get("annotations", [])
                ]
                self.image_status = {
                    str(image): dict(status)
                    for image, status in payload.get("image_status", {}).items()
                }
                self.load_errors = {
                    str(image): str(error)
                    for image, error in payload.get("load_errors", {}).items()
                }
            except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                QMessageBox.warning(self, "Session not loaded", str(exc))

        def _write_csv(self, path: Path) -> None:
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
                writer.writeheader()
                writer.writerows(asdict(annotation) for annotation in self.annotations)

        @staticmethod
        def _write_dict_rows(path: Path, fieldnames: Sequence[str], rows: Sequence[dict[str, Any]]) -> None:
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
                writer.writeheader()
                writer.writerows(rows)

        def _metadata_columns(self, image: str) -> dict[str, str]:
            metadata = self.metadata.get(image) or {}
            profile = self.config.metadata_profile
            return {
                "subject": metadata.get(profile.subject_field, "") if profile else "",
                "stain": metadata.get("stain", ""),
                "batch": metadata.get("batch", ""),
                "region": metadata.get("region", ""),
                "layer": metadata.get("layer", ""),
                "repetition": metadata.get("repetition", ""),
            }

        def _image_rows(self) -> list[dict[str, Any]]:
            rows = []
            for path in self.files:
                image = path.name
                status = self.image_status.get(image, {})
                image_annotations = [a for a in self.annotations if a.image == image]
                row: dict[str, Any] = {
                    **self._metadata_columns(image),
                    "image": image,
                    "status": "done" if status.get("done", False) else (
                        "flagged_for_review"
                        if status.get("flagged_for_review", False)
                        else ("in_progress" if image_annotations else "not_reviewed")
                    ),
                    "cell_count": len(image_annotations),
                    "completed_at": status.get("completed_at", ""),
                    "flagged_at": status.get("flagged_at", ""),
                    "modified_after_completion": status.get("modified_after_completion", False),
                    "load_error": self.load_errors.get(image, ""),
                    "parse_ok": self.metadata.get(image) is not None
                    if self.config.metadata_profile is not None
                    else "",
                }
                for marker in MARKER_COLORS:
                    row[f"marker_{marker}_count"] = sum(
                        annotation.marker == marker for annotation in image_annotations
                    )
                rows.append(row)
            return rows

        def _qc_rows(self) -> list[dict[str, str]]:
            flags: list[dict[str, str]] = []

            def add(subject: str, image: str, flag: str, details: str) -> None:
                flags.append(
                    {"subject": subject, "image": image, "flag": flag, "details": details}
                )

            profile = self.config.metadata_profile
            if profile is not None:
                for path in self.files:
                    if self.metadata.get(path.name) is None:
                        add("", path.name, "filename_parse_error", "Filename does not match profile")
                subjects = sorted(
                    {
                        self._subject_for_image(path.name)
                        for path in self.files
                        if self._subject_for_image(path.name) is not None
                    }
                )
                for subject in subjects:
                    seen: dict[tuple[str, str, int], list[str]] = {}
                    for path in self.files:
                        metadata = self.metadata.get(path.name)
                        if metadata is None or metadata.get(profile.subject_field) != subject:
                            continue
                        try:
                            key = (
                                metadata.get("region", "").lower(),
                                metadata.get("layer", ""),
                                int(metadata.get("repetition", "0")),
                            )
                        except ValueError:
                            add(subject, path.name, "invalid_repetition", metadata.get("repetition", ""))
                            continue
                        seen.setdefault(key, []).append(path.name)
                    for region, layers in profile.expected_images.items():
                        for layer, expected_count in layers.items():
                            for repetition in range(1, expected_count + 1):
                                matches = seen.get((region, layer, repetition), [])
                                if not matches:
                                    add(
                                        subject,
                                        "",
                                        "missing_image",
                                        f"{region}_{layer}_{repetition}",
                                    )
                                elif len(matches) > 1:
                                    add(
                                        subject,
                                        ";".join(matches),
                                        "duplicate_image",
                                        f"{region}_{layer}_{repetition}",
                                    )
                    for (region, layer, repetition), images in seen.items():
                        expected = profile.expected_images.get(region, {}).get(layer)
                        if expected is None or not 1 <= repetition <= expected:
                            add(
                                subject,
                                ";".join(images),
                                "unexpected_image",
                                f"{region}_{layer}_{repetition}",
                            )
            for path in self.files:
                image = path.name
                subject = self._subject_for_image(image) or ""
                status = self.image_status.get(image, {})
                if not status.get("done", False):
                    add(subject, image, "not_reviewed", "Image is not marked complete")
                if status.get("flagged_for_review", False):
                    add(subject, image, "flagged_for_review", "User deferred this image for review")
                if status.get("modified_after_completion", False):
                    add(subject, image, "modified_after_completion", "Image must be reviewed again")
                if image in self.load_errors:
                    add(subject, image, "load_error", self.load_errors[image])
            return flags

        def _summary_rows(self) -> list[dict[str, Any]]:
            profile = self.config.metadata_profile
            if profile is None:
                return []
            rows = []
            subjects = sorted(
                {
                    self._subject_for_image(path.name)
                    for path in self.files
                    if self._subject_for_image(path.name) is not None
                }
            )
            qc = self._qc_rows()
            for subject in subjects:
                for region, layers in profile.expected_images.items():
                    for layer, expected_count in layers.items():
                        images = [
                            path.name
                            for path in self.files
                            if self._subject_for_image(path.name) == subject
                            and self.metadata[path.name].get("region", "").lower() == region
                            and self.metadata[path.name].get("layer") == layer
                        ]
                        annotations = [a for a in self.annotations if a.image in images]
                        row: dict[str, Any] = {
                            "subject": subject,
                            "region": region,
                            "layer": layer,
                            "expected_images": expected_count,
                            "found_images": len(images),
                            "completed_images": sum(
                                bool(self.image_status.get(image, {}).get("done", False))
                                for image in images
                            ),
                            "zero_cell_images": sum(
                                bool(self.image_status.get(image, {}).get("done", False))
                                and not any(a.image == image for a in self.annotations)
                                for image in images
                            ),
                            "total_cells": len(annotations),
                            "qc_flag_count": sum(
                                flag["subject"] == subject
                                and (not flag["image"] or flag["image"] in images)
                                and (f"{region}_{layer}_" in flag["details"] or flag["image"] in images)
                                for flag in qc
                            ),
                        }
                        for marker in MARKER_COLORS:
                            row[f"marker_{marker}_count"] = sum(
                                annotation.marker == marker for annotation in annotations
                            )
                        rows.append(row)
            return rows

        def _annotation_rows_with_metadata(self) -> list[dict[str, Any]]:
            return [
                {**self._metadata_columns(annotation.image), **asdict(annotation)}
                for annotation in self.annotations
            ]

        def _write_structured_outputs(self) -> None:
            metadata_fields = ("subject", "stain", "batch", "region", "layer", "repetition")
            annotation_rows = self._annotation_rows_with_metadata()
            self._write_dict_rows(
                self.config.output_folder / "annotations.csv",
                (*metadata_fields, *CSV_FIELDS),
                annotation_rows,
            )
            image_fields = (
                *metadata_fields,
                "image",
                "status",
                "cell_count",
                "marker_1_count",
                "marker_2_count",
                "marker_3_count",
                "marker_4_count",
                "completed_at",
                "flagged_at",
                "modified_after_completion",
                "load_error",
                "parse_ok",
            )
            image_rows = self._image_rows()
            self._write_dict_rows(self.config.output_folder / "images.csv", image_fields, image_rows)
            qc_rows = self._qc_rows()
            self._write_dict_rows(
                self.config.output_folder / "qc_flags.csv",
                ("subject", "image", "flag", "details"),
                qc_rows,
            )
            summary_rows = self._summary_rows()
            summary_fields = (
                "subject",
                "region",
                "layer",
                "expected_images",
                "found_images",
                "completed_images",
                "zero_cell_images",
                "total_cells",
                "marker_1_count",
                "marker_2_count",
                "marker_3_count",
                "marker_4_count",
                "qc_flag_count",
            )
            self._write_dict_rows(
                self.config.output_folder / "subject_summary.csv", summary_fields, summary_rows
            )
            subjects_folder = self.config.output_folder / "subjects"
            subjects_folder.mkdir(exist_ok=True)
            subjects = sorted({row["subject"] for row in image_rows if row["subject"]})
            for subject in subjects:
                folder = subjects_folder / f"subject_{subject}"
                folder.mkdir(exist_ok=True)
                self._write_dict_rows(
                    folder / "annotations.csv",
                    (*metadata_fields, *CSV_FIELDS),
                    [row for row in annotation_rows if row["subject"] == subject],
                )
                self._write_dict_rows(
                    folder / "images.csv",
                    image_fields,
                    [row for row in image_rows if row["subject"] == subject],
                )
                self._write_dict_rows(
                    folder / "summary.csv",
                    summary_fields,
                    [row for row in summary_rows if row["subject"] == subject],
                )

        def _log_event(self, event: str, details: str = "") -> None:
            try:
                self.config.output_folder.mkdir(parents=True, exist_ok=True)
                exists = self.activity_csv.exists()
                metadata = self._metadata_columns(self.current_image_name) if self.files else {}
                with self.activity_csv.open("a", newline="", encoding="utf-8") as handle:
                    writer = csv.DictWriter(
                        handle,
                        fieldnames=("timestamp", "event", "subject", "image", "details"),
                    )
                    if not exists:
                        writer.writeheader()
                    writer.writerow(
                        {
                            "timestamp": datetime.now().isoformat(timespec="seconds"),
                            "event": event,
                            "subject": metadata.get("subject", ""),
                            "image": self.current_image_name if self.files else "",
                            "details": details,
                        }
                    )
            except OSError:
                LOGGER.exception("Could not write activity log")

        def export_csv(self) -> None:
            filename, _ = QFileDialog.getSaveFileName(
                self, "Export annotations", str(self.config.output_folder / "annotations.csv"), "CSV (*.csv)"
            )
            if filename:
                self._write_csv(Path(filename))
                self._set_status(f"Exported {Path(filename).name}")

        def export_json(self) -> None:
            filename, _ = QFileDialog.getSaveFileName(
                self, "Export session", str(self.config.output_folder / "annotations.json"), "JSON (*.json)"
            )
            if filename:
                Path(filename).write_text(json.dumps(self._session_payload(), indent=2), encoding="utf-8")
                self._set_status(f"Exported {Path(filename).name}")

        def snapshot(self) -> None:
            filename, _ = QFileDialog.getSaveFileName(
                self, "Save snapshot", str(self.config.output_folder / "snapshot.png"), "PNG (*.png)"
            )
            if filename:
                self.viewer.screenshot(filename, canvas_only=False, flash=False)
                self._set_status(f"Saved {Path(filename).name}")

        def _set_status(self, message: str) -> None:
            self.status.setText(
                f"{message}\n1-4 marker | Space target | Enter add | A add at cursor | "
                "U undo | [/] Z | arrows: previous/next image"
            )


def choose_input_folder() -> Path | None:
    """Show a native folder picker, creating a Qt application if required."""

    if QApplication is None:
        return None
    application = QApplication.instance() or QApplication(sys.argv)
    selected = QFileDialog.getExistingDirectory(None, "Choose microscopy image folder")
    return Path(selected) if selected else None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_folder", nargs="?", type=Path, help="Folder containing TIFF images")
    parser.add_argument(
        "--output-folder",
        type=Path,
        help="Session/export folder (default: INPUT_FOLDER/cell_counter_output)",
    )
    parser.add_argument(
        "--profile",
        type=Path,
        help="JSON metadata profile (defaults to mpfc_syn_profile.json beside this script when present)",
    )
    parser.add_argument("--downsample", type=int, choices=(1, 2, 4), default=1)
    parser.add_argument("--grid", type=int, nargs=2, metavar=("ROWS", "COLS"), default=(4, 4))
    parser.add_argument("--percentile-max", type=float, default=99.0)
    parser.add_argument("--floor-high", type=float, default=255.0)
    parser.add_argument(
        "--visible-channel",
        type=int,
        choices=(1, 2, 3, 4),
        default=1,
        help="Only this 1-based channel is visible after each image load (default: 1/PVALB)",
    )
    parser.add_argument(
        "--single-z-points",
        action="store_true",
        help="Show annotations only on their recorded Z plane",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if _GUI_IMPORT_ERROR is not None:
        raise SystemExit(f"napari and qtpy are required: {_GUI_IMPORT_ERROR}")
    input_folder = args.input_folder or choose_input_folder()
    if input_folder is None:
        return 0
    input_folder = input_folder.expanduser().resolve()
    output_folder = (args.output_folder or input_folder / "cell_counter_output").expanduser().resolve()
    profile_path = args.profile
    if profile_path is None:
        adjacent_profile = Path(__file__).resolve().with_name("mpfc_syn_profile.json")
        profile_path = adjacent_profile if adjacent_profile.exists() else None
    metadata_profile = MetadataProfile.from_json(profile_path) if profile_path else None
    config = AppConfig(
        input_folder=input_folder,
        output_folder=output_folder,
        percentile_max=args.percentile_max,
        floor_high=args.floor_high,
        downsample=args.downsample,
        grid_rows=args.grid[0],
        grid_cols=args.grid[1],
        annotations_all_z=not args.single_z_points,
        initial_visible_channel=args.visible_channel - 1,
        metadata_profile=metadata_profile,
    )
    files = discover_images(config.input_folder)
    if not files:
        raise SystemExit(f"No TIFF images found in {config.input_folder}")
    viewer = napari.Viewer()
    dock = CellCounterDock(viewer, files, config)
    viewer.window.add_dock_widget(dock, area="right", name="Four-marker cell counter")
    napari.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

