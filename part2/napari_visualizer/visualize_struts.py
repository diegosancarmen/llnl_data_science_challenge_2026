"""Dual-window, memory-bounded PyVista strut visualizer.

The FastAPI/Streamlit broker exchanges only strut IDs and tabular metadata.
This desktop process alone memory-maps and renders the CT TIFF.
"""

from __future__ import annotations

import argparse
import json
import os
import threading
from pathlib import Path

import numpy as np
import pandas as pd
import pyvista as pv
import tifffile
from PyQt6.QtCore import QObject, Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDockWidget,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)
from pyvistaqt import QtInteractor
from vtkmodules.vtkRenderingCore import vtkCellPicker

from part2.napari_visualizer.scene_data import (
    TRANSPARENT_RGBA,
    classification_color,
    image_grid,
    make_line_mesh,
    parse_strut_ids,
    unit_cell_context_color,
)


SELECTED_STRUT_COLOR = "yellow"
SELECTED_STRUT_WIDTH = 6
ALL_STRUT_WIDTH = 3
DEFAULT_MACRO_DOWNSAMPLE = 2
DEFAULT_MICRO_DOWNSAMPLE = 1

# The selection and micro-context layers start empty and are populated after
# the user selects a strut. PyVista 0.48+ rejects empty meshes by default.
pv.global_theme.allow_empty_mesh = True


def format_strut_metadata(strut_row: pd.Series | None, geometry_row: pd.Series) -> str:
    """Format concise identity, defect, and structural metadata for the inspector."""

    def value(*columns: str):
        for column in columns:
            for row in (strut_row, geometry_row):
                if row is not None and column in row.index and pd.notna(row[column]):
                    return row[column]
        return "Unavailable"

    return "\n".join(
        [
            "=== STRUT SUMMARY ===",
            f"Strut ID: {value('strut_id')}",
            f"Classification: {value('stage2_classification', 'inventory_classification')}",
            f"Primary defect: {value('primary_defect')}",
            f"Needs review: {value('needs_review')}",
            f"Expected by CAD: {value('expected_by_0point5_cad', 'inventory_expected_by_0point5_cad')}",
            "",
            "=== STRUCTURE ===",
            f"Unit cell: {value('unit_cell_ids', 'inventory_unit_cell_ids')}",
            f"Junctions: {value('junction0_id', 'inventory_junction0_id')} -> {value('junction1_id', 'inventory_junction1_id')}",
            f"Junction degrees: {value('junction0_degree', 'inventory_junction0_degree')} -> {value('junction1_degree', 'inventory_junction1_degree')}",
            f"Length (um): {value('inventory_length_um', 'length_um_from_centerline')}",
            f"Start (vox ZYX): {value('start_z_vox')}, {value('start_y_vox')}, {value('start_x_vox')}",
            f"End (vox ZYX): {value('end_z_vox')}, {value('end_y_vox')}, {value('end_x_vox')}",
        ]
    )


class ApiSelectionBridge(QObject):
    """Transfer websocket events from its worker thread to the Qt event loop."""

    selection_received = pyqtSignal(list)
    connection_error = pyqtSignal(str)


class StrutVisualizer:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.volume = tifffile.memmap(args.scan)
        if self.volume.ndim != 3:
            raise ValueError(f"Expected a 3-D TIFF scan, got shape {self.volume.shape}")
        self.volume_shape = np.asarray(self.volume.shape, dtype=int)

        self.table = pd.read_csv(args.centerlines)
        self._validate_centerlines()
        self.defect_strut_index = self._load_defect_summary(args.defect_by_strut)
        self.defect_station_index = self._load_station_details(args.defect_by_station)

        self.strut_ids = self.table["strut_id"].astype(str).tolist()
        self.strut_index = {strut_id: index for index, strut_id in enumerate(self.strut_ids)}
        self.base_colors = [classification_color(value) for value in self.table["inventory_classification"]]
        starts = self.table[["start_z_vox", "start_y_vox", "start_x_vox"]].to_numpy(dtype=float)
        directions = self.table[["direction_z_vox", "direction_y_vox", "direction_x_vox"]].to_numpy(dtype=float)
        self.vectors_zyx = np.stack((starts, starts + directions), axis=1)
        self.all_lines = make_line_mesh(self.vectors_zyx, self.base_colors)
        self.unit_cell_to_indices = self._unit_cell_indices()
        self.selected_indices: list[int] = []
        self.macro_ct_actors: list[object] = []
        self.micro_ct_actors: list[object] = []
        self.macro_layer_visibility = {
            "ct": True,
            "all_centerlines": True,
            "selected_centerline": True,
        }
        self.micro_layer_visibility = {
            "ct": True,
            "context_centerlines": True,
            "selected_centerline": True,
        }

        self.app = QApplication.instance() or QApplication([])
        self.macro_window, self.macro_plotter = self._new_window("Full Lattice (Macro)")
        self.micro_window, self.micro_plotter = self._new_window("Selected Unit Cell (Micro)")
        self._build_controls()
        self._add_macro_scene()
        self._add_micro_scene()
        self._enable_shift_pick()
        self.api_bridge = ApiSelectionBridge()
        self.api_bridge.selection_received.connect(self._receive_api_selection)
        self.api_bridge.connection_error.connect(lambda error: self._set_status(f"API connection closed: {error}"))
        self.api_thread = self._start_api_selection_client()

    def _validate_centerlines(self) -> None:
        required = {
            "strut_id", "inventory_unit_cell_ids", "inventory_classification",
            "start_z_vox", "start_y_vox", "start_x_vox",
            "direction_z_vox", "direction_y_vox", "direction_x_vox",
            "center_z_vox", "center_y_vox", "center_x_vox",
            "bbox_z_min_inclusive", "bbox_y_min_inclusive", "bbox_x_min_inclusive",
            "bbox_z_max_exclusive", "bbox_y_max_exclusive", "bbox_x_max_exclusive",
        }
        missing = sorted(required.difference(self.table.columns))
        if missing:
            raise ValueError("Centerlines CSV is missing required columns: " + ", ".join(missing))
        if self.table["strut_id"].astype(str).duplicated().any():
            raise ValueError("Centerlines CSV must contain one row per strut_id")

    @staticmethod
    def _load_defect_summary(path: Path) -> dict[str, pd.Series]:
        frame = pd.read_csv(path)
        if "strut_id" not in frame.columns:
            raise ValueError(f"Defect strut CSV must contain strut_id: {path}")
        return {str(strut_id): row for strut_id, row in frame.set_index("strut_id").iterrows()}

    @staticmethod
    def _load_station_details(path: Path) -> dict[str, pd.DataFrame]:
        frame = pd.read_csv(path)
        if "strut_id" not in frame.columns:
            raise ValueError(f"Defect station CSV must contain strut_id: {path}")
        return {str(strut_id): rows.drop(columns=["strut_id"]) for strut_id, rows in frame.groupby("strut_id", sort=False)}

    def _unit_cell_indices(self) -> dict[str, list[int]]:
        result: dict[str, list[int]] = {}
        for index, unit_cell in enumerate(self.table["inventory_unit_cell_ids"]):
            if pd.notna(unit_cell) and str(unit_cell).strip():
                result.setdefault(str(unit_cell), []).append(index)
        return result

    @staticmethod
    def _new_window(title: str) -> tuple[QMainWindow, QtInteractor]:
        window = QMainWindow()
        window.setWindowTitle(title)
        plotter = QtInteractor(window, auto_update=False)
        plotter.set_background("#161616")
        window.setCentralWidget(plotter)
        window.resize(1040, 760)
        return window, plotter

    @staticmethod
    def _add_dock(window: QMainWindow, title: str, content: QWidget) -> None:
        dock = QDockWidget(title, window)
        dock.setWidget(content)
        window.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, dock)

    def _build_controls(self) -> None:
        self.ids_input = QLineEdit()
        self.ids_input.setPlaceholderText("Comma- or space-separated strut IDs")
        apply_button = QPushButton("Apply")
        clear_button = QPushButton("Clear")
        self.selection_status = QLabel("No struts selected")
        apply_button.clicked.connect(self.apply_selection)
        clear_button.clicked.connect(self.clear_selection)
        full_content = QWidget()
        full_layout = QVBoxLayout(full_content)
        full_layout.addWidget(QLabel("Strut IDs:"))
        full_layout.addWidget(self.ids_input)
        button_row = QHBoxLayout()
        button_row.addWidget(apply_button)
        button_row.addWidget(clear_button)
        full_layout.addLayout(button_row)
        full_layout.addWidget(self.selection_status)
        full_layout.addWidget(QLabel("Shift + left-click a centerline to append it."))
        full_layout.addWidget(QLabel("Visible layers:"))
        self._add_layer_checkbox(full_layout, "Full CT Scan", self.macro_layer_visibility, "ct", self._set_macro_layer_visibility)
        self._add_layer_checkbox(full_layout, "All Centerlines", self.macro_layer_visibility, "all_centerlines", self._set_macro_layer_visibility)
        self._add_layer_checkbox(full_layout, "Selected Centerline", self.macro_layer_visibility, "selected_centerline", self._set_macro_layer_visibility)
        full_layout.addStretch()
        self._add_dock(self.macro_window, "Strut Selection", full_content)

        self.dropdown = QComboBox()
        self.dropdown.currentTextChanged.connect(self._on_dropdown_changed)
        self.metadata_display = QTextEdit("Select a strut...")
        self.metadata_display.setReadOnly(True)
        micro_content = QWidget()
        micro_layout = QFormLayout(micro_content)
        micro_layout.addRow("Selected Strut:", self.dropdown)
        micro_layout.addRow("Metadata:", self.metadata_display)
        layer_controls = QWidget()
        layer_layout = QVBoxLayout(layer_controls)
        layer_layout.setContentsMargins(0, 0, 0, 0)
        layer_layout.addWidget(QLabel("Visible layers:"))
        self._add_layer_checkbox(layer_layout, "Unit Cell 3D CT", self.micro_layer_visibility, "ct", self._set_micro_layer_visibility)
        self._add_layer_checkbox(layer_layout, "Unit Cell Centerlines", self.micro_layer_visibility, "context_centerlines", self._set_micro_layer_visibility)
        self._add_layer_checkbox(layer_layout, "Selected Centerline", self.micro_layer_visibility, "selected_centerline", self._set_micro_layer_visibility)
        micro_layout.addRow(layer_controls)
        self._add_dock(self.micro_window, "Strut Inspector", micro_content)

    @staticmethod
    def _add_layer_checkbox(
        layout: QVBoxLayout,
        label: str,
        visibility: dict[str, bool],
        layer: str,
        callback,
    ) -> None:
        checkbox = QCheckBox(label)
        checkbox.setChecked(visibility[layer])
        checkbox.toggled.connect(lambda visible, layer_name=layer: callback(layer_name, visible))
        layout.addWidget(checkbox)

    @staticmethod
    def _remove_actors(plotter: QtInteractor, actors: list[object]) -> None:
        for actor in actors:
            plotter.remove_actor(actor, render=False)
        actors.clear()

    @staticmethod
    def _set_actors_visibility(actors: list[object], visible: bool) -> None:
        for actor in actors:
            actor.SetVisibility(visible)

    def _set_macro_layer_visibility(self, layer: str, visible: bool) -> None:
        self.macro_layer_visibility[layer] = visible
        if layer == "ct":
            self._set_actors_visibility(self.macro_ct_actors, visible)
        elif layer == "all_centerlines":
            self.all_actor.SetVisibility(visible)
        else:
            self.selected_actor.SetVisibility(visible)
        self.macro_plotter.render()

    def _set_micro_layer_visibility(self, layer: str, visible: bool) -> None:
        self.micro_layer_visibility[layer] = visible
        if layer == "ct":
            self._set_actors_visibility(self.micro_ct_actors, visible)
        elif layer == "context_centerlines":
            self.micro_context_actor.SetVisibility(visible)
        else:
            self.micro_selected_actor.SetVisibility(visible)
        self.micro_plotter.render()

    def _add_volume(self, plotter: QtInteractor, grid: pv.ImageData) -> list[object]:
        # Napari renders both CT layers as a grayscale maximum-intensity
        # projection. VTK's maximum blend mode requires a named transfer
        # function; a numeric opacity array leaves this projection black.
        intensities = grid.point_data["ct_intensity"]
        contrast_limits = (float(np.min(intensities)), float(np.max(intensities)))
        actor = plotter.add_volume(
            grid,
            scalars="ct_intensity",
            cmap="gray",
            opacity="linear",
            clim=contrast_limits,
            blending="maximum",
            show_scalar_bar=False,
            pickable=False,
        )
        return actor if isinstance(actor, list) else [actor]

    def _add_macro_scene(self) -> None:
        print(f"Loading {self.args.macro_downsample}x-downsampled macro CT volume...", flush=True)
        grid = image_grid(self.volume, self.args.macro_downsample, np.zeros(3, dtype=int))
        self.macro_ct_actors = self._add_volume(self.macro_plotter, grid)
        self.all_actor = self.macro_plotter.add_mesh(
            self.all_lines, scalars="rgba", rgba=True, line_width=ALL_STRUT_WIDTH, name="All Centerlines", pickable=True
        )
        self.selected_actor = self.macro_plotter.add_mesh(
            make_line_mesh(np.empty((0, 2, 3))), color=SELECTED_STRUT_COLOR, line_width=SELECTED_STRUT_WIDTH,
            name="Selected Centerlines", pickable=False
        )
        self._set_actors_visibility(self.macro_ct_actors, self.macro_layer_visibility["ct"])
        self.all_actor.SetVisibility(self.macro_layer_visibility["all_centerlines"])
        self.selected_actor.SetVisibility(self.macro_layer_visibility["selected_centerline"])
        self.macro_plotter.add_text("CT volume + all centerlines + selected centerlines", font_size=10)
        self.macro_plotter.reset_camera()

    def _add_micro_scene(self) -> None:
        self.micro_context_actor = self.micro_plotter.add_mesh(
            make_line_mesh(np.empty((0, 2, 3))), color="gray", line_width=ALL_STRUT_WIDTH,
            name="Unit Cell Centerlines", pickable=False
        )
        self.micro_selected_actor = self.micro_plotter.add_mesh(
            make_line_mesh(np.empty((0, 2, 3))), color=SELECTED_STRUT_COLOR, line_width=SELECTED_STRUT_WIDTH,
            name="Selected Centerline", pickable=False
        )
        self.micro_context_actor.SetVisibility(self.micro_layer_visibility["context_centerlines"])
        self.micro_selected_actor.SetVisibility(self.micro_layer_visibility["selected_centerline"])
        self.micro_plotter.add_text("Select a strut to load a bounded CT crop", font_size=10)

    def _replace_micro_context_actor(self, mesh: pv.PolyData) -> None:
        self.micro_plotter.remove_actor(self.micro_context_actor, render=False)
        self.micro_context_actor = self.micro_plotter.add_mesh(
            mesh,
            scalars="rgba",
            rgba=True,
            line_width=ALL_STRUT_WIDTH,
            name="Unit Cell Centerlines",
            pickable=False,
        )
        self.micro_context_actor.SetVisibility(self.micro_layer_visibility["context_centerlines"])

    def _reset_micro_context_actor(self) -> None:
        self.micro_plotter.remove_actor(self.micro_context_actor, render=False)
        self.micro_context_actor = self.micro_plotter.add_mesh(
            make_line_mesh(np.empty((0, 2, 3))),
            color="gray",
            line_width=ALL_STRUT_WIDTH,
            name="Unit Cell Centerlines",
            pickable=False,
        )
        self.micro_context_actor.SetVisibility(self.micro_layer_visibility["context_centerlines"])

    def _enable_shift_pick(self) -> None:
        picker = vtkCellPicker()
        picker.SetTolerance(0.0005)
        interactor = self.macro_plotter.iren.interactor
        interactor.SetPicker(picker)

        def on_left_button_press(vtk_interactor, _event):
            if not vtk_interactor.GetShiftKey():
                return
            x, y = vtk_interactor.GetEventPosition()
            picker.Pick(x, y, 0, self.macro_plotter.renderer)
            if picker.GetActor() != self.all_actor:
                return
            cell_id = picker.GetCellId()
            if 0 <= cell_id < len(self.strut_ids):
                current = parse_strut_ids(self.ids_input.text())
                strut_id = self.strut_ids[cell_id]
                if strut_id not in current:
                    current.append(strut_id)
                self.ids_input.setText(", ".join(current))
                self.apply_selection()

        interactor.AddObserver("LeftButtonPressEvent", on_left_button_press)

    def _context_indices_for(self, index: int) -> tuple[list[int], str | None]:
        unit_cell = self.table.iloc[index]["inventory_unit_cell_ids"]
        if pd.isna(unit_cell) or not str(unit_cell).strip():
            return [index], None
        return self.unit_cell_to_indices.get(str(unit_cell), [index]), str(unit_cell)

    def _context_bounds(self, indices: list[int]) -> tuple[np.ndarray, np.ndarray]:
        rows = self.table.iloc[indices]
        lower = rows[["bbox_z_min_inclusive", "bbox_y_min_inclusive", "bbox_x_min_inclusive"]].to_numpy(dtype=int).min(axis=0)
        upper = rows[["bbox_z_max_exclusive", "bbox_y_max_exclusive", "bbox_x_max_exclusive"]].to_numpy(dtype=int).max(axis=0)
        lower = np.maximum(lower - self.args.unit_cell_padding_vox, 0)
        upper = np.minimum(upper + self.args.unit_cell_padding_vox, self.volume_shape)
        if np.any(lower >= upper):
            raise ValueError("Selected strut's context crop does not intersect the scan")
        return lower, upper

    def apply_selection(self) -> None:
        requested = parse_strut_ids(self.ids_input.text())
        unknown = [strut_id for strut_id in requested if strut_id not in self.strut_index]
        if unknown:
            self._set_status("Unknown strut ID(s): " + ", ".join(unknown))
            return
        if not requested:
            self._set_status("Enter at least one strut ID")
            return
        self.selected_indices = [self.strut_index[strut_id] for strut_id in requested]
        self.ids_input.setText(", ".join(requested))
        all_colors = list(self.base_colors)
        for index in self.selected_indices:
            all_colors[index] = TRANSPARENT_RGBA
        self.all_actor.mapper.SetInputData(make_line_mesh(self.vectors_zyx, all_colors))
        self.selected_actor.mapper.SetInputData(make_line_mesh(self.vectors_zyx[self.selected_indices]))
        self.dropdown.blockSignals(True)
        self.dropdown.clear()
        self.dropdown.addItems(requested)
        self.dropdown.blockSignals(False)
        self.dropdown.setCurrentText(requested[0])
        self._set_status(f"{len(requested)} strut(s) selected")
        self.update_micro_view(requested[0])
        self.macro_plotter.render()

    def clear_selection(self) -> None:
        self.selected_indices = []
        self.ids_input.clear()
        self.all_actor.mapper.SetInputData(self.all_lines)
        self.selected_actor.mapper.SetInputData(make_line_mesh(np.empty((0, 2, 3))))
        self.dropdown.blockSignals(True)
        self.dropdown.clear()
        self.dropdown.blockSignals(False)
        self._remove_actors(self.micro_plotter, self.micro_ct_actors)
        self._reset_micro_context_actor()
        self.micro_selected_actor.mapper.SetInputData(make_line_mesh(np.empty((0, 2, 3))))
        self.metadata_display.setPlainText("Select a strut...")
        self._set_status("No struts selected")
        self.macro_plotter.render()
        self.micro_plotter.render()

    def _on_dropdown_changed(self, strut_id: str) -> None:
        if strut_id:
            self.update_micro_view(strut_id)

    def update_micro_view(self, strut_id: str) -> None:
        index = self.strut_index[str(strut_id)]
        row = self.table.iloc[index]
        context_indices, unit_cell = self._context_indices_for(index)
        lower, upper = self._context_bounds(context_indices)
        crop = np.asarray(self.volume[lower[0]:upper[0], lower[1]:upper[1], lower[2]:upper[2]])
        crop_grid = image_grid(crop, self.args.micro_downsample, lower)
        self._remove_actors(self.micro_plotter, self.micro_ct_actors)
        center = row[["center_z_vox", "center_y_vox", "center_x_vox"]].to_numpy(dtype=float)
        self.micro_ct_actors = self._add_volume(self.micro_plotter, crop_grid)
        self._set_actors_visibility(self.micro_ct_actors, self.micro_layer_visibility["ct"])
        context_colors = [
            TRANSPARENT_RGBA
            if context_index == index
            else unit_cell_context_color(self.table.iloc[context_index]["inventory_classification"])
            for context_index in context_indices
        ]
        self._replace_micro_context_actor(make_line_mesh(self.vectors_zyx[context_indices], context_colors))
        self.micro_selected_actor.mapper.SetInputData(make_line_mesh(self.vectors_zyx[index:index + 1]))
        metadata = format_strut_metadata(self.defect_strut_index.get(str(strut_id)), row)
        metadata += f"\n\n=== VISUALIZER CONTEXT ===\nUnit Cell: {unit_cell or 'Unavailable'} ({len(context_indices)} struts)"
        self.metadata_display.setPlainText(metadata)
        self.micro_plotter.reset_camera()
        self.micro_plotter.render()
        # Reset before zooming so switching inspector entries cannot compound
        # the zoom factor on every selection.
        self.macro_plotter.reset_camera()
        self.macro_plotter.camera.focal_point = tuple(center[::-1])
        self.macro_plotter.camera.zoom(self.args.macro_zoom)
        self.macro_plotter.render()

    def _set_status(self, text: str) -> None:
        self.selection_status.setText(text)
        print(text, flush=True)

    def _receive_api_selection(self, strut_ids: list) -> None:
        self.ids_input.setText(", ".join(str(value) for value in strut_ids))
        self.apply_selection()

    def _start_api_selection_client(self) -> threading.Thread | None:
        if not self.args.api_url:
            return None
        try:
            import websocket
        except ImportError as exc:
            self._set_status(f"API client unavailable: install websocket-client ({exc})")
            return None
        websocket_url = self.args.api_url.rstrip("/").replace("https://", "wss://").replace("http://", "ws://") + "/ws"

        def receive_events() -> None:
            try:
                connection = websocket.create_connection(websocket_url, timeout=5)
                connection.settimeout(None)
                print(f"API connected: {websocket_url}", flush=True)
                while True:
                    event = json.loads(connection.recv())
                    if event.get("event_type") == "STRUTS_SELECTED":
                        ids = event.get("data", {}).get("strut_ids", [])
                        if ids:
                            self.api_bridge.selection_received.emit(ids)
            except Exception as exc:
                self.api_bridge.connection_error.emit(str(exc))

        thread = threading.Thread(target=receive_events, daemon=True, name="pyvista-api-selection")
        thread.start()
        self._set_status(f"Listening for dashboard selections at {websocket_url}")
        return thread

    def show(self) -> int:
        self.macro_window.show()
        self.micro_window.show()
        return self.app.exec()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dual-window PyVista strut visualizer")
    parser.add_argument("--scan", required=True, help="Path to the 3-D TIFF scan")
    parser.add_argument("--centerlines", required=True, help="Path to napari_centerlines.csv")
    parser.add_argument("--unit-cell-padding-vox", type=int, default=10, help="Source-voxel border around a unit-cell crop")
    parser.add_argument("--macro-zoom", type=float, default=6.0, help="Macro camera zoom after selecting a strut")
    parser.add_argument(
        "--macro-downsample",
        type=int,
        default=DEFAULT_MACRO_DOWNSAMPLE,
        help=f"Integer CT stride for macro MIP rendering (default: {DEFAULT_MACRO_DOWNSAMPLE})",
    )
    parser.add_argument(
        "--micro-downsample",
        type=int,
        default=DEFAULT_MICRO_DOWNSAMPLE,
        help=f"Integer CT stride for bounded micro MIP rendering (default: {DEFAULT_MICRO_DOWNSAMPLE})",
    )
    parser.add_argument("--api-url", default=os.getenv("NAPARI_API_URL"), help="Optional FastAPI base URL for dashboard selection events")
    default_defect_dir = Path(__file__).resolve().parents[2] / "part2/stage_3_defect_analysis/output/station_export_20260727T193004Z"
    parser.add_argument("--defect-by-strut", type=Path, default=default_defect_dir / "defect_analysis_by_strut.csv")
    parser.add_argument("--defect-by-station", type=Path, default=default_defect_dir / "defect_analysis_by_station.csv")
    args = parser.parse_args()
    if args.unit_cell_padding_vox < 0:
        parser.error("--unit-cell-padding-vox must be non-negative")
    if args.macro_zoom <= 0:
        parser.error("--macro-zoom must be greater than zero")
    if args.macro_downsample < 1 or args.micro_downsample < 1:
        parser.error("CT downsample strides must be greater than zero")
    return args


def main() -> None:
    args = parse_args()
    print(f"Memory-mapping TIFF from {args.scan}...", flush=True)
    visualizer = StrutVisualizer(args)
    raise SystemExit(visualizer.show())


if __name__ == "__main__":
    main()
