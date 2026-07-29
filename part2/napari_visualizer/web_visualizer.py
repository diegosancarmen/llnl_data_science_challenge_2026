"""Browser-native, dual-pane PyVista viewer synchronized through the dashboard broker.

Run this process beside FastAPI and Streamlit.  It owns the TIFF memory map;
the broker continues to exchange only small metadata and selection messages.
"""

from __future__ import annotations

import argparse
import json
import os
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyvista as pv
import requests
import tifffile

from part2.napari_visualizer.scene_data import (
    TRANSPARENT_RGBA,
    classification_color,
    image_grid,
    make_line_mesh,
    unit_cell_context_color,
)

SELECTED_STRUT_COLOR = "yellow"
SELECTED_STRUT_WIDTH = 6
ALL_STRUT_WIDTH = 3


class WebStrutVisualizer:
    """Own the two PyVista scenes and mirror selection through FastAPI."""

    def __init__(self, args: argparse.Namespace):
        started_at = time.perf_counter()
        try:
            from trame.app import get_server
            from trame.ui.vuetify3 import SinglePageLayout
            from trame.widgets import html, vuetify3
            from trame.widgets import vtk as vtk_widgets
        except ImportError as exc:  # pragma: no cover - depends on optional runtime
            raise RuntimeError("Install trame, trame-vtk, and trame-vuetify to run the web visualizer.") from exc

        self.get_server = get_server
        self.SinglePageLayout = SinglePageLayout
        self.html = html
        self.vuetify3 = vuetify3
        self.vtk_widgets = vtk_widgets
        self.args = args
        self.api_url = args.api_url.rstrip("/")
        print(f"[viewer] memory-mapping CT scan: {args.scan}", flush=True)
        self.volume = tifffile.memmap(args.scan)
        if self.volume.ndim != 3:
            raise ValueError(f"Expected a 3-D TIFF scan, got shape {self.volume.shape}")
        self.volume_shape = np.asarray(self.volume.shape, dtype=int)
        self.table = pd.read_csv(args.centerlines)
        self._validate_centerlines()
        self.strut_ids = self.table["strut_id"].astype(str).tolist()
        self.strut_index = {strut_id: index for index, strut_id in enumerate(self.strut_ids)}
        self.base_colors = [classification_color(value) for value in self.table["inventory_classification"]]
        starts = self.table[["start_z_vox", "start_y_vox", "start_x_vox"]].to_numpy(dtype=float)
        directions = self.table[["direction_z_vox", "direction_y_vox", "direction_x_vox"]].to_numpy(dtype=float)
        self.vectors_zyx = np.stack((starts, starts + directions), axis=1)
        self.all_lines = make_line_mesh(self.vectors_zyx, self.base_colors)
        self.all_lines.cell_data["strut_index"] = np.arange(len(self.strut_ids), dtype=np.int64)
        self.unit_cell_to_indices = self._unit_cell_indices()
        self.selected_id = self.strut_ids[0]

        self.server = self.get_server("strut-viewer", client_type="vue3")
        self.state = self.server.state
        self.state.strut_options = [{"title": strut_id, "value": strut_id} for strut_id in self.strut_ids]
        self.state.selected_strut_id = self.selected_id
        self.state.macro_ct_visible = True
        self.state.macro_lines_visible = True
        self.state.macro_active_line_visible = True
        self.state.micro_ct_visible = True
        self.state.micro_lines_visible = True
        self.state.micro_active_line_visible = True


        self.macro_plotter = pv.Plotter(off_screen=True, window_size=(900, 600))
        self.micro_plotter = pv.Plotter(off_screen=True, window_size=(900, 600))
        print(f"[viewer] building PyVista scenes (elapsed {time.perf_counter() - started_at:.1f}s)", flush=True)
        self._build_scenes()
        print(f"[viewer] scenes ready (elapsed {time.perf_counter() - started_at:.1f}s)", flush=True)
        self._build_ui()
        self._observe_state()
        self._start_broker_listener()
        print(f"[viewer] initialization complete (elapsed {time.perf_counter() - started_at:.1f}s)", flush=True)

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

    def _unit_cell_indices(self) -> dict[str, list[int]]:
        groups: dict[str, list[int]] = {}
        for index, unit_cell in enumerate(self.table["inventory_unit_cell_ids"]):
            if pd.notna(unit_cell) and str(unit_cell).strip():
                groups.setdefault(str(unit_cell), []).append(index)
        return groups

    def _add_volume(self, plotter: pv.Plotter, grid: pv.ImageData) -> list[Any]:
        values = grid.point_data["ct_intensity"]
        actor = plotter.add_volume(
            grid, scalars="ct_intensity", cmap="gray", opacity="linear",
            clim=(float(np.min(values)), float(np.max(values))), blending="maximum",
            show_scalar_bar=False, pickable=False,
        )
        return actor if isinstance(actor, list) else [actor]

    def _build_scenes(self) -> None:
        self.macro_plotter.set_background("#161616")
        macro_grid = image_grid(self.volume, self.args.macro_downsample, np.zeros(3, dtype=int))
        self.macro_ct_actors = self._add_volume(self.macro_plotter, macro_grid)
        self.macro_all_actor = self.macro_plotter.add_mesh(
            self.all_lines, scalars="rgba", rgba=True, line_width=ALL_STRUT_WIDTH, pickable=True,
        )
        self.macro_selected_actor = self.macro_plotter.add_mesh(
            make_line_mesh(np.empty((0, 2, 3))), color=SELECTED_STRUT_COLOR,
            line_width=SELECTED_STRUT_WIDTH, pickable=False,
        )
        self.macro_plotter.add_text("Full lattice", font_size=12)
        # Element picking is handled by the VTK interactor used by VtkRemoteView.
        self.macro_plotter.enable_element_picking(callback=self._macro_picked, mode="cell", show=True, show_message=False)
        self.macro_plotter.add_text(
            "Right-click or press P to select strut", 
            position="lower_left", 
            font_size=9, 
            color="gray"
        )
        self.macro_plotter.reset_camera()
        self._setup_camera_bounds(self.macro_plotter, min_dist_factor=0.33, max_dist_factor=2.0)

        self.micro_plotter.set_background("#161616")
        self.micro_context_actor = self.micro_plotter.add_mesh(
            make_line_mesh(np.empty((0, 2, 3))), color="gray", line_width=ALL_STRUT_WIDTH, pickable=False,
        )
        self.micro_selected_actor = self.micro_plotter.add_mesh(
            make_line_mesh(np.empty((0, 2, 3))), color=SELECTED_STRUT_COLOR,
            line_width=SELECTED_STRUT_WIDTH, pickable=False,
        )
        self.micro_plotter.add_text("Selected unit cell", font_size=12)
        self._update_selection(self.selected_id, publish=False)

        self.micro_plotter.reset_camera()
        self._setup_camera_bounds(self.micro_plotter, min_dist_factor=0.33, max_dist_factor=2.0)

    def _build_ui(self) -> None:
        with self.SinglePageLayout(self.server, full_height=True) as layout:
            layout.theme = "dark"
            layout.title.set_text("Lattice PyVista Viewer")
            with layout.toolbar:
                self.vuetify3.VSelect(
                    v_model=("selected_strut_id",), items=("strut_options",), label="Active strut",
                    density="compact", style="max-width: 280px",
                )
            with layout.content:
                with self.vuetify3.VContainer(fluid=True, classes="fill-height pa-2"):
                    with self.vuetify3.VRow(dense=True, classes="fill-height"):
                        with self.vuetify3.VCol(cols=12, md=6,
                                                style="height: 100%; display: flex; flex-direction: column;"):
                            self.vuetify3.VCheckbox(v_model=("macro_ct_visible",), label="Full CT scan", density="compact", hide_details=True)
                            self.vuetify3.VCheckbox(v_model=("macro_lines_visible",), label="All centerlines", density="compact", hide_details=True)
                            self.vuetify3.VCheckbox(v_model=("macro_active_line_visible",), label="Selected centerline", 
                            density="compact", hide_details=True)

                            with self.html.Div(style="flex: 1 1 auto; height: 100%; min-height: 0; position: relative;"):
                                self.macro_view = self.vtk_widgets.VtkRemoteView(
                                    self.macro_plotter.ren_win, 
                                    interactive_ratio=1,
                                    style="height: 100%; width: 100%;"
                                )

                            self.vuetify3.VSlider(
                                v_model=("macro_zoom", 1.0),
                                min=0.5, max=3.0, step=0.1,
                                label="Macro Zoom", density="compact", hide_details=True
                            )


                        with self.vuetify3.VCol(cols=12, md=6,
                                                style="height: 100%; display: flex; flex-direction: column;"):
                            self.vuetify3.VCheckbox(v_model=("micro_ct_visible",), label="Unit-cell CT", density="compact", hide_details=True)
                            self.vuetify3.VCheckbox(v_model=("micro_lines_visible",), label="Unit-cell centerlines", density="compact", hide_details=True)
                            self.vuetify3.VCheckbox(v_model=("micro_active_line_visible",), label="Selected centerline", 
                            density="compact", hide_details=True)


                            with self.html.Div(style="flex: 1 1 auto; height: 100%; min-height: 0; position: relative;"):
                                self.micro_view = self.vtk_widgets.VtkRemoteView(
                                    self.micro_plotter.ren_win, interactive_ratio=1, 
                                    style="height: 100%; width: 100%;"
                                )

                            self.vuetify3.VSlider(
                                v_model=("micro_zoom", 1.0),
                                min=0.5, max=3.0, step=0.1,
                                label="Micro Zoom", density="compact", hide_details=True
                            )
    def _setup_camera_bounds(
        self, 
        plotter: pv.Plotter, 
        min_dist_factor: float = 0.33,  # Max zoom in (1 / 3.0)
        max_dist_factor: float = 2.0     # Max zoom out (1 / 0.5)
    ) -> None:
        """Clamps mouse wheel and right-click drag zooming directly in VTK."""
        
        # Store initial baseline camera distance after scene setup
        base_dist = plotter.camera.distance
        min_dist = base_dist * min_dist_factor
        max_dist = base_dist * max_dist_factor

        def clamp_mouse_zoom(obj, event):
            cam = plotter.camera
            current_dist = cam.distance
            
            if current_dist < min_dist or current_dist > max_dist:
                # Clamp the camera distance while preserving focal point and angle
                clamped_dist = max(min_dist, min(max_dist, current_dist))
                
                focal_pt = np.array(cam.focal_point)
                cam_pos = np.array(cam.position)
                direction = cam_pos - focal_pt
                norm = np.linalg.norm(direction)
                
                if norm > 0:
                    direction = direction / norm
                    cam.position = tuple(focal_pt + direction * clamped_dist)

        # Attach to VTK's EndInteractionEvent (fires after mouse drag/scroll)
        plotter.iren.add_observer("EndInteractionEvent", clamp_mouse_zoom)
        # Attach to InteractionEvent (fires continuously during mouse drag)
        plotter.iren.add_observer("InteractionEvent", clamp_mouse_zoom)
        
    def _observe_state(self) -> None:
        @self.state.change("selected_strut_id")
        def selected_strut_changed(selected_strut_id: str, **_kwargs: Any) -> None:
            if selected_strut_id in self.strut_index and selected_strut_id != self.selected_id:
                self._update_selection(selected_strut_id, publish=True)

        @self.state.change("macro_ct_visible", "macro_lines_visible", "macro_active_line_visible", "micro_ct_visible", "micro_lines_visible", "micro_active_line_visible")
        def visibility_changed(**_kwargs: Any) -> None:
            self._set_visibility()

        MIN_ZOOM = 0.5
        MAX_ZOOM = 3.0

        @self.state.change("macro_zoom")
        def _on_macro_zoom_change(macro_zoom, **kwargs):
            if hasattr(self, "macro_plotter") and self.macro_plotter:
                # 1. Clamp the state value between MIN and MAX
                clamped_zoom = max(MIN_ZOOM, min(MAX_ZOOM, float(macro_zoom)))
                
                # 2. Reset camera to baseline focal fit, then apply absolute zoom factor
                self.macro_plotter.reset_camera()
                self.macro_plotter.camera.zoom(clamped_zoom)
                
                # 3. Update view
                self.macro_view.update()

        @self.state.change("micro_zoom")
        def _on_micro_zoom_change(micro_zoom, **kwargs):
            if hasattr(self, "micro_plotter") and self.micro_plotter:
                # 1. Clamp the state value between MIN and MAX
                clamped_zoom = max(MIN_ZOOM, min(MAX_ZOOM, float(micro_zoom)))
                
                # 2. Reset camera to baseline focal fit, then apply absolute zoom factor
                self.micro_plotter.reset_camera()
                self.micro_plotter.camera.zoom(clamped_zoom)
                
                # 3. Update view
                self.micro_view.update()

    def _macro_picked(self, picked: pv.DataSet) -> None:
        """Publish a clicked macro centerline when PyVista reports its source cell."""
        try:
            index = int(np.asarray(picked.cell_data["strut_index"])[0])
        except (KeyError, IndexError, TypeError, ValueError):
            return
        if 0 <= index < len(self.strut_ids):
            self.state.selected_strut_id = self.strut_ids[index]

    def _context_for(self, index: int) -> tuple[list[int], str | None]:
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
            raise ValueError("Selected strut context crop does not intersect the scan")
        return lower, upper

    def _replace_actor(self, plotter: pv.Plotter, actor: Any, mesh: pv.PolyData, **kwargs: Any) -> Any:
        plotter.remove_actor(actor, render=False)
        return plotter.add_mesh(mesh, **kwargs)

    def _update_selection(self, strut_id: str, *, publish: bool) -> None:
        self.selected_id = strut_id
        index = self.strut_index[strut_id]
        colors = list(self.base_colors)
        colors[index] = TRANSPARENT_RGBA
        macro_lines = make_line_mesh(self.vectors_zyx, colors)
        macro_lines.cell_data["strut_index"] = np.arange(len(self.strut_ids), dtype=np.int64)
        self.macro_all_actor.mapper.SetInputData(macro_lines)
        self.macro_selected_actor.mapper.SetInputData(make_line_mesh(self.vectors_zyx[index:index + 1]))

        indices, _unit_cell = self._context_for(index)
        lower, upper = self._context_bounds(indices)
        crop = np.asarray(self.volume[lower[0]:upper[0], lower[1]:upper[1], lower[2]:upper[2]])
        for actor in getattr(self, "micro_ct_actors", []):
            self.micro_plotter.remove_actor(actor, render=False)
        self.micro_ct_actors = self._add_volume(self.micro_plotter, image_grid(crop, self.args.micro_downsample, lower))
        context_colors = [TRANSPARENT_RGBA if item == index else unit_cell_context_color(self.table.iloc[item]["inventory_classification"]) for item in indices]
        self.micro_context_actor = self._replace_actor(
            self.micro_plotter, self.micro_context_actor, make_line_mesh(self.vectors_zyx[indices], context_colors),
            scalars="rgba", rgba=True, line_width=ALL_STRUT_WIDTH, pickable=False,
        )
        self.micro_selected_actor.mapper.SetInputData(make_line_mesh(self.vectors_zyx[index:index + 1]))
        self.micro_plotter.reset_camera()
        self.macro_plotter.reset_camera()
        center = self.table.iloc[index][["center_z_vox", "center_y_vox", "center_x_vox"]].to_numpy(dtype=float)
        self.macro_plotter.camera.focal_point = tuple(center[::-1])
        self.macro_plotter.camera.zoom(self.args.macro_zoom)
        self._set_visibility()
        self._render()
        if publish:
            try:
                requests.post(f"{self.api_url}/select_struts", json={"strut_ids": [int(strut_id)]}, timeout=3).raise_for_status()
            except requests.RequestException as exc:
                print(f"Could not publish selected strut: {exc}", flush=True)

    def _set_visibility(self) -> None:
        # Macro pane visibility
        for actor in self.macro_ct_actors:
            actor.SetVisibility(self.state.macro_ct_visible)
        self.macro_all_actor.SetVisibility(self.state.macro_lines_visible)
        
        # Check if macro_active_line_visible exists in state; default to True if not set
        macro_active = getattr(self.state, "macro_active_line_visible", True)
        self.macro_selected_actor.SetVisibility(macro_active)

        # Micro pane visibility
        for actor in getattr(self, "micro_ct_actors", []):
            actor.SetVisibility(self.state.micro_ct_visible)
        self.micro_context_actor.SetVisibility(self.state.micro_lines_visible)
        
        # Check if micro_active_line_visible exists in state; default to True if not set
        micro_active = getattr(self.state, "micro_active_line_visible", True)
        self.micro_selected_actor.SetVisibility(micro_active)

        self._render()

    def _render(self) -> None:
        self.macro_plotter.render()
        self.micro_plotter.render()
        if hasattr(self, "macro_view"):
            self.macro_view.update()
            self.micro_view.update()

    def _start_broker_listener(self) -> None:
        websocket_url = self.api_url.replace("https://", "wss://").replace("http://", "ws://") + "/ws"

        def listen() -> None:
            try:
                import websocket
                connection = websocket.create_connection(websocket_url, timeout=5)
                connection.settimeout(None)
                while True:
                    event = json.loads(connection.recv())
                    if event.get("event_type") == "INIT_STATE":
                        selected = event.get("data", {}).get("active_strut_id")
                    elif event.get("event_type") == "STRUTS_SELECTED":
                        selected = (event.get("data", {}).get("strut_ids") or [None])[0]
                    else:
                        continue
                    if str(selected) in self.strut_index:
                        self.state.selected_strut_id = str(selected)
            except Exception as exc:
                print(f"Broker selection listener stopped: {exc}", flush=True)

        threading.Thread(target=listen, daemon=True, name="pyvista-web-broker").start()

    def start(self) -> None:
        print(f"[viewer] serving on http://{self.args.host}:{self.args.port}", flush=True)
        self.server.start(host=self.args.host, port=self.args.port, open_browser=False)


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Browser-native dual-pane PyVista strut visualizer")
    parser.add_argument("--scan", type=Path, default=repo_root / "part2/data/missing_struts/tif_stacks/210127_Brian_Tran_strut_lattices_0point5dash1 1 Slices.tif")
    parser.add_argument("--centerlines", type=Path, default=repo_root / "part2/napari_visualizer/napari_centerlines.csv")
    parser.add_argument("--api-url", default=os.getenv("DASHBOARD_API_URL", "http://127.0.0.1:8000"))
    parser.add_argument("--host", default=os.getenv("PYVISTA_VIEWER_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.getenv("PYVISTA_VIEWER_PORT", "8081")))
    parser.add_argument("--unit-cell-padding-vox", type=int, default=10)
    parser.add_argument("--macro-zoom", type=float, default=6.0)
    parser.add_argument("--macro-downsample", type=int, default=2)
    parser.add_argument("--micro-downsample", type=int, default=1)
    args = parser.parse_args()
    if args.unit_cell_padding_vox < 0 or args.macro_downsample < 1 or args.micro_downsample < 1 or args.macro_zoom <= 0:
        parser.error("padding/downsample values must be non-negative/positive and macro zoom must be positive")
    return args


def main() -> None:
    args = parse_args()
    WebStrutVisualizer(args).start()


if __name__ == "__main__":
    main()
