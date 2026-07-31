"""Browser-native, dual-pane PyVista viewer synchronized through the dashboard broker.

Run this process beside FastAPI and Streamlit.  It owns the TIFF memory map;
the broker continues to exchange only small metadata and selection messages.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import queue
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
    parse_strut_ids,
    placeholder_line_mesh,
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
        self.selected_ids = [self.selected_id]
        self._selection_state_update = False
        self._broker_selection_queue: queue.SimpleQueue[tuple[list[str], str | None]] = queue.SimpleQueue()

        self.server = self.get_server("strut-viewer", client_type="vue3")
        self.state = self.server.state
        self.state.strut_ids_input = self.selected_id
        self.state.selected_strut_options = [{"title": self.selected_id, "value": self.selected_id}]
        self.state.selection_status = "1 strut selected"
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
        self._register_broker_selection_consumer()
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
            placeholder_line_mesh(), color=SELECTED_STRUT_COLOR,
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
            placeholder_line_mesh("gray"), scalars="rgba", rgba=True,
            line_width=ALL_STRUT_WIDTH, pickable=False,
        )
        self.micro_selected_actor = self.micro_plotter.add_mesh(
            placeholder_line_mesh(), color=SELECTED_STRUT_COLOR,
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
            with layout.content:
                with self.vuetify3.VContainer(fluid=True, classes="fill-height pa-2"):
                    with self.vuetify3.VRow(dense=True, classes="fill-height"):
                        with self.vuetify3.VCol(cols=12, md=6,
                                                style="height: 100%; display: flex; flex-direction: column;"):
                            self.vuetify3.VTextField(
                                v_model=("strut_ids_input",), label="Strut IDs",
                                placeholder="Comma- or space-separated IDs", density="compact",
                                hide_details=True,
                            )
                            with self.html.Div(classes="d-flex align-center mb-1"):
                                self.vuetify3.VBtn("Apply", click=self._apply_input_selection, density="compact")
                                self.vuetify3.VBtn("Clear", click=self._clear_selection, density="compact", classes="ml-2")
                                self.html.Div("{{ selection_status }}", classes="ml-3 text-caption")

                            with self.vuetify3.VContainer(classes="d-flex flex-row align-center ga-4 pa-0 mb-2"):
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
                            self.vuetify3.VSelect(
                                v_model=("selected_strut_id",), items=("selected_strut_options",),
                                label="Selected strut", density="compact", hide_details=True,
                            )

                            # Empty div matching the height visually via padding
                            self.html.Div(classes="py-3 mb-1")

                            with self.vuetify3.VContainer(classes="d-flex flex-row align-center ga-4 pa-0 mb-2"):
                                self.vuetify3.VCheckbox(
                                    v_model=("micro_ct_visible",),
                                    label="Unit-cell CT",
                                    density="compact",
                                    hide_details=True,
                                )
                                self.vuetify3.VCheckbox(
                                    v_model=("micro_lines_visible",),
                                    label="Unit-cell centerlines",
                                    density="compact",
                                    hide_details=True,
                                )
                                self.vuetify3.VCheckbox(
                                    v_model=("micro_active_line_visible",),
                                    label="Selected centerline",
                                    density="compact",
                                    hide_details=True,
                                )

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
            if (
                not self._selection_state_update
                and selected_strut_id in self.strut_index
                and selected_strut_id in self.selected_ids
                and selected_strut_id != self.selected_id
            ):
                self._update_selection(selected_strut_id, publish=False)
                self._publish_active_strut(selected_strut_id)

        @self.state.change("macro_ct_visible", "macro_lines_visible", "macro_active_line_visible", "micro_ct_visible", "micro_lines_visible", "micro_active_line_visible")
        def visibility_changed(**_kwargs: Any) -> None:
            self._set_visibility()
            self._render()

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
            self.state.strut_ids_input = self.strut_ids[index]
            self._apply_input_selection(publish=True)

    def _set_selection_status(self, message: str) -> None:
        self.state.selection_status = message

    def _apply_input_selection(self, *_args: Any, publish: bool = True, **_kwargs: Any) -> None:
        requested = parse_strut_ids(self.state.strut_ids_input)
        self._apply_ids(requested, publish=publish)

    def _apply_ids(
        self, requested: list[str], *, publish: bool, active_strut_id: str | None = None
    ) -> None:
        requested = list(dict.fromkeys(str(value) for value in requested))
        unknown = [strut_id for strut_id in requested if strut_id not in self.strut_index]
        if unknown:
            self._set_selection_status("Unknown strut ID(s): " + ", ".join(unknown))
            return
        if not requested:
            self._set_selection_status("Enter at least one strut ID")
            return
        active_id = requested[0] if active_strut_id is None else str(active_strut_id)
        if active_id not in requested:
            self._set_selection_status("Active strut must be one of the selected IDs")
            return

        self.selected_ids = requested
        self._selection_state_update = True
        try:
            # Broker events arrive outside a browser UI callback.  Batch and
            # flush their state changes so Trame immediately updates both the
            # ID input and the active-ID selector in connected clients.
            with self.state:
                self.state.strut_ids_input = ", ".join(requested)
                self.state.selected_strut_options = [
                    {"title": strut_id, "value": strut_id} for strut_id in requested
                ]
                self.state.selected_strut_id = active_id
            self.state.flush()
        finally:
            self._selection_state_update = False
        self._set_selection_status(f"{len(requested)} strut(s) selected")
        self._update_selection(active_id, publish=publish)

    def _clear_selection(self, *_args: Any, **_kwargs: Any) -> None:
        self.selected_ids = []
        self.state.strut_ids_input = ""
        self.state.selected_strut_options = []
        self.state.selected_strut_id = None
        self._set_selection_status("No struts selected")
        self.macro_all_actor.mapper.SetInputData(self.all_lines)
        self._set_visibility()
        self._render()

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

    def _set_micro_volume(self, grid: pv.ImageData) -> None:
        """Update the existing bounded CT actor without replacing it mid-render."""
        if not getattr(self, "micro_ct_actors", []):
            self.micro_ct_actors = self._add_volume(self.micro_plotter, grid)
            return
        for actor in self.micro_ct_actors:
            actor.mapper.SetInputData(grid)

    def _publish_active_strut(self, strut_id: str) -> None:
        try:
            requests.post(
                f"{self.api_url}/select_active_strut",
                json={"strut_id": int(strut_id)},
                timeout=3,
            ).raise_for_status()
        except requests.RequestException as exc:
            print(f"Could not publish active strut: {exc}", flush=True)

    def _update_selection(self, strut_id: str, *, publish: bool) -> None:
        self.selected_id = strut_id
        index = self.strut_index[strut_id]
        colors = list(self.base_colors)
        for selected_index in (self.strut_index[item] for item in self.selected_ids):
            colors[selected_index] = TRANSPARENT_RGBA
        macro_lines = make_line_mesh(self.vectors_zyx, colors)
        macro_lines.cell_data["strut_index"] = np.arange(len(self.strut_ids), dtype=np.int64)
        self.macro_all_actor.mapper.SetInputData(macro_lines)
        selected_indices = [self.strut_index[item] for item in self.selected_ids]
        self.macro_selected_actor.mapper.SetInputData(make_line_mesh(self.vectors_zyx[selected_indices]))

        indices, _unit_cell = self._context_for(index)
        lower, upper = self._context_bounds(indices)
        crop = np.asarray(self.volume[lower[0]:upper[0], lower[1]:upper[1], lower[2]:upper[2]])
        self._set_micro_volume(image_grid(crop, self.args.micro_downsample, lower))
        context_colors = [TRANSPARENT_RGBA if item == index else unit_cell_context_color(self.table.iloc[item]["inventory_classification"]) for item in indices]
        self.micro_context_actor.mapper.SetInputData(make_line_mesh(self.vectors_zyx[indices], context_colors))
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
                requests.post(
                    f"{self.api_url}/select_struts",
                    json={
                        "strut_ids": [int(item) for item in self.selected_ids],
                        "active_strut_id": int(strut_id),
                    },
                    timeout=3,
                ).raise_for_status()
            except requests.RequestException as exc:
                print(f"Could not publish selected strut: {exc}", flush=True)

    def _set_visibility(self) -> None:
        # Macro pane visibility
        for actor in self.macro_ct_actors:
            actor.SetVisibility(self.state.macro_ct_visible)
        self.macro_all_actor.SetVisibility(self.state.macro_lines_visible)
        
        # Check if macro_active_line_visible exists in state; default to True if not set
        macro_active = getattr(self.state, "macro_active_line_visible", True) and bool(self.selected_ids)
        self.macro_selected_actor.SetVisibility(macro_active)

        # Micro pane visibility
        for actor in getattr(self, "micro_ct_actors", []):
            actor.SetVisibility(self.state.micro_ct_visible and bool(self.selected_ids))
        self.micro_context_actor.SetVisibility(self.state.micro_lines_visible and bool(self.selected_ids))
        
        # Check if micro_active_line_visible exists in state; default to True if not set
        micro_active = getattr(self.state, "micro_active_line_visible", True) and bool(self.selected_ids)
        self.micro_selected_actor.SetVisibility(micro_active)

    def _render(self) -> None:
        self.macro_plotter.render()
        self.micro_plotter.render()
        if hasattr(self, "macro_view"):
            self.macro_view.update()
            self.micro_view.update()

    def _register_broker_selection_consumer(self) -> None:
        @self.server.controller.add_task("on_server_ready")
        async def consume_broker_selections(**_kwargs: Any) -> None:
            while True:
                selected_ids, active_id = await asyncio.to_thread(self._broker_selection_queue.get)
                self._apply_ids(selected_ids, publish=False, active_strut_id=active_id)

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
                        data = event.get("data", {})
                        selected_ids = data.get("active_strut_ids") or ([data.get("active_strut_id")] if data.get("active_strut_id") is not None else [])
                        active_id = data.get("active_strut_id")
                    elif event.get("event_type") == "STRUTS_SELECTED":
                        data = event.get("data", {})
                        selected_ids = data.get("strut_ids") or []
                        active_id = data.get("active_strut_id")
                    elif event.get("event_type") == "STRUT_ACTIVE_CHANGED":
                        data = event.get("data", {})
                        selected_ids = data.get("strut_ids") or []
                        active_id = data.get("strut_id")
                    else:
                        continue
                    selected_ids = [str(value) for value in selected_ids]
                    active_id = str(active_id) if active_id is not None else None
                    if (
                        selected_ids
                        and all(value in self.strut_index for value in selected_ids)
                        and (active_id is None or active_id in selected_ids)
                    ):
                        self._broker_selection_queue.put((selected_ids, active_id))
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
