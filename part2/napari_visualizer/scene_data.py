"""Shared, renderer-independent data helpers for the strut visualizers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pyvista as pv


TRANSPARENT_RGBA = (0, 0, 0, 0)

# PyVista 0.48+ rejects the intentionally empty placeholder meshes used by
# both the desktop and browser visualizers until a strut is selected.
pv.global_theme.allow_empty_mesh = True


def parse_strut_ids(value: str) -> list[str]:
    """Return unique comma- or whitespace-separated strut IDs in entry order."""
    return list(dict.fromkeys(str(value or "").replace(",", " ").split()))


def classification_color(value: object) -> str:
    """Return the established full-lattice color for a classification."""
    if pd.isna(value):
        return "magenta"
    label = str(value).lower()
    if "missing" in label:
        return "red"
    if "partial" in label:
        return "orange"
    if "nominal" in label:
        return "cyan"
    return "yellow"


def unit_cell_context_color(value: object) -> str:
    """Color unit-cell context struts as nominal (blue) or defective (red)."""
    return "blue" if not pd.isna(value) and "nominal" in str(value).lower() else "red"


def zyx_to_xyz(points_zyx: np.ndarray) -> np.ndarray:
    """Convert source TIFF coordinates to PyVista's XYZ world coordinates."""
    return np.ascontiguousarray(points_zyx[:, ::-1], dtype=float)


def make_line_mesh(endpoints_zyx: np.ndarray, colors: list[object] | None = None) -> pv.PolyData:
    """Create one pickable line cell per strut, retaining its input row ID."""
    endpoints_xyz = zyx_to_xyz(endpoints_zyx.reshape(-1, 3))
    line_count = len(endpoints_zyx)
    mesh = pv.PolyData(endpoints_xyz)
    mesh.verts = np.empty(0, dtype=np.int64)
    mesh.lines = np.column_stack(
        (np.full(line_count, 2, dtype=np.int64), np.arange(2 * line_count).reshape(line_count, 2))
    ).ravel()
    if colors is not None:
        mesh.cell_data["rgba"] = np.asarray([pv.Color(color).int_rgba for color in colors], dtype=np.uint8)
    return mesh


def placeholder_line_mesh(color: object | None = None) -> pv.PolyData:
    """Return a valid line mesh for hidden, not-yet-populated VTK actors.

    VTK remote rendering cannot safely bind a completely empty ``PolyData`` to
    an OpenGL mapper.  Callers should hide the resulting actor until it holds
    meaningful geometry.
    """
    colors = [color] if color is not None else None
    return make_line_mesh(np.array([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]]), colors)


def image_grid(volume_zyx: np.ndarray, downsample: int, origin_zyx: np.ndarray) -> pv.ImageData:
    """Create source-aligned PyVista image data from an explicitly downsampled TIFF array."""
    sampled = np.ascontiguousarray(volume_zyx[::downsample, ::downsample, ::downsample])
    grid = pv.ImageData()
    grid.dimensions = tuple(np.asarray(sampled.shape[::-1], dtype=int))
    grid.spacing = (downsample, downsample, downsample)
    grid.origin = tuple(np.asarray(origin_zyx[::-1], dtype=float))
    grid.point_data["ct_intensity"] = np.ascontiguousarray(sampled.transpose(2, 1, 0)).ravel(order="F")
    return grid
