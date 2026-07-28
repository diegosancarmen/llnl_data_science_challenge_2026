#!/usr/bin/env python3
"""Interactive PyVista visualization of EDT diameter tail locations.

Usage:
    python pyvista_tail_visualization.py tail_measurements.csv
"""
from pathlib import Path
import argparse

import numpy as np
import pyvista as pv


def build_visualization(csv_path: Path, output_dir: Path) -> None:
    data = np.genfromtxt(csv_path, delimiter=",", names=True)
    points = np.column_stack((data["x"], data["y"], data["z"]))
    diam = data["diameter_voxels"]

    cloud = pv.PolyData(points)
    cloud["diameter_voxels"] = diam

    plotter = pv.Plotter(window_size=(1400, 900))
    plotter.set_background("white")
    plotter.add_points(
        cloud,
        scalars="diameter_voxels",
        cmap="magma",
        point_size=5,
        render_points_as_spheres=True,
        opacity=0.85,
        scalar_bar_args={"title": "Diameter (voxels)"},
    )
    plotter.add_axes(line_width=2)
    plotter.show_grid(color="lightgray", xlabel="X voxel", ylabel="Y voxel", zlabel="Z slice")
    plotter.add_text(
        f"Upper-tail EDT measurements: {len(points):,} points\n"
        f"Diameter range: {diam.min():.2f}–{diam.max():.2f} voxels",
        position="upper_left",
        font_size=12,
        color="black",
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    plotter.export_html(str(output_dir / "tail_locations_pyvista.html"))
    plotter.screenshot(str(output_dir / "tail_locations_pyvista.png"), transparent_background=False)
    plotter.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", type=Path, help="CSV containing z,y,x,diameter_voxels columns")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    build_visualization(args.csv, args.output_dir)
    print(f"Wrote PyVista outputs to {args.output_dir}")
