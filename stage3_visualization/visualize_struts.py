#!/usr/bin/env python3
"""Interactive Napari viewer for the registered strut graph.

The project registration JSON stores junction positions as XYZ source-voxel
coordinates and struts as junction-ID pairs. Napari indexes the CT volume and
shape coordinates as ZYX, so this module performs that conversion explicitly.
Classifications are read from the Stage 2 inventory CSV because they are not
stored in the registration JSON.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import napari
import numpy as np
import tifffile
from magicgui import magicgui


CLASSIFICATION_COLORS = {
    "Missing_Unintentional": "red",
    "Missing_Intentional": "blue",
    "Expected_Missing_But_Material_Present": "orange",
    "Nominal": "green",
    "Unclassified": "gray",
}


def _load_registration(path: Path) -> dict[str, Any]:
    with path.open() as handle:
        registration = json.load(handle)
    if not isinstance(registration.get("junctions"), list) or not isinstance(registration.get("struts"), list):
        raise ValueError("Registration JSON must contain 'junctions' and 'struts' lists")
    return registration


def _load_classifications(path: Path | None) -> dict[int, str]:
    if path is None:
        return {}
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or "strut_id" not in reader.fieldnames:
            raise ValueError("Classification CSV must contain a 'strut_id' column")
        return {
            int(row["strut_id"]): row.get("classification", "Unclassified") or "Unclassified"
            for row in reader
        }


def _parse_centerlines(
    registration: dict[str, Any],
    classifications: dict[int, str],
) -> tuple[list[np.ndarray], list[str], list[np.ndarray], list[str]]:
    """Return Napari ZYX lines, IDs, midpoints, and classification colors."""
    junctions = {
        int(junction["id"]): np.asarray(junction["position"], dtype=float)
        for junction in registration["junctions"]
    }
    lines: list[np.ndarray] = []
    strut_ids: list[str] = []
    centers: list[np.ndarray] = []
    colors: list[str] = []

    for strut in registration["struts"]:
        strut_id = int(strut["id"])
        try:
            p0_xyz = junctions[int(strut["junction0"])]
            p1_xyz = junctions[int(strut["junction1"])]
        except KeyError as exc:
            raise ValueError(f"Strut {strut_id} references a missing junction") from exc
        if p0_xyz.shape != (3,) or p1_xyz.shape != (3,):
            raise ValueError(f"Strut {strut_id} has a non-3D junction position")

        # Registration coordinates are XYZ; Napari data coordinates are ZYX.
        p0_zyx = p0_xyz[[2, 1, 0]]
        p1_zyx = p1_xyz[[2, 1, 0]]
        classification = classifications.get(strut_id, "Unclassified")
        lines.append(np.asarray([p0_zyx, p1_zyx]))
        strut_ids.append(str(strut_id))
        centers.append((p0_zyx + p1_zyx) / 2.0)
        colors.append(CLASSIFICATION_COLORS.get(classification, "gray"))

    return lines, strut_ids, centers, colors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scan",
        "--tiff",
        dest="scan",
        required=True,
        help="3-D TIFF CT scan (the --tiff spelling is retained as an alias)",
    )
    parser.add_argument(
        "--registration",
        "--json",
        dest="registration",
        required=True,
        help="registration JSON containing 'junctions' and 'struts'",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        help="optional inventory/defect CSV containing strut_id and classification",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=58.09,
        help="isotropic voxel size in microns for Napari display (default: 58.09)",
    )
    args = parser.parse_args()
    if args.scale <= 0:
        parser.error("--scale must be positive")

    scan_path = Path(args.scan)
    registration_path = Path(args.registration)
    print(f"Loading TIFF from {scan_path} (memory-mapped)...")
    volume = tifffile.memmap(scan_path)
    if volume.ndim != 3:
        raise ValueError(f"Expected a 3-D TIFF, got shape {volume.shape}")

    print(f"Loading registration from {registration_path}...")
    registration = _load_registration(registration_path)
    classifications = _load_classifications(args.csv)
    lines, strut_ids, centers, colors = _parse_centerlines(registration, classifications)

    viewer = napari.Viewer(ndisplay=3)
    viewer.add_image(
        volume,
        name=f"CT Scan ({scan_path.name})",
        colormap="gray",
        rendering="mip",
        depiction="volume",
        scale=(args.scale, args.scale, args.scale),
    )
    shapes_layer = viewer.add_shapes(
        lines,
        shape_type="line",
        edge_width=2,
        edge_color=colors,
        name="Registered Centerlines",
        properties={"strut_id": strut_ids},
    )

    @magicgui(strut_id={"choices": strut_ids, "label": "Select Strut:"})
    def navigate_to_strut(strut_id: str):
        index = strut_ids.index(strut_id)
        viewer.camera.center = centers[index]
        viewer.camera.zoom = 5.0
        shapes_layer.selected_data = {index}

    viewer.window.add_dock_widget(navigate_to_strut, area="right")
    print(f"Launching Napari viewer with {len(strut_ids):,} registered struts...")
    napari.run()


if __name__ == "__main__":
    main()


"""
Example:

python stage3_visualization/visualize_struts.py \
    --scan "part2/data/missing_struts/tif_stacks/210127_Brian_Tran_strut_lattices_0point5dash1 1 Slices.tif" \
    --registration "part2/data/missing_struts/registered_jsons/210127_Brian_Tran_strut_lattices_0point5dash1 1 Slices.json" \
    --csv part2/stage_2a_developer_output/all_struts_inventory.csv
"""
