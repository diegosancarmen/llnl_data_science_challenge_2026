#!/usr/bin/env python3
"""Map registered strut IDs into a TIFF volume and render their CT region.

The registration files used by this project store positions as ``[x, y, z]``
source-voxel coordinates.  TIFF volumes are indexed ``volume[z, y, x]``.
Keeping that conversion in one module prevents the common x/z swap when
working with the registered graph.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any

# Keep Matplotlib's cache inside a writable temporary location on shared
# compute environments.
os.environ.setdefault("MPLCONFIGDIR", "/tmp/stage3_visualization-mpl")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tifffile


def _load_registration(path: Path) -> dict[str, Any]:
    with path.open() as handle:
        registration = json.load(handle)
    if not isinstance(registration.get("junctions"), list) or not isinstance(registration.get("struts"), list):
        raise ValueError("Registration JSON must contain 'junctions' and 'struts' lists")
    return registration


def _strut_geometry(registration: dict[str, Any], strut_id: int) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    matches = [s for s in registration["struts"] if int(s["id"]) == strut_id]
    if not matches:
        raise KeyError(f"Strut ID {strut_id} is not present in the registration JSON")
    strut = matches[0]
    junctions = {int(j["id"]): np.asarray(j["position"], dtype=float) for j in registration["junctions"]}
    try:
        p0 = junctions[int(strut["junction0"])]
        p1 = junctions[int(strut["junction1"])]
    except KeyError as exc:
        raise ValueError(f"Strut {strut_id} references a missing junction") from exc
    if p0.shape != (3,) or p1.shape != (3,):
        raise ValueError("Junction positions must be 3-element XYZ coordinates")
    return p0, p1, strut


def _read_csv_row(csv_path: Path | None, strut_id: int) -> dict[str, str] | None:
    if csv_path is None:
        return None
    with csv_path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if int(row["strut_id"]) == strut_id:
                return row
    return None


def extract_strut(
    scan_path: str | Path,
    registration_path: str | Path,
    strut_id: int,
    margin_voxels: float = 24.0,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Extract a registration-aware crop around one strut.

    Returns ``(crop, metadata)``. ``crop`` is indexed ZYX, and metadata
    includes half-open source-volume bounds in both XYZ and ZYX order.
    ``tifffile.memmap`` means the full CT volume is not loaded into RAM.
    """
    if margin_voxels < 0:
        raise ValueError("margin_voxels must be non-negative")
    scan_path, registration_path = Path(scan_path), Path(registration_path)
    scan = tifffile.memmap(scan_path)
    if scan.ndim != 3:
        raise ValueError(f"Expected a 3-D TIFF, got shape {scan.shape}")
    registration = _load_registration(registration_path)
    p0, p1, strut = _strut_geometry(registration, int(strut_id))

    xyz_lo = np.floor(np.minimum(p0, p1) - margin_voxels).astype(int)
    xyz_hi = np.ceil(np.maximum(p0, p1) + margin_voxels).astype(int) + 1
    shape_zyx = np.asarray(scan.shape, dtype=int)
    xyz_shape = shape_zyx[::-1]
    clipped_lo = np.maximum(xyz_lo, 0)
    clipped_hi = np.minimum(xyz_hi, xyz_shape)
    if np.any(clipped_lo >= clipped_hi):
        raise ValueError(f"Strut {strut_id} does not intersect the TIFF volume")

    # NumPy/TIFF indexing is ZYX, whereas registration geometry is XYZ.
    zyx_lo, zyx_hi = clipped_lo[::-1], clipped_hi[::-1]
    crop = np.asarray(scan[zyx_lo[0]:zyx_hi[0], zyx_lo[1]:zyx_hi[1], zyx_lo[2]:zyx_hi[2]])
    metadata: dict[str, Any] = {
        "strut_id": int(strut_id),
        "junction0_id": int(strut["junction0"]),
        "junction1_id": int(strut["junction1"]),
        "endpoint0_xyz_source_voxels": p0.tolist(),
        "endpoint1_xyz_source_voxels": p1.tolist(),
        "requested_bounds_xyz_source_voxels_half_open": [xyz_lo.tolist(), xyz_hi.tolist()],
        "clipped_bounds_xyz_source_voxels_half_open": [clipped_lo.tolist(), clipped_hi.tolist()],
        "clipped_bounds_zyx_tiff_indices_half_open": [zyx_lo.tolist(), zyx_hi.tolist()],
        "crop_shape_zyx": list(crop.shape),
        "scan_shape_zyx": list(scan.shape),
        "coordinate_convention": "registration XYZ source voxels; TIFF/NumPy ZYX",
    }
    metadata.update({k: v for k, v in strut.items() if k not in {"id", "junction0", "junction1"}})
    return crop, metadata


def _render(crop: np.ndarray, metadata: dict[str, Any], output: Path, threshold: float, downsample: int) -> None:
    mask = crop >= threshold
    step = max(1, int(downsample))
    display = mask[::step, ::step, ::step]
    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(111, projection="3d")
    if display.any():
        ax.voxels(display, facecolors="#e45756", edgecolor="none", alpha=0.85)
    else:
        ax.text2D(0.05, 0.95, "No voxels above threshold", transform=ax.transAxes)
    # Plot endpoints in local XYZ, converted to the plot's (x, y, z) axes.
    origin_xyz = np.asarray(metadata["clipped_bounds_xyz_source_voxels_half_open"][0], dtype=float)
    points = (np.asarray([metadata["endpoint0_xyz_source_voxels"], metadata["endpoint1_xyz_source_voxels"]]) - origin_xyz) / step
    ax.plot(points[:, 0], points[:, 1], points[:, 2], color="#1f77b4", linewidth=2.5, label="registered centerline")
    ax.set(xlabel="X crop voxels", ylabel="Y crop voxels", zlabel="Z crop voxels", title=f"Registered strut {metadata['strut_id']} (threshold {threshold:g})")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def visualize_strut(
    scan_path: str | Path,
    registration_path: str | Path,
    strut_id: int,
    output_dir: str | Path,
    csv_path: str | Path | None = None,
    margin_voxels: float = 24.0,
    threshold: float = 40000.0,
    downsample: int = 2,
) -> dict[str, Any]:
    """Extract, save, and render one registered strut region."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    crop, metadata = extract_strut(scan_path, registration_path, int(strut_id), margin_voxels)
    row = _read_csv_row(Path(csv_path) if csv_path else None, int(strut_id))
    if row:
        metadata["csv_row"] = row
    stem = f"strut_{int(strut_id)}"
    crop_path = output_dir / f"{stem}_ct_crop.tif"
    mask_path = output_dir / f"{stem}_mask.tif"
    image_path = output_dir / f"{stem}_3d.png"
    tifffile.imwrite(crop_path, crop, metadata={"axes": "ZYX", "strut_id": int(strut_id)})
    tifffile.imwrite(mask_path, (crop >= threshold).astype(np.uint8), metadata={"axes": "ZYX", "threshold": threshold})
    _render(crop, metadata, image_path, threshold, downsample)
    metadata_path = output_dir / f"{stem}_metadata.json"
    metadata.update({"threshold": threshold, "output_files": {"ct_crop": str(crop_path), "mask": str(mask_path), "render_3d": str(image_path), "metadata": str(metadata_path)}})
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scan", required=True, help="3-D TIFF CT scan")
    parser.add_argument("--registration", required=True, help="registration JSON")
    parser.add_argument("--strut-id", required=True, type=int)
    parser.add_argument("--output-dir", default="stage3_visualization/output")
    parser.add_argument("--csv", help="optional inventory CSV to attach to metadata")
    parser.add_argument(
        "--margin-voxels",
        type=float,
        default=24.0,
        help="border around the registered strut endpoints in source voxels (default: 24)",
    )
    parser.add_argument("--threshold", type=float, default=40000.0, help="CT intensity threshold for the mask/render")
    parser.add_argument("--downsample", type=int, default=2, help="render-only voxel downsampling")
    args = parser.parse_args()
    result = visualize_strut(
        scan_path=args.scan,
        registration_path=args.registration,
        strut_id=args.strut_id,
        output_dir=args.output_dir,
        csv_path=args.csv,
        margin_voxels=args.margin_voxels,
        threshold=args.threshold,
        downsample=args.downsample,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
