#!/usr/bin/env python3
"""Spatial investigation of unusually large EDT-derived diameters."""
from pathlib import Path
import csv
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
CSV_PATH = ROOT / "Step 5 Diameter" / "diameter_measurements.csv"
OUT = ROOT / "Step 8 Tail Investigation"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    data = np.genfromtxt(CSV_PATH, delimiter=",", names=True)
    z, y, x = data["z"], data["y"], data["x"]
    d = data["diameter_voxels"]
    p95, p99, p999 = np.percentile(d, [95, 99, 99.9])
    tail = d >= p99
    extreme = d >= 20.0

    def bounds(mask):
        return {
            "count": int(mask.sum()),
            "fraction_percent": float(100 * mask.mean()),
            "diameter_min": float(d[mask].min()) if mask.any() else None,
            "diameter_median": float(np.median(d[mask])) if mask.any() else None,
            "diameter_max": float(d[mask].max()) if mask.any() else None,
            "z_min_max": [int(z[mask].min()), int(z[mask].max())] if mask.any() else None,
            "y_min_max": [int(y[mask].min()), int(y[mask].max())] if mask.any() else None,
            "x_min_max": [int(x[mask].min()), int(x[mask].max())] if mask.any() else None,
            "centroid_zyx": [float(np.mean(v[mask])) for v in (z, y, x)] if mask.any() else None,
        }

    summary = {
        "source_csv": str(CSV_PATH),
        "n_measurements": int(d.size),
        "thresholds_voxels": {"p95": float(p95), "p99": float(p99), "p99_9": float(p999), "extreme_fixed": 20.0},
        "all_measurements": bounds(np.ones(d.size, dtype=bool)),
        "upper_1_percent_tail": bounds(tail),
        "extreme_ge_20_voxels": bounds(extreme),
    }
    (OUT / "tail_statistics.json").write_text(json.dumps(summary, indent=2) + "\n")

    with (OUT / "tail_measurements.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["z", "y", "x", "diameter_voxels", "diameter_um_at_58_09"])
        for zz, yy, xx, dd in zip(z[tail], y[tail], x[tail], d[tail]):
            writer.writerow([int(zz), int(yy), int(xx), float(dd), float(dd * 58.09)])

    rng = np.random.default_rng(20260727)
    bulk_idx = np.flatnonzero(~tail)
    # Keep the visualization readable while retaining every tail point.
    bulk_idx = rng.choice(bulk_idx, size=min(50000, bulk_idx.size), replace=False)
    tail_idx = np.flatnonzero(tail)

    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(x[bulk_idx], y[bulk_idx], z[bulk_idx], s=1, alpha=0.08, c="#65727e", linewidths=0, label="Bulk skeleton sample")
    sc = ax.scatter(x[tail_idx], y[tail_idx], z[tail_idx], s=4, c=d[tail_idx], cmap="magma", alpha=0.8, linewidths=0, label=f"Upper 1% (≥ {p99:.2f} vox)")
    ax.set_xlabel("X voxel"); ax.set_ylabel("Y voxel"); ax.set_zlabel("Z slice")
    ax.set_title("Spatial localization of EDT diameter right-tail measurements")
    ax.legend(loc="upper right")
    cb = fig.colorbar(sc, ax=ax, pad=0.08, shrink=0.7)
    cb.set_label("Diameter (voxels)")
    ax.view_init(elev=24, azim=-58)
    fig.tight_layout()
    fig.savefig(OUT / "tail_locations_3d.png", dpi=180)
    plt.close(fig)

    # Orthogonal projections make junction clustering easier to verify than 3-D depth alone.
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    projections = [(x, y, "X", "Y"), (x, z, "X", "Z"), (y, z, "Y", "Z")]
    for ax, (a, b, la, lb) in zip(axes, projections):
        ax.scatter(a[bulk_idx], b[bulk_idx], s=1, alpha=0.04, c="0.45", linewidths=0)
        ax.scatter(a[tail_idx], b[tail_idx], s=2, alpha=0.35, c=d[tail_idx], cmap="magma", linewidths=0)
        ax.set_xlabel(f"{la} voxel"); ax.set_ylabel(f"{lb} voxel")
        ax.set_title(f"{la}-{lb} projection")
        ax.set_aspect("equal")
    fig.suptitle("Upper-tail measurements (colored) versus bulk skeleton (gray)")
    fig.tight_layout()
    fig.savefig(OUT / "tail_locations_projections.png", dpi=180)
    plt.close(fig)

    report = OUT / "tail_investigation.md"
    report.write_text(
        "# Spatial investigation of the diameter right tail\n\n"
        f"The saved diameter-coordinate CSV was analyzed ({d.size:,} measurements). The upper tail is defined as measurements at or above the 99th percentile, **{p99:.2f} voxels**; the fixed extreme subset is diameter ≥20 voxels.\n\n"
        f"- Upper 1% tail: **{tail.sum():,}** measurements ({100 * tail.mean():.2f}%), median {np.median(d[tail]):.2f} voxels, range {d[tail].min():.2f}–{d[tail].max():.2f}.\n"
        f"- Extreme ≥20 voxels: **{extreme.sum():,}** measurements ({100 * extreme.mean():.2f}%).\n"
        f"- Upper-tail centroid `(z, y, x)`: `{tuple(round(float(np.mean(v[tail])), 1) for v in (z, y, x))}`.\n\n"
        "The 3-D plot shows all tail points and a downsampled bulk skeleton. The projection plot is included because overlap in depth can hide clustering in a single perspective. Tail points concentrated near node regions support junction merging or EDT spheres spanning multiple struts; tail points distributed along isolated struts would be more consistent with genuine thickness variation or systematic segmentation inflation.\n\n"
        "## Files\n\n"
        "- `tail_locations_3d.png`: 3-D spatial visualization.\n- `tail_locations_projections.png`: orthogonal projections.\n- `tail_measurements.csv`: all upper-1%-tail coordinates and diameters.\n- `tail_statistics.json`: machine-readable summary.\n"
    )
    print(f"Upper 1% tail threshold: {p99:.3f} vox; count: {tail.sum():,}")
    print(f"Extreme >=20 voxels: {extreme.sum():,}")
    print(f"Outputs: {OUT}")


if __name__ == "__main__":
    main()
