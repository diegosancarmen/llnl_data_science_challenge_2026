# Spatial investigation of the diameter right tail

The saved diameter-coordinate CSV was analyzed (853,058 measurements). The upper tail is defined as measurements at or above the 99th percentile, **49.03 voxels**; the fixed extreme subset is diameter ≥20 voxels.

- Upper 1% tail: **8,611** measurements (1.01%), median 58.82 voxels, range 49.03–73.81.
- Extreme ≥20 voxels: **70,479** measurements (8.26%).
- Upper-tail centroid `(z, y, x)`: `(48.8, 314.5, 539.8)`.

The 3-D plot shows all tail points and a downsampled bulk skeleton. The projection plot is included because overlap in depth can hide clustering in a single perspective. Tail points concentrated near node regions support junction merging or EDT spheres spanning multiple struts; tail points distributed along isolated struts would be more consistent with genuine thickness variation or systematic segmentation inflation.

## Files

- `tail_locations_3d.png`: 3-D spatial visualization.
- `tail_locations_projections.png`: orthogonal projections.
- `tail_measurements.csv`: all upper-1%-tail coordinates and diameters.
- `tail_statistics.json`: machine-readable summary.
