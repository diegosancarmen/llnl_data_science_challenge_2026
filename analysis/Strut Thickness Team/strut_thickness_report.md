# Strut thickness report

## Procedure

The segmented mask was loaded page-by-page, binarized, transformed with a 3-D Euclidean distance transform, skeletonized with scikit-image, and filtered to retain skeleton voxels with one or two 26-neighbors. Diameter measurements are twice the EDT radius at retained skeleton voxels.

## Input and outputs

- Input TIFF: `analysis/Strut Thickness/Step 0 Segmentation/segmentation/segmented_mask.tif`
- Loaded shape: `[761, 815, 837]`
- Binary foreground voxels: `93477525`
- Measurements: `853,058`
- CSV: `analysis/Strut Thickness/Step 5 Diameter/diameter_measurements.csv`
- Histogram: `analysis/Strut Thickness/Step 7 Plot/diameter_histogram.png`

## Statistics

- Median diameter: **7.483 voxels** (434.71 µm at 58.09 µm/voxel).
- Mean ± SD: 9.584 ± 9.093 voxels.
- Nominal 350 µm diameter at 58.09 µm/voxel: 6.025 voxels.
- Implied pitch from 350 µm / median: **46.771 µm/voxel**; divergence from 58.09: **-19.49%**.

## Potential issues

- EDT thickness assumes a circular cross-section and a well-segmented mask; non-circular or oblique sections can bias the diameter.
- Skeletonization can create breaks, spurs, or merged junction branches. Junction voxels were removed using the 26-neighbor rule, but nearby artifacts can remain.
- The implied voxel pitch is a consistency check, not a replacement for physical calibration. Threshold sensitivity and TIFF axis/orientation metadata should be reviewed if the divergence is large.
