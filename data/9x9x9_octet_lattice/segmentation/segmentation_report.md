# Segmentation report: 9x9x9 octet lattice

## Inputs and method

- Input: `9x9x9_octet_lattice.tif`
- Volume shape (Z, Y, X): `761 × 815 × 837`
- Input dtype: `uint16`; pages are read one at a time
- Output mask: `segmented_mask.tif` (`uint8`, binary, compressed TIFF)
- Slice visualization: `slice_380_visualization.png` (slice index 380)
- Optimization history: `optimization_history.csv`
- Per-slice threshold parameter selected: intensity percentile **95**
- Tiny-component cleanup: 8-connected components smaller than **4** pixels removed per 2-D page
- Sparse foreground prior used for optimization: **5.00%** median foreground fraction

The sampled histogram used 17 interior slices. Its intensity range was
`26730`–`61497`, with mean page intensity
`33531.65` and representative global Otsu threshold
`37850`. A global threshold was not used because page intensity
levels vary strongly near the stack ends; each page therefore receives its own percentile
threshold while remaining memory-safe.

## Bounded optimization

The search stopped after **7** iterations (maximum 10); it stops
after 3 consecutive non-improving candidates. The selected candidate
maximized a score combining sparse-lattice density and retention of non-tiny connected
components on sampled interior pages.

| Run | Percentile | Score | Median foreground fraction | Component retention | Improved |
|---:|---:|---:|---:|---:|:---:|
| 1 | 92 | 0.55130 | 0.08000 | 0.99882 | true |
| 2 | 93 | 0.66537 | 0.07000 | 0.99921 | true |
| 3 | 94 | 0.81187 | 0.06000 | 0.99955 | true |
| 4 | 95 | 0.99993 | 0.05000 | 0.99957 | true |
| 5 | 96 | 0.81190 | 0.04000 | 0.99947 | false |
| 6 | 97 | 0.66547 | 0.03000 | 0.99955 | false |
| 7 | 98 | 0.55136 | 0.01999 | 0.99942 | false |

## Final voxel statistics

| Quantity | Value |
|---|---:|
| Total voxels | 519,119,955 |
| Foreground voxels (mask = 1) | 25,855,146 |
| Background voxels (mask = 0) | 493,264,809 |
| Foreground fraction | 4.980573% |
| Per-slice threshold minimum | 37680.30 |
| Per-slice threshold median | 39048.30 |
| Per-slice threshold mean | 40911.30 |
| Per-slice threshold maximum | 58441.00 |

## Slice 380 inspection

- Mask foreground voxels: **34,054**
- Mask foreground fraction: **4.992121%**
- Raw intensity range: **28949–59641**
- Raw mean / 99th percentile: **33068.40 / 54449.46**

The slice-380 visualization shows a sparse, periodic set of bright lattice-member
cross-sections, with the connected diagonal/strut structure retained on the left side
and repeated node-like sections across the field. The background remains predominantly
zero in the binary view, while the intensity view confirms that the green contour follows
the high-intensity lattice signal rather than the low-intensity background. The adaptive
threshold also limits bright edge-plane artifacts that would dominate a single global
threshold.

## Validation performed

- Reopened the output TIFF after writing and checked page count, page shape, dtype, and binary value set.
- Checked slice index 380 exists in both input and output and was used for the visualization.
- Checked `foreground_voxels + background_voxels == total_voxels`.
- All processing paths use page-at-a-time reads and explicit garbage collection; no full 3-D input array is created.
