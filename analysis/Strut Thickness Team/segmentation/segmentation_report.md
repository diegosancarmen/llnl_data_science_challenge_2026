# TIFF lattice segmentation report

- Input: `/Users/samanthazhu/Desktop/DSC 2026/llnl_data_science_challenge_2026/data/missing_struts/tif_stacks/210127_Brian_Tran_strut_lattices_0point5dash1 1 Slices.tif`
- Output mask: `/Users/samanthazhu/Desktop/DSC 2026/llnl_data_science_challenge_2026/analysis/Strut Thickness/Step 0 Segmentation/segmentation/segmented_mask.tif`
- Shape: `(761, 815, 837)`
- Output dtype: `uint8` (binary values 0 and 1)
- Source dtype/range/mean: `uint16`, 0.0 to 65535.0, mean 34296.1
- Final threshold: `35439`
- Method: global intensity threshold selected from exact uint16 histogram; Otsu initialization plus bounded candidate sweep, scored on slice 380 by plausible foreground fraction, 2-D connectivity, and component noise
- Foreground voxels: `93477525`
- Background voxels: `425642430`
- Foreground fraction: `18.006922%`
- Iterations run: `7`
- Stop reason: Stopped after candidate sweep; selected highest-quality slice-380 score within the maximum of 10 iterations.

The histogram and per-iteration slice-380 previews are saved alongside this report. The selected mask has a connected-strut appearance and a plausible foreground fraction under the best candidate score; no source file was modified.
