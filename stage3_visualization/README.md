# Stage 3: registration-aware strut visualization

This module maps a `strut_id` from the registration JSON to its two registered
junctions, converts registration XYZ coordinates to TIFF ZYX indexing, and
extracts a bounded CT crop. It writes the crop, a thresholded mask, a 3-D PNG
render, and traceable coordinate metadata.

From the repository root, for the included data:

```bash
python -m stage3_visualization.strut_visualizer \
  --scan "part2/data/missing_struts/tif_stacks/210127_Brian_Tran_strut_lattices_0point5dash1 1 Slices.tif" \
  --registration "part2/data/missing_struts/registered_jsons/210127_Brian_Tran_strut_lattices_0point5dash1 1 Slices.json" \
  --strut-id 9 \
  --csv part2/stage_2a_developer_output/all_struts_inventory.csv
```

Use `--threshold` to change the CT segmentation cutoff and `--margin-voxels`
to enlarge or reduce the extracted region. The default border is 24 source
voxels (three times the original 8-voxel border), and applies to the CT crop,
binary mask, and 3-D PNG render. The default threshold is 40000 for the
included uint16 scan; it should be adjusted for scans with a different
intensity scale.
