# Napari centerline data preparation

## Output

[`napari_centerlines.csv`](napari_centerlines.csv) contains one row for each of
the 18,468 candidate-registered struts. It is a new derived file; neither the
candidate JSON nor `all_struts_inventory.csv` is modified.

The table uses the candidate affine registration from
[`../../candidate_affine_registered.json`](../../candidate_affine_registered.json)
and the regenerated adaptive Stage 2a inventory from
[`../all_struts_inventory.csv`](../all_struts_inventory.csv). The source CT
scan is opened only to read its shape `(Z, Y, X) = (761, 815, 837)`; no CT
voxels are loaded or written during preparation.

## Coordinate convention

The registration JSON stores positions as `XYZ` source voxels. Every geometry
column intended for Napari is precomputed as `ZYX`, matching NumPy/TIFF and
Napari array coordinates. The scan is isotropic at `58.09 µm/voxel`.

The output includes both source-voxel coordinates and micrometer coordinates:

- Use the `*_vox` columns with an image layer in source-voxel coordinates.
- Use the `*_um` columns if the Napari image layer is added with
  `scale=(58.09, 58.09, 58.09)`.

## Important columns

| Columns | Purpose |
|---|---|
| `start_z_vox` … `start_x_vox` | Napari vector origin / line start in ZYX |
| `direction_z_vox` … `direction_x_vox` | Napari vector direction in ZYX (`end - start`) |
| `end_z_vox` … `end_x_vox` | Shapes-line endpoint in ZYX |
| `napari_vector_zyx_json` | Preassembled `[[origin_zyx], [direction_zyx]]` vector record |
| `napari_line_zyx_json` | Preassembled `[[start_zyx], [end_zyx]]` line record |
| `center_*`, `length_vox`, `length_um_from_centerline` | Fast picking, labels, and measurements |
| `bbox_*` | A 10-voxel-margin, scan-clipped ZYX crop for focus/inspection |
| `napari_color_hex`, `napari_color_rgba_json`, `napari_layer` | Precomputed class styling and nominal/defect layer routing |
| `inventory_*` | All fields from the regenerated Stage 2a inventory, including classification, defect metrics, unit-cell index, and original tube occupancy |

## Minimal Napari loading example

```python
import numpy as np
import pandas as pd
import tifffile
import napari

root = "registration/alignment_check/stage2a_candidate_registration_output"
table = pd.read_csv(f"{root}/napari_visualizer/napari_centerlines.csv")
scan = tifffile.memmap(
    "data/missing_struts/tif_stacks/"
    "210127_Brian_Tran_strut_lattices_0point5dash1 1 Slices.tif"
)

viewer = napari.Viewer(ndisplay=3)
viewer.add_image(scan, name="CT", scale=(1, 1, 1))

origin = table[["start_z_vox", "start_y_vox", "start_x_vox"]].to_numpy()
direction = table[["direction_z_vox", "direction_y_vox", "direction_x_vox"]].to_numpy()
vectors = np.stack((origin, direction), axis=1)  # (N, 2, 3), already ZYX

viewer.add_vectors(
    vectors,
    name="registered strut centerlines",
    edge_color=table["napari_color_hex"].tolist(),
    properties=table,
)

# Optional Shapes-line representation instead of Vectors:
end = table[["end_z_vox", "end_y_vox", "end_x_vox"]].to_numpy()
lines = np.stack((origin, end), axis=1)  # (N, 2, 3), already ZYX
# viewer.add_shapes(lines, shape_type="line", edge_color=table["napari_color_hex"].tolist(), properties=table)
```

For a physically scaled view, multiply the six coordinate columns by `58.09`
or use the precomputed `*_um` columns and set the CT image scale to
`(58.09, 58.09, 58.09)`.

## Reproducibility

- [Preparation script](prepare_napari_centerlines.py)
- [Preparation summary](napari_centerlines_summary.json)
- [Candidate registration provenance](../candidate_registration_rerun_provenance.json)

Run the script from `part2` with:

```bash
/home/hannahdc/anaconda3/bin/python \
  registration/alignment_check/stage2a_candidate_registration_output/napari_visualizer/prepare_napari_centerlines.py
```
