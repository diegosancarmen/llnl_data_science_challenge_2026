# PyVista tail visualization

This folder contains the minimal shareable inputs for visualizing the large-EDT-diameter measurements:

- `tail_measurements.csv`: `(x, y, z)` locations and diameter values for the upper 1% tail.
- `pyvista_tail_visualization.py`: standalone PyVista renderer.
- `tail_locations_pyvista.html`: interactive export, when generated.
- `tail_locations_pyvista.png`: static screenshot, when generated.

Install PyVista in the target environment, then run:

```bash
python pyvista_tail_visualization.py tail_measurements.csv
```

The points are colored by estimated diameter in voxels. Coordinates use the TIFF volume convention: `z` is the page/slice index.
