# Missing-Strut Methodology: Graph-Edge Tubular-Occupancy Analysis with Dual-CAD Comparison

## Method name

This workflow uses **graph-edge tubular-occupancy analysis with dual-CAD
comparison**. It is an edge-wise CT metrology method: the registered lattice
graph supplies one nominal centerline per strut, a cylindrical sampling tube
measures CT material along that centerline, and a comparison of the 0% and
0.5% CAD models distinguishes designed removals from unintended ones.

It does not identify defects by globally segmenting empty CT blobs. Instead,
it treats every registered CAD strut as a graph edge between two junctions and
asks whether material is present where that particular edge should be.

## 1. Register the graph to CT

The alignment check selected an affine XY correction using held-out,
high-occupancy nominal struts. It improved material support from 0.885 to
0.930. The final extraction reruns the sampler with that candidate
registration.

## 2. Segment and sample each strut

CT intensity is thresholded with Otsu at 40,081. For each of the 18,468
edges, the pipeline samples a cylindrical tube with a radius of 6 source
voxels (348.54 micrometres) at 21 positions along the edge. It excludes the
outer 20% at both ends so that material at junctions is not mistaken for an
otherwise present strut.

## 3. Measure absence robustly

For each trimmed tube, the pipeline computes CT material occupancy,
per-station occupancy and the longest low-material run, plus estimated missing
volume and physical gap length.

A station is considered empty when its material occupancy is below 5%. A
strut must have a long empty run--at least 16 of the 21 stations--and very low
total tube occupancy before it qualifies as missing. This guards against a
local artifact, slight registration error, or a thin region being called a
missing strut.

## 4. Separate intentional and unintentional removals

The 0% and 0.5% CAD meshes are compared directly. Their facet differences are
spatially clustered, excluding large remeshed-junction artifacts, to identify
92 intended-removal regions. Those regions are mapped to graph edges using
clearance and CT-supported symmetry selection.

## 5. Assign the final label

- `Missing_Intentional`: the edge is CAD-designated as removed and passes the
  CT absence test.
- `Missing_Unintentional`: the edge is not CAD-designated but passes the same
  absence test.
- `Expected_Missing_But_Material_Present`: CAD says the strut is removed, but
  CT shows too much material.
- `Nominal`: every other strut.

## Authoritative final result

The authoritative final rerun is
`registration/alignment_check/stage2a_candidate_registration_output`:

- 87 intentional missing struts
- 331 unintentional missing struts
- 5 expected-missing-but-material-present struts
- 18,045 nominal struts

This supersedes the original developer run, which called 630 unintentional
missing struts, and the alignment sensitivity table's 337 affine estimate. The
final rerun uses the affine registration and recomputes the expected-removal
support cutoff to 0.02532.
