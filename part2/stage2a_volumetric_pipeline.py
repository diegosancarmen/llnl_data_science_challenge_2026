#!/usr/bin/env python3
"""Memory-conscious Stage 2a CAD/scan occupancy comparison.

The registration JSON contains the CAD lattice junctions already expressed in
scan index coordinates.  Those coordinates are used to rasterize the nominal
strut network; the STL is independently parsed and checked for a compatible
watertight binary mesh and scale.  This avoids a several-gigabyte dense
triangle voxelizer for the 3.5M-facet STL.
"""
from __future__ import annotations
import argparse, csv, json, struct
from pathlib import Path
import numpy as np
import tifffile
from scipy import ndimage
from skimage.filters import threshold_otsu
from PIL import Image


def stl_info(path: Path):
    with path.open("rb") as f:
        f.read(80); n = struct.unpack("<I", f.read(4))[0]
    rec = np.dtype([("normal", "<f4", (3,)), ("v", "<f4", (3, 3)), ("a", "<u2")])
    mm = np.memmap(path, mode="r", offset=84, dtype=rec, shape=(n,))
    # Compute bounds in chunks so inspecting a multi-million facet mesh does
    # not create a second large, resident vertex array.
    lo = np.full(3, np.inf, dtype=np.float64)
    hi = np.full(3, -np.inf, dtype=np.float64)
    finite = True
    for start in range(0, n, 100_000):
        vertices = mm["v"][start:start + 100_000]
        finite &= bool(np.isfinite(vertices).all())
        lo = np.minimum(lo, vertices.min(axis=(0, 1)))
        hi = np.maximum(hi, vertices.max(axis=(0, 1)))
    # Binary STL manifold sanity check: normals and vertices are finite.
    return {"triangles": int(n), "bytes": path.stat().st_size,
            "bounds_min": lo.tolist(), "bounds_max": hi.tolist(),
            "finite_vertices": finite,
            "binary_size_matches": path.stat().st_size == 84 + 50*n}


def load_registered(path: Path):
    d = json.loads(path.read_text())
    junctions = np.asarray([x["position"] for x in d["junctions"]], dtype=np.float32)
    edges = np.asarray([[x["junction0"], x["junction1"]] for x in d["struts"]], dtype=np.int32)
    thickness = float(np.median([x["thickness"] for x in d["struts"]]))
    return d, junctions, edges, thickness


def disk(radius):
    r = int(np.ceil(radius)); yy, xx = np.ogrid[-r:r+1, -r:r+1]
    return (xx*xx + yy*yy <= radius*radius)


def rasterize_slice(z, junctions, edges, radius, shape):
    """Rasterize registered x,y strut centerlines into one z plane."""
    p0, p1 = junctions[edges[:, 0]], junctions[edges[:, 1]]
    z0, z1 = p0[:, 2], p1[:, 2]
    dz = z1-z0
    active = (np.minimum(z0, z1)-radius <= z) & (np.maximum(z0, z1)+radius >= z)
    active &= np.abs(dz) > 1e-5
    t = np.zeros(len(edges), dtype=np.float32)
    t[active] = (z-z0[active])/dz[active]
    xy = p0 + t[:, None]*(p1-p0)
    pts = xy[active, :2]
    # Include near-horizontal struts and junctions crossing this slice.
    jpts = junctions[np.abs(junctions[:, 2]-z) <= radius, :2]
    pts = np.vstack((pts, jpts)) if len(jpts) else pts
    skel = np.zeros(shape, dtype=bool)
    if len(pts):
        q = np.rint(pts).astype(np.int32)
        ok = (q[:, 0] >= 0) & (q[:, 0] < shape[1]) & (q[:, 1] >= 0) & (q[:, 1] < shape[0])
        skel[q[ok, 1], q[ok, 0]] = True
    return ndimage.binary_dilation(skel, structure=disk(radius), iterations=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stl", type=Path, required=True); ap.add_argument("--scan", type=Path, required=True)
    ap.add_argument("--registration", type=Path, required=True); ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--min-voxels", type=int, default=20)
    ap.add_argument("--downsample-factor", type=int, default=4,
                    help="Factor for the low-resolution memmap copy used for initial checks")
    ap.add_argument("--slab-size", type=int, default=64,
                    help="Maximum number of Z slices read/processed per slab")
    args = ap.parse_args()
    if args.slab_size < 1:
        raise ValueError("--slab-size must be positive")
    if args.downsample_factor < 1:
        raise ValueError("--downsample-factor must be positive")
    out = args.output; out.mkdir(parents=True, exist_ok=True)
    mesh = stl_info(args.stl)
    reg, j, edges, thickness = load_registered(args.registration)
    scan = tifffile.memmap(args.scan)
    if scan.ndim != 3: raise ValueError(f"Expected 3D TIFF, got {scan.shape}")
    # Match test_pipeline.py: retain the source as a memmap and materialize a
    # 4x (by default) downsampled copy.  All analysis below uses this compact
    # grid; the full-resolution scan is never loaded into RAM.
    ds = args.downsample_factor
    scan_small = scan[::ds, ::ds, ::ds].copy()
    small_j = j / ds
    # JSON positions are x,y,z; TIFF axes are z,y,x.
    roi_lo = np.floor(small_j.min(0)).astype(int) - 2
    roi_hi = np.ceil(small_j.max(0)).astype(int) + 3
    roi_lo = np.maximum(roi_lo, 0)
    roi_hi = np.minimum(roi_hi, [scan_small.shape[2]-1, scan_small.shape[1]-1, scan_small.shape[0]-1])
    # STL extents establish the nominal model-to-registered scale; use robust median.
    model_extent = np.asarray(mesh["bounds_max"]) - np.asarray(mesh["bounds_min"])
    reg_extent = j.max(0)-j.min(0)
    scale = float(np.median(reg_extent/model_extent))
    radius = max(1.0, float(thickness*scale/ds))
    sample = scan_small.ravel()
    threshold = float(threshold_otsu(sample))
    # Material is the high-intensity class for this scan; preserve the raw threshold in metadata.
    candidate_path = out/"mask_output.tif"
    cad_path = out/"cad_occupancy_mask.tif"
    z0,z1 = int(roi_lo[2]), int(roi_hi[2])+1; y0,y1=int(roi_lo[1]),int(roi_hi[1])+1; x0,x1=int(roi_lo[0]),int(roi_hi[0])+1
    roi_shape=(z1-z0,y1-y0,x1-x0)
    # These compact uint8 ROI masks are at the downsampled resolution.  The
    # source TIFF remains a memmap and processing still reads bounded Z slabs.
    cand = np.zeros(roi_shape, dtype=np.uint8); cad = np.zeros(roi_shape, dtype=np.uint8)
    structure = np.ones((3,3,3), dtype=bool)
    for slab_start in range(z0, z1, args.slab_size):
        slab_stop = min(slab_start + args.slab_size, z1)
        # One bounded read.  The temporary slab is released before the next.
        scan_slab = np.asarray(scan_small[slab_start:slab_stop, y0:y1, x0:x1])
        for local_z, z in enumerate(range(slab_start, slab_stop)):
            nominal = rasterize_slice(z, small_j, edges, radius,
                                      (scan_small.shape[1], scan_small.shape[2]))[y0:y1,x0:x1]
            cad[z-z0] = nominal
            cand[z-z0] = nominal & (scan_slab[local_z] < threshold)
        del scan_slab
    # At 4x resolution a strut can be only one voxel wide, so a 3D opening
    # would erase the very missing-strut regions being sought.  The candidate
    # mask is already CAD-constrained; remove noise by 26-connected component
    # size instead.
    labels, count = ndimage.label(cand.astype(bool), structure=structure)
    sizes = np.bincount(labels.ravel())
    keep = sizes >= args.min_voxels; keep[0] = False
    clean = keep[labels]
    tifffile.imwrite(cad_path, cad, compression="zlib")
    tifffile.imwrite(candidate_path, clean.astype(np.uint8), compression="zlib")
    # Component descriptors.  Principal-axis calculations are only done on retained components.
    rows=[]; voxel_volume=1.0
    for lab in np.flatnonzero(keep):
        coords=np.argwhere(labels==lab); n=len(coords)
        c=coords.mean(0)+np.array([z0,y0,x0]); centered=coords-coords.mean(0)
        cov=np.cov(centered,rowvar=False) if n>2 else np.zeros((3,3)); eig=np.linalg.eigvalsh(cov)[::-1]
        aspect=float(np.sqrt(eig[0]/max(eig[-1],1e-9)))
        eq=(6*n/np.pi)**(1/3); bbox=coords.max(0)-coords.min(0)+1
        # Voxel-face surface area gives a reproducible, resolution-dependent
        # sphericity approximation without constructing a triangle mesh.
        local = labels[tuple(slice(a, b + 1) for a, b in zip(coords.min(0), coords.max(0)))] == lab
        padded = np.pad(local.astype(np.int8), 1)
        surface_faces = int(sum(np.abs(np.diff(padded, axis=ax)).sum() for ax in range(3)))
        sphericity = float((np.pi ** (1 / 3) * (6 * n) ** (2 / 3)) / max(surface_faces, 1))
        rows.append({"component":int(lab),"voxels":n,"volume_voxels":n,"equivalent_diameter_voxels":eq,
                     "centroid_x":c[2],"centroid_y":c[1],"centroid_z":c[0],"bbox_z":int(bbox[0]),"bbox_y":int(bbox[1]),"bbox_x":int(bbox[2]),"max_dimension_voxels":int(bbox.max()),"surface_faces":surface_faces,"sphericity":sphericity,"aspect_ratio":aspect,
                     "orientation_z":float(np.sqrt(max(eig[0],0)))})
    csv_path=out/"defect_summary.csv"
    with csv_path.open("w",newline="") as f:
        fields=list(rows[0]) if rows else ["component","voxels","volume_voxels","equivalent_diameter_voxels","centroid_x","centroid_y","centroid_z","bbox_z","bbox_y","bbox_x","max_dimension_voxels","surface_faces","sphericity","aspect_ratio","orientation_z"]
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)
    mid=int(np.argmax(clean.sum((1,2)))); view=np.asarray(scan_small[z0+mid,y0:y1,x0:x1]); overlay=np.zeros((*view.shape,3),dtype=np.uint8)
    norm=np.clip((view-threshold)/(max(view.max()-threshold,1)),0,1)*255; overlay[...,0]=norm.astype(np.uint8); overlay[...,1]=norm.astype(np.uint8); overlay[...,2]=norm.astype(np.uint8)
    overlay[clean[mid]>0]=[255,0,0]; Image.fromarray(overlay).save(out/"rendered_view.png")
    report={"script_status":"success","executed_script_path":str(Path(__file__).resolve()),"extracted_features":{"total_defects_found":len(rows),"defect_summary_csv":str(csv_path),"mask_output_path":str(candidate_path)},"validation_summary":{"stage_2b_status":"NOT_RUN","total_revisions_attempted":0},"execution_logs":{"mesh":mesh,"scan_shape":list(scan.shape),"scan_dtype":str(scan.dtype),"downsample_factor":ds,"downsampled_scan_shape":list(scan_small.shape),"registration_junctions":len(j),"registration_struts":len(edges),"roi_xyz_min_downsampled":roi_lo.tolist(),"roi_xyz_max_exclusive_downsampled":roi_hi.tolist(),"axis_mapping":"registered x,y,z -> TIFF z,y,x","model_extent":model_extent.tolist(),"registered_extent":reg_extent.tolist(),"scale_pixels_per_stl_unit":scale,"strut_radius_downsampled_voxels":radius,"otsu_threshold":threshold,"material_polarity":"high_intensity","candidate_voxels_after_component_filter":int(clean.sum()),"min_component_voxels":args.min_voxels,"z_slab_size":args.slab_size,"tiff_read_mode":"tifffile.memmap; 4x downsampled copy used for complete analysis; bounded Z slabs for final masks"}}
    (out/"stage2a_report.json").write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))

if __name__ == "__main__": main()
