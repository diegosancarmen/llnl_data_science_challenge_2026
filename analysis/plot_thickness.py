# strut_thickness_histogram.py
import sys
import numpy as np
import tifffile
from scipy import ndimage
from skimage.morphology import skeletonize
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---- Reference values ----
NOMINAL_PAPER_UM = 424.0        # Tran et al. nominal strut diameter
NOMINAL_DOC_UM = 350.0          # challenge doc's stated value (kept for comparison)
VOXEL_SIZE_EST_UM = 50.0        # independent geometry estimate (~9 cells * 4.56mm / axis)

def main(mask_path):
    print(f"Loading mask: {mask_path}")
    with tifffile.TiffFile(mask_path) as _tf:
        n_pages = len(_tf.pages)
    vol = tifffile.imread(mask_path, key=range(n_pages))
    mask = np.asarray(vol) > 0
    print(f"Shape {mask.shape}, foreground voxels {int(mask.sum()):,}")
    if mask.ndim != 3:
        print(f"WARNING: expected a 3D volume, got {mask.ndim}D. Stopping.")
        return

    print("Computing distance transform...")
    edt = ndimage.distance_transform_edt(mask).astype(np.float32)  # local radius in voxels

    print("Skeletonizing (slow on large volumes)...")
    skel = skeletonize(mask)

    print("Removing junction voxels...")
    kernel = np.ones((3, 3, 3), dtype=np.uint8)
    kernel[1, 1, 1] = 0
    neighbors = ndimage.convolve(skel.astype(np.uint8), kernel, mode="constant", cval=0)
    strut_interior = skel & (neighbors <= 2)   # keep path voxels, drop junctions
    print(f"skeleton voxels: {int(skel.sum()):,}  |  strut-interior kept: {int(strut_interior.sum()):,}")

    # Diameter in VOXELS (primary, unit-free)
    diam_vox = 2.0 * edt[strut_interior]
    diam_vox = diam_vox[diam_vox > 0]

    med_vox = float(np.median(diam_vox))
    print("\n--- Strut diameter (VOXELS) ---")
    print(f"sampled points: {diam_vox.size:,}")
    print(f"median: {med_vox:.2f}   mean: {diam_vox.mean():.2f}   std: {diam_vox.std():.2f}")
    print(f"5th/95th pct: {np.percentile(diam_vox,5):.2f} / {np.percentile(diam_vox,95):.2f}")

    # --- Self-calibration consistency check ---
    print("\n--- Voxel-size consistency check ---")
    if med_vox > 0:
        implied_paper = NOMINAL_PAPER_UM / med_vox
        implied_doc = NOMINAL_DOC_UM / med_vox
        print(f"voxel size implied by 424um/median : {implied_paper:.1f} um/voxel")
        print(f"voxel size implied by 350um/median : {implied_doc:.1f} um/voxel")
        print(f"independent geometry estimate      : {VOXEL_SIZE_EST_UM:.1f} um/voxel")
        print("If 424um/median is close to ~50 -> voxel size AND segmentation both consistent.")
        print("If it's notably higher than 50   -> struts thinner than nominal "
              "(threshold may have eroded material).")

    # --- Histogram: voxels primary, microns on secondary axis ---
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(diam_vox, bins=80, color="steelblue", edgecolor="none")
    ax.axvline(med_vox, color="black", linestyle="-", label=f"median {med_vox:.1f} vox")
    ax.axvline(NOMINAL_PAPER_UM / VOXEL_SIZE_EST_UM, color="red", linestyle="--",
               label=f"424um nominal ({NOMINAL_PAPER_UM/VOXEL_SIZE_EST_UM:.1f} vox)")
    ax.axvline(NOMINAL_DOC_UM / VOXEL_SIZE_EST_UM, color="orange", linestyle=":",
               label=f"350um doc ({NOMINAL_DOC_UM/VOXEL_SIZE_EST_UM:.1f} vox)")
    ax.set_xlabel("Strut diameter (voxels)")
    ax.set_ylabel("Count (skeleton points)")
    ax.set_title("Strut thickness distribution (distance-transform, junctions excluded)")
    ax.legend(loc="upper right")

    def v2u(v): return v * VOXEL_SIZE_EST_UM
    def u2v(u): return u / VOXEL_SIZE_EST_UM
    secax = ax.secondary_xaxis("top", functions=(v2u, u2v))
    secax.set_xlabel(f"Approx. diameter (um)  [at {VOXEL_SIZE_EST_UM:.0f} um/voxel, ESTIMATE]")

    plt.tight_layout()
    out = "strut_thickness_histogram.png"
    plt.savefig(out, dpi=150)
    print(f"\nSaved {out}")

if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else \
        "data/9x9x9_octet_lattice/segmentation/9x9x9_octet_lattice_mask.tif"
    main(path)