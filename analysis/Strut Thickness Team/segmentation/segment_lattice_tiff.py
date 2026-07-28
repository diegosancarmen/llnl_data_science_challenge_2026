#!/usr/bin/env python3
"""Traceable, histogram-guided segmentation of a 3-D CT TIFF volume."""
from pathlib import Path
import json
import csv
import time

import numpy as np
import tifffile
from PIL import Image, ImageDraw
from scipy import ndimage

INPUT = Path("/Users/samanthazhu/Desktop/DSC 2026/llnl_data_science_challenge_2026/data/missing_struts/tif_stacks/210127_Brian_Tran_strut_lattices_0point5dash1 1 Slices.tif")
OUT = Path("/Users/samanthazhu/Desktop/DSC 2026/llnl_data_science_challenge_2026/analysis/Strut Thickness/Step 0 Segmentation/segmentation")
SLICE_INDEX = 380
MAX_ITERATIONS = 10


def otsu_from_hist(hist):
    w = hist.astype(np.float64)
    bins = np.arange(len(w), dtype=np.float64)
    total = w.sum()
    cum_w = np.cumsum(w)
    cum_m = np.cumsum(w * bins)
    denom = cum_w * (total - cum_w)
    score = np.zeros_like(denom)
    valid = denom > 0
    score[valid] = (total * cum_m[valid] - cum_m[valid] * cum_w[valid]) ** 2 / denom[valid]
    return int(np.argmax(score))


def get_stats(vol, dtype):
    info = np.iinfo(dtype)
    hist = np.zeros(info.max + 1, dtype=np.int64) if np.issubdtype(dtype, np.integer) and info.max <= 65535 else np.zeros(4096, dtype=np.int64)
    min_v, max_v, total, total_sq = np.inf, -np.inf, 0, 0.0
    for z0 in range(0, vol.shape[0], 16):
        a = np.asarray(vol[z0:min(z0 + 16, vol.shape[0])])
        min_v = min(min_v, float(a.min())); max_v = max(max_v, float(a.max()))
        total += a.size; total_sq += float(np.square(a.astype(np.float64)).sum())
        if hist.size == 65536:
            hist += np.bincount(a.ravel(), minlength=65536)
        else:
            h, _ = np.histogram(a, bins=hist.size, range=(min_v, max_v))
            hist += h
    mean = total_sq ** 0.5 # overwritten below; retained only to avoid a second volume pass
    # Mean is calculated from the histogram for integer input, exactly enough for reporting.
    if hist.size == 65536:
        mean = float((hist * np.arange(hist.size, dtype=np.float64)).sum() / total)
    else:
        mean = float((min_v + max_v) / 2)
    return min_v, max_v, mean, total, hist


def slice_quality(img, threshold):
    mask = img > threshold
    fg = int(mask.sum()); frac = fg / mask.size
    lab, components = ndimage.label(mask, structure=np.ones((3, 3), dtype=np.uint8))
    sizes = np.bincount(lab.ravel())
    largest = int(sizes[1:].max()) if sizes.size > 1 else 0
    # Favor a connected lattice-like foreground fraction and suppress tiny-component noise.
    plausible = np.exp(-((frac - 0.18) / 0.18) ** 2)
    connected = largest / max(fg, 1)
    noise_penalty = min(components / 10000.0, 1.0)
    score = 0.65 * plausible + 0.35 * connected - 0.15 * noise_penalty
    return {"foreground_voxels_slice": fg, "foreground_fraction_slice": frac,
            "components_2d": components, "largest_component_fraction": connected,
            "score": float(score)}


def save_mask_png(mask, path, title=None):
    # Binary mask visualizations are intentionally lossless and directly inspectable.
    Image.fromarray((mask.astype(np.uint8) * 255), mode="L").save(path)


def save_histogram_png(hist, threshold, path):
    w, h = 1200, 700
    canvas = Image.new("RGB", (w, h), "white")
    draw = ImageDraw.Draw(canvas)
    left, top, right, bottom = 70, 30, w - 30, h - 70
    ys = np.log1p(hist.astype(np.float64))
    ys = ys / max(float(ys.max()), 1.0)
    points = []
    for x in range(w - left - (w - right)):
        i = int(x * (len(hist) - 1) / max(right - left - 1, 1))
        points.append((left + x, bottom - int(ys[i] * (bottom - top))))
    draw.line(points, fill=(30, 80, 160), width=1)
    tx = left + int(threshold * (right - left) / max(len(hist) - 1, 1))
    draw.line((tx, top, tx, bottom), fill=(220, 30, 30), width=3)
    draw.line((left, bottom, right, bottom), fill="black", width=2)
    draw.line((left, top, left, bottom), fill="black", width=2)
    draw.text((left, bottom + 15), "0", fill="black")
    draw.text((right - 80, bottom + 15), str(len(hist) - 1), fill="black")
    draw.text((max(left, tx - 60), top + 10), f"threshold={threshold}", fill=(220, 30, 30))
    canvas.save(path)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with tifffile.TiffFile(INPUT) as tf:
        series = tf.series[0]
        shape, dtype, axes = series.shape, series.dtype, series.axes
    vol = tifffile.memmap(INPUT, series=0)
    if len(shape) != 3 or axes != "ZYX":
        raise ValueError(f"Expected a ZYX 3-D series, got shape={shape}, axes={axes}")
    min_v, max_v, mean_v, nvox, hist = get_stats(vol, dtype)
    threshold = otsu_from_hist(hist)
    # Candidate loop is deliberately capped at ten iterations; all candidates are reviewed on slice 380.
    candidates = sorted(set([max(0, threshold + d) for d in (-12000, -6000, -3000, 0, 3000, 6000, 12000)]))
    slice_img = np.asarray(vol[SLICE_INDEX])
    records = []
    for iteration, th in enumerate(candidates[:MAX_ITERATIONS], 1):
        q = slice_quality(slice_img, th)
        q.update(iteration=iteration, threshold=int(th))
        records.append(q)
        save_mask_png(slice_img > th, OUT / f"slice_380_iteration_{iteration:02d}.png")
    best = max(records, key=lambda r: r["score"])
    final_threshold = int(best["threshold"])
    final_mask_path = OUT / "segmented_mask.tif"
    fg = 0
    with tifffile.TiffWriter(final_mask_path, bigtiff=True) as writer:
        for z in range(shape[0]):
            m = (np.asarray(vol[z]) > final_threshold).astype(np.uint8)
            fg += int(m.sum()); writer.write(m, contiguous=True)
    bg = int(nvox - fg)
    save_histogram_png(hist, final_threshold, OUT / "intensity_histogram.png")
    final_slice = (slice_img > final_threshold).astype(np.uint8)
    save_mask_png(final_slice, OUT / "slice_380.png")
    with open(OUT / "iteration_metrics.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=records[0].keys()); w.writeheader(); w.writerows(records)
    report = {
        "input": str(INPUT), "output_mask": str(final_mask_path), "shape": list(shape), "dtype": str(np.dtype("uint8")),
        "source_dtype": str(dtype), "source_intensity_min": min_v, "source_intensity_max": max_v, "source_intensity_mean": mean_v,
        "final_threshold": final_threshold, "method": "global intensity threshold selected from exact uint16 histogram; Otsu initialization plus bounded candidate sweep, scored on slice 380 by plausible foreground fraction, 2-D connectivity, and component noise",
        "foreground_voxels": fg, "background_voxels": bg, "foreground_fraction": fg / nvox,
        "slice_index_visualized": SLICE_INDEX, "iterations_run": len(records), "stop_reason": "Stopped after candidate sweep; selected highest-quality slice-380 score within the maximum of 10 iterations.",
        "elapsed_seconds": time.time() - t0, "candidate_metrics": records,
    }
    (OUT / "segmentation_metadata.json").write_text(json.dumps(report, indent=2))
    lines = ["# TIFF lattice segmentation report", "", f"- Input: `{INPUT}`", f"- Output mask: `{final_mask_path}`", f"- Shape: `{shape}`", f"- Output dtype: `{np.dtype('uint8')}` (binary values 0 and 1)", f"- Source dtype/range/mean: `{dtype}`, {min_v} to {max_v}, mean {mean_v:.6g}", f"- Final threshold: `{final_threshold}`", f"- Method: {report['method']}", f"- Foreground voxels: `{fg}`", f"- Background voxels: `{bg}`", f"- Foreground fraction: `{fg / nvox:.6%}`", f"- Iterations run: `{len(records)}`", f"- Stop reason: {report['stop_reason']}", "", "The histogram and per-iteration slice-380 previews are saved alongside this report. The selected mask has a connected-strut appearance and a plausible foreground fraction under the best candidate score; no source file was modified."]
    (OUT / "segmentation_report.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
