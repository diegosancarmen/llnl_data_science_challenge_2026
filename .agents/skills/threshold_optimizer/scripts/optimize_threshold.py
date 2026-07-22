# .agents/skills/threshold_optimizer/scripts/optimize_threshold.py
import sys
import os
import numpy as np

def optimize(input_path):
    if not os.path.exists(input_path):
        print(f"Error: file not found at {input_path}")
        return

    data = np.load(input_path)
    total = data.size
    print(f"Loaded {input_path}, shape {data.shape}")
    print(f"min={data.min():.5f} max={data.max():.5f} mean={data.mean():.5f}\n")

    # Derive candidates from the data's own distribution — never assume [0,1]
    pcts = [90, 95, 99, 99.5, 99.9]
    values = np.percentile(data, pcts)
    candidates = sorted(set(round(float(v), 5) for v in values))

    print(f"{'threshold':>10} | {'fg voxels':>12} | {'fraction':>9} | verdict")
    print("-" * 55)
    results = []
    for t in candidates:
        fg = int((data >= t).sum())
        frac = fg / total
        if fg == 0:
            verdict = "EMPTY - too high"
        elif frac > 0.5:
            verdict = "OVERFULL - too low"
        elif 0.02 <= frac <= 0.10:
            verdict = "plausible lattice"
        else:
            verdict = "check visually"
        results.append((t, fg, frac, verdict))
        print(f"{t:>10.5f} | {fg:>12,} | {frac:>8.2%} | {verdict}")

    plausible = [r for r in results if "plausible" in r[3]]
    pick = plausible[0] if plausible else min(
        (r for r in results if r[1] > 0), key=lambda r: abs(r[2] - 0.04), default=None)
    print()
    if pick:
        print(f"Recommended threshold: {pick[0]} ({pick[2]:.2%} foreground)")
    else:
        print("No valid threshold found in candidate set.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python optimize_threshold.py <path_to_npy>")
        sys.exit(1)
    optimize(sys.argv[1])