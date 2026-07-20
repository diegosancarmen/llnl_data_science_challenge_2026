#!/usr/bin/env python3
"""Prepare and validate a multi-threshold segmentation sweep.

The Codex agent performs the actual segment_ct_dataset MCP calls. This helper
creates deterministic output paths, prints the calls to make, and validates
the resulting .npy files after the calls complete.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np


def format_threshold(value: float) -> str:
    """Return a stable filename label for a threshold."""
    if not np.isfinite(value):
        raise ValueError(f"Threshold must be finite: {value}")
    return f"{value:.12g}"


def build_requests(input_path: Path, output_dir: Path, thresholds: Iterable[float]) -> list[dict]:
    """Build one MCP request description per threshold."""
    requests = []
    for threshold in thresholds:
        threshold = float(threshold)
        label = format_threshold(threshold)
        output_path = output_dir / f"{input_path.stem}_segmented_{label}.npy"
        requests.append({
            "input_filepath": str(input_path.resolve()),
            "output_filepath": str(output_path.resolve()),
            "threshold": threshold,
        })
    return requests


def validate_outputs(input_path: Path, requests: Iterable[dict]) -> list[dict]:
    """Check output existence and shape compatibility with the input volume."""
    input_array = np.load(input_path, mmap_mode="r")
    if input_array.ndim != 3:
        raise ValueError(f"Input must be 3D; got shape {input_array.shape}")

    results = []
    for request in requests:
        output_path = Path(request["output_filepath"])
        result = {
            "threshold": request["threshold"],
            "output_filepath": str(output_path),
            "exists": output_path.exists(),
        }
        if output_path.exists():
            array = np.load(output_path, mmap_mode="r")
            result.update({
                "shape": list(array.shape),
                "shape_compatible": array.shape == input_array.shape,
                "foreground_voxels": int(np.count_nonzero(array)),
            })
        results.append(result)
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Input 3D CT .npy file")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--threshold", dest="thresholds", type=float, action="append")
    parser.add_argument("--validate", action="store_true", help="Validate outputs after MCP calls")
    args = parser.parse_args()

    thresholds = args.thresholds or [0.3, 0.5, 0.7]
    input_path = args.input.resolve()
    output_dir = (args.output_dir or input_path.parent).resolve()
    if not input_path.exists():
        parser.error(f"Input does not exist: {input_path}")
    output_dir.mkdir(parents=True, exist_ok=True)

    requests = build_requests(input_path, output_dir, thresholds)
    if args.validate:
        print(json.dumps(validate_outputs(input_path, requests), indent=2))
    else:
        print(json.dumps(requests, indent=2))
        print("Use one segment_ct_dataset MCP call for each request above.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
