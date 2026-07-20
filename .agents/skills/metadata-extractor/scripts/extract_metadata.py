#!/usr/bin/env python3
"""Print basic metadata for a NumPy .npy array without modifying it."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_file", type=Path, help="Path to a .npy file")
    args = parser.parse_args()

    input_file = args.input_file
    if input_file.suffix.lower() != ".npy":
        parser.error("input_file must have a .npy extension")
    if not input_file.is_file():
        parser.error(f"file not found: {input_file}")

    try:
        array = np.load(input_file, mmap_mode="r", allow_pickle=False)
    except (OSError, ValueError) as exc:
        parser.error(f"could not load {input_file}: {exc}")

    print(f"File: {input_file}")
    print(f"Shape: {array.shape}")
    print(f"Data type: {array.dtype}")
    print(f"Minimum: {array.min()}")
    print(f"Maximum: {array.max()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
