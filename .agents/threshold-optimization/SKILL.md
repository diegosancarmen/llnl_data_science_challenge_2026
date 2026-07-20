---
name: threshold-optimization
description: Compare multiple CT segmentation thresholds by calling the segment_ct_dataset MCP tool once per threshold and saving each binary mask to a separate .npy file. Use when a user wants threshold sweeps, segmentation comparison, or threshold sensitivity analysis for a 3D CT .npy dataset.
---

# Threshold Optimizer

Use this skill to run a threshold sweep over a 3D CT `.npy` volume.

## Workflow

1. Identify the input volume and choose thresholds. If none are supplied, use `0.3`, `0.5`, and `0.7`.
2. Run `segment_ct_dataset()` once for every threshold. Do not reuse one output path: each result must be independently saved.
3. Name outputs deterministically as `<stem>_segmented_<threshold>.npy`, formatting thresholds without unnecessary trailing zeroes.
4. Verify that every output exists and has the same shape as the input. Report the output paths and, when useful, foreground voxel counts for comparison.

## MCP calls

For each threshold, call:

```text
segment_ct_dataset(
  input_filepath="/absolute/path/to/input.npy",
  output_filepath="/absolute/path/to/output_segmented_0.3.npy",
  threshold=0.3,
)
```

Use the bundled `scripts/threshold_optimizer.py` to normalize threshold labels, construct output paths, print a call plan, and validate completed outputs. The script does not replace the MCP call; the agent must invoke the MCP tool for each planned threshold.

## Guardrails

- Require a 3D `.npy` input and numeric thresholds.
- Keep outputs in the requested output directory unless the user specifies otherwise.
- Preserve existing files unless the user explicitly requests replacement; if a target exists, report it before overwriting.
- If one MCP call fails, continue the remaining thresholds and report the individual failure.
