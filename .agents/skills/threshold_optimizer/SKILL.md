---
name: threshold_optimizer
description: Use this skill when the user asks to find, optimize, tune, or compare segmentation thresholds for a CT .npy volume. Runs a data-driven threshold sweep and recommends the best value.
---

# Threshold Optimizer

When asked to find or optimize a segmentation threshold for a .npy volume:

1. Run the bundled sweep script on the user's file:
   `python .agents/skills/threshold_optimizer/scripts/optimize_threshold.py <input_filepath>`
2. Read the printed table. The script derives candidate thresholds from the data's
   actual percentiles — do NOT assume the data lies in [0, 1].
3. Report the recommended threshold and the full table to the user.
4. If the user wants to see it, offer to segment at the recommended threshold using
   the `segment_ct_dataset` MCP tool and visualize the central slice.