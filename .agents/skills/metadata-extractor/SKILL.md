---
name: metadata-extractor
description: "Inspect NumPy `.npy` arrays and print basic metadata: file path, shape, dtype, minimum, and maximum. Use when a user asks for metadata, dimensions, data type, or value range of a generated or existing `.npy` dataset."
---

# Metadata Extractor

Run the bundled script on the requested `.npy` file:

```bash
python .agents/skills/metadata-extractor/scripts/extract_metadata.py <path-to-file.npy>
```

Report the script output verbatim or summarize its fields. Do not modify the
input file. If the path is missing or is not a readable `.npy` file, report the
error and request a valid path.
