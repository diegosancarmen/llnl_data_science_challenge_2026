# Segmentation Quality Rubric (Slice 380)

You are evaluating a predicted segmentation slice against a ground-truth slice of an
octet lattice cross-section. The first image is the GROUND TRUTH; the second is the
RESULT to be scored. Both show the same physical slice (index 380). Foreground =
lattice material; background = empty space.

## Criteria to assess

1. **Structural integrity / connectivity.** In the ground truth, are struts that connect
   nodes present and continuous in the result? Penalize results where struts are broken
   into disconnected dots that are connected in the ground truth.
2. **False positives / false negatives.** Identify over-segmentation (extra foreground/
   noise not in the ground truth) and under-segmentation (missing struts or nodes present
   in the ground truth).
3. **Topology / node preservation.** Are the junction points (nodes) present at the correct
   locations and roughly the correct count compared to the ground truth?
4. **Noise and artifacts.** Does the result contain speckle, stray foreground pixels, or
   artifacts absent from the clean ground truth?

## Scoring (0-5)

- **5:** Essentially identical to ground truth. Connectivity, nodes, and strut coverage
  all match; no meaningful false positives or negatives.
- **4:** Excellent, only very minor differences (e.g., a few slightly thinned struts).
- **3:** Main topology correct, but noticeable problems — some struts missing/broken or
  moderate noise.
- **2:** Fair; significant differences such as large regions of missing struts or many
  disconnected components where the ground truth is connected.
- **1:** Major structural failure or excessive noise; the lattice is barely recognizable.
- **0:** Blank, unrelated, or completely wrong output.

## Output format

Return ONLY a JSON object:
{
  "reasoning": "<2-4 sentences citing specific differences between the two images>",
  "score": <integer 0-5>
}