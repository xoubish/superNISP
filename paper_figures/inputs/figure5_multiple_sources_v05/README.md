# Figure 5: fields with multiple sources

The user requested more interesting examples, including fields with more than
one galaxy, while retaining the face-on disk in row b. New frozen IDs in row
order: **749105, 677525, 592591, 744769, 997288**.

Rows a/c/d/e illustrate a close projected pair, overlapping elongated sources,
a crowded field with multiple neighbors, and a close pair with different
orientations. These descriptions come from observed NISP/JWST images. Physical
association is not established. The selection is illustrative, not random or
statistically representative.

`candidates.json` records the 64 reviewed fields. The search found 369 qualifying
fields among 2,118 validation objects and kept 313 spatially distinct centers.
It ranks separated peaks in observed JWST crops; peaks are a browsing aid, not
a deblending or galaxy catalogue. Full thresholds, source hash, candidate-record
hash, IDs, display scales and noise seeds are in
`../galaxy_examples_selection_v05.json`. No model outputs were inspected for
the new selection. Row b retains the previous object and display scale.

Reproduce the candidate sheets from the repository root:

```bash
MPLCONFIGDIR=/tmp/supernisp-mpl python paper_figures/inspect_multiple_sources.py
```

Render the observation-only five-row preview:

```bash
MPLCONFIGDIR=/tmp/supernisp-mpl python paper_figures/inspect_multiple_sources.py \
  --selection paper_figures/inputs/galaxy_examples_selection_v05.json
```

Preview: `paper_figures/exports/fig05_observed_selection_v05.png` and `.pdf`.
The user explicitly approved the W&B upload. Model comparison and residual
exports are complete: `paper_figures/exports/fig05_galaxy_examples_v05.*` and
`fig05_galaxy_residuals_v05.*`. Run `runs/figure5-v05` completed with exit 0;
[W&B 50peiezt](https://wandb.ai/AI-Astro/superNISP-corrected/runs/50peiezt)
contains the figures and prediction artifact. All row-b arrays and display
scales exactly match v04, with unchanged checkpoint hashes. Source, selection,
bundle and export hashes verified; main and residual figures visually checked.
