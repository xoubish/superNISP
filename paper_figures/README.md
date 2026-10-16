# Paper figures workspace

One notebook per proposed figure. Start with the figure, agree on its content
and visual choices, then refine the caption and manuscript text. Numbering is
provisional; Figures 2 and 3 retain the current paper's diffusion-then-residual
architecture order. The manuscript and its existing figures are not replaced.

| Figure | Notebook | Proposed content | Initial state |
| --- | --- | --- | --- |
| 1 | [01_sample.ipynb](01_sample.ipynb) | Footprint/splits and matched ~49-arcsec NISP/JWST field | Horizontal layout; larger field awaits original mosaics |
| 1b | [01b_sample_properties.ipynb](01b_sample_properties.ipynb) | Magnitude and redshift distributions | Separate horizontal figure; rendered |
| 2 | [02_diffusion_architecture.ipynb](02_diffusion_architecture.ipynb) | Corrected conditioning, UNet, training and sampling | Vector schematic draft |
| 3 | [03_residual_architecture.ipynb](03_residual_architecture.ipynb) | Shared residual trunk, legacy/direct-5× heads, dense-block detail | Vector schematic draft |
| 4 | [04_galaxy_examples.ipynb](04_galaxy_examples.ipynb) | NISP, JWST, models and residuals on the same galaxies | Wait for completed predictions |
| 5 | [05_model_comparison.ipynb](05_model_comparison.ipynb) | Matched reconstruction errors and computational cost | Budget table ready; benchmark pending |
| 6 | [06_observed_shapes.ipynb](06_observed_shapes.ipynb) | Observed e1/e2 recovery and residual distributions | Frozen legacy validation preview |
| 7 | [07_shape_dependence.ipynb](07_shape_dependence.ipynb) | Shape gains versus input S/N, magnitude and redshift; success counts | Frozen legacy validation preview |
| 8 | [08_shape_robustness.ipynb](08_shape_robustness.ipynb) | Measurement choices, adaptive moments and failure coverage | Frozen legacy validation preview |

Figures 7–8 may move to an appendix or be combined. Diffusion sample variability
and the individual-versus-mean-image comparison can become an additional figure
after the full run finishes; their place in the story is not fixed yet.

The sample overview and property distributions are now **two separate figures**,
each in one row, without figure/panel titles. `1b` is a working filename; final
numbering is deferred so existing architecture notebook names remain stable.
For the larger overview field, set `MOSAIC_ROOT` in `01_sample.ipynb` on the
machine holding the original mosaics. `sample_region.py` extracts 164×164 native
NISP pixels and an aligned 820×820 JWST field (about 49 arcsec across). The
notebook also accepts explicit paths for the two images and caches the extracted
field under `inputs/` for reuse elsewhere. No training or inference is involved.

## Working together

Open any notebook in Jupyter or VS Code and run from top to bottom. Each has:
the scientific question, explicit inputs/status, editable style/settings,
plotting cells, exports, a working caption and a short list of review choices.
Plot code lives in the notebook so edits are easy to discuss. Shared colors,
fonts, paths and provenance helpers live in `figure_utils.py`.

Exports go to `exports/` as PDF, editable-text SVG and PNG, with a JSON sidecar
recording input hashes, notebook source hash and settings. Change the notebook's
`VERSION` before preserving a new review iteration; rerunning the same version
updates its exports. The notebooks are the source of truth—there is no generator
to rerun over hand edits. Data is resolved from `SUPERNISP_DATA` when set,
otherwise from the sibling `supernisp_nir_y_f115w_v1` folder.

The notebooks read metadata, observed images and completed saved diagnostics. They contain
no model construction, checkpoint loading, training, inference or W&B API calls.
They can run while the GPUs are busy. Architecture labels are checked against
recorded source hashes without importing the models.

## Scientific boundaries for the drafts

- Figure 1 uses metadata, including test positions/counts for the split design.
  The separate magnitude/redshift figure includes train, validation and test
  catalog distributions at the user's request; this does not evaluate test
  predictions or change the split. The overview
  opens images only within the selected validation block and performs no held-out
  evaluation. This is an optically
  selected COSMOS subset, not a NISP-complete galaxy sample.
- Results previews are explicitly labeled **validation / legacy baseline**.
  They do not automatically add an unfinished model or claim a final result.
- Figure 4 waits for one aligned, physical-intensity prediction bundle from a
  separate tmux/W&B export job after model choices are ready. It does not crop
  screenshots or rerun models inside the notebook. Its proposed selection is
  based on input properties, with exact IDs saved before comparing outputs.
- Figure 5 requires per-object metrics on the same validation objects and a
  controlled inference timing benchmark. Differently normalized training losses
  and differently sized validation subsets must not be ranked as comparable
  reconstruction metrics. Tiny debugging pilots stay out of the main comparison.
- Shape panels describe **observed-shape recovery**, not calibrated shear.
  Include matching object IDs, failed measurements, and conditional spatial-block
  uncertainty. Keep the Y/F115W band difference and instrumental PSFs explicit.
- The test set remains reserved until models and measurement choices are frozen.

`inputs/README.md` defines the two pending figure input contracts. No synthetic
placeholder results are substituted when those artifacts are missing.

## Rendered drafts and current sample revision

The current [sample properties figure](exports/fig01b_sample_properties_v02.png)
shows train, validation and test distributions without titles. Test is dashed
gray; each histogram has unit integral and shares the original bin edges.
Its notebook explains training versus validation versus final test evaluation.
The current overview will export
`fig01_sample_v03` after its ~49-arcsec field is extracted from original mosaics.
Earlier combined layouts are preserved as
[v01: accounting panel](exports/fig01_sample_v01.png) and
[v02: smaller 12-arcsec images](exports/fig01_sample_v02.png), with corresponding
notebook snapshots in `revisions/`; neither is the requested final layout.

Architecture drafts are [Figure 2: diffusion](exports/fig02_diffusion_architecture_v01.png) and
[Figure 3: residual architectures](exports/fig03_residual_architecture_v01.png).
Each also has a PDF and an editable-text SVG with the same filename stem.

The legacy-only validation previews are
[Figure 6: observed shapes](exports/fig06_observed_shapes_v01.png),
[Figure 7: property dependence](exports/fig07_shape_dependence_v01.png), and
[Figure 8: measurement robustness](exports/fig08_shape_robustness_v01.png).
Their input analyses are frozen; rerunning these notebooks will not silently
switch to newer model results. Figures 4–5 intentionally have no exported plot
until their required prediction/benchmark bundles are available. Their waiting
paths have been executed; the populated plotting paths await those real inputs.
