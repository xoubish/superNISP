# Paper figures workspace

One notebook per proposed figure. Start with the figure, agree on its content
and visual choices, then refine the caption and manuscript text. Figure 2 contains the sample distributions; the diffusion and residual
architectures follow as Figures 3 and 4. The architecture notebooks are now `03_diffusion_architecture.ipynb` and
`04_residual_architecture.ipynb`; result notebook filenames retain their original prefixes.

The approved sample overview v07 is now included as the manuscript's first
figure in `paper/survey_enhancements.tex`, with label `fig:sample_overview` and
a caption in Observations. Its frozen PDF is `paper/figs/sample_overview_v07.pdf`.
This version shows the 24.3-arcsec field around COSMOS2020 734405 with 5-arcsec
scale bars. The adjoining sample/split paragraph was updated; the remaining
historical manuscript text and results still require revision. Local full-PDF
verification is blocked by missing TeX `etex.sty` (and latexmk's Perl
`Time::HiRes` dependency); the included PDF and figure ordering were checked.

| Figure | Notebook | Proposed content | Initial state |
| --- | --- | --- | --- |
| 1 | [01_sample.ipynb](01_sample.ipynb) | Footprint/splits and matched ~24-arcsec NISP/JWST field | True image footprint with connector lines; v07 |
| 2 | [02_sample_properties.ipynb](02_sample_properties.ipynb) | Magnitude and redshift distributions | Magnitude-only COSMOS parent; v05 rendered |
| 3 | [03_diffusion_architecture.ipynb](03_diffusion_architecture.ipynb) | Corrected conditioning, UNet, training and sampling | Vector schematic draft |
| 4 | [04_residual_architecture.ipynb](04_residual_architecture.ipynb) | Shared residual trunk, legacy/direct-5× heads, dense-block detail | Vector schematic draft |
| 5 | [04_galaxy_examples.ipynb](04_galaxy_examples.ipynb) | NISP, JWST, models and residuals on the same galaxies | Wait for completed predictions |
| 6 | [05_model_comparison.ipynb](05_model_comparison.ipynb) | Matched reconstruction errors and computational cost | Budget table ready; benchmark pending |
| 7 | [06_observed_shapes.ipynb](06_observed_shapes.ipynb) | Observed e1/e2 recovery and residual distributions | Frozen legacy validation preview |
| 8 | [07_shape_dependence.ipynb](07_shape_dependence.ipynb) | Shape gains versus input S/N, magnitude and redshift; success counts | Frozen legacy validation preview |
| 9 | [08_shape_robustness.ipynb](08_shape_robustness.ipynb) | Measurement choices, adaptive moments and failure coverage | Frozen legacy validation preview |

Figures 8–9 may move to an appendix or be combined. Diffusion sample variability
and the individual-versus-mean-image comparison can become an additional figure
after the full run finishes; their place in the story is not fixed yet.

The sample overview and property distributions are now **two separate figures**,
each in one row, without figure/panel titles, numbered Figures 1 and 2.
The zoomed overview field is cached in `inputs/sample_region_v06.npz`, with
source checksums and extraction metadata in the accompanying JSON. To rebuild,
set `MOSAIC_ROOT` in `01_sample.ipynb` on the machine holding the original mosaics.
`sample_region.py` extracts 82×82 native
NISP pixels and an aligned 410×410 JWST field (about 24 arcsec across). The
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
otherwise from the sibling `supernisp_nir_y_f115w_v1` folder or, when that package
is absent, the local `data/ready/supernisp_nir_y_f115w_v1` folder.

The notebooks read metadata, observed images and completed saved diagnostics. They contain
no model construction, checkpoint loading, training, inference or W&B API calls.
They can run while the GPUs are busy. Architecture labels are checked against
recorded source hashes without importing the models.

## Scientific boundaries for the drafts

- Figure 1 uses metadata, including test positions/counts for the split design.
  The separate magnitude/redshift figure includes train, validation and test
  catalog distributions at the user's request; this does not evaluate test
  predictions or change the split. The overview
  shows a field within the selected validation block and performs no held-out
  evaluation. This is an optically
  selected COSMOS subset, not a NISP-complete galaxy sample.
- Results previews are explicitly labeled **validation / legacy baseline**.
  They do not automatically add an unfinished model or claim a final result.
- Figure 5 waits for one aligned, physical-intensity prediction bundle from a
  separate tmux/W&B export job after model choices are ready. It does not crop
  screenshots or rerun models inside the notebook. Its proposed selection is
  based on input properties, with exact IDs saved before comparing outputs.
- Figure 6 requires per-object metrics on the same validation objects and a
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

Figure 2's [current v05](exports/fig02_sample_properties_v05.png) uses the full
COSMOS2020 catalog with **only valid ACS F814W magnitude < 25**, yielding
233,784 parent objects. There is no classification, footprint, redshift, size,
quality, or buffer cut. Gray shading sits behind train (teal), validation
(orange), and test (dashed purple), with one legend in the left panel.
The magnitude axis ends at 25; the redshift axis extends to 6. All histograms
use common bins and normalization over all valid measurements, including
values outside the displayed ranges. Missing magnitude values cannot satisfy
the magnitude cut; invalid redshifts are excluded only from that histogram,
not from the parent population. The redshift histogram has 225,567 valid
measurements. Counts, exclusions, classification counts, and source checksums
are recorded in `inputs/parent_sample_v05.json`.

The notebook is executed and the v05 PDF/caption installed in the manuscript.
The catalog checksum matches the sample manifest, and all 22,810 selected IDs
belong to the parent. To rerun elsewhere, set `COSMOS_CATALOG` or environment
variable `SUPERNISP_COSMOS_CATALOG` to the full catalog. `UPDATE_PAPER` controls
manuscript installation. The earlier footprint/classification parent v04,
its executed notebook and helper snapshot are preserved in `revisions/` and
`exports/` for comparison. The notebook explains training versus validation
versus final test evaluation.

The current [overview](exports/fig01_sample_v07.png) is exported as
`fig01_sample_v07` (PDF/SVG/PNG). It shows real NISP Y and JWST F115W observations
of the same 24.3-arcsec field centred on COSMOS2020 object **734405**, a larger
spiral in validation block `3:2`, using a shared
intensity stretch, 5-arcsec scale bars and WCS-derived direction arrows.
The field was selected by catalog size/brightness and visual inspection of
observed training/validation images, without model outputs. It illustrates
resolved structure rather than representative model performance. Change
`CENTER_OBJECT_ID` and `BLOCK` in the notebook to choose another galaxy, and
reset `JWST_PATH` if needed. The A4 tile is retained from v05, where it covered
the wider field better than the catalog-linked A3 tile.
Source mosaic checksums match the sample manifest; valid coverage is 100%
for both NISP and JWST. The notebook includes the rendered figure and can
reuse the cached extraction without reopening the mosaics. The NPZ is local
and git-ignored; copy it with its JSON sidecar when moving to another machine.
Panel (a) has its legend in the upper-right corner; the field-size label below
the paired images has been removed. v06 uses three adjacent, similarly sized
axes to reduce whitespace. The image width is halved relative to v05, with
its shared intensity stretch retained; NISP uses native pixels with nearest
neighbor display, without additional blur.
v07 removes the 180-arcsec block outline and circled-arrow annotation. Only
the true 24.3-arcsec WCS footprint remains, with two lines joining corresponding
corners to the left edge of the enlarged NISP view. The opaque legend sits
above the connectors. v07 reuses the v06 image cache unchanged.
The [v06 compact layout](exports/fig01_sample_v06.png) and notebook snapshot
are preserved.
The [v05 wider field](exports/fig01_sample_v05.png) and notebook snapshot remain
available for comparison.
The previous [v04 training field](exports/fig01_sample_v04.png) and its notebook
snapshot are preserved alongside the [v03 validation field](exports/fig01_sample_v03.png),
its cache, and its notebook/extractor snapshots in `revisions/`.
Earlier combined layouts are preserved as
[v01: accounting panel](exports/fig01_sample_v01.png) and
[v02: smaller 12-arcsec images](exports/fig01_sample_v02.png), with corresponding
notebook snapshots in `revisions/`; neither is the requested final layout.

Architecture drafts are [Figure 3: diffusion](exports/fig03_diffusion_architecture_v04.png) and
[Figure 4: residual architectures](exports/fig04_residual_architecture_v03.png).
Each also has a PDF and an editable-text SVG with the same filename stem.
Figure 3 v03 replaces the historical diffusion PNG in the manuscript with
`paper/figs/diffusion_architecture_v04.pdf` and an implementation-based caption.
It separates training, the shared U-Net, and generation; the residual-block
inset shows how NISP conditioning and timestep information enter all 14 blocks.
The notebook checks source hashes against the full run before drawing, without
constructing a model. The v01 and detailed v02 notebooks are archived in `revisions/`.
v03 keeps the topology and channel widths while removing equations, repeated
tensor dimensions, parameter counts, and operation lists from the drawing.
Full implementation details remain in the notebook notes and export provenance.
The user approved simplified v03 for the paper. Its caption and diffusion
subsection now explain the implemented model in plain scientific language;
the diffusion training table matches the recorded full run. Other historical
methods/results prose still needs revision. These changes await an Overleaf push. Exporting a later iteration requires copying its PDF into
`paper/figs/` and updating the manuscript's figure reference/caption.

The legacy-only validation previews are
[Figure 7: observed shapes](exports/fig06_observed_shapes_v01.png),
[Figure 8: property dependence](exports/fig07_shape_dependence_v01.png), and
[Figure 9: measurement robustness](exports/fig08_shape_robustness_v01.png).
Their input analyses are frozen; rerunning these notebooks will not silently
switch to newer model results. Figures 5–6 intentionally have no exported plot
until their required prediction/benchmark bundles are available. Their waiting
paths have been executed; the populated plotting paths await those real inputs.

Figure 4 v02 uses the same simplified style as Figure 3: common feature
extraction, two alternative enlargement paths, and a compact dense-block inset.
The model, trainer and loader sources match both recorded residual full runs.
The PDF is installed as `paper/figs/residual_architecture_v03.pdf` with label
`fig:resnet_model`. The caption and residual subsection explain feature reuse,
residual additions, edge/channel weighting and pixel shuffling in plain scientific
language; the training table matches the 50-epoch, batch-8, masked-L1 runs.
The original detailed notebook is preserved in `revisions/03_residual_architecture_v01.ipynb`.
Historical results figures/text remain for later review. Figure 4 has not been
pushed to Overleaf.

Latest architecture palette: Figure 3 v04 and Figure 4 v03 use stronger blue,
teal, purple and amber fills with darker text and matching colored borders.
The approved layouts, labels and captions are retained. Both PDFs are installed
in the local manuscript; earlier pale versions are preserved in exports/revisions.
These color updates have not been pushed to Overleaf.
