# Paper figures workspace

One notebook per proposed figure. Start with the figure, agree on its content
and visual choices, then refine the caption and manuscript text. Figure 2 contains the sample distributions; the diffusion and residual
architectures follow as Figures 3 and 4. The architecture notebooks are now `03_diffusion_architecture.ipynb` and
`04_residual_architecture.ipynb`; all active notebook filenames now follow the Figure 1–9 sequence.

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
| 4 | [04_residual_architecture.ipynb](04_residual_architecture.ipynb) | Shared residual trunk, ResNet/ResNet-D5 heads, dense-block detail | Named variants; v05 |
| 5 | [05_galaxy_examples.ipynb](05_galaxy_examples.ipynb) | NISP, JWST, models and residuals on the same galaxies | Isolated source first, then pairs/disk/crowded field; v07 |
| 6 | [06_model_comparison.ipynb](06_model_comparison.ipynb) | Matched reconstruction errors and computational cost | Source/background accuracy and speed; local-only v03 |
| 7 | [07_observed_shapes.ipynb](07_observed_shapes.ipynb) | Observed e1/e2 recovery and residual distributions | Frozen legacy validation preview |
| 8 | [08_shape_dependence.ipynb](08_shape_dependence.ipynb) | Shape gains versus input S/N, magnitude and redshift; success counts | Frozen legacy validation preview |
| 9 | [09_shape_robustness.ipynb](09_shape_robustness.ipynb) | Measurement choices, adaptive moments and failure coverage | Frozen legacy validation preview |

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

## Current model names

Use **ResNet** for the two-stage residual network and **ResNet-D5** for the
direct 5× upsampling variant. Figure 4 v05, Figure 5 v07 and Figure 6 v02
apply these names throughout labels and captions:
[architecture](exports/fig04_residual_architecture_v05.png),
[examples](exports/fig05_galaxy_examples_v07.png),
[residuals](exports/fig05_galaxy_residuals_v07.png),
[accuracy and speed](exports/fig06_model_comparison_v02.png), and
[error distributions](exports/fig06_error_distributions_v02.png).
These are label-only revisions: Figure 5 reuses bundle v06 (`BUNDLE_VERSION`),
and Figure 6 reuses benchmark v01 (`BENCHMARK_VERSION`). Internal `legacy`
and `direct5` keys and archived snapshots retain their original identities.
The manuscript uses the new architecture and example PDFs; Figure 6 remains local.
The sections below retain the history of earlier figure versions.

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
- Shape-results previews are explicitly labeled **validation / legacy baseline**.
  Figure 5 is a provisional best-checkpoint visual comparison with frozen inputs.
- Figure 5 reads a versioned physical-intensity prediction bundle from a
  separate tmux/W&B export job. Checkpoints, input-selected galaxy IDs, scales
  and noise seeds are frozen for review; diffusion may still be training.
  The notebook displays arrays and does not rerun models.
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

Architecture drafts are [Figure 3: diffusion](exports/fig03_diffusion_architecture_v05.png) and
[Figure 4: residual architectures](exports/fig04_residual_architecture_v04.png).
Each also has a PDF and an editable-text SVG with the same filename stem.
Figure 3 v03 replaces the historical diffusion PNG in the manuscript with
`paper/figs/diffusion_architecture_v05.pdf` and an implementation-based caption.
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
[Figure 7: observed shapes](exports/fig07_observed_shapes_v01.png),
[Figure 8: property dependence](exports/fig08_shape_dependence_v01.png), and
[Figure 9: measurement robustness](exports/fig09_shape_robustness_v01.png).
Their input analyses are frozen; rerunning these notebooks will not silently
switch to newer model results. Figure 6 awaits its matched prediction/benchmark bundle. Figure 5 now uses
versioned best-checkpoint exports; its earlier waiting notebook is archived.

Figure 4 v02 uses the same simplified style as Figure 3: common feature
extraction, two alternative enlargement paths, and a compact dense-block inset.
The model, trainer and loader sources match both recorded residual full runs.
The PDF is installed as `paper/figs/residual_architecture_v04.pdf` with label
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

The current architecture versions (Figure 3 v05 and Figure 4 v04) share
`architecture_style.py`: blue for learned operations, teal for NISP inputs
and conditioning, yellow for JWST/F115W images, purple for diffusion steps
and timestep information, and gray for other operations. The shared palette
is included in both export provenance records.

The current architecture PDFs, captions, method subsections and bibliography
updates were pushed to Overleaf in commit `f16fbf5` at the user’s request.

The result notebooks, headings, and current export stems are numbered 5–9.
Earlier notebook names are archived in `revisions/*_before_renumber.ipynb`;
old export files are retained as historical versions. The manuscript uses
automatic LaTeX figure numbering and has no duplicate counter assignments.

## Figure 5: first current-checkpoint comparison

[Figure 5 v01](exports/fig05_galaxy_examples_v01.png) and its
[companion residual sheet](exports/fig05_galaxy_residuals_v01.png) show four
validation galaxies (IDs 992795, 1024714, 463235, 1106391). ResNet and direct-5×
use their best epoch-49 checkpoints; diffusion uses the best available epoch-35
checkpoint, snapshotted while training continued. These are visual examples,
not a final model ranking. The main grid uses 21-native/105-target-pixel central
crops (~6.3 arcsec), identical per-row stretches, and 2-arcsec scale bars.

The input-brightness selection and display scales are frozen in
`inputs/galaxy_examples_selection.json`. Predictions and checkpoint provenance
are in `inputs/galaxy_examples_v01.npz` and its JSON sidecar. The saved bundle
also contains three independently seeded diffusion samples and their linear
mean, available as an optional notebook column. Snapshot files remain under
`runs/figure5-v01/`. The plot notebook constructs no models.

The completed job ran in `supernisp-figure5-v01` with
[online W&B monitoring](https://wandb.ai/AI-Astro/superNISP-corrected/runs/napgcdal),
including both figures and a versioned prediction artifact. See `inputs/README.md`
for the checkpoint-update workflow. This draft has not replaced the historical
Results figures in the manuscript and has not been pushed to Overleaf.

## Figure 5 v02: reordered columns and clearer morphology examples

The current [Figure 5](exports/fig05_galaxy_examples_v02.png) is ordered
**NISP Y, JWST F115W, Bilinear, Diffusion, ResNet, ResNet-D5**.
The [residual sheet](exports/fig05_galaxy_residuals_v02.png) follows the same
method order. Bilinear uses NISP in linear MJy/sr and its own interpolation
support mask, without a band conversion.

At the user's request, the examples were replaced by four bright validation
galaxies selected visually from observed NISP/JWST pairs: 734405 (barred spiral),
677525 (more face-on disk), 671723 (edge-on disk), and 679385 (compact smooth
source). The candidate pool was the 60 brightest validation objects satisfying
F814W<23 and 5<catalog FLUX_RADIUS<25. No model predictions or shape-success
cuts entered this new selection. These illustrate morphology and are not a
representative random sample. The frozen IDs, candidate pool and shared
observed-image display rule are in `inputs/galaxy_examples_selection_v02.json`.

Epochs remain ResNet 49, direct5 49 and diffusion 35. The completed export used
tmux `supernisp-figure5-v02`, with
[W&B wh0bth5m](https://wandb.ai/AI-Astro/superNISP-corrected/runs/wh0bth5m).
V01's notebook, selection and predictions are preserved. Future checkpoint
updates should reuse the v02 selection via the exporter's `--selection` option.
The v02 figures are not yet inserted into the manuscript or pushed to Overleaf.

## Figure 5 v03: typical replacements for rows a and c

[Current Figure 5](exports/fig05_galaxy_examples_v03.png) keeps rows b and d
exactly as in v02 (all arrays, checkpoint hashes and display limits verified).
Rows a/c now use validation objects 722361 and 1033198, with F814W magnitudes
22.88/23.11 and catalog radii 5.03/4.84 pixels. They were selected from observed
NISP/JWST pairs in a seeded 40-object pool with 22.5<F814W<24 and
4.3<FLUX_RADIUS<6, before inspecting predictions. These are closer to the sample
middle in size/brightness; the four-panel selection is still illustrative.

The six-column order is unchanged. [Residuals](exports/fig05_galaxy_residuals_v03.png)
use the same new rows. Reuse `inputs/galaxy_examples_selection_v03.json` for
future updates. The export completed in tmux `supernisp-figure5-v03`, with
[W&B p9068svz](https://wandb.ai/AI-Astro/superNISP-corrected/runs/p9068svz).
The notebook and previous selection are archived; v03 has not been pushed.

## Figure 5 v04: replacement c and fifth row

[Current Figure 5](exports/fig05_galaxy_examples_v04.png) retains a/b/d exactly
from v03, replaces c with a modest inclined galaxy (504234), and adds an
asymmetric compact galaxy as e (1126532). Both new objects were chosen using
only the observed NISP/JWST images in the same seeded 40-object candidate pool.
Their F814W magnitudes are 22.61/22.80 and catalog radii 4.92/5.18 pixels.
The selection is illustrative; apparent sharpness does not establish accurate
shape recovery. Shared per-row intensity scales and column order are unchanged.

The [residual sheet](exports/fig05_galaxy_residuals_v04.png) also has five rows.
Reuse `inputs/galaxy_examples_selection_v04.json` for subsequent exports.
Checkpoint epochs remain 49/49/35. All retained-row arrays match v03 exactly;
source, input and export hashes were verified, and the main grid visually reviewed.
Completed in tmux `supernisp-figure5-v04` with
[W&B xqwixuq9](https://wandb.ai/AI-Astro/superNISP-corrected/runs/xqwixuq9), exit 0.
V03 notebook archived. V04 has not yet been inserted into the manuscript.

## Figure 5 v05: pairs and crowded fields

The [observed-image preview](exports/fig05_observed_selection_v05.png) retains
the face-on disk in b and replaces a/c/d/e with close projected pairs,
overlapping elongated sources and a crowded field. IDs are 749105, 677525,
592591, 744769, 997288. The new examples were chosen before inspecting their
predictions from 64 observed-only candidate fields; this is an illustrative
selection, not a statistically representative sample. Search details and
reproduction commands are in [the candidate record](inputs/figure5_multiple_sources_v05/README.md).

[Current Figure 5](exports/fig05_galaxy_examples_v05.png) and its
[residual sheet](exports/fig05_galaxy_residuals_v05.png) are exported, with active
notebook VERSION v05. All b arrays and display scales exactly match v04.
Checkpoint hashes remain unchanged (epochs ResNet49/direct5 49/diffusion35);
all training runs are now complete. Selection, input, source and export hashes
were verified and both figures visually reviewed.

The user explicitly approved the W&B upload. Completed tmux
`supernisp-figure5-v05`, exit 0, with
[W&B 50peiezt](https://wandb.ai/AI-Astro/superNISP-corrected/runs/50peiezt).
Reuse `inputs/galaxy_examples_selection_v05.json` for subsequent updates.
V04 notebook is archived. V05 has not been inserted into the manuscript.

## Figure 5 v06: isolated source first

[Current Figure 5](exports/fig05_galaxy_examples_v06.png) replaces v05 row e
with the modest isolated galaxy 504234 from v04 and puts it first. V05 rows a-d
follow in their original relative order. The face-on disk is now c; the crowded
field is now e. All image arrays and display scales exactly match the source
rows, with the same checkpoint hashes. The caption describes this as an editorial
rearrangement of previously viewed examples.

[Residuals](exports/fig05_galaxy_residuals_v06.png) use the same row order.
Reuse `inputs/galaxy_examples_selection_v06.json` for future updates. Active
notebook VERSION v06; v05 archived. Export and provenance checks passed, and
the main grid was visually reviewed. Completed tmux `supernisp-figure5-v06`,
exit 0, with [W&B 1kwf3s0z](https://wandb.ai/AI-Astro/superNISP-corrected/runs/1kwf3s0z).
The user approved v06 for the manuscript. Its exact PDF is installed at
`paper/figs/galaxy_examples_v06.pdf` as Figure 5, label `fig:galaxy_examples`,
with a caption describing the fields, columns, display scale and selection.
It replaces both historical image galleries; associated references and visual
descriptions are updated. The other historical numerical results still need
revision. PDF checksum, figure order and references were checked, and the figure
with its caption compiles in a standalone one-page proof. Full manuscript
compilation remains blocked by missing `etex.sty`. The local Overleaf archive
is refreshed. The user subsequently requested publication: Figure 5 and both
manuscript sources were pushed to Overleaf in commit `13ff60c`. All 36 publication
files match the synchronized checkout. Overleaf compilation was not inspected.

## Figure 6 v01: accuracy and speed, local only

[Figure 6](exports/fig06_model_comparison_v01.png) compares mean per-object MAE,
mean per-object pixel RMSE and measured inference time for bilinear, bicubic,
single-sample diffusion, the mean of three diffusion samples, ResNet and
ResNet-D5. All 2,118 validation objects use identical support within the
central 105×105 target-pixel crop, matching Figure 5. Accuracy whiskers are 95%
paired spatial-block bootstrap intervals; timing whiskers are the min/max of
ten repetitions, with median points, on an L40S at batch size eight.

The [companion sheet](exports/fig06_error_distributions_v01.png) shows error CDFs
and paired reductions relative to bilinear. The residual models reduce mean
MAE by about 26% and mean pixel RMSE by 46–47%; their inference times are about
0.47/0.64 ms per object versus 125 ms for one diffusion draw and 378 ms for the
mean of three. These are pixel agreement and runtime results, not a shape
recovery ranking. See [the benchmark report](../paper/figure6_reconstruction_benchmark.md)
for exact values, definitions and limitations.

Frozen inputs: `inputs/model_benchmark_v01.json`. Exporter:
`code/export_model_benchmark.py`; pure metric/bootstrap helper and regression
checks: `code/reconstruction_benchmark.py`, `code/test_reconstruction_benchmark.py`.
Run `runs/figure6-v01` completed in tmux with exit 0. All hashes and 120 cached
Figure5 error comparisons passed; figures visually reviewed. The user chose
**Keep Figure 6 local**: no W&B initialization or upload. Run with `--local`
and `WANDB_MODE=disabled`. The figures are review drafts, not inserted or pushed.

## Figure 6 v03: source masks and separate background errors

[Current Figure 6](exports/fig06_model_comparison_v03.png) uses source-only
MAE and RMSE alongside the original inference timings. The
[source/background comparison](exports/fig06_source_background_v03.png)
shows paired gains for all valid pixels, source pixels and outside-source
pixels. Also available: [source distributions](exports/fig06_error_distributions_v03.png)
and [exact masks for Figure 5 examples](exports/fig06_source_masks_v03.png).

Masks are derived only from JWST, include neighbouring sources, and are
identical for every method. Primary detection uses 2 sigma, at least 10
connected pixels, and a 3-pixel dilation; see the full definition and caveats
in [the experiment report](../paper/figure6_source_mask_experiment.md).
All 2,118 validation objects retain both regions. Median source coverage is
18.85%. Results at 1.5 and 3 sigma preserve the learned-model source ordering.

Single-sample diffusion improves source MAE by 9.01% relative to bilinear,
while outside-source MAE is 3.93% worse. ResNet-D5 source gains are 29.95%
(MAE) and 46.56% (RMSE); ResNet is close at 29.74% and 45.63%.
This separates source reconstruction from background behaviour.
The v02 notebook and original all-pixel plots are preserved.

Frozen bundle: `inputs/model_benchmark_v03.json`. Run: `runs/figure6-v03-b8`;
console: `runs/figure6-v03-b8-console`. The central predictions are now cached
for future mask-only revisions without rerunning inference. No Figure 6
artifacts have been uploaded to W&B or Overleaf.

## Figure 6 companion: spatial structure (local v01)

The [Fourier diagnostic](exports/fig06_spatial_structure_v01.png) measures power
ratios and signed reference cross-correlation from the existing v03 cache.
[Window checks](exports/fig06_spectral_window_checks_v01.png) and an
[outside-source control](exports/fig06_spectral_background_v01.png) accompany it.
Notebook: [06b_spatial_structure.ipynb](06b_spatial_structure.ipynb).
The source-error/speed Figure 6 remains unchanged.

Primary curves use 1,363 complete-valid cutouts; a validity-taper check includes
all 2,118. Diffusion has power amplitudes closer to observed JWST but lower
reference correlation than the residual networks. This trade-off persists with
alternative tapers, a soft source weight and reference-contrast selections.
In the 944-object reference-contrast subset at 6–15 target pixels (~0.36–0.89
arcsec), single diffusion / ResNet-D5 have power ratios 0.884 / 0.443 and
correlations 0.643 / 0.783. Ideal values are 1 for both metrics.

Diffusion also produces substantial uncorrelated background power. Neither
raw power agreement nor the background proxy establishes noise-free morphological
recovery. Full protocol, numbers, caveats and proposed wording are in the
[spatial-structure report](../paper/figure6_spatial_structure.md).
Run `runs/spectra-v01` completed locally on CPU with no new inference or uploads.

## Figure 6 companion: strict source-only spatial structure (local v01)

[Source-only Fourier plot](exports/fig06_source_spatial_structure_v01.png),
[boundary checks](exports/fig06_source_window_checks_v01.png), and
[notebook](06c_source_spatial_structure.ipynb) repeat the comparison on exactly
zero-outside-source windows. The same 1,363 fields are used; masks include all
detected neighbours. At 0.36–0.89 arcsec, diffusion / ResNet-D5 power ratios are
0.912 / 0.419, and correlations 0.604 / 0.738. Source masking raises correlation
but preserves the power-versus-alignment trade-off. Boundary checks show that
finest-scale values depend on the mask taper. See the
[full protocol and interpretation](../paper/figure6_source_spatial_structure.md).
All outputs remain local; the main Figure 6 and previous companion are preserved.

## Matched observed shapes for all models (local v01)

[Shape error and success summary](exports/fig07_model_shapes_v01.png) and
[component agreement](exports/fig07_model_shape_agreement_v01.png) compare the
frozen Figure 6 predictions using fixed and adaptive moments. Notebook:
[07b_model_shapes.ipynb](07b_model_shapes.ipynb). ResNet and ResNet-D5 have nearly
identical errors and outperform single diffusion and the mean-of-3 image with
both estimators. The primary common-success populations contain 1,447 and 1,546
objects; the shared 1,313-object check preserves the ordering. Full success counts
use all 2,118 attempted fields. These are observed cross-band shapes, not shear.
See [the report and proposed figure selection](../paper/model_shape_comparison.md).
Older shape notebooks/results remain preserved; no manuscript changes or uploads.
