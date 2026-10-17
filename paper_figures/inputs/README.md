# Figure inputs

No model is executed by the notebooks. Create these artifacts in a separate
persistent export/benchmark job using frozen checkpoints. Figure 5 uses online
W&B. The user explicitly requested **local-only Figure 6**: use `--local` and
do not initialize W&B or upload its measurements/plots.
Figure 5 may use a provisional best-so-far diffusion
checkpoint while training continues; its metadata records that status.

## Figure 5: `galaxy_examples_vNN.npz` and `galaxy_examples_vNN.json`

NPZ arrays, all sharing the first axis and ordering:

- `object_ids`, `rows`: unique catalog IDs and corrected-sample rows.
- `nisp`, `nisp_valid`: `(N, 41, 41)`, sky-subtracted linear MJy/sr and bool mask.
- `bilinear`, `bilinear_valid`: `(N, 205, 205)`, linear-intensity NISP
  interpolation and its strict input-support mask (from v02 onward).
- `jwst`, `jwst_valid`: `(N, 205, 205)`, same physical units and target mask.
- `legacy`, `direct5`, `diffusion`, `diffusion_mean3`: `(N, 205, 205)` predictions
  from frozen selected checkpoints. The main grid shows the three individual
  models; the mean image is available as an optional column.
- `diffusion_samples`: `(N, 3, 205, 205)`, the three seeded diffusion draws.

JSON: `complete: true`, `split: "validation"`, `units: "MJy/sr"`, `object_ids`
matching NPZ order, `checkpoint_sha256` keyed by the four prediction names,
`selection` describing the input-only rule and its seed, `sampling_seeds`,
`nisp_pixel_scale_arcsec` (one value per object, from actual stored pixel area),
`data_hashes` and `npz_sha256`. `checkpoints` records epochs, selection metrics, source hashes, snapshot paths
and whether training is complete. `limits_mjy_sr` stores frozen display limits.
Use fixed shared per-object display scales and
symmetric residual limits; retain negative pixels and invalid support.

`galaxy_examples_selection.json` retains the first input-ranked selection.
The current observed-morphology selection is `galaxy_examples_selection_v06.json`;
pass it with `--selection paper_figures/inputs/galaxy_examples_selection_v06.json`
and reuse it for subsequent checkpoint updates. It fixes validation IDs, observed-image selection, display limits,
and diffusion noise seeds. A new bundle version preserves older predictions.
Run `code/export_paper_examples.py` with `--version vNN --output runs/figure5-vNN`
through `code/launch_paired.py`, after setting both `VERSION` and `BUNDLE_VERSION` to that fresh version in the notebook.
The export job checks for tmux/screen and online W&B before loading any model,
snapshots each `best.pt` with a stable-copy check, verifies recorded code/data
hashes, saves predictions, executes the plotting notebook and logs both the
main figure and its companion residual sheet to W&B. The notebook itself only
reads saved arrays. Use a fresh output and console directory for each update.

## Figure 6: `model_benchmark_vNN.json`

Top-level: `complete: true`, `split: "validation"`, `units: "MJy/sr"`,
`object_ids` (unique, common to every method), `population_rule`, `data_hashes`,
and `timing_protocol` (GPU, batch size, precision, warmup, repeats, whether
preprocessing and transfers are included, diffusion steps/seeds/ensemble size).

`methods` maps method keys to records with `object_ids` in the same order,
`mae_mjy_sr` and `rmse_mjy_sr` (per-object arrays of physical pixel errors on
the same valid target pixels), `inference_ms_per_object` and
`checkpoint_sha256` (or `null` for interpolation). Include legacy, direct5,
diffusion single sample and mean image, and bilinear/bicubic baselines.

Primary errors use the central 105×105 target-pixel crop (21 native pixels,
approximately 6.3 arcsec) to match Figure 5. Every method uses the same mask:
JWST validity intersected with strict support for both interpolations. No
shape-success or morphology cut is applied. Each object has equal weight;
reported RMSE is the mean of per-object pixel RMSE, not the square root of a
population-wide MSE. Values remain signed physical intensities without
clipping or per-object flux fitting. `full_field_check` stores the same errors
on the full 205×205 field for sensitivity to crop/background weighting.

`summary` stores object-weighted means, percentile 95% spatial-block bootstrap
intervals (2,000 draws), and paired reductions relative to bilinear. These
intervals are conditional on the field, split and fixed trained models.
`timing` stores all ten synchronized batch timings per method; the plot uses
their median and min/max range, not statistical confidence intervals. Two
warm-up batches precede timing on device-resident NISP inputs. Representation
conversion, diffusion noise generation and ensemble averaging are included;
disk IO, host/device transfers, metric and validity-mask calculations are not.

`code/export_model_benchmark.py` uses the immutable checkpoints recorded by
Figure 5 v06. Launch through `code/launch_paired.py` with `--local` and
`WANDB_MODE=disabled` for this user's local-only Figure 6 workflow. Local
console/progress/config files provide monitoring; the plotting notebook reads
only the completed versioned JSON bundle. Preserve the prior version when
changing checkpoints, population, error contract or timing protocol.

These are export contracts, not results. Keep large image bundles local; the
small sidecar/provenance files should accompany any shared rendered figures.

## Display names and label-only revisions

Display `legacy` as **ResNet** and `direct5` as **ResNet-D5** (direct 5×
upsampling). Preserve these internal keys, checkpoints and frozen data hashes.
Figure 5 display v07 reads bundle v06 via `BUNDLE_VERSION`. Figure 6 display
v06 reads benchmark v03 and both spectral v01 bundles through explicit input
paths in its setup cell. For a new analysis, update those paths together only
when the benchmark, masks, predictions and spectral provenance agree; the
notebook validates that agreement. For a presentation-only revision, increase
`VERSION` and reuse the existing input bundles.

## Figure 6 source-mask experiment (v03)

`code/prepare_source_masks.py` freezes masks before reconstruction errors are
computed. It reads validation JWST cutouts only, smooths with a sigma-1-pixel
Gaussian, estimates clipped smoothed sky noise at radii 70–95, detects all
8-connected regions of at least 10 pixels at 1.5, 2 and 3 sigma, fills holes,
and dilates by a radius-3 disk. Primary threshold: 2 sigma. Detection occurs
on the full 205×205 reference before the central 105×105 crop; all neighbours
are retained. Intersect with the same JWST/bilinear/bicubic validity support.

Frozen masks and their complete detection diagnostics are stored locally in
`runs/figure6-source-masks-v03/source_masks.npz` and `.json`. The benchmark embeds
the protocol and metadata and records the NPZ hash. `source_regions` contains
per-object MAE/RMSE and paired spatial-block summaries separately for source and
outside-source pixels at every threshold. Empty regions must be excluded with
explicit IDs, consistently across methods; all 2,118 objects have both regions
in this experiment. The outside-source region may contain undetected faint light.
Reference-based detection also couples mask selection to noise in the reference.

Use exporter `--source-masks`, `--reuse-timing` and `--cache-predictions` together
with `--local`. The unchanged models/sampler retain v01 inference timings, with
the original batch size and timing-bundle hash recorded. Central signed
predictions and the reference are cached locally in `predictions.h5` for future
mask-only revisions without another inference run. Cache completeness and hash
are recorded; do not use the incomplete batch-size-two attempt.

## Spatial Fourier companion: `spatial_spectra_v01.json` / `.npz`

CPU-only `code/analyze_spatial_spectra.py` reads the immutable v03 prediction
cache, frozen source masks and object/block IDs. Main curves use complete-valid
cutouts only; the same objects and windows are used for every model. The JSON
stores the predeclared contract, source/input/array hashes, exclusion IDs,
per-case summaries, paired diffusion-minus-residual differences, and fixed-band
reference-contrast subset IDs. The local NPZ retains all per-object powers,
reference powers, signed cross powers, log ratios and correlations for every
window, plus window areas and validity/background coverage flags. Undefined
background measurements are NaN in the NPZ and explicitly excluded in summaries.

Primary power ratios are geometric means; signed Fourier correlations are
arithmetic means. Bounds are paired 2,000-draw spatial-block percentile intervals.
The outside-source spectrum is a diagnostic proxy, not a calibrated noise PSD.
No noise subtraction or PSF deconvolution is applied. No new model inference
or external upload is performed. These saved results now feed the consolidated `06_model_comparison.ipynb` (v04);
the original whole-cutout notebook is archived in `revisions/`.

## Strict-source Fourier companion: `spatial_spectra_source_v01.json` / `.npz`

Same frozen v03 predictions, 2σ masks, object/block IDs, bins and bootstrap as
the whole-cutout comparison. Cases `source3`, `source1`, `source5` are strictly
zero outside source ∩ validity, with inward cosine taper widths 3/1/5 pixels
and outer Tukey alpha 0.5. `whole` and `background` reproduce prior controls.
The primary case is `source3`; `all_source3` includes incomplete fields using
valid source support. Reference/background contrast uses the source3 reference
numerator, so its reference-only subsets differ from the whole-cutout subsets.
No model-based cuts. Contract, hashes, cases, exclusions and paired summaries
are in JSON; all per-object powers/correlations/window areas are in local NPZ.
The consolidated `06_model_comparison.ipynb` (v04) plots these saved source and
outside-source measurements alongside benchmark v03 RMSE. The original source-only
notebook is archived in `revisions/`.

## Cached shape comparison: `cached_shapes_v01.json` / `.npz`

Frozen v03 predictions and common-valid masks; no inference or segmentation-based
shape clipping. Existing fixed moments and GalSim HSM adaptive moments are
measured for JWST and all six methods. JSON contains the frozen contract, per-
estimator and cross-estimator common-success summaries/IDs, full success counts,
overlapping failure reasons, paired block RMSE differences and provenance hashes.
NPZ retains all per-object measurements and fit statuses. Primary plots use the
matched sample within each estimator and report full-denominator success beside
errors. Notebook `07b_model_shapes.ipynb`; local-only analysis with no W&B usage.

## Exploratory PSNR and SSIM (v01)

`psnr_ssim_v01.json` stores the fixed protocol, input/source hashes, per-method
means and paired 95% spatial-block intervals, plus paired differences relative
to bilinear for all-valid, source and outside-source regions. The companion
local `.npz` stores per-object PSNR/SSIM, shared JWST ranges, retained SSIM window
counts, object IDs and spatial blocks. Produced by `code/analyze_psnr_ssim.py`
from the original v03 prediction cache and 2-sigma masks. The premeasurement
contract and source snapshots are in `runs/psnr-ssim-v01/`.

All 2,118 objects retain valid 7×7 windows in all three regions. PSNR reproduces
the benchmark's regional RMSE with a shared per-object reference range. SSIM
is averaged only over fully contained windows, without zero-masking sources.
The consolidated `06_model_comparison.ipynb` reads this optional bundle to
produce the separate `fig06_psnr_ssim_v01` review plot. No inference or uploads.
