# Figure 6: source-mask experiment

Local review experiment requested by the user after identifying that the
original MAE/RMSE include both source and background pixels. No W&B or Overleaf
upload is authorized for Figure 6. This document supplements the unchanged
all-pixel benchmark in `figure6_reconstruction_benchmark.md`.

## Definition fixed before evaluating reconstruction errors

Reference: signed, sky-subtracted JWST F115W, all 2,118 validation cutouts.
Construct masks on the full 205×205 cutout and crop only afterward to the same
central 105×105 field as the original Figure 6. Smooth with a Gaussian of sigma
1 target pixel (truncated at 4 sigma); require essentially complete valid
kernel support. Estimate the smoothed sky median and standard deviation with
3-sigma clipping for five iterations in the annulus at radii 70–95 pixels.

Detect every eight-connected region with at least 10 pixels above sky median
plus 2 times the smoothed sky standard deviation. Fill enclosed holes and
expand each region by a circular radius of 3 target pixels (approximately
0.18 arcsec). Include all detected neighbours; no deblending is needed for the
union of source pixels. These operations follow the threshold/connected-region
and dilated-mask approach described in the
[Photutils segmentation documentation](https://photutils.readthedocs.io/en/stable/user_guide/segmentation.html).
The numerical parameters are exploratory choices, not fitted to model rankings.

Intersect with the identical common-valid support used for the original
benchmark. Source and outside-source masks partition those valid pixels.
Measure MAE and RMSE on original, unsmoothed, signed physical intensities:
the detection median does not change images used in evaluation. Give each
object equal weight; calculate RMSE per object before averaging. Use the same
2,000 paired spatial-block bootstrap draws over the 18 blocks.

Sensitivity thresholds of 1.5 and 3 sigma were frozen alongside the primary
2-sigma mask. Median source-area fractions are 31.89%, 18.85% and 10.92%,
respectively. All 2,118 objects retain both nonempty source and outside-source
regions at every threshold; no object is excluded from these comparisons.

## Interpretation

This is an equal-pixel source mask, not intensity weighting. It reduces the
influence of the large number of sky pixels without concentrating the score
on bright cores. Detection and dilation can still miss faint outskirts or
include positive noise fluctuations. Outside-source pixels are not guaranteed
to be source-free sky. Mask selection uses the noisy evaluation reference,
so these are conditional source-region agreement metrics, not noise-free
reconstruction errors or a galaxy morphology measurement.

The checkpoints, validation population, physical units, crop and diffusion
seeds match the original benchmark. The 50-step diffusion sampler is evaluated
for the same three seeds, with the first draw and linear mean of three retained.
Original inference timings are reused, because masks are part of evaluation
and the inference algorithms have not changed. No new timings are measured
while the GPUs are sharing other workloads.

## Reproduction and artifacts

Masks: `runs/figure6-source-masks-v03/source_masks.npz` and `.json`.
Preparation: `code/prepare_source_masks.py`; algorithm: `code/source_masks.py`.
Benchmark: `code/export_model_benchmark.py --local --source-masks ...
--reuse-timing paper_figures/inputs/model_benchmark_v01.json --cache-predictions`.
Active run: `runs/figure6-v03-b8`; console: `runs/figure6-v03-b8-console`.
An initial batch-size-two attempt was stopped for throughput and has an
incomplete cache; use only the completed batch-size-eight run.

The new benchmark saves per-object errors at all three thresholds, mask and
checkpoint hashes, source snapshots, original timing provenance and a central
prediction cache for future mask revisions without repeated inference.
Plot notebook: `paper_figures/06_model_comparison.ipynb`, display/bundle v03.
Previous display v02 is archived in `paper_figures/revisions/`.

## Results

All 2,118 objects are retained in both regions. These are means of per-object errors.

| Method | Source MAE | Source RMSE | Outside-source MAE | Outside-source RMSE |
| --- | ---: | ---: | ---: | ---: |
| Bilinear | 0.028902 | 0.110684 | 0.013389 | 0.017103 |
| Bicubic | 0.028897 | 0.107888 | 0.014279 | 0.018246 |
| Diffusion (1 sample) | 0.026297 | 0.081233 | 0.013916 | 0.018152 |
| Diffusion (mean of 3) | 0.022781 | 0.069622 | 0.012050 | 0.015720 |
| ResNet | 0.020306 | 0.060173 | 0.010863 | 0.013957 |
| ResNet-D5 | 0.020247 | 0.059152 | 0.010860 | 0.013985 |

Errors are in MJy/sr.

| Method | All-pixel MAE gain | Source MAE gain | Outside-source MAE gain | All-pixel RMSE gain | Source RMSE gain | Outside-source RMSE gain |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Bilinear | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% |
| Bicubic | -3.52% | 0.02% | -6.65% | 1.47% | 2.53% | -6.68% |
| Diffusion (1 sample) | 4.51% | 9.01% | -3.93% | 26.90% | 26.61% | -6.13% |
| Diffusion (mean of 3) | 17.39% | 21.18% | 10.00% | 37.25% | 37.10% | 8.08% |
| ResNet | 26.05% | 29.74% | 18.87% | 46.11% | 45.63% | 18.39% |
| ResNet-D5 | 26.16% | 29.95% | 18.89% | 46.95% | 46.56% | 18.23% |

Positive gains mean lower error relative to bilinear; all comparisons use the same population.

## Threshold sensitivity

| Threshold | Method | Source MAE gain | Source RMSE gain |
| --- | --- | ---: | ---: |
| 1.5 sigma | Bicubic | -1.15% | 2.21% |
| 1.5 sigma | Diffusion (1 sample) | 8.31% | 26.94% |
| 1.5 sigma | Diffusion (mean of 3) | 20.41% | 37.32% |
| 1.5 sigma | ResNet | 28.79% | 45.85% |
| 1.5 sigma | ResNet-D5 | 28.96% | 46.77% |
| 2 sigma | Bicubic | 0.02% | 2.53% |
| 2 sigma | Diffusion (1 sample) | 9.01% | 26.61% |
| 2 sigma | Diffusion (mean of 3) | 21.18% | 37.10% |
| 2 sigma | ResNet | 29.74% | 45.63% |
| 2 sigma | ResNet-D5 | 29.95% | 46.56% |
| 3 sigma | Bicubic | 1.32% | 2.84% |
| 3 sigma | Diffusion (1 sample) | 10.15% | 26.39% |
| 3 sigma | Diffusion (mean of 3) | 22.61% | 36.99% |
| 3 sigma | ResNet | 31.49% | 45.51% |
| 3 sigma | ResNet-D5 | 31.74% | 46.45% |

The learned-model source ordering is unchanged at all three thresholds:
ResNet-D5, ResNet, diffusion mean of three, and single-sample diffusion, in
increasing mean error. This ordering does not establish a significant difference
between the two residual networks. Bicubic's source MAE comparison crosses zero
with threshold and is not robustly better than bilinear.

Single-sample diffusion illustrates why the separation helps: its source MAE
improves by 9.01%, while outside-source MAE worsens by 3.93%. The original
all-pixel average combines these into a 4.51% gain. Source RMSE gains remain
similar to the original all-pixel results.

## Local outputs

- Main figure: `paper_figures/exports/fig06_model_comparison_v03.pdf` / `.png`.
- Source distributions: `paper_figures/exports/fig06_error_distributions_v03.pdf` / `.png`.
- Source/background comparison: `paper_figures/exports/fig06_source_background_v03.pdf` / `.png`.
- Mask examples: `paper_figures/exports/fig06_source_masks_v03.pdf` / `.png`.
- Frozen benchmark: `paper_figures/inputs/model_benchmark_v03.json`.
- Central prediction cache: `runs/figure6-v03-b8/predictions.h5`.
- Verification: `runs/figure6-v03-b8/verification.json`.


## Verification

Run completed with exit 0. Four analytical CPU tests passed, covering signed
masked errors, source/background error partitioning, neighbour detection with
invalid support, and paired object-weighted block resampling. All per-object
source/background metrics at all three thresholds reproduce from the cached
predictions and masks. Paired bootstrap summaries reproduce exactly. All-pixel
scores match v01 to numerical precision (maximum absolute difference
8.89e-16 MJy/sr). Timings are reused exactly, with their source-bundle hash.
Checkpoint, source snapshot, input, cache, notebook and export hashes were
verified. All four plots were visually reviewed. The masks were additionally
reviewed on 12 randomly selected cutouts before final results were available.
The verification script and software environment are preserved with the run.
