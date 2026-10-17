# Figure 6: matched reconstruction accuracy and inference time

Local review draft, 2026-10-16. User requested accuracy and speed together and
explicitly chose to keep Figure 6 local. No W&B run or upload was performed.

Main figure: `paper_figures/exports/fig06_model_comparison_v02.pdf` / `.png`.
Companion distributions and paired gains:
`paper_figures/exports/fig06_error_distributions_v02.pdf` / `.png`.

A subsequent [source-mask experiment](figure6_source_mask_experiment.md) is
available as display/bundle v03. This report preserves the original all-pixel
results and v02 plots; the new main figure uses source-only errors and retains
these inference timings.

## Population and measurement

All **2,118 validation galaxies**, with no additional morphology or shape-success
cuts. Checkpoints are the immutable Figure 5 v06 snapshots: epoch 49 for both
residual models, epoch 35 for diffusion. Each method is evaluated on exactly the
same pixels per object: observed JWST validity intersected with bilinear and
strict bicubic input support. Primary measurements use the central 105×105
target pixels, equivalent to 21 native NISP pixels or about 6.3 arcsec, matching
the example figure. Full 205×205-field measurements are saved as a sensitivity
check. Intensities are signed, sky-subtracted MJy/sr without clipping, flux
fitting or band conversion. Interpolation retains the NISP band and response.

MAE and RMSE are first computed within each object, then averaged with equal
object weights. Thus mean RMSE is **not** a pooled population RMSE. Accuracy
intervals use 2,000 paired resamples of the 18 spatial blocks, preserving all
objects within selected blocks. Percentile intervals are conditional on this
field, sample and fixed trained models.

## Results

| Method | Mean MAE [MJy/sr] | Mean pixel RMSE [MJy/sr] | MAE reduction vs bilinear | RMSE reduction vs bilinear | Median ms/object |
| --- | ---: | ---: | ---: | ---: | ---: |
| Bilinear | 0.018034 | 0.067231 | 0% | 0% | 0.00315 |
| Bicubic | 0.018669 | 0.066244 | −3.52% | 1.47% | 0.00578 |
| Diffusion, one draw | 0.017220 | 0.049145 | 4.51% | 26.90% | 125.31 |
| Diffusion, mean of three | 0.014898 | 0.042188 | 17.39% | 37.25% | 377.99 |
| ResNet | 0.013337 | 0.036232 | 26.05% | 46.11% | 0.4717 |
| ResNet-D5 | 0.013316 | 0.035668 | 26.16% | 46.95% | 0.6382 |

The ordering of the learned methods is unchanged for full-field errors.
Full-field MAE/RMSE reductions are 6.91%/26.69% for diffusion, 19.21%/36.07%
for its mean of three, 27.38%/45.30% for ResNet and 27.44%/46.14% for
ResNet-D5. Bicubic changes from a small central RMSE improvement to a
0.59% full-field RMSE degradation, so its comparison depends on the crop.

## Runtime definition

NVIDIA L40S, float32, no autocast or TF32, batch size eight. Two warm-up batches,
then ten wall-clock timings bracketed by CUDA synchronization. Divide batch
times by eight and report the median, with min/max whiskers. This is amortized
batch time, not single-request latency; the whiskers are not confidence
intervals. Timings include representation conversion, diffusion noise creation,
physical-intensity output conversion and ensemble averaging. They exclude
loading data/models, host/device transfers, metrics and interpolation support
mask construction. The timing batch IDs and all repeats are saved.

Diffusion uses 50 DDIM sampling steps and object-specific seeds 314159, 314160,
314161. The single sample uses the first seed. The ensemble averages the three
images in linear physical intensity before measuring errors.

## Interpretation and verification

These are validation-selected checkpoints with one training seed per model.
They differ in representation, loss, capacity and inference process, so this
is a comparison of the trained configurations rather than an isolated
architecture experiment. Errors assess pixel agreement with a noisy observed
reference across different bands and PSFs. They do not establish calibrated
shape recovery, shear accuracy or the reality of individual generated features.
The held-out test set remains untouched.

Two analytical CPU regression checks passed for signed/masked errors and paired
block resampling with unequal block sizes. Run completed with exit 0. Checkpoint,
code, data, bundle, notebook and export hashes passed verification. All 120
central/full-field errors checked against the five cached Figure 5 examples
agree within 2.11e-8 MJy/sr (small GPU batching differences). All validation IDs
are unique and have positive common support. Both figures were visually reviewed.

Data/provenance: `paper_figures/inputs/model_benchmark_v01.json`.
Run, source snapshots and verification: `runs/figure6-v01/`.
Console: `runs/figure6-v01-console/console.log`.
Reproduction uses `code/export_model_benchmark.py --local` in tmux with
`WANDB_MODE=disabled`, a fresh output/version, and the corresponding notebook
`VERSION` and `BENCHMARK_VERSION`. Display v02 renames the direct 5× variant
to ResNet-D5 and reuses the unchanged benchmark v01. These drafts have not been inserted into the manuscript or pushed.
