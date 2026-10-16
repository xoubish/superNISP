# Validation shape comparisons during full training

2026-10-16. This analysis runs independently of training, using an immutable
copy of a completed checkpoint epoch. It does not alter model selection or
use held-out test examples. The first snapshot is epoch 11 of
[full training nifpxes6](https://wandb.ai/AI-Astro/superNISP-corrected/runs/nifpxes6).

## Implemented and verified

`code/shape_diagnostics.py` computes the same signed, sky-subtracted fixed
Gaussian-weighted moments as the trainer on the target grid. It adds explicit,
overlapping failure flags: nonfinite aperture input, incomplete aperture,
nonpositive weighted flux, nonpositive Qxx/Qyy, nonpositive determinant, and
nonfinite ellipticity. It retains failed objects and their numeric moments.
The tests confirm exact agreement with the existing target-grid implementation.

Native NISP uses five target-pixel coordinate units per native pixel, with the
same Gaussian sigma (10 target pixels) and aperture radius (25 target pixels).
This matches the angular region with pixel-center quadrature; discretization
still differs. Bilinear and bicubic use the common 205-pixel target grid.
Bilinear output requires valid interpolation support. Bicubic conservatively
requires all 16 input stencil pixels, including zero-weight neighbors, because
signed interpolation weights cannot safely propagate boolean masks. Each
instrument retains its own mask; the final common success population requires
all methods and the JWST reference to pass their measurement checks.

Component ellipticity RMSE is
`sqrt(mean(((e1-e1_ref)^2 + (e2-e2_ref)^2)/2))`. Every reported method uses the
same objects within a comparison. A round-shape control predicts `(0,0)` for
every object. The baseline/reference eligible population is defined without
SR; SR failures on that population are reported separately. This prevents a
lower score from concealing dropped difficult objects, though the resulting
score is still conditional on successful measurement.

The analysis joins catalog properties by retained row and object ID and verifies
both. It saves per-object results (`measurements.csv` and `.npz`), catalog
properties, spatial blocks, raw moments, flags and coordinates of the weighted
centroid. Fixed bins cover input S/N proxy, ACS F814W magnitude, redshift, and
catalog FLUX_RADIUS. S/N is Gaussian-weighted NISP flux divided by quadrature
RMS; it ignores correlated noise. FLUX_RADIUS retains original catalog units,
not a newly inferred angular size. No new quality cut is imposed.

Uncertainty intervals use 1,000 paired bootstrap resamples of the 18 saved
spatial blocks. The same block draws and objects are used for both errors in
an RMSE ratio. These are conditional diagnostics within one field and under
this moment/mask contract. Small bins can contain very few objects/blocks;
inspect their counts, and do not interpret unavailable intervals as zero error.

Four regression checks passed in tmux/W&B: target-grid equivalence, native/target
angular scaling and failure flags, interpolation intensity/masks, and common
populations with known paired-bootstrap ratios. Two additional comparison
checks passed for reordered IDs, fixed cross-checkpoint populations, and
rejection of altered references or duplicate IDs.

## Epoch 11 result

[Analysis run od7fm46d](https://wandb.ai/AI-Astro/superNISP-corrected/runs/od7fm46d)
evaluated all 2,118 validation objects on GPU 1 while training continued on
GPU 0. The evaluator copied `last.pt`, whose atomic replacement by the trainer
allows the copy to capture one completed checkpoint. Source and sample hashes
were verified against the checkpoint. No active training source was changed.

- Reference plus all interpolation/native baselines succeed on **1,527** objects.
- SR fails on **43** of these, leaving **1,484** common successful objects.
- Across all 2,118 objects, successes are JWST 1,900; native 1,741; bilinear
  1,739; bicubic 1,628; SR 1,892. Different masks/algorithms explain some of
  these differing populations; these totals are not used as separate RMSE sets.

| Method | Component RMSE on the same 1,484 objects | SR RMSE reduction relative to method | Conditional 95% block interval |
| --- | ---: | ---: | ---: |
| Native NISP | 0.20698 | 14.20% | 10.49–17.41% |
| Bilinear | 0.20074 | 11.53% | 8.39–14.33% |
| Bicubic | 0.20741 | 14.38% | 10.68–17.71% |
| SR, epoch 11 | 0.17759 | — | — |
| Round-shape control | 0.28872 | 38.49% | 35.71–41.12% |

These are intermediate **validation diagnostics**, not the final scientific
improvement percentage. They compare observed weighted shapes across Y/F115W
bands without PSF correction and are not calibrated gravitational shear.
They remain conditional on successful measurement, a single checkpoint,
a single seed and a single field. Test data remain untouched.

All **43** SR failures on baseline/reference eligible objects have input S/N
proxy **1.33–3.75** and ACS F814W **24.15–24.99**. Across all 226 SR failures,
there are 7 incomplete apertures, 60 nonpositive weighted fluxes, 132 nonpositive
Qxx, 130 nonpositive Qyy, and 94 nonpositive determinants; counts overlap.
There are no nonfinite inputs or nonfinite ellipticities. This localizes the
problem to faint-source moment stability/negative signed intensity rather
than NaNs. It does not prove a unique model or measurement cause, and it does
not justify clipping negative intensity or quietly discarding faint galaxies.

[Machine-readable report](diagnostics/validation_epoch11_shapes.json) includes
bin results, failure counts, conditional confidence intervals, W&B URL and
checkpoint hash. Full per-object results and the immutable checkpoint are
under `runs/validation-shapes-full50-snapshot/`. The W&B analysis run includes
a summary figure and a table of SR failures with IDs, flags and S/N proxies.

## Completion workflow

`code/compare_shape_runs.py` aligns two analyses by unique object ID, verifies
row/block metadata and unchanged reference/baseline measurements, and then
uses the **intersection of successful objects across both checkpoints** for
both sets of scores. It saves those exact IDs, counts recovered/lost SR
measurements on the fixed baseline/reference eligible population, and reports
a paired block-bootstrap interval for the SR RMSE change. This addresses the
changing-success-population caveat in the earlier training logs.

`code/finish_baseline_analysis.py` waits for the full training process to exit
successfully and certify 50 completed epochs, then evaluates its **best
validation-MAE checkpoint**, not necessarily epoch 50. It uses GPU 1, logs a
separate W&B analysis, and compares that checkpoint with epoch 11 on identical
objects. It refuses to run after a failed/incomplete training process and
times out after 24 hours. It does not retrain, change checkpoint selection,
or evaluate the test split.

```bash
# Follow queued final analysis:
tmux attach -t supernisp-final-analysis
tail -f runs/final-analysis-console/console.log

# Run another independent validation analysis later, with fresh output/log names:
python code/launch_paired.py validation-example runs/validation-example-console -- \
  python -u code/evaluate_paired_shapes.py \
  --data ../supernisp_nir_y_f115w_v1 \
  --checkpoint runs/residual-full50-s20261016/best.pt \
  --output runs/validation-example --device cuda:1
```

The command using `best.pt` should run after training exits because the trainer
copies `best.pt` nonatomically. During training, use the atomically replaced
`last.pt` as in the completed snapshot analysis. The queued workflow follows
this rule. Final result directories will be `runs/validation-shapes-full50-best/`
and `runs/full50-shape-comparison/`. Their reports include W&B links.

Next, inspect the completed run's fixed-population comparison and failure
changes before choosing a controlled architecture experiment. Loss/shape
estimator changes should be separate ablations; do not mix them into the
running baseline or select final quality cuts using test results.
