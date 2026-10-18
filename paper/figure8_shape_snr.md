# Figure 8: current-model shape recovery versus input NISP S/N

The user selected a compact Figure 8 on input quality and measurement coverage,
with notebook 9 retained as supporting checks. The executed
[Figure 8 notebook](../paper_figures/08_shape_dependence.ipynb) and
[preview](../paper_figures/exports/fig08_shape_dependence_v03.png) replace the
legacy-only magnitude/redshift/S/N notebook as the active draft. Presentation v03
moves the estimator names to the y-axes and removes panel titles, reusing the
unchanged v02 measurements. The v02 notebook is archived in `revisions/`. PDF and editable
SVG exports share the same stem. The old notebook and exports are preserved.
This new figure is not yet installed in the manuscript; its current Figure 8 is
still the historical residual-plane plot. No external upload was performed.

## Frozen measurement protocol

`code/analyze_shape_snr.py` reads `cached_shapes_v02.json/.npz`, the same per-object
measurements and success flags used by current Figure 7. Model checkpoints remain
epoch 49 for ResNet and ResNet-D5, epoch 35 for diffusion. Additional training
seeds and continuation checkpoints do not replace these selected models.
All eight images (JWST, native NISP and six cached methods) must be measurable
for an object to enter an estimator's error comparison. The sample is then
partitioned by the input-only S/N bins. No per-model selection or remeasurement.

Native S/N is signed background-subtracted Gaussian-weighted flux divided by
quadrature weighted pixel RMS, with Gaussian sigma 2 native pixels and central
radius 5 native pixels, using the native validity mask. Correlated noise is
ignored: this is a relative proxy, not a calibrated detection significance.
It was recomputed in physical units from the original validation rows. The
maximum difference from the old float32 implementation is 0.0000187; all bin
assignments match exactly. No old model/reference shape measurements are reused.
The original left-closed/right-open edges are unchanged: -infinity, 5, 10, 20,
50, infinity. Negative finite S/N belongs in the first bin. All 2,118 objects
have finite S/N; none is excluded by this step.

The analysis contract and source snapshots were saved before computing new
results in `runs/shape-snr-v02/`. Report and arrays: `shape_snr_v02.json/.npz`.
Copy the git-ignored NPZ and its referenced shape bundle when moving machines.
Gain is 100*(1 - RMSE_model/RMSE_native), with the combined e1/e2 RMSE on the
identical objects. Confidence intervals use 2,000 paired spatial-block draws,
seed 20261017, with equal object weighting. They condition on success and this
field; they are marginal intervals, not a simultaneous significance test of
bin-to-bin trends. Both-estimator common-ID results and all interpolation and
mean-of-three-image comparisons are retained in the bundle and notebook tables.

## Coverage

| NISP S/N bin | All attempted | Fixed common | Adaptive common | Both estimators common |
|---|---:|---:|---:|---:|
| <5 | 626 | 179 (28.6%) | 297 (47.4%) | 129 |
| 5–10 | 547 | 441 (80.6%) | 419 (76.6%) | 386 |
| 10–20 | 439 | 387 (88.2%) | 363 (82.7%) | 363 |
| 20–50 | 371 | 322 (86.8%) | 304 (81.9%) | 304 |
| ≥50 | 135 | 117 (86.7%) | 115 (85.2%) | 115 |

The low-S/N gain therefore describes a selected subset; it cannot be interpreted
as useful shapes for all faint galaxies. Coverage is the all-image intersection,
not the success rate of an individual model. The notebook reports individual
success counts separately. Fixed and adaptive samples differ, so the report
also retains their shared 1,297-object sample, partitioned into the same bins.

## Native-relative RMSE reductions

Values are percent reduction with conditional 95% paired spatial-block intervals.

| Estimator | S/N bin | Diffusion (one sample) | ResNet | ResNet-D5 |
|---|---|---:|---:|---:|
| fixed | <5 | -15.6 [-24.4, -8.8] | +19.9 [+13.8, +25.3] | +20.3 [+14.0, +25.8] |
| fixed | 5–10 | -14.8 [-20.6, -7.9] | +17.8 [+12.9, +22.0] | +17.2 [+12.2, +21.4] |
| fixed | 10–20 | -11.6 [-20.2, -4.1] | +17.4 [+10.9, +23.1] | +17.4 [+11.1, +23.2] |
| fixed | 20–50 | +4.4 [-5.0, +13.3] | +32.3 [+27.3, +37.3] | +34.5 [+29.7, +39.1] |
| fixed | ≥50 | +38.5 [+26.6, +47.2] | +63.0 [+57.3, +68.3] | +61.0 [+54.0, +67.1] |
| adaptive | <5 | -25.8 [-36.0, -18.0] | +7.8 [+3.2, +11.4] | +7.6 [+2.8, +11.5] |
| adaptive | 5–10 | -21.9 [-30.4, -13.8] | +9.0 [+4.7, +12.9] | +9.0 [+5.2, +12.6] |
| adaptive | 10–20 | -21.4 [-27.8, -14.1] | +13.5 [+7.8, +18.9] | +13.6 [+8.1, +18.8] |
| adaptive | 20–50 | -4.9 [-16.1, +4.6] | +29.1 [+21.2, +36.8] | +29.0 [+21.2, +36.4] |
| adaptive | ≥50 | -8.0 [-26.6, +4.1] | +39.8 [+29.1, +47.6] | +39.1 [+29.0, +46.6] |

Both ResNets have positive gains in each bin under both estimators. The largest
fractional reductions occur at high S/N. Single-draw diffusion improves in the
highest-S/N bin under fixed moments, but the corresponding adaptive interval
crosses zero. Do not claim diffusion improves shapes uniformly or interpret
fixed/adaptive RMSE as the same measurement. The shared-estimator population
check retains positive ResNet gains throughout, with a larger high-S/N gain.
S/N shares native pixels with the baseline measurement, and size, brightness,
crowding and redshift covary with S/N; this is a descriptive trend, not evidence
of a causal S/N effect or improved calibrated shear recovery.

## Proposed caption

Observed-shape recovery as a function of input NISP weighted S/N. Left and middle:
percentage reduction in ellipticity-component RMSE relative to native NISP for
fixed weighted and adaptive moments, respectively. Positive values indicate
lower error against observed JWST F115W. Each estimator/bin uses the same galaxies
measurable in JWST, native NISP and all six cached methods. Points are offset
within each categorical bin for visibility; connecting lines guide the eye.
Whiskers are conditional 95% intervals from 2,000 paired spatial-block resamples.
They are marginal intervals, not a simultaneous test of a trend across bins.
Right: the fraction of all attempted galaxies in each bin entering these
comparisons; numbers inside bars give common successes and $N$ below each bin
gives its total count. The fixed/adaptive totals are 1,446/1,498 successes out of
2,118 attempted validation fields. The predefined bins are $(-\infty,5)$,
$[5,10)$, $[10,20)$, $[20,50)$ and $[50,\infty)$.

The S/N proxy is native signed Gaussian-weighted flux divided by quadrature
weighted pixel RMS, ignoring correlated noise. It is not a calibrated detection
significance. Fixed and adaptive moments measure different weighted shapes and
have different successful samples. These observed cross-band comparisons include
PSF and pixel-sampling differences and do not establish calibrated shear recovery.


## Verification

Three tests cover bin boundaries/nonfinite values, weighted S/N, physical-unit
scaling, negative signals and invalid pixels. All pass. Every frozen input/source
hash, ID/block alignment, bin total and individual success count was checked.
Independent object-level spatial-block bootstraps reproduce the gain and interval
for native plus all three displayed models in all ten estimator/bin combinations
(40 comparisons). Weighted recombination of squared bin RMSEs reproduces Figure 7
exactly; averages of percentage gains are not used. The notebook executes without
error, the plot was visually inspected, and export provenance records the input
and notebook hashes. These checks require no GPU inference or test-set data.
