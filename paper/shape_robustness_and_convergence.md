# Epoch-49 measurement robustness and convergence

The user authorized the observed-shape robustness work and asked whether the
50-epoch training run had converged. The original model, sample, loss and run
artifacts remain frozen. The independent analysis uses validation only.

## Convergence assessment

The best validation MAE was 0.96457083 at epoch 49. Improvement remaining from
earlier checkpoints to that best value was:

| Starting epoch | Validation MAE | Reduction to epoch 49 |
| --- | ---: | ---: |
| 30 | 0.98196416 | 1.7713% |
| 40 | 0.97012003 | 0.5720% |
| 45 | 0.96605847 | 0.1540% |
| 48 | 0.96462141 | 0.00524% |
| 50 | 0.96479028 | Epoch 50 is 0.02275% above best, relative to epoch 50 |

The last optimizer learning rate was approximately 4.0004e-10, following the
completed OneCycle schedule with maximum 1e-4. Epoch 49 ended near 2.015e-7.
The run reached a practical plateau under this schedule. A best epoch near the
end is therefore not by itself evidence that substantial training was missed.
It also does not prove that the model cannot improve under another schedule.
Simply appending epochs to the exhausted OneCycle schedule would be ineffective
or invalid; a meaningful continuation would require a new, recorded schedule.

Keep epoch 49 as the completed baseline and assess the observed-shape metrics
before investing in longer training. A later bounded continuation experiment
with a reset low learning rate could test remaining improvement. Such a fork
must preserve the original baseline, use validation-only selection and report
its extra optimizer-update budget. It should not be silently counted as the
same 50-epoch architecture comparison.

## Predeclared robustness suite

`code/run_shape_robustness.py` evaluates the same checkpoint on all 2,118
validation objects. The complete 21-scenario protocol was saved before any
scenario results were examined. The run uses tmux session `supernisp-robustness`
and [W&B lb3m3r9k](https://wandb.ai/AI-Astro/superNISP-corrected/runs/lb3m3r9k).
Local artifacts are in `runs/shape-robustness-best49/`; console and exit status
are under `runs/robustness-console/`.

- Original fixed moments, Gaussian weight sigma 10 and aperture radius 25
  target pixels, reproduced against the previously saved epoch-49 analysis.
- Weight sigma 8 and 12 at fixed aperture; aperture radii 20 and 30 at fixed
  weight. These isolate weight/aperture dependence.
- Weight/aperture centers offset by ±2 target pixels in each coordinate,
  applied consistently to native and target sampling.
- Measurement-only background shifts of ±0.1 times each object's outer sky
  RMS, applied separately to NISP baselines, SR, and JWST. These are stress
  amplitudes, not measured uncertainties on background estimation. They do
  not rerun the network for the NISP-background scenarios.
- Independent GalSim 2.7.1 HSM adaptive moments in matched 105-target-pixel
  (21-native-pixel) crops; a 155/31-pixel crop variation; and an initial-guess
  size variation from 10 to 15 target pixels. HSM fits elliptical weights,
  centroids and sizes iteratively. Inputs remain signed linear intensities;
  no PSF correction or shear calibration is attempted.
- Three fixed-seed input-noise trials add independent Gaussian noise with
  standard deviation 0.25 times the supplied NISP RMS. SR and interpolations
  are recomputed from the perturbed input, while JWST remains unchanged.
  Under the independent-noise approximation, this increases input variance
  by 6.25%. It does not model original noise correlations or JWST noise.

HSM receives explicit bad-pixel masks, requires complete coverage of the
central radius-25 region, and records convergence status/error, fitted size,
centroid and shape. A predeclared maximum centroid displacement of 10 target
pixels guards against capturing a neighbor. Physical sizes and HSM limit
settings are converted consistently to each image's pixel units. The 105- and
155-pixel crops have exactly matched native-grid footprints. HSM failure reasons
are stored under distinct labels rather than recycled fixed-moment labels.
See [GalSim HSM documentation](https://galsim-developers.github.io/GalSim/_build/html/hsm.html)
for the independent estimator's definition and API.

Every comparison uses the same successful objects for all methods. For each
scenario the report additionally restricts both that scenario and the nominal
measurement to their shared object IDs; it also reports the intersection across
all scenarios. Full-sample success rates are retained independently. Confidence
intervals use the same paired spatial-block resampling as the baseline.
Different estimator definitions measure different weighted shapes, so absolute
RMSE values should not be compared as though their targets were identical.

Crowding is inspected with a fixed input-only indicator: a local peak exceeding
five times the supplied RMS between 3 and 9 native pixels from stamp center.
The report separates objects with/without this indicator. This is neither a
complete neighbor catalog nor a deblending correction; extended source
structure can also trigger it.

Four new regression checks passed before inference: exact nominal equivalence
and consistent centering, known adaptive Gaussian shapes at both samplings and
mask failures, neighbor-displacement/peak checks, and correct HSM failure labels.
The full run must also reproduce the prior nominal per-object results before
writing a completed report. Tests and source hashes are stored with the run.

## Completed robustness results

All 21 scenarios completed on all 2,118 validation objects. The console exit
status is 0; W&B synced the convergence figure, robustness figure and scenario
table. The original nominal per-object shapes and success flags were reproduced.

| Measurement | Objects shared with nominal and all methods | SR RMSE reduction vs native | Conditional 95% spatial-block interval | Reduction vs bilinear |
| --- | ---: | ---: | --- | ---: |
| Original fixed moments | 1,523 | 19.20% | 16.53–21.85% | 16.42% |
| Independent HSM, 105-pixel crop | 1,351 | 12.20% | 9.02–15.10% | 16.45% |
| HSM, larger 155-pixel crop | 1,347 | 12.61% | 9.19–15.95% | 16.81% |
| HSM, initial guess 15 instead of 10 | 1,259 | 13.36% | 9.24–16.98% | 17.43% |

All reported gains are positive under the tested settings. To eliminate changes
in the successful population across scenarios, the report also compares every
scenario on the **same 1,098 objects** measurable in all 21 scenarios. On this
strict intersection, SR error reductions range from 8.23–25.03% versus native,
8.94–22.48% versus bilinear, and 8.30–25.64% versus bicubic. Every individual
95% block interval has a positive lower bound. These intervals are marginal,
not simultaneous, and the 21 scenarios are highly correlated rather than
independent confirmations. The intersection is a selected, more measurable
subset of the validation population, not a replacement for full failure counts.

The independent HSM check succeeds on 2,020 JWST, 1,685 native, 1,707 bilinear,
1,652 bicubic and 1,987 SR images, out of all 2,118. Its nominal-shared comparison
contains 1,351 objects. Changing the initial size guess reduces HSM success
counts substantially even though the advantage remains. This sensitivity and
the centroid-displacement rejection must be retained in any HSM result.

Background offsets have the largest practical effect on the reported gain and
failure rate. Subtracting 0.1 outer-sky RMS from SR reduces its all-object
success count from 2,045 to 1,759; on the strict 1,098-object intersection its
gain versus native falls from 19.03% to 8.23% (interval 3.22–12.80%). The stress
offset is not a calibrated uncertainty, so these ranges cannot be relabeled
as a systematic-error confidence interval. Better background characterization
remains relevant to the final measurement contract.

The added-input-noise trials retain an advantage: nominal-shared comparisons
give reductions of 14.04–17.10% versus native and 12.10–14.85% versus bilinear.
This tests modest independent input noise, not a realistic complete correlated
noise model. The nearby-input-peak subsets also retain positive gains, but the
indicator is only a crowding proxy and does not validate deblending.

These checks support the direction of the observed-shape improvement across
independent estimators and the tested perturbations. Its size remains estimator-
and population-dependent. No settings were selected to maximize the gain, and
no test images were evaluated. Keep fixed moments as the declared reference
and HSM as an independent cross-check for the controlled head comparison.

Tracked evidence: [robustness report](diagnostics/shape_robustness_best49.json)
and [convergence summary](diagnostics/full50_convergence.json). Full scenario
object IDs, HSM errors, fitted centroids/sizes and per-object flags remain in
the run's `report.json` and `measurements.npz`. The next experiment is the
controlled ResNet-D5 head comparison with the same 50-epoch update budget;
a baseline continuation would be a separate budget/schedule experiment.
