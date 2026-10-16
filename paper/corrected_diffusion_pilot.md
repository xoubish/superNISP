# Corrected diffusion pilot

The user authorized revisiting diffusion after the source audit found an
invalid shortened sampler, target clipping incompatible with normalization,
refitted train/validation normalizers, and model selection using noisy-target
reconstruction instead of deployed NISP-only generation. This experiment repairs
the training/inference path and checks whether it learns. It does not establish
that any one defect explains the historical figures.

Historical files under `code/diffusion/` are preserved. The new path imports
their SR3 UNet and implements its wrapper in `code/diffusion_corrected.py`.
No source used by the active direct-5x pipeline was modified.

## Model and measurement contract

- The same corrected, spatially separated dataset provides 41x41 NISP inputs,
  205x205 JWST targets, input/target validity masks and saved normalization.
  This pilot reads training and validation images only.
- Model coordinates use `asinh(linear_normalized_intensity / 3)` from the
  existing corrected loader. Scales were fitted on training data and are
  copied/hashed with the run; neither split refits them. Generated and target
  images are inverted to signed MJy/sr before selection and shape measurement.
  This differs from the residual model's linear training coordinates and is
  explicitly part of the new diffusion recipe, not a sampler-only ablation.
- The existing UNet uses width 32, three encoder scales and a 256-dimensional
  timestep embedding. It has 4,910,593 parameters with an added NISP-validity
  condition channel. Invalid NISP values are zero filled and distinguished by
  that channel. No JWST image or mask enters `sample()`.
- The cosine schedule has 1,000 timesteps and exactly zero terminal signal.
  Training predicts `v = sqrt(alpha_bar) * noise - sqrt(1-alpha_bar) * x0`.
  This permits stable conversion to x0/epsilon at zero terminal signal without
  division by a tiny alpha. It requires fresh weights; old epsilon-prediction
  checkpoints are not compatible. See [Lin et al.](https://arxiv.org/abs/2305.08891).
- Sampling begins from pure standard Gaussian noise and uses deterministic
  DDIM transitions with both current and next alpha values; the final
  destination has alpha=1. No output clipping, LR-centered initialization,
  classifier-free guidance or perceptual/VGG loss is used. Compare the
  [DDIM authors' implementation](https://github.com/ermongroup/ddim/blob/main/functions/denoising.py).
- Loss is per-object valid-pixel v MSE plus 0.1 times x0 L1 in model coordinates.
  Unknown target pixels contribute no loss. Their zero fills can still affect
  spatial context during training; this is not a full missing-data likelihood.
- Adam, maximum LR 1e-4, per-update OneCycle, gradient-norm clipping at 1, batch
  size 4, seed 20261016, float32. The pilot applies common D4 transforms to
  images and masks; the overfit debugging stage does not augment.
- Model selection uses masked MAE in MJy/sr from NISP-only generation on fixed
  evaluation IDs and initial-noise seed 314159. No noisy JWST reconstruction
  metric selects the checkpoint. Object-specific noise streams are independent
  of batch partitioning. The repeated-sample check uses seeds 314159–314161;
  these seeds do not select the checkpoint.
- Fixed signed Gaussian-weighted shapes, native/interpolated baselines and
  explicit failure flags use the existing shape code. Every method in each
  comparison shares successful objects. Repeated-sample scatter is measured on
  the intersection across all seeds and baselines. It is not calibrated
  uncertainty, a shear measurement or a claim about final test performance.

## Verification and execution

`code/check_diffusion_corrected.py` ran in tmux `supernisp-diffusion-check` with
[online W&B](https://wandb.ai/AI-Astro/superNISP-corrected/runs/gokjvp87), exit 0.
Five regression tests passed:

1. An analytic velocity oracle recovers unbounded targets through full and
   skipped trajectories, verifying every intermediate destination and endpoint.
2. Velocity/x0/noise round trips work, including at zero terminal signal.
3. Masked reduction ignores invalid values and gradients; object noise seeds
   reproduce across batch partitions and differ across seeds.
4. The actual sampling loop visits the expected timesteps, ignores invalid
   NISP values, repeats deterministically and recovers values above 1.
5. The UNet handles odd image sizes, produces finite gradients and is invariant
   to changing unknown target values, including NaNs.

A separate width-32 CUDA check on an actual training pair passed 41-to-205
geometry, finite forward/backward, bitwise repeated sampling and physical-unit
normalization inversion. Model construction began only after online W&B started.
Artifacts are in `runs/diffusion-corrected-check/`.

`code/run_diffusion_pilot.py` completed in tmux `supernisp-diffusion-pilot` on
GPU 1, exit status 0. Commands and source snapshots are in
`runs/diffusion-pilot-pipeline/`; its status records `pilot_completed`.
Source hashes were checked before each stage.

1. Overfit 32 fully valid training pairs for 40 epochs / 320 updates, with
   20-step generation checks every 10 epochs. Interrupt after epoch 2, restore
   model/optimizer/scheduler/RNG states and finish the same budget. Require
   generated physical MAE to improve by at least 10% from initialization.
2. If that gate passes, train a fresh model on 512 training galaxies for five
   epochs / 640 updates, evaluate 50-step generation on 64 fixed validation
   objects every epoch, and select the best generated physical MAE.
3. Reload that best checkpoint, reproduce its selection metric, then generate
   three samples per validation galaxy. Save per-object shapes/failures and
   common-population scatter, plus a W&B image comparing repeated samples.

Each stage has online W&B, fixed galaxy/residual panels with physical display
scales, source/data hashes, exact row selections and restartable last/best
checkpoints. The overfit stage's metrics are training-set debugging results.
Only the fresh pilot uses validation for selection. No full diffusion run or
hyperparameter sweep is queued. Pilot uncertainties use 200 block bootstrap
draws and are conditional debugging diagnostics on a small selected subset.

Console: `runs/diffusion-pilot-console/console.log`; overall exit status appears
in `runs/diffusion-pilot-console/exit_status`. Outputs:
`runs/diffusion-corrected-overfit/` and `runs/diffusion-corrected-pilot/`.
Each stores its W&B link in `wandb_url.txt`. Do not duplicate a running stage.

## Interpretation

The experiment and saved-measurement assessment completed successfully.
[Pilot W&B 6rqj99e8](https://wandb.ai/AI-Astro/superNISP-corrected/runs/6rqj99e8)
contains all epoch galaxy panels and the final three-seed image comparison.
[Assessment W&B d28ha2le](https://wandb.ai/AI-Astro/superNISP-corrected/runs/d28ha2le)
contains learning curves and comparisons restricted to identical galaxies.

The overfit/resume check completed 320 updates and reduced generated physical
MAE from 0.0647860 to 0.0172460 MJy/sr (73.38%). The fresh pilot completed 640
updates; best epoch 5 reduced generated validation MAE from 0.0716054 to
0.0203956 MJy/sr (71.52%). Reloading the best checkpoint reproduced its fixed
seed selection metric. These reductions are relative to random initialization,
not relative to the residual model or interpolation.

On the same 36 validation galaxies measurable in all three samples and all
reference methods, component shape RMSE was:

| Method | Component shape RMSE |
| --- | ---: |
| Native NISP | 0.18772 |
| Bilinear | 0.18615 |
| Frozen legacy full50, epoch 49 | 0.15956 |
| Diffusion seed 314159 | 0.21503 |
| Diffusion seed 314160 | 0.21432 |
| Diffusion seed 314161 | 0.22616 |

The three diffusion samples fail measurement on 7, 12 and 11 of all 64 objects,
respectively. Across the fixed 36-object intersection their RMS component
scatter is 0.15711. This is substantial sampling variation, not calibrated
uncertainty. On the 34 objects successful at every epoch, shape RMSE also
fluctuates rather than following the steadily improving pixel MAE. Generated
fine structure in the displayed examples is not reliably the observed JWST
structure. The corrected pilot learns but does not improve observed shapes
over native/interpolation in this small comparison. The legacy result has
vastly more training (111,350 updates versus 640); this cannot rank architectures.

Two additional saved-array regression tests passed for common-population
selection/scatter and rejecting changed references/handling empty populations.
Together with the five model/sampler checks, seven regression tests passed.
`code/summarize_diffusion_pilot.py` saves the assessment with input hashes.
Tracked evidence: [pilot assessment](diagnostics/diffusion_corrected_pilot.json).
Full per-object measurements and source snapshots remain in `runs/`.

Before a long diffusion run, the next useful experiment is to assess shapes
from a mean of independently generated images, in linear intensity, alongside
individual samples and their failures. This would test how much sampling noise
limits shape recovery. It has not been run, and would not establish unbiased
shapes or remove systematic morphology errors. Any larger pilot should track
generated shape quality explicitly and predeclare its selection rule.

Passing these tests establishes numerical and pipeline correctness for the
tested cases. Learning gates establish that the small experiment learns;
neither establishes superiority over the full residual baseline. A larger run
requires reviewing generation quality, shape errors, failure rates and sample
variation. Any subsequent architecture comparison must disclose differing
training representations, objectives, parameter counts, updates and inference
costs. Repeated seeds quantify sampling variation, not uncertainty coverage.
