# Corrected residual training

Prepared 2026-10-16 on ipacgpu3. This is a pipeline/debugging experiment, not
an architecture ranking or a final shape-improvement measurement.

## Full training and galaxy images

The user requested longer training and visual galaxy monitoring after the pilot.
The new run `residual-full50-s20261016` uses all 17,816 saved training and all
2,118 validation objects for 50 epochs (111,350 optimizer updates at batch size
8). It starts from scratch with a new 50-epoch OneCycleLR schedule; extending
the completed pilot's schedule would be a different experiment. Architecture,
linear normalization, masked L1, Adam maximum LR 1e-4, augmentation, and seed
20261016 are unchanged. Test objects remain reserved. Best/last checkpoints
are saved each epoch, and `progress.json` records the latest completed epoch.

```bash
tmux attach -t supernisp-full50
tail -f runs/full50-console/console.log
cat runs/residual-full50-s20261016/wandb_url.txt
```

W&B `galaxies/full_1`, `galaxies/full_2`, `galaxies/central_1`, and
`galaxies/central_2` show eight fixed evaluation objects at initialization and
each epoch. Columns are native-sampling NISP Y, bilinear NISP Y, SR F115W,
JWST F115W, and SR-minus-JWST. Central views cover 81 target pixels on each
side. Panels include object IDs, physical display limits, and moment-check
pass/fail labels. These display examples span input-brightness ranks in a
seeded pool of up to 256 validation objects; they are not a random population
estimate and do not change validation metrics or checkpoint selection.

Images are sky-subtracted MJy/sr. NISP and JWST scales are inverted separately.
Each object's reference-derived asinh display limits are fixed across epochs
and shared among the four intensity columns; the signed residual has a fixed
symmetric scale. These are display transforms only. Invalid pixels are gray;
interpolation is masked unless its input support is fully valid. The two bands
are different, so interpolation is a visual reference, not a cross-band forward
model. Exact object selection and limits are saved in `gallery_selection.json`.
PNGs are retained locally under the run's `galaxies/` directory and synced to
W&B as image media. Rendering preserves training random state.

[Visual check sfgwa5ku](https://wandb.ai/AI-Astro/superNISP-corrected/runs/sfgwa5ku)
passed on the saved pilot model in tmux and synced eight media files: fixed
examples plus examples of moment failures. Images were inspected locally;
the panels show recognizable galaxies and the pilot's smooth predictions.
The 20 prediction moment failures were reproduced; 13 have passing reference
moments. Failure examples include faint predictions, and their quantitative
cause still needs analysis. Source code for the image check is
`code/check_galaxy_visuals.py`; results and IDs are in
`runs/galaxy-visual-check/{checks,pilot_moment_failures}.json`.

The added visualization module is copied and hashed with every new run's
training source. Older pilot checkpoints retain their original source snapshots;
use those snapshots if resuming an old run, since the current entrypoint rejects
changed source/configuration hashes.

## Contract and implementation

`code/train_paired.py` consumes the frozen `PairedSample` splits. The reference
uses linear training-only normalization, per-object valid-pixel L1, Adam,
OneCycleLR stepped per successful optimizer update, and no AMP or clipping.
Training pairs receive consistent rotations/reflections including masks and RMS;
overfit checks and validation are not augmented. Invalid target pixels are
excluded from loss. Targets and predictions are never intensity-clamped.

`code/residual_baseline.py` isolates the historical eight single residual dense
blocks, 64 features, growth rate 32, edge enhancer, and two 2x PixelShuffle
stages followed by bilinear resize. Its 2,332,177 parameters, state dict, and
CUDA outputs match the original class exactly. The optional `--head direct5`
changes only the upsampling head; its additional parameters should be reported
when compared. Only the legacy head is trained in this initial pilot.

Pilot checkpoint selection uses validation MAE exclusively. Overfit checkpoints
are explicitly selected on the training subset and are not scientific models.
No test examples are used. Run directories store configs, exact training source
copies and hashes, git revision/status, a working diff, manifest/normalization
hashes and copies, selected HDF5 rows, environment versions, local metric logs,
W&B IDs/URLs, and last/best checkpoints. Checkpoints include optimizer, scheduler,
Python/NumPy/Torch/CUDA RNG and sampler-generator state. Resume is at completed
epoch boundaries and rejects changed configuration, source, or data hashes.

## Monitoring and persistence

Every model execution requires tmux or screen and an online W&B initialization;
failed initialization aborts before constructing a model. Runs use
[AI-Astro / superNISP-corrected](https://wandb.ai/AI-Astro/superNISP-corrected).
The launcher saves a console log and process exit status, then leaves an
interactive shell in the named tmux session. Detach with Ctrl-b then d.
A lost SSH connection does not end the run. A host restart still requires
checkpoint resume. W&B may buffer telemetry locally during a network outage.

```bash
# Inspect current work:
tmux attach -t supernisp-pilot
tail -f runs/pilot-console/console.log

# A new independent run; choose fresh session/output/log names:
python code/launch_paired.py supernisp-example runs/example-console -- \
  python -u code/train_paired.py \
  --data ../supernisp_nir_y_f115w_v1 --output runs/example \
  --mode pilot --epochs 3 --train-limit 1024 --validation-limit 256
```

To resume, use the identical training arguments plus `--resume`, the same output
directory, and a fresh launcher session/log directory. The original total epoch
budget is fixed for OneCycleLR. An interrupted epoch is replayed from the most
recent completed epoch; local/W&B metrics from an unsaved epoch can remain.

## Checks and initial experiment

All 46 extracted sample files passed `SHA256SUMS`. CUDA is accessible outside
the execution sandbox: PyTorch 2.6.0+cu124, driver 590.48.01, two NVIDIA L40S
GPUs with 46,068 MiB each. The initial experiment uses CUDA device 0.
W&B SDK is 0.19.7. No environment installation was needed.

[Integration checks](https://wandb.ai/AI-Astro/superNISP-corrected/runs/lkg2d7mu)
passed in tmux: worker loading, physical-intensity round trips and tiny CPU
updates for both representations; original/new residual equivalence; finite CUDA
forward/backward for both heads; masked-pixel zero gradients; paired augmentation;
circular/negative-flux shape fixtures; exact next-update checkpoint restoration,
including optimizer, scheduler and RNG state.

`code/run_paired_pilot.py` orchestrates:

1. A seeded, fully valid 32-pair training subset, 40 epochs / 160 updates,
   batch size 8, learning-rate maximum 1e-4, seed 20261016. Stop after epoch 2,
   then resume the same checkpoint and W&B run.
2. Require all 40 epochs and 160 updates, and a best training MAE below 90% of
   its initial value. This is a debugging gate, not a scientific effect target.
3. A fresh three-epoch pilot with 1,024 seeded training examples and 256 seeded
   validation examples, batch size 8, same seed and learning-rate maximum.
   These subsets come from the saved spatial splits; their rows are recorded.
   Total pilot budget: 384 optimizer updates. Full-population training and
   multi-seed comparisons remain later work.

## Fixed shape diagnostic

Validation also logs linear-intensity Gaussian-weighted central quadrupole
ellipticity components `(Qxx-Qyy)/(Qxx+Qyy)` and `2 Qxy/(Qxx+Qyy)`. The circular
weight has sigma 10 target pixels and an aperture radius of 25 target pixels
(approximately 0.59 and 1.48 arcsec). It uses signed sky-subtracted intensities,
requires complete aperture coverage, positive weighted flux and positive-definite
moments, and reports prediction/reference failures separately. Component RMSE
uses paired successes only, so changing failure populations can bias it. This
simple fixed diagnostic cannot establish calibrated shear or an SR improvement
percentage. It does not select checkpoints. Native/interpolated baselines,
matched populations, uncertainty estimation and final shape analysis remain
future work. Timing is elapsed training/evaluation wall time; peak allocated
CUDA memory is logged. No GPU-kernel-only timing is claimed.

## Completed pilot outcomes

The tmux orchestration exited with status 0 and W&B finished syncing both runs.
All intended epochs and optimizer updates are present in the local logs.

| Run | Budget | Initial MAE | Final MAE | Measured train/evaluation time |
| --- | --- | --- | --- | --- |
| [Overfit emr5ag36](https://wandb.ai/AI-Astro/superNISP-corrected/runs/emr5ag36) | 40 epochs, 160 updates, 32 train objects | 1.175426 | 0.924286 (training) | 22.1 s |
| [Pilot 9c47c2of](https://wandb.ai/AI-Astro/superNISP-corrected/runs/9c47c2of) | 3 epochs, 384 updates, 1,024 train / 256 validation | 1.442872 | 1.108224 (validation) | 71.5 s |

MAE is in the fixed normalized target representation; the initial value is
the randomly initialized network, not an interpolation/native baseline. Pilot
physical-intensity RMSE decreased from 0.427578 to 0.372955 MJy/sr. Epoch 3 is
the best validation checkpoint. Peak allocated CUDA memory was 0.577 GiB.
Timing excludes process setup and W&B initialization/final sync.

Prediction moment failures increased from 0 to 20/256, while reference failures
remained 25/256; paired successes decreased from 231 to 218. The diagnostic
component RMSE changed from 0.285975 to 0.219382, but those differing populations
prevent interpreting this as a shape-improvement percentage. Inspect the failed
objects and use fixed paired populations with native/interpolation diagnostics
before scientific comparisons. No held-out test objects were evaluated.

[Machine-readable results](diagnostics/corrected_training_pilot.json) contain
epoch metrics, run links, source hashes, and checkpoint SHA256 values. The pilot
best/last checkpoints are identical (epoch 3); their hash was verified after
completion. The 32-pair run resumed after epoch 2 and completed every epoch
exactly once. It demonstrated learning, not convergence to a memorized sample.
