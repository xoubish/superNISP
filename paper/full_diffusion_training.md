# Full corrected diffusion training

The user authorized a full run on the second GPU after the corrected pilot
passed its numerical and learning checks. This fresh run retains the corrected
diffusion core, representation, objective and maximum learning rate from the
[pilot](corrected_diffusion_pilot.md). Pilot weights are discarded.

## Predeclared budget and selection

- All 17,816 training galaxies, 50 epochs, batch size 8: 111,350 optimizer
  updates, matching the residual full runs' epochs, examples and updates.
  Batch size increases from the pilot's 4; a real-data GPU check verifies the
  batch-8 forward/backward and sampling geometry before training starts.
- Width-32 SR3 UNet with NISP mask conditioning, 4,910,593 parameters;
  v prediction; 1,000-step cosine schedule with zero terminal signal;
  unbounded, invertible asinh coordinates with saved training-only scales;
  masked v MSE + 0.1 x0 L1, Adam, max LR 1e-4, per-update OneCycle,
  gradient-norm clipping at 1, float32 and D4 augmentation. Seed 20261016.
- Each epoch generates from NISP alone using 50-step deterministic DDIM on
  256 fixed validation galaxies and per-object initial-noise seed 314159.
  Checkpoint selection is generated masked MAE in physical MJy/sr on this
  monitor subset. Shape errors, failures and eight fixed galaxy/residual
  panels are logged every epoch. The subset avoids spending most training
  time repeatedly sampling the full validation split.
- At completion, reload the selected checkpoint and reproduce its monitoring
  metric. Generate all 2,118 validation galaxies with seeds 314159–314161.
  Report sample variation and the mean of these three images in linear
  intensity, measured after averaging. This ensemble requires no additional
  model sampling. It is a predeclared additional estimator, not a replacement
  selected after looking at its results.
- Automatically compare individual diffusion samples, the mean image, the
  frozen legacy epoch-49 result, and the completed ResNet-D5 result when
  available. Use identical common-success galaxies, full-population failure
  counts and paired spatial-block intervals. Small numerical differences in
  reference shapes from the asinh round trip are checked and recorded.

Validation monitoring differs from the residual run's full-validation epoch
selection. The architectures also differ in representation, loss, parameter
count, sampling cost and GPU time. Equal updates alone do not make this a
single-factor architecture ablation. No test images are opened.

## Execution and GPU sharing

`code/run_diffusion_full.py` runs in persistent tmux session
`supernisp-diffusion-full50`. All model construction happens after online W&B
initialization in `AI-Astro/superNISP-corrected`.

Training: [W&B gepl2iv2](https://wandb.ai/AI-Astro/superNISP-corrected/runs/gepl2iv2).
The [batch-8 preflight](https://wandb.ai/AI-Astro/superNISP-corrected/runs/vlziq0gw)
passed both regression checks and actual-data GPU forward/backward/sampling,
with 3.13 GiB peak tensor allocation. The saved full training rows exactly
match the residual baseline; all 256 monitor rows belong to validation. Both
pipelines' queued source hashes were verified unchanged after launch.

The existing ResNet-D5 pipeline trains on GPU 0 but already has analysis stages
queued for GPU 1. The new diffusion wrapper checks that pipeline at every
training/sampling batch boundary. Once ResNet-D5 training finishes, diffusion
waits without GPU compute until its analyses exit, then resumes automatically.
It retains its small model allocation while waiting. A failed ResNet-D5 exit
is explicitly logged; a two-hour unresolved wait aborts rather than silently
waiting forever. This does not modify the active ResNet-D5 sources or commands.

Two regression checks cover the wait/release decisions (including incomplete
status writes and failed exits) and averaging in linear intensity rather than
asinh coordinates. Both passed before launch and run again within the online
preflight. The previously verified diffusion core remains unchanged.

Artifacts:

- `runs/diffusion-full50-pipeline/`: exact commands, protocol, source snapshots,
  preflight checks, and pipeline status. All source hashes are checked before
  queued stages; do not edit queued sources while the run is active.
- `runs/diffusion-full50-s20261016/`: W&B URL, selected rows, normalization,
  model/source/data provenance, metrics, per-epoch measurements, fixed gallery,
  restartable `last.pt` and selected `best.pt`.
- `runs/diffusion-full50-comparison/`: final matched comparison, created only
  after full training and repeated-sample validation complete.
- `runs/diffusion-full50-console/`: persistent console and eventual exit status.

`training_status.json` distinguishes finished training from finished final
sampling; `status.json` marks the trainer complete only after final diagnostics.
The pipeline status becomes complete after the matched comparison also exits.
Attach with `tmux attach -t supernisp-diffusion-full50`.
