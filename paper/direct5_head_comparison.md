# Direct-5x head comparison

The authorized experiment changes the upsampling head from two 2x pixel-shuffle
blocks plus a final 164-to-205 bilinear resize to one learned 5x pixel-shuffle
block. The target is improved observed-shape recovery on validation data.

The reference is the completed legacy run `residual-full50-s20261016`, selected
at epoch 49 by validation masked L1. The new run starts from scratch and uses
the same saved train/validation rows, linear representation and normalization,
trunk architecture, seed 20261016, D4 augmentation, masked L1 objective, Adam,
batch size 8, maximum learning rate 1e-4 and 50-epoch OneCycle schedule. Both
full runs receive 111,350 optimizer updates. Best-checkpoint selection remains
validation masked L1; shape diagnostics do not select the epoch.

The full trainer configuration differs only in `head` and output directory.
All four training source hashes match the baseline. The trunk initializes
identically under the shared seed. Total parameters increase from 2,332,177 to
2,959,953. The new head's
initialization shifts the Torch RNG position used by the final reconstruction
layers. Consequently this is a comparison at a fixed update budget, not a
parameter- or compute-matched experiment. It is one seed, so conditional spatial
bootstrap intervals do not measure training-seed variability.

## Execution and monitoring

`code/run_direct5_comparison.py` runs these stages sequentially in tmux session
`supernisp-direct5`, failing immediately if a process or gate fails:

1. Fit 32 fully valid training pairs for 40 epochs / 160 updates. Require at
   least a 10% masked-MAE decrease from initialization.
2. Pilot on 1,024 training / 256 validation objects for 3 epochs / 384 updates.
   Require finite validation MAE and improvement over initialization.
3. Discard the debugging-stage weights and train the full fresh model for
   50 epochs. Save the best validation checkpoint and each epoch's last state.
4. Evaluate the best checkpoint on all 2,118 validation objects and compare
   with the frozen legacy best on identical common-success object IDs.
5. Run the same 21 predeclared measurement-robustness scenarios, including
   GalSim adaptive moments, then compare both architectures on identical IDs
   separately within each scenario. Retain lost/recovered measurements and
   spatial-block intervals, including unfavorable results.

Every model process requires tmux/screen and initializes online W&B in
`AI-Astro/superNISP-corrected` before constructing the model. The full run logs
eight fixed galaxies at initialization and each epoch, showing native NISP,
bilinear, SR, JWST and residuals with fixed physical display scales and central
crops. The same data, seed and gallery settings select the baseline's galaxies.

The immutable execution protocol, commands and source snapshots are under
`runs/direct5-comparison/`. Source hashes are rechecked before every stage;
do not edit queued sources while the pipeline runs. Current stage is in its
`status.json`. Console and overall exit code are under `runs/direct5-console/`.
Attach with `tmux attach -t supernisp-direct5`.

Full training artifacts: `runs/direct5-full50-s20261016/`, including
`wandb_url.txt`, `progress.json`, `metrics.jsonl`, `last.pt` and `best.pt`.
Queued result directories:

- `runs/validation-shapes-direct5-best/`
- `runs/direct5-fixed-comparison/`
- `runs/shape-robustness-direct5-best/`
- `runs/direct5-robust-comparison/`

## Validation before launch

Six existing comparison/robustness regression tests passed. All 21 saved
legacy robustness scenarios were also compared against themselves with reversed
object ordering: every change and interval was exactly zero, with zero lost or
recovered measurements. Adaptive-moment comparisons retain HSM-specific failure
labels. The generic robustness runner checks that its nominal reference belongs
to the same checkpoint and reproduces IDs, rows, blocks, shapes and failure
flags before accepting the new report. The old baseline reports and source
snapshots are preserved.

Both learning gates passed: the 32-pair MAE fell 22.10%, and pilot validation
MAE fell 24.65% from initialization (1.44575 to 1.08933). Full training is
running in [W&B ilaim0sg](https://wandb.ai/AI-Astro/superNISP-corrected/runs/ilaim0sg).
Remote verification confirmed the run is online and all four galaxy-image
panels are uploaded. Training/validation selections and gallery selections
exactly match the baseline. These debugging results are not architecture gains.

Training and analysis are ongoing; no architecture improvement is established
yet. Test images remain reserved. Results concern observed Y-to-F115W shapes,
with differing bands and PSFs, and do not establish calibrated shear recovery.
