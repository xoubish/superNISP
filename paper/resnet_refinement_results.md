# Paired ResNet refinement: first-parent results

Completed 2026-10-17. Both corrected `v03` branches finished 10 extra epochs
(22,270 updates) and exited successfully. Parent weights/checkpoint remain
unchanged. Batch and augmentation stream hashes match across all ten epochs.
The automatic comparison completed successfully in
`runs/resnet-refinement-comparison-v01/`.

The balanced loss improves source errors but worsens outside-source errors.
It does not establish an overall better model or a shape improvement. Keep
the approved paper model and Figure 6 unchanged. The completed seed check and
the user's subsequent decision confirm this choice; the manuscript now includes
Figure 6 and a concise main-text summary of these tests. The appendix was later
removed at the user's request; its detail is preserved in
`diagnostics/resnet_validation_archive/`.

| Variant | Selected extra epoch | Source MAE | Source RMSE | Outside RMSE |
|---|---:|---:|---:|---:|
| Original | 0 | 0.0203059 | 0.0601735 | 0.0139575 |
| Original-loss continuation, best | 5 | 0.0202381 | 0.0594465 | 0.0139444 |
| Balanced loss, best | 5 | 0.0198781 | 0.0588579 | 0.0142922 |
| Original-loss continuation, last | 10 | 0.0202892 | 0.0600835 | 0.0139518 |
| Balanced loss, last | 10 | 0.0199232 | 0.0594867 | 0.0142954 |

Pixel metrics are MJy/sr; RMSE is averaged per object. Selection uses mean source
MAE on the frozen 2,118-object validation sample, with epoch zero eligible.
Compared with the original, the balanced selected checkpoint improves source
MAE by 2.11% and source RMSE by 2.19%, but worsens outside RMSE by 2.40%.
The selected control improves source RMSE by 1.21% and outside RMSE by 0.094%.
Best-versus-last differences matter; these are validation-selected results.

On the 1,363 complete fields, the fine-scale (3–6 target pixels) signed source
correlation changes by +0.000575 for balanced versus original, with paired
95% spatial-block interval [-0.000631, +0.001669]. For control versus original,
the change is +0.002476 [0.000798, 0.004095]. Balanced versus control is
-0.001902 [-0.003108, -0.000721]. Increased power alone is not better recovered
structure.

Fixed-shape comparisons use the same 1,628 successes across reference and all
five variants. Balanced minus original ellipticity component RMSE is +0.000309
[-0.002941, +0.003315]. Adaptive shapes use 1,677 common successes; the same
change is -0.000737 [-0.004583, +0.002737]. Neither supports a clear shape
improvement. Measurement success counts must also be reported: fixed successes
increase from 1,786 to 1,845, whereas adaptive successes decrease from 1,742 to
1,723 for balanced best. These are observed-image diagnostics without PSF or
noise correction, not calibrated shear results.

All intervals use 2,000 paired spatial-block draws, seed 20261017, with equal
object weighting. They condition on one trained parent and validation selection;
they do not include training-seed or checkpoint-selection uncertainty.

Local artifacts:

- [Full comparison report](../runs/resnet-refinement-comparison-v01/report.md)
- [Errors, power and correlation](../runs/resnet-refinement-comparison-v01/comparison.png)
- [Learning curves](../runs/resnet-refinement-comparison-v01/learning_curves.png)
- [Observed shapes](../runs/resnet-refinement-comparison-v01/shapes.png)
- [Fixed observed-image-selected examples](../runs/resnet-refinement-comparison-v01/fixed_examples.png)
- [Machine-readable estimates and provenance](../runs/resnet-refinement-comparison-v01/comparison.json)

The `runs/` artifacts are ignored by Git and must be transferred separately.

## Independent-seed replication completed

After the user asked for more tests and said “continue,” two fresh baseline
trainings were launched and completed, one on each GPU. These reproduce the original
50-epoch recipe with seeds **20261018** and **20261019**; they are not forks of
the old checkpoint. Original seed: **20261016**. All historical source and data
hashes match. The frozen replication contract is
`runs/resnet-seed-replication-v01/contract.json`.

| Seed | GPU | Directory | W&B |
|---|---|---|---|
| 20261018 | 0 | `runs/residual-full50-s20261018` | [l0kbzuub](https://wandb.ai/AI-Astro/superNISP-corrected/runs/l0kbzuub) |
| 20261019 | 1 | `runs/residual-full50-s20261019` | [8vh9u6nb](https://wandb.ai/AI-Astro/superNISP-corrected/runs/8vh9u6nb) |

Each uses the original legacy head, batch 8, masked L1, fixed linear
normalization, Adam and per-update OneCycle schedule with max LR 1e-4,
50 epochs / 111,350 updates, paired D4 augmentation and original full-field
validation-MAE checkpoint selection. Runs execute in tmux with online W&B;
their exact commands/logs are in the corresponding `-console/` directories.
Both completed all 50 epochs / 111,350 updates with exit 0, in about 71.7 minutes
each, concurrently. Their selected checkpoints are epochs 46 and 48.

`code/evaluate_residual_seed_replicates.py --wait` compares all three seeds'
best and last models on the frozen validation masks, with errors, spectra,
fixed/adaptive shapes and success counts. It uses online monitoring for new
inference, while comparison artifacts remain local. Completed output:
`runs/resnet-seed-replication-v01/evaluation/`. It reports descriptive
between-seed variation separately from conditional spatial-bootstrap intervals.
The evaluator completed with exit 0 in tmux `supernisp-seed-replication-eval-v01`,
with console `runs/resnet-seed-replication-eval-v01-console/`. Its source snapshots
and provenance were frozen when the waiting process started.

Post-launch verification confirmed that both runs differ from the original
configuration only in seed, output directory and explicit W&B entity. All
source/data hashes and train/validation row selections match. All completion
flags and exit statuses were checked.

The three selected seeds have source RMSE 0.0601735, 0.0601034 and 0.0600667
MJy/sr (mean 0.0601145; sample standard deviation 0.0000542). Outside RMSE is
0.0139575, 0.0139342 and 0.0139416 MJy/sr. This supports reproducibility under
the fixed recipe. The user approved retaining the original model, ending the
training experiments, and adding the evidence to the paper. Figure 6 is installed
unchanged. The seed plot and selected/final checkpoint tables were subsequently
archived in `diagnostics/resnet_validation_archive/`, with essential methods and
results retained in the main text. The held-out test evaluation remains pending.
