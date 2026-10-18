# Paired ResNet refinement experiments

Prepared 2026-10-17 (user timezone: America/Los_Angeles).

## Latest update: all experiments complete and original model retained

The two new 50-epoch seeds and their automatic comparison also finished with
exit 0. The user approved keeping the original epoch-49 ResNet and the existing
three-panel Figure 6, with seed reproducibility in the appendix. This decision
and the refinement trade-off are now in `survey_enhancements.tex` and
`resnet_validation_appendix.tex`; the approved PDF is installed unchanged.
All numerical provenance is frozen in `diagnostics/resnet_model_selection_v01.json`.
The held-out test remains unevaluated. No further training is queued; the older
proposed continuation/ablation stages below are historical plans, superseded by
this decision.

## Previous update: refinements completed, independent-seed tests launched

Both corrected `v03` runs completed all ten epochs and the local comparison
finished with exit 0. Balanced loss improves selected source RMSE by 2.19%,
worsens outside RMSE by 2.40%, and does not clearly improve observed shapes.
See [the results and replication note](resnet_refinement_results.md) for
paired intervals, plots, success counts and limitations.

Following the user's “continue,” two fresh 50-epoch baseline runs with seeds
20261018 and 20261019 are running on the now-free GPUs. Their only scientific
change from the original baseline is the seed. They test reproducibility before
any further model-selection claim. This supersedes the launch status below.

## Earlier update: full paired shared training launched

The user subsequently said “do it.” The new trainer is implemented and both
10-epoch refinements are running on the current host:

- GPU 0 control: `runs/resnet-refine-control-v03`, tmux
  `supernisp-refine-control-v03`, [W&B sap2hiib](https://wandb.ai/AI-Astro/superNISP-corrected/runs/sap2hiib).
- GPU 1 balanced: `runs/resnet-refine-balanced-v03`, tmux
  `supernisp-refine-balanced-v03`, [W&B 609wkepy](https://wandb.ai/AI-Astro/superNISP-corrected/runs/609wkepy).

Exact commands and logs are in the corresponding `-console/launch.sh` and
`console.log`. Both runs use `code/train_residual_refinement.py`, batch 8,
22,270 extra updates, the same parent weights/Adam/RNG and fresh cosine
schedule. Epoch zero is saved as an eligible best checkpoint. All five CPU
loss, mask/augmentation, schedule and resume tests passed. CPU diagnostics
reproduced the frozen Figure 6 baseline statistics. Both GPU startup checks
passed finite losses/gradients, exact model/Adam/scheduler/RNG restoration and
numerical equivalence of the next update. Parent inference exactly reproduced
all cached validation predictions on common support.

The first `v01` attempts stopped before experiment training because the check
incorrectly demanded bitwise equality of repeated CUDA gradients. The unchanged
historical model ends with bilinear interpolation; PyTorch documents its CUDA
backward nondeterminism. The revised test separately requires exact saved-state
restoration and next-update agreement at rtol 2e-6, atol 1e-8. Observed maximum
weight differences were 3.73e-9 in both branches, with identical
loss and LR. The historical kernels and trainer were not changed. Failed-run
artifacts are retained. The `v02` runs were subsequently stopped after the first
epoch: preflight updates had advanced aliased CPU Adam step counters by three.
Those runs are marked invalid and must not be used for comparison. The parent
on disk was unchanged. `restore_parent()` now deep-copies optimizer state; a
regression test and runtime assertions verify the exact inherited counter.
The corrected `v03` runs are the actual experiments.
Read-back checks of both saved `v03` epoch-zero checkpoints confirmed exact
agreement with every parent model/Adam state tensor, including step 109,123.

Each allocator is capped at 1 GiB; peak reserved at startup 814 MiB and sampled
free device memory 972 MiB. Both Ray workers continue. The trainer checks free
memory every 200 updates and before validation, and stops below 256 MiB while
retaining completed epoch checkpoints. There is no guaranteed compute share.

Use each run's `live.json`, `progress.json`, `status.json` and console exit status
to establish current progress/completion. Resumption is at an epoch boundary:
use the exact original command with `--resume` in a **new** tmux/console session.
Do not change the trainer/source snapshots, GPU UUID or configuration mid-run;
the saved signature must match.

`code/compare_residual_refinements.py --wait` provides automatic local CPU
analysis after both statuses report completion. Output:
`runs/resnet-refinement-comparison-v01/`, containing `report.md`,
`comparison.json`, shape arrays, source snapshots, and `comparison`,
`learning_curves`, `shapes`, `fixed_examples` PNG/PDF files. It verifies equal
batch/augmentation stream hashes across all ten epochs and compares selected
best and final weights with the parent. Confidence intervals use the frozen
2,000 paired spatial-block draws. Shape comparisons use common successes
across reference and all five evaluated variants and report failure counts.
Example fields/display scales reuse the existing observed-image-selected
Figure 5 v06 set; they are illustrative, not a population performance sample.
Approved Figure 6 inputs/exports remain unchanged and local-only.
The waiting comparison process is in tmux `supernisp-refine-comparison-v02`,
console `runs/resnet-refinement-comparison-v02-console/`. The first comparison
watcher exited on the deliberately stopped invalid runs and produced no report.
The comparison's analytic paired-RMSE check, adaptive-shape smoke check and
four PNG/PDF plot exports passed using cached baseline data in `/tmp`.

## Previous update: shared-GPU trial passed

The user subsequently authorized a short sharing trial. It completed 100
batch-eight ResNet training steps on occupied GPU 0: median 0.05955 s/step,
peak sampled total process memory 1,308 MiB, and minimum free memory 982 MiB.
PyTorch's allocator was capped at 1 GiB. All losses/gradients were finite;
trial weights were discarded and the original checkpoint is unchanged.
Both Ray workers remained running; their throughput impact was not measured.
GPU memory returned to its pre-trial level after exit. See
[the measured trial report](shared_gpu_probe.md) for evidence and limitations.
At this stage no full refinements were running; the trainer and checks were
still pending. Shared execution became feasible under the observed load.

## Earlier status and user request (superseded by launch above)

The user approved testing the suggested ResNet improvements with two GPUs in
parallel, then asked to write the plan in notes because they will move to another
machine. **No full refinement training has started and no refinement jobs are queued.**
On the current host, both L40S GPUs were occupied by Ray workers at 100%
utilization and approximately 43 GB each. Do not stop those workers.

The new continuation trainer, paired launcher, and final comparison have **not
been implemented yet**. `code/train_paired.py` is the frozen historical trainer;
its `--resume` path is not a way to append a new schedule to a completed run.
CPU mask preparation **completed successfully** using
`code/prepare_refinement_masks.py`. The verified output is
`runs/residual-refinement-masks-v01/`, including `source_masks.h5`,
`metadata.json`, the frozen contract and source snapshots. It contains all
17,816 training and 2,118 validation objects; every object has nonempty source
and outside-source regions. Median source fractions on the full training and
validation fields are 13.34% and 13.27%, respectively. Validation masks reproduce
the existing Figure 6 masks exactly after cropping/intersection. HDF5 completion,
metadata and file hash were verified. This preprocessing used neither GPU.

Mask HDF5 SHA256: `62ebbff02160dd2cba3c416adbf7700db6612ed83fa9b9d6f923fb56eab5ec93`.
After transfer, check both the HDF5 completion flag and metadata/hash; file
existence alone does not establish a valid completed bundle.

## Why these two runs

Current ResNet/ResNet-D5 perform best on source errors, SSIM, Fourier correlation,
and matched observed-shape recovery, while suppressing observed small-scale power.
The existing ResNet reached a practical plateau under its 50-epoch OneCycle
schedule: validation MAE improved 0.57% from epoch 40 to 49 and 0.15% from 45 to
49. This does not exclude improvement with a fresh schedule or a different loss.
The objective is to distinguish optimization headroom from the effect of source
weighting, without changing architecture or increasing only one model's budget.

## Frozen starting point and shared settings

Use **ResNet**, internal head `legacy`, not ResNet-D5. Both branches start from
exactly the same epoch-49 weights **and Adam optimizer state**:

- Checkpoint: `runs/residual-full50-s20261016/best.pt`.
- SHA256: `be747f0cd0b02bb81ff0ab42a9ce8ea81a0e473c28c58d8406a428f758ea6b56`.
- Starting epoch: 49; completed optimizer updates: 109,123.
- Architecture: unchanged 2,332,177-parameter residual model.
- Data: portable `supernisp_nir_y_f115w_v1` package, normally beside the repo.
- Use all 17,816 saved training and 2,118 saved validation objects.
- Test split remains reserved; do not load test images or fit new normalizers.
- Linear, signed intensities, fixed training-only normalization, same full
  205×205 target field, batch size 8, and paired D4 augmentation.
- **10 additional epochs per branch**, 2,227 updates per epoch = 22,270 extra
  updates each. These are new refinement experiments, not part of the original
  matched 50-epoch architecture comparison.
- New schedule: cosine decay per successful optimizer update, initial LR
  **1e-5**, final LR **1e-6**, over exactly 22,270 updates, without restarts.
  Preserve Adam moments/step counts; replace the exhausted OneCycle schedule
  and its LR metadata. Use the same optimizer beta settings in both branches.
- Restore the same starting training sampler/RNG state in both branches so
  minibatches and augmentations match. The parent seed is 20261016. Map CUDA RNG
  state correctly to the selected GPU on a different machine; do not assume
  the old checkpoint's two-device CUDA-state list matches the new machine.
- Retain float32, deterministic cuDNN settings, no AMP/TF32, and no gradient
  clipping, matching the original run. Check gradients for nonfinite values.

These are the proposed fixed settings for the next implementation; freeze them
in a machine-readable experiment contract before looking at new results.

## Two simultaneous GPU jobs

| Branch | GPU | Training objective | Purpose |
| --- | --- | --- | --- |
| A: low-LR control | first free GPU | Original per-object valid-pixel masked L1, averaged over objects | Does additional optimization improve the existing model? |
| B: source-balanced loss | second free GPU | Per object: `0.5 * source_MAE + 0.5 * outside_source_MAE`, then average objects | Does source emphasis improve fidelity at the same extra training budget? |

Every regional MAE is divided by that object's region pixel count. This is equal
weight between regions, not equal weight per source pixel across the dataset.
Use exactly the same normalized intensities as branch A. Source and outside
masks partition the target-valid support. If a region is empty, renormalize
weights over the present regions for that object, report its count, and do not
drop the object. Apply precisely the same rotation/reflection to images and
source/validity masks. No extra gradient, Fourier, perceptual, or shape loss in
this first ablation: those would introduce another experimental change.

Build source masks from each split's own JWST targets with the existing frozen
2-sigma recipe in `code/source_masks.py`: sigma-1-pixel Gaussian detection,
clipped sky statistics at radii 70–95, at least 10 connected pixels, filled
holes, radius-3-pixel dilation, all neighbours included. For training, retain
the **full 205×205 mask**, intersected with JWST validity. Training masks must
come from training targets; validation labels never define a training weight.
No mask parameter tuning against new model results.

The new preparation script supports these train/validation masks:

```bash
python code/prepare_refinement_masks.py \
  --data ../supernisp_nir_y_f115w_v1 \
  --output runs/residual-refinement-masks-v02
```

Use a fresh output name if v01 is partial; do not overwrite a completed bundle.
The script verifies that its validation masks, cropped to 105×105 and
intersected with common support, reproduce the existing Figure 6 source masks.
It expects `runs/figure6-source-masks-v03/source_masks.npz` to be available.

## Selection, monitoring, and comparison

1. Evaluate the starting checkpoint as **epoch 0** in each branch. Reproduce the
   current cached source/outside scores before allowing updates. If the match
   fails, resolve normalization/support/model differences first.
2. Use the **same primary checkpoint criterion for both branches: mean source
   MAE** on the fixed central-105×105 validation common support and existing
   2-sigma masks. Include epoch 0 as a candidate: keep the original checkpoint
   if none of the ten new epochs improves it. This is intentionally a new
   shared selection criterion, distinct from the historical full-field MAE.
3. Track original full-field masked MAE, central source/outside MAE and RMSE,
   source Fourier power and signed correlation, observed shapes and failure
   counts. Spectral monitoring should use the existing 1,363 complete fields
   and strict 3-pixel source taper; foreground RMSE uses all 2,118 objects.
   Do not let epoch-by-epoch measurement failures silently change the apparent
   shape comparison population. Shape and spectrum scores are diagnostics,
   not additional checkpoint selection criteria.
4. Log the same fixed galaxy and residual gallery in both runs. Include bright,
   resolved sources in a separate reference/input-selected diagnostic sheet
   to inspect coherent missed features versus noise or misregistration.
5. Save last/best checkpoints with optimizer, new scheduler, RNG and loader
   state. Record parent checkpoint hash, exact source snapshots, sample and mask
   hashes, initial metrics, per-epoch metrics, run time, and extra update budget.
   Preserve all original model checkpoints and paper inputs.
6. Compare original, A, and B on identical object IDs, masks and successful-shape
   intersections. Use 2,000 paired spatial-block bootstrap draws over 18 blocks,
   equal object weighting, seed 20261017. Report source/outside errors,
   source power **and** reference correlation, fixed/adaptive shape RMSE,
   success counts and paired changes. Higher observed power by itself is not
   evidence of improvement. Show last-epoch results alongside selected best
   results so selection differences are visible.
7. Keep new reconstruction-analysis bundles and plots separate from the approved
   Figure 6. Do not replace its inputs or install a new model in the manuscript
   based only on this exploratory validation experiment.

If a branch looks useful, the next experiment is an independent-seed replication
of the baseline and promising configuration. Forks from one parent checkpoint
are **not** independent training-seed replicates. Only after that should a small
multiscale/gradient-loss ablation be considered. Do not launch an unrestricted
architecture/loss search or evaluate the held-out test set at this stage.

## Implementation/check contract (completed for the launch above)

- Add a separate refinement trainer/runner rather than altering the historical
  trainer used by checkpoint source-hash verification.
- Add a source-mask dataset wrapper with worker-safe HDF5 access and paired
  augmentation. Validate object IDs/rows and mask/data hashes.
- Test 50/50 per-object weighting, empty-region handling, exclusion of invalid
  pixels and their gradients, mask augmentation, and source-mask reproduction.
- Test fresh schedule length/LR endpoints and an interrupted/resumed run's
  equivalence, including optimizer and random-state restoration.
- Run a small finite-loss/gradient and checkpoint-reload check before both full
  refinements. Preserve the preflight and parent states so these checks cannot
  consume the actual experiment's RNG sequence or alter its starting weights.
- Recheck available memory immediately before launch and pin each experiment
  to its assigned GPU. The user authorized the bounded sharing probe described
  above; retain memory limits and free-memory monitoring for any shared run.
  Do not stop or reconfigure other workloads. Full paired shared training has
  not yet been launched or validated.
- **Existing user requirement:** launch model executions inside tmux/screen,
  with online W&B initialized before model execution. Use the established
  `AI-Astro/superNISP-corrected` project. Log new training diagnostics/gallery;
  do not upload the local-only Figure 6 artifacts. Do not silently fall back
  to offline monitoring or foreground training.
- `code/launch_paired.py` is the existing persistent-session launcher; the new
  refinement command still needs implementing. Do not invoke the old trainer's
  `--resume --epochs 60`: it correctly rejects a changed signature/schedule.

## Files to bring to the new machine

The repository alone is insufficient: `runs/` and large analysis arrays are
ignored by Git, and the latest local code/notes/figure changes are not committed.
Transfer or otherwise preserve those working-tree changes deliberately.

Required training assets:

- Portable `supernisp_nir_y_f115w_v1` package, with manifest, normalization,
  HDF5 sample, object catalog, split indices and checksum files.
- `runs/residual-full50-s20261016/best.pt`, `provenance.json`, `selection.json`,
  `metrics.jsonl`, and its `source/` snapshots (parent model and optimizer/RNG).
- A completed `runs/residual-refinement-masks-v*/` bundle, or regenerate it.
- `runs/figure6-source-masks-v03/source_masks.npz` and `.json`.
- `paper_figures/inputs/model_benchmark_v03.json`.

Assets for exact current-reference diagnostics and final comparisons:

- `runs/figure6-v03-b8/predictions.h5` (original frozen predictions and targets).
- `paper_figures/inputs/spatial_spectra_source_v01.json` and `.npz`.
- `paper_figures/inputs/spatial_spectra_v01.json` and `.npz` as needed for controls.
- `paper_figures/inputs/cached_shapes_v01.json` and `.npz` for fixed/adaptive
  shape comparison populations; associated frozen contracts/source snapshots
  in `runs/cached-shapes-v01/` are useful for exact reproduction.

Before running, verify the transferred data/checkpoint hashes, inspect installed
Torch/CUDA and scientific packages, and run `nvidia-smi` outside any execution
sandbox that hides the driver. Set the new data directory explicitly; do not
reuse the old absolute `/home/shemmati/...` paths blindly.

## Current paper/figure state

The sole active Figure 6 notebook is `paper_figures/06_model_comparison.ipynb`.
Its current export is **v06**, one row with combined source/outside RMSE,
power and signed correlation. Filled circles/open squares encode RMSE regions;
solid/dashed curves encode spectral regions. Panel titles and footer notes were
removed at the user's request; their information is in the working caption.
The notebook also contains a separate exploratory PSNR/SSIM plot (`v01`).
Earlier three notebooks and successive layouts are archived in `revisions/`.
Do not restore the old three-notebook layout from older handoff sections.
