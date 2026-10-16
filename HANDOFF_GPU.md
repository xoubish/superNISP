# superNISP GPU handoff

Updated 2026-10-16. Read this first when continuing on `ipacgpu3`.
This records completed work and the next experiment; it does not certify
the old paper results or claim a trained model is ready.

## Validation analysis and queued completion workflow

While full50 training continues, the user asked to work on the other steps.
Implemented standalone validation analysis without changing any active training
source (all four training source hashes were rechecked and still match).

- `code/shape_diagnostics.py`: the existing signed moment definition with
  explicit failure reasons; native/bilinear/bicubic baselines in matched angular
  regions; common-success comparisons, round-shape control, paired spatial-block
  bootstrap intervals and fixed property bins.
- `code/evaluate_paired_shapes.py`: immutable checkpoint copy, verified code/data
  hashes, full validation only, row/ID catalog joins, per-object CSV/NPZ, report,
  plot and W&B failure table. Runs on GPU 1 in tmux with online W&B.
- Four metric/geometry/mask/bootstrap regression checks passed. Two comparison
  regression checks passed (IDs/reordering/unchanged baselines). Real-data
  same-checkpoint comparison correctly returned zero change and zero lost IDs.
- [Epoch-11 analysis od7fm46d](https://wandb.ai/AI-Astro/superNISP-corrected/runs/od7fm46d)
  completed on all 2,118 validation objects. The immutable snapshot and detailed
  results are in `runs/validation-shapes-full50-snapshot/`.
- 1,527 objects pass JWST and every baseline; 43 fail SR, leaving 1,484 common
  successes. On those same objects, component ellipticity RMSE is native
  0.20698, bilinear 0.20074, bicubic 0.20741, SR 0.17759, round control 0.28872.
  These are intermediate validation diagnostics, not final test/shear results.
  All 43 SR failures on baseline/reference eligible objects have input S/N proxy
  1.33–3.75 and ACS F814W magnitude 24.15–24.99. Failure flags retain negative
  flux/indefinite moments; no clipping or new sample cut was applied.

**A final-analysis job is queued in `supernisp-final-analysis`.** It waits for
the existing training to exit successfully with 50 completed epochs, then:

1. Evaluates `best.pt` on validation using GPU 1, into
   `runs/validation-shapes-full50-best/` with a separate online W&B run.
2. Uses `code/compare_shape_runs.py` to compare it with epoch 11 on identical
   common object IDs, saving recovered/lost measurements and paired uncertainty
   estimates into `runs/full50-shape-comparison/` with another W&B run.

Follow `runs/final-analysis-console/console.log` or attach with
`tmux attach -t supernisp-final-analysis`. Check its `exit_status` when done.
Do not duplicate this pending job. The watcher aborts after a failed/incomplete
training exit or a 24-hour wait. It does not touch test data or train models.

Read [validation diagnostics](paper/validation_shape_diagnostics.md) for the
measurement contract, limitations, commands and preliminary results. The tracked
machine-readable snapshot is `paper/diagnostics/validation_epoch11_shapes.json`.
Next: inspect the queued best-checkpoint analysis and fixed-population comparison
when training completes, then choose one controlled architecture ablation.

## Active full-split training and image monitoring

The user requested a longer run and galaxy images in W&B after the pilot.
**A 50-epoch run is now launched; inspect its status before starting anything
else on CUDA device 0.**

- W&B: [residual-full50-s20261016 / nifpxes6](https://wandb.ai/AI-Astro/superNISP-corrected/runs/nifpxes6).
  Remote API verification confirmed state `running`, 1,120 optimizer updates,
  and all four epoch-0 image panels. This is a startup snapshot, not completion;
  current status is in W&B and the run directory.
- Persistent session: `tmux attach -t supernisp-full50`.
- Console: `runs/full50-console/console.log`; `exit_status` appears on exit.
- Artifacts: `runs/residual-full50-s20261016/`, including per-epoch
  `progress.json`, `last.pt`, `best.pt`, and final `status.json` on completion.
- Uses all 17,816 train / 2,118 validation examples, 50 epochs / 111,350
  optimizer updates, batch 8, seed 20261016, max LR 1e-4. It starts from scratch
  with a matching 50-epoch schedule and the unchanged legacy residual model.
  Best model selection remains validation MAE. Test data remain reserved.
- Eight fixed evaluation galaxies appear under W&B `galaxies/full_1`,
  `full_2`, `central_1`, and `central_2`, at initialization and every epoch.
  Panels show NISP native sampling, bilinear NISP, SR, JWST and signed residual.
  Per-object physical-intensity/asinh display scales stay fixed across epochs;
  central crops, object IDs, masks and moment-check failures are visible.
- `code/galaxy_visuals.py` is included in source snapshots and hashes.
  Rendering preserves Python/NumPy/Torch/CUDA RNG state, so it does not alter
  training/resume random streams. The trainer supports `--gallery-count`,
  `--image-every`, and `--mode train`.

[Visual checks sfgwa5ku](https://wandb.ai/AI-Astro/superNISP-corrected/runs/sfgwa5ku)
passed in tmux and synced eight image files. Locally inspected panels show
the pilot predictions and a separate set of failed-moment examples. All 20
pilot prediction failures were reproduced; 13 have passing JWST moments.
This does not yet diagnose their quantitative cause. The longer run retains
the existing metric and failure-count logging.

Next: inspect full-run learning curves and fixed galaxy panels, confirm best
checkpoint selection and shape failure rates, then plan architecture comparisons.
See [the training guide](paper/corrected_training.md) for image conventions and
resume instructions. Older completed pilot results are recorded below.

## GPU continuation — 2026-10-16

The extracted sibling package is present and all 46 `SHA256SUMS` entries pass.
No rebuild or transfer is needed. CUDA works outside the Codex execution
sandbox: PyTorch 2.6.0+cu124, NVIDIA driver 590.48.01, two L40S GPUs.
An in-sandbox `nvidia-smi` failure does not mean the host driver is broken.

**User requirement:** run models inside tmux/screen so disconnects do not
terminate work, and attach online W&B monitoring before running models.
The user approved project `superNISP-corrected` under the existing account;
it resolves to [AI-Astro/superNISP-corrected](https://wandb.ai/AI-Astro/superNISP-corrected).
Do not fall back silently to offline monitoring or foreground training.

Implemented `code/residual_baseline.py`, `code/train_paired.py`,
`code/launch_paired.py`, `code/check_paired_training.py`, and
`code/run_paired_pilot.py`. See [training contract and commands](paper/corrected_training.md).
The new baseline matches the original eight-block, 64-feature model exactly.
The trainer uses the corrected loader, masked loss, update-level OneCycleLR,
validation-only pilot selection, provenance, and resumable epoch checkpoints.
The direct-5x head is an explicit option but has not been trained or ranked.

Integration checks passed in persistent session `supernisp-check`, tracked at
[W&B lkg2d7mu](https://wandb.ai/AI-Astro/superNISP-corrected/runs/lkg2d7mu).
Checks cover both loader representations, worker loading/flux round trips,
legacy output equivalence, finite CUDA gradients for both heads, masked-pixel
gradients, paired augmentation, fixed moment diagnostics, and exact optimizer
update restoration from checkpoint including scheduler and RNG state.

The gated initial experiment was launched in tmux session `supernisp-pilot`:

```bash
tmux attach -t supernisp-pilot
tail -f runs/pilot-console/console.log
```

Sequence: 32 fully valid training pairs, 40 epochs / 160 updates; deliberately
stop after epoch 2 and resume; require a training-loss decrease; then a fresh
3-epoch pilot on 1,024 training / 256 validation objects (384 updates).
Both experiments completed successfully; the console exit status is 0 and
W&B finished syncing. The tmux session remains open at an idle shell.

- [Overfit emr5ag36](https://wandb.ai/AI-Astro/superNISP-corrected/runs/emr5ag36):
  all 40 epochs / 160 updates, planned resume succeeded, masked training MAE
  1.1754255 at initialization to 0.9242856; about 22 seconds of measured
  train/evaluation time, excluding process setup and W&B startup/sync.
- [Pilot 9c47c2of](https://wandb.ai/AI-Astro/superNISP-corrected/runs/9c47c2of):
  all 3 epochs / 384 updates, validation MAE 1.4428717 at initialization to
  1.1082243, physical-intensity RMSE 0.4275776 to 0.3729545 MJy/sr.
  Best checkpoint is epoch 3; about 71.5 seconds train/evaluation wall time,
  peak allocated CUDA memory 0.577 GiB.
- Prediction moment-diagnostic failures rose from 0 to 20/256; reference
  failures remained 25/256, and paired successes fell from 231 to 218.
  The apparent shape RMSE decrease cannot establish shape improvement with
  this changing population. No test data were used and no scientific
  SR-improvement percentage was measured.

Checkpoints and provenance are in `runs/residual-{overfit,pilot}-s20261016/`.
Tracked metrics, checkpoint/source hashes, and checks are in
`paper/diagnostics/corrected_training_pilot.json`. Runs and weights are gitignored.

Next: inspect pilot predictions and the 20 moment failures, add fixed matched
interpolation/native comparison diagnostics, then choose a full-split baseline
budget before architecture comparisons. The original startup and implementation
checklist below describes the state before this continuation.

Pre-existing local edits in `code/test_model.ipynb` and untracked
`code/older/{losses_olddiff,model_olddiff,train_olddiff}.py` were left untouched.

## Objective and agreed order

Finalize the Euclid NISP / JWST superresolution paper with a reproducible
answer to whether SR improves galaxy shapes, by how much, and for which
galaxies. The user asked to review architecture, code, and sample **before**
improving downstream measurements. They then asked to prepare the correct
sample locally so only a compact package needs transferring to the GPU.

The sample rebuild is complete. Next: integrate the new loader into a
correct training pipeline, establish a residual baseline, and compare a
small set of architectures under controlled conditions. Shape-estimator
development and final paper numbers come later. The improvement percentage
is an experimental outcome, not a target to force.

## Locations and repository state

Expected GPU layout (transfer/extraction has not been confirmed in this chat):

```text
/home/shemmati/Work/Projects/
├── superNISP/                         # https://github.com/xoubish/superNISP.git
└── supernisp_nir_y_f115w_v1/           # extracted portable sample
```

Local source workspace: `/Users/shemmati/Dropbox/superNISP`.
Local package: `data/ready/supernisp_nir_y_f115w_v1/`.
Archive: `data/ready/supernisp_nir_y_f115w_v1.tar`, 3,585,710,080 bytes.
Archive SHA256:

```text
91cb6b09cd1cfd25cf99cb68441edcdd062bb73b82fa76eb2a272e955430bf33
```

The audit and sample code are committed in `61705a3` (`paper1`), following
the original audited baseline `5e2fe67`. At handoff preparation, local main
and its cached origin/main both pointed to `61705a3`; no live remote check
was needed for this document. This handoff is a subsequent change.
Earlier chat statements that the sample code was uncommitted are superseded.
Data are gitignored: cloning does not download the prepared sample.
The tar includes copies of the sample code, but active development should
use the repository's `code/` files and record the code revision per run.
Keep run outputs under the repo in run-specific directories, with large
artifacts gitignored and their hashes/provenance recorded in tracked reports.

## Completed checks — do not restart these without new evidence

- Inspected local branches and reachable history, plus live remote branch
  refs at the time of the audit. The exact checkpoints and result arrays
  behind the paper figures were not recovered. A diffusion-named checkpoint
  in another branch was actually the same blob as a residual checkpoint.
  Do not repeat this hunt unless a new location or artifact becomes available.
- Reviewed residual and diffusion source, loaded compatible old residual
  checkpoints, and ran finite forward/backward checks. These were execution
  checks, not architecture rankings or scientific training results.
- Found old extraction defects: RMS mistaken for DQ, reflected mask transfer,
  padding/cropping issues, and lost accepted-object IDs. Old array versions
  largely overlap and must not be treated as independent populations.
- Rebuilt from original NIR-Y stacks and COSMOS-Web F115W mosaics using named
  SCI/RMS/DQ layers, explicit WCS mapping, independent masks, and retained IDs.
- Verified all stored stamps for finite values, invalid fill, coverage,
  central validity, IDs, and split consistency. Confirmed no cross-split
  object centers within 20 arcsec.
- Passed five synthetic extraction regression tests and a loader smoke test
  with two spawned workers, physical-intensity round trips, masked loss,
  and one tiny CPU optimizer step in both linear and asinh representations.
- Checked WCS/image alignment on a seeded subset: 86 usable bright fits from
  512 sampled objects; median shifts essentially zero, largest fitted
  component 0.341 NISP pixel. No empirical shifts were applied. This is an
  approximate diagnostic, not precision astrometric calibration for all objects.
- Verified all 46 archived regular files against their package checksums.
  Transfer integrity and GPU execution still need checking on the destination.

Evidence and details:

- [Corrected sample build](paper/corrected_sample_build.md) — completed rebuild.
- [Architecture/sample review](paper/architecture_sample_review.md) — findings
  and proposed comparisons. Its original final extraction step is now complete.
- [Shape-validation audit](paper/shape_validation_audit.md) — paper/evaluation
  issues and the eventual quantitative science test.
- [Machine-readable original audit](paper/diagnostics/training_pipeline_audit.json).
- Package `README.md`, `manifest.json`, `validation.json`, `BUILD_REPORT.md`.

## Frozen sample and training contract

**One band pair:** Euclid NISP Y input, JWST NIRCam F115W target.
41×41 native NISP pixels → 205×205 target pixels on exactly the same WCS
footprint. Fivefold pixel sampling is not a demonstrated fivefold gain in
resolved detail. NISP sampling is about 0.297 arcsec; target about 0.0593 arcsec.

| Split | Galaxies |
| --- | ---: |
| Train | 17,816 |
| Validation | 2,118 |
| Test | 2,876 |
| Total | 22,810 |

39,024 candidates verified against COSMOS2020. Selection: `lp_type == 0`,
ACS F814W < 25, 0.01 < photo-z < 3, and 4 < FLUX_RADIUS < 500 in catalog
units. Rejections: 8,229 spatial buffer; 7,043 coverage; 723 central invalid
region; 219 partial NISP cutouts. This is an optically selected COSMOS sample.

Spatial splits use 180-arcsec blocks, seed 20261016, nominal 80/10/10
assignments and 10-arcsec border exclusion. Use saved indices; do not randomly
re-split. At least 95% valid coverage in each instrument and a fully valid
central 5×5 NISP / 25×25 JWST region. No additional S/N cut was imposed.

Use `code/paired_sample.py`:

```python
from paired_sample import PairedSample, masked_l1

train = PairedSample(data_dir, split='train', representation='linear')
validation = PairedSample(data_dir, split='validation', representation='linear')
# batch['nisp']: B,1,41,41; batch['jwst']: B,1,205,205
# loss = masked_l1(model(batch['nisp']), batch['jwst'], batch['jwst_valid'])
```

The import works in scripts inside `code/`; otherwise add `code/` to the
Python module search path. Existing training loaders expect a different
schema and cannot simply be pointed at this HDF5 file.

- HDF5 stores NISP science/RMS in microJy per native pixel, JWST in MJy/sr,
  independent validity masks, NISP DQ, IDs, splits, and calibration metadata.
- Loader converts NISP to MJy/sr, subtracts per-image outer-region background,
  and divides by fixed, instrument-specific **training-only** scales.
  Do not refit normalization or use per-image target variance scaling.
- `representation='asinh'` applies `asinh(z/3)`; it is unbounded. Neither
  representation may be clamped to [-1,1]. Start with linear as the reference;
  compare asinh separately if needed.
- Invalid pixels are zero filled but are not zero-flux observations. Apply
  `jwst_valid` to the loss. Input mask/noise conditioning can be an explicit
  ablation; the current loader exposes both. RMS always refers to linear space.
- `jwst_to_mjy_sr()` inverts normalization. Default background zero gives
  sky-subtracted predictions. The reference's `jwst_background` is available
  for round-trip checks, not as an input for NISP-only deployment.
- Do not manually flip or rotate to align the images. Augment training pairs
  consistently, including masks and any PSF information; do not augment validation.
- Retrain from scratch. Old checkpoints may have seen the new test objects.

## GPU startup: verify transfer and environment only

If the tar is still present, verify before extracting:

```bash
cd /home/shemmati/Work/Projects
sha256sum -c supernisp_nir_y_f115w_v1.tar.sha256
# Only after OK, and if not already extracted:
tar -xf supernisp_nir_y_f115w_v1.tar
```

With the extracted directory:

```bash
cd /home/shemmati/Work/Projects/supernisp_nir_y_f115w_v1
sha256sum -c SHA256SUMS
cd /home/shemmati/Work/Projects/superNISP
git status -sb
git log -1 --oneline
nvidia-smi
python -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No CUDA device")'
python code/smoke_test_sample.py ../supernisp_nir_y_f115w_v1
```

Inspect the existing Python environment first. Loader dependencies are NumPy,
h5py, and a GPU-compatible PyTorch. Package `environment-used.json` records
the Mac verification environment, not a CUDA installation prescription.
The existing smoke test is **CPU-only** even on a GPU host; separately verify
a CUDA forward/backward step when integrating the real training entrypoint.
Do not download original mosaics or rerun the full audit/build just to start.
The optional full validator regenerates `validation.json` and `preview.png`;
verify checksums first if rerunning it for a concrete reason.

## Next implementation work (not yet done)

1. Create a portable training entrypoint using `PairedSample`, explicit data
   and output paths, masked loss, validation-only model selection, and saved
   configs/seeds. Avoid import-time sweeps or hardcoded NERSC paths in old files.
   Record manifest/normalization hashes, code revision, environment, optimizer
   and scheduler state, and best-checkpoint selection in each run directory.
2. Establish a corrected current residual baseline. `claude_model_NIR_2.py`
   used eight single residual dense blocks, 64 features, two 2× pixel-shuffle
   stages then resize to 205 (2,332,177 parameters). These are not nested ESRGAN
   RRDBs. `claudemodel.py` has an existing direct-5× design with a different
   trunk; comparing those two unchanged does not isolate the upsampling head.
   Fix OneCycleLR stepping to each successful optimizer update if using it;
   the old HDF5 path steps it once per epoch.
3. Run a small training-only overfit experiment (about 32 clean pairs), finite
   CUDA gradients, masked-pixel checks, and checkpoint reload/resume checks.
   Then run a short train/validation pilot before committing to a long run.
4. Compare the current head with direct-5× using the same trunk, sample,
   normalization, augmentation, loss, seeds, and optimizer-update budget.
   Test residual scaling or removing the edge enhancer separately. Consider
   a similarly sized residual UNet after this baseline. Report GPU time and
   memory as well as quality; repeat finalists with at least three seeds.
5. Keep a fixed, simple linear-intensity shape diagnostic on validation
   objects during comparisons, alongside reconstruction errors. Do not use
   sharper appearance alone to pick a model. Reserve final test evaluation
   until model and analysis choices are frozen.

Diffusion is a later comparator: its old shortened sampler uses adjacent-step
coefficients while skipping timesteps; its output clamp conflicts with asinh;
normalizers refit on held-out data; and validation uses noisy HR reconstruction
instead of full LR-only generation. Repair and test these before judging its
architecture. Production training code has not yet been repaired by this handoff.

## Scientific boundaries and eventual result

JWST SCI-only coverage masks use finite/nonzero support; no full JWST RMS/DQ
was supplied. Bilinear resampling affects PSF and noise correlations. Supplied
NISP PSFs retain their original sampling headers, which need inspection before
use. Y and F115W differ: do not impose naive blur/downsample equality as a
physical constraint without a cross-band model.

The old paper figures/numbers remain unverified. GalSim `observed_shape.g`
is a shape parameterization, not calibrated gravitational shear. Measure
agreement with the JWST reference in linear intensity, with matched angular
regions, object populations, and explicit measurement failure rates.
Ultimately compare native NISP and interpolation with SR using ellipticity
RMSE, e.g. `100 * (1 - RMSE_e(SR) / RMSE_e(baseline))`, with paired spatial
uncertainty estimates and bins in galaxy properties. Regression slopes alone
do not establish that percentage. No shape-improvement result has yet been
measured on this rebuilt sample.

Update this document with run IDs, changes, checks, and the next concrete
step as work proceeds, so another session can continue without restarting.
