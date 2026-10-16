# Architecture, training, and sample review

2026-10-16; source baseline `5e2fe67`. This review precedes improvements to
downstream shape estimators. It combines source inspection, local checkpoint
loading, forward/backward execution, synthetic extraction fixtures, FITS
header inspection, and seeded diagnostics of the actual training arrays.
It does not rank trained architectures or establish a final science result.

## Decision

There is no evidence yet that the current network configuration is the best
we can achieve. More immediately, there are confirmed input and training
problems that make a larger architecture search premature. Repair and freeze
the sample construction, normalization, and training contract first; then
compare a small set of architectures under the same conditions on a GPU.
The current residual models are usable starting points, not designs to discard.

## Reproducible evidence

Run from the repository root:

```bash
python code/audit_training_pipeline.py
```

The script writes `paper/diagnostics/training_pipeline_audit.json` and
`paper/diagnostics/training_input_examples.png`. It records source hashes,
versions, random seed, and sampled row indices. It does not modify inputs,
load training services, or run optimization. The network/extraction checks
execute selected original definitions without their modules' import side
effects; they are not full training-entrypoint integration tests.

### Sample and preprocessing

| Check | Result | Consequence |
| --- | --- | --- |
| Finite/nonconstant stamps | All 256 sampled pairs per array version passed | Encouraging basic integrity; not a full pixel scan |
| Exact zeros | 163/256 undated and 168/256 dated pairs contain zeros | Missing-data treatment affects many sampled examples |
| Mask orientation | After 5x block reduction, every sampled JWST zero mask equals the horizontally flipped NISP zero mask | Confirmed mismatch in the stored pairs, consistent with the extra flip in `clipping.py` |
| Same-orientation mask disagreement | 4.71% and 5.22% of sampled LR-grid positions | This is not just a hypothetical extraction concern |
| Overlap between versions | 34,044/34,058 dated LR stamps have byte-identical undated counterparts | These are overlapping versions, not independent training/testing populations |
| HR overlap check | 255/256 sampled corresponding HR stamps are exactly equal | Most sampled matched pairs are identical; some targets differ, so versions cannot be silently interchanged |
| Within-version exact LR duplicates | None among all 53,229 undated or 34,058 dated LR hashes | Does not rule out overlapping sky footprints or repeated objects in different cutouts |

The mask check uses exact zeros as a proxy, not original quality flags.
However, exact agreement after reflection across both samples, plus the
synthetic reproduction, makes this a concrete issue to fix. `clip_images()`
flips the NISP data and mask together, then flips the mask again before
applying it to the rotated JWST target. The stored zero locations agree with
that behavior. This is evidence of a mask mismatch, not evidence that the
entire JWST intensity image should be flipped.

Do not attempt to recover zeroed JWST flux by moving its existing zero pixels.
Regenerate targets from the original mosaics, retaining each instrument's
validity mask and transforming it with its image geometry. Pass input masks
to the model if needed and exclude invalid target pixels from the loss.
Zero-valued missing pixels must not be treated as measured zero flux.

Additional synthetic fixtures reproduce two extraction failures:

- `mask=False` accesses an undefined `clip_mask` and raises `UnboundLocalError`.
- With zero rotation and nonzero padding, the JWST image is not cropped to
  its requested size; masking then raises a shape-mismatch error.

The metadata uses `pad_jwst`, while `clip_images()` accepts `pad`; the former
is swallowed by `**kwargs`. Fix this interface instead of relying on defaults.
The helper `pixel_scale()` also adds rather than subtracts pixel positions;
prefer Astropy's WCS utilities and verify any callers before reuse.

### Sample provenance and sky geometry

The two candidate catalogs have unique IDs within each catalog, but their
row counts do not match the saved cutout counts. Extraction groups by image
pair, drops rejected objects, and saves only the image arrays. Recover the
accepted order or regenerate with an accepted-object manifest; never assign
IDs by simply joining on row position.

The undated catalog references MER products (16 NISP tiles, 20 JWST tiles);
the dated catalog references NIR stacks (4 NISP images, 19 JWST tiles).
Four inspected MER files have 0.10-arcsec sampling and negative WCS matrix
determinants; the four dated NIR stacks have approximately 0.2967-arcsec
sampling and positive determinants. The inspected COSMOS-Web mosaic has
0.06-arcsec sampling. This corrects the initial suggestion that 0.10 in the
evaluation must be wrong: it depends on the product. The near-complete array
overlap also shows that catalog filenames alone do not establish provenance.

A 41-to-205 array-size ratio does not guarantee the same field of view.
Use WCS to define a common angular footprint and orientation and measure
residual registration on well-detected, isolated sources, by tile. For NIR
stacks, the nominal factor of five is only approximately the local WCS scale
ratio. A fixed rotation plus flip is not a substitute for checking that grid.
Avoid deriving noisy per-object shifts from faint or blended sources.

Selection in the current `cut_catalog()` is ACS F814W < 25,
0.01 < photometric redshift < 3, and 4 < FLUX_RADIUS < 500, plus the catalog
type filter. This is not an explicitly NISP-selected sample. Confirm which
historical selection produced each candidate catalog. Join NISP S/N,
coverage, size, redshift, crowding, and quality flags to accepted IDs. Start
with a clean, well-covered subset for debugging, but retain the full intended
population and document selection effects before generalizing the paper.

### Normalization and training targets

The diffusion normalizer is not bounded to [-1, 1], but the diffusion
reconstruction is clamped to that interval during both training and sampling.
In a diagnostic fit using 64 JWST stamps and checking the 256 sampled central
105-pixel targets, 234/256 undated and 178/256 dated targets exceeded +1.
Only about 0.62%/0.43% of all sampled target pixels exceed the interval, but
these include the bright cores. Median image peaks were 1.57/1.21.
These are new diagnostic normalization fits, not recovered training states;
the numbers must not be described as historical-run saturation rates.

The clamp prevents the reconstruction from matching such targets and gives
zero reconstruction-loss gradients where the prediction is saturated.
Choose a consistent target representation and output range. Preserve fitted
normalization state in checkpoints and invert it for scientific diagnostics.
Fit global statistics on training images only; exclude invalid pixels.
Asinh is a possible representation, not intrinsically a defect.

Per-image z-scoring of JWST targets in the residual pipeline removes the
target's absolute scale. It can support normalized morphology learning but
does not by itself define an absolute-flux prediction from NISP. Decide this
contract explicitly. Avoid treating histogram equalization or tanh clipping
as flux-preserving preprocessing.

## What the networks actually implement

| Path | Executed configuration | Trainable parameters | Upsampling |
| --- | --- | ---: | --- |
| `claude_model_NIR_2.py` | Local `best_sweep_model_tbra2akh.pth`, 8 dense blocks, 64 features | 2,332,177 | Two 2x PixelShuffle stages, then bilinear resize to fixed 205x205 |
| `claudemodel.py` | Local `final_best.pth`, 6 dense blocks, 48 features | 1,383,985 | Direct 5x PixelShuffle; handles 21/25/41 input widths as 105/125/205 |
| `diffusion/model_sr3.py` | Random initialization, base width 32, three encoder scales | 4,910,305 | Conditional UNet on the high-resolution grid |

Both residual checkpoints loaded strictly after stripping the saved compile
prefix where needed. Forward checks were finite for input widths 21, 25, and
41; the fixed-output model always returned 205. Backpropagation produced
finite gradients. The diffusion model produced a finite 105x105 output and
finite gradients. These checks show executable graphs, not scientific quality.

The residual blocks are single residual dense blocks, not the nested RRDB
units described in ESRGAN. They omit ESRGAN's residual scaling. This is a
legitimate variant, but the paper should describe the implementation accurately;
adding true RRDBs or scaling is an experiment, not a guaranteed repair.
See [ESRGAN](https://arxiv.org/abs/1809.00219).

The fixed-output model's edge enhancer adds a Laplacian-like feature filter
and channel attention. Its benefit has not been isolated. Its final bilinear
resize could limit detail, but that is a hypothesis to test against the
already-available direct-5x design. Do not change width, depth, head, loss,
normalization, and crop simultaneously and attribute the gain to one choice.

The spatial-transformer variant learns translations, uses batch normalization,
and also ends with a fixed 205-pixel output. It does not resolve known WCS
geometry or masks automatically. Keep it as a later alignment ablation rather
than a substitute for correcting the training pairs.

The tested diffusion UNet has more parameters than either residual checkpoint.
Parameter count is not effective capacity, but it reinforces that the current
failure cannot be diagnosed as simply too few parameters. Correct sampling,
target range, conditioning, and validation first.

## Training-code findings

- **HDF5 ResNet scheduler:** `OneCycleLR` is configured with batches per epoch
  but stepped once per epoch. Move its step to successful optimizer updates.
  This finding applies to `claudemodel.py`; the sweep path uses an epoch-based
  cosine schedule. [PyTorch documentation](https://docs.pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.OneCycleLR)
- **Diffusion sampling:** shortened trajectories skip timesteps while using
  adjacent-step posterior coefficients. Implement a valid jump-aware sampler
  and test it against an analytic noise-prediction oracle before benchmarking.
  Review initialization around the LR image and the output clamp separately.
- **Diffusion model selection:** sweep ranking uses `val/psnr_x0` from a noisy
  HR target, whereas deployment generates from NISP alone. This denoising
  diagnostic is valid to log but insufficient to select the best SR sampler.
  Save fixed validation seeds and assess full sampling on a fixed validation
  subset; keep an independent test set untouched.
- **Diffusion normalization:** separate train/validation instances randomly
  refit statistics, including on held-out images. Inference mode does not
  restore a fitted normalization state. Make preprocessing an explicit saved
  part of the trained model contract.
- **Mask/noise handling:** current single-channel losses treat zero fills as
  truth. There is no propagated valid-pixel mask or instrument uncertainty
  weighting in the principal reconstruction objectives.
- **HDF5 subsampling:** the loader selects random keys but reads by the new
  positional index, so `sample_fraction` does not load the selected rows.
  Persist row indices and verify paired keys. Existing split JSON files also
  need input-identity, range, and disjointness checks before reuse.
- **Augmentation:** paired transform support exists in one residual loader,
  but the inspected sweep does not enable it. Use consistent 90-degree
  rotations/reflections of inputs, targets, masks, and PSF information. Check
  augmentation against instrument anisotropy; do not augment validation.
- **Losses:** SSIM constants assume a unit range despite z-scored inputs.
  VGG features receive repeated single-channel asinh data without the usual
  image preprocessing. Neither objective has demonstrated an advantage for
  this task. Establish masked L1/Charbonnier and linear-space diagnostics
  first; add structural or gradient terms as controlled ablations.
- **Run artifacts:** checkpoint names shared across sweep runs can collide.
  Use run-specific paths and save model configuration, optimizer/scheduler,
  seeds, normalizers, data manifest hash, and validation protocol. Verify the
  selected checkpoint, rather than the last epoch, generates comparison plots.

## Proposed GPU comparison

First fix correctness and run a tiny-subset overfit check (e.g. 32 clean pairs),
masked-pixel loss checks, normalization round trips, and input/target geometry
tests locally or on the GPU. Failure here is not evidence for a larger model.

Then use one versioned sample, spatial train/validation/test split, training
representation, valid-pixel loss, augmentation policy, and common output grid:

1. **Corrected current residual model:** establish the reference training run.
2. **Direct-5x residual model:** isolate the upsampling change using the same
   trunk; separately test residual scaling and removing the edge enhancer.
3. **A multiscale residual UNet:** a different inductive bias for extended
   structure and neighbors, with an explicit 5x output. Compare at a similar
   parameter budget, reporting compute as well as reconstruction quality.
4. **Corrected SR3:** retain a probabilistic comparator if useful to the paper,
   using valid sampling, consistent target range, saved normalizers, and
   multiple samples to distinguish conditional mean and sample variability.

Use the same training objects and optimizer-update budget for initial tests;
report parameter count, GPU time, and peak memory since equal updates do not
mean equal compute. Repeat finalists over at least three seeds. Fix a simple
shape diagnostic during model comparison so that model selection does not
reward sharp-looking but inaccurate images; defer estimator development.
Define gains on validation data, then evaluate the untouched test once the
model and analysis choices are frozen.

An image-level residual connection, PSF/noise conditioning, and a constrained
forward-model consistency term are sensible later ablations. The last requires
a justified cross-band model: JWST F115W is not the same sky brightness field
as NISP Y, so simple PSF blur/downsampling need not reproduce NISP exactly.
Do not impose an incorrect physical constraint just because it is convenient.

EDSR is a useful residual baseline reference, and SwinIR an optional transformer
comparison if the corrected CNNs plateau. Their natural-image results do not
establish superiority for NISP. References: [EDSR](https://arxiv.org/abs/1707.02921),
[SwinIR](https://arxiv.org/abs/2108.10257).
The April 2026 [Neo paper](https://arxiv.org/abs/2604.20195) is particularly
relevant prior work on HSC-to-HST morphological recovery with a conditional
GAN; it should inform the paper's literature comparison, not be taken as a
reason to adopt an adversarial objective without testing fidelity here.

## Immediate next step

Correct extraction and regenerate a small, fully traceable set of pairs from
local raw mosaics, with masks and WCS preserved. Compare those pairs with the
existing versions, establish the intended sample/product, and only then
regenerate the training set. This addresses demonstrated limitations before
spending GPU time. Existing arrays, checkpoints, and article figures remain
unchanged; this review adds diagnostic code and artifacts only.
