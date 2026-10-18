# PSF-aware ResNet redesign: validation experiment v01

User request: test a PSF-aware redesign for RMSE, visual quality, sharpness and
shapes before finalizing the paper. This is a bounded architecture comparison,
not a guarantee of a globally optimal network. The original paper model remains
available and unchanged. No reserved test images are used for model selection.

## Models and fair comparison

Two new branches run on the two free GPUs:

1. **Conditioned:** original NISP image plus normalized input RMS, validity mask,
   and a learned encoding of the associated NISP stack PSF.
2. **Deconv:** identical architecture and inputs, plus a conservatively regularized
   inverse-filter residual as an extra feature. The original image is retained.

Both inherit every original epoch-49 legacy ResNet weight. A small context
adapter adds features after the original first convolution. Its final layer is
initialized to zero, making both initial predictions exactly those of the parent.
The network learns how much of the additional information to use. Both branches
have the same parameter count; the inverse-feature channel is zero in Conditioned.
The old architecture and preprocessing files are not edited.

The original frozen model and the already completed **original-loss control
continuation v03** are both baselines. The new runs reuse that control's exact
parent RNG, batch order, eight-object batches, D4 augmentation sequence, masked-L1
objective, and ten-epoch / 22,270-update budget. An epoch is accepted only if its
recorded batch/augmentation stream hash matches the corresponding old control.
Original parameters retain Adam moments, betas, and step counters, with cosine LR
1e-5 → 1e-6. New context parameters get fresh Adam state with the same betas/eps
and cosine LR 1e-4 → 1e-5. Thus this is a warm-start redesign test, not independent
from-scratch training. The existing control has no new context parameters.

Checkpoint selection is fixed before training: smallest mean **source MAE** on
the frozen central 105 × 105 validation mask, including the unchanged epoch-zero
parent. This matches the control selection rule. Report both selected and final
checkpoints; retain every new epoch checkpoint/prediction for audit. RMSE,
sharpness and shapes are evaluation outcomes, not separately cherry-picked epochs.

## PSF and noise inputs

`objects.csv` supplies the exact per-object `psf_image` association. All four
PSF hashes match the dataset manifest. The training counts by stack are
2 / 6128 / 5913 / 5773; validation counts are 0 / 597 / 1204 / 317. The four PSFs
are very similar, so this dataset cannot establish generalization to a wide
range of PSFs. PSF identity is also associated with field position: gains cannot
automatically be attributed to physical PSF correction rather than contextual
information. Report validation results by stack and S/N as diagnostics.

The native image grid is rotated relative to the PSF header's linear CD axes.
The preprocessing uses `inverse(PSF CD) * native CD`, recenters on the PSF's
adaptive centroid, samples an 11 × 11 kernel with cubic interpolation, and
normalizes its signed sum. No additional pixel response is convolved. This is
an explicit effective-PSF **proxy convention**, not a validated alias-free
intrinsic-image model. Header orientation, stacking, and pixel-response uncertainty
therefore remain limitations; the original image is always retained.

The inverse feature uses transfer `H* (1+λ) / (|H|²+λ)` with fixed λ=0.05, preserving
DC gain and limiting Fourier gain to approximately 2.35. It uses 20-pixel
reflection padding and keeps signed intensities. Invalid input pixels are zero
filled as before; the validity map is explicitly supplied to the model. The
feature can still contain mask-boundary or interpolation artifacts, which must
be assessed through validation. It is not presented as a calibrated deconvolution.
No filtering strength is fitted to validation or target images. RMS is the
original input-only uncertainty map, transformed as `log1p(normalized RMS)`;
invalid RMS is zero and marked invalid.

PSF, image, masks and RMS receive the same D4 augmentation. Targets remain observed
JWST F115W on the original aligned grid. No JWST PSF or target-derived quantity
is supplied at inference. Inputs are not claimed to remove cross-band differences.

PSFEx's [pixel-basis model and interpolation convention](https://psfex.readthedocs.io/en/latest/Working.html)
supports treating its oversampled tables as models of observed point-source
pixels, rather than automatically applying a second pixel convolution. This does
not resolve every convention of these particular Euclid stack products.

## Evaluation fixed before results

- All 2,118 validation objects, existing source/outside masks and equal object
  weights. Source/outside MAE and RMSE, plus full common-field RMSE.
- Source and outside-source Fourier power relative to observed JWST and signed
  cross-correlation, on matched windows and frequency bands. More power alone
  does not count as greater recovered sharpness; power fidelity and structural
  correlation must be considered together.
- Existing fixed-weight and GalSim adaptive shape measurements on identical
  successful populations. Keep all per-method failure counts, paired differences,
  and both selected/final checkpoints; these are observed shapes, not shear.
- Paired spatial-block bootstrap intervals (2,000 draws, seed 20261017). These
  exploratory intervals are conditional on validation selection and one parent
  seed; they do not remove model-selection optimism or measure training variance.
- Existing Figure 5 example IDs and fixed display limits, shared across all
  methods, with image and residual sheets. No selection based on new predictions.
- S/N-bin and PSF-stack summaries to check whether an aggregate improvement hides
  failures in faint objects or one field. Do not choose a model from one favorable
  subset or from appearance alone.

Only call a candidate an overall improvement if source RMSE improves versus
both original and matched continuation, while outside error, structural
correlation, both shape errors and measurement success show no material loss.
Report trade-offs explicitly. A promising candidate requires further seed and
PSF-proxy sensitivity checks before a strong paper claim. If neither candidate
improves the joint evidence, retain the original model rather than replacing it
because the redesign is newer. The scientific comparison will determine whether
additional training or a different loss is justified; no open-ended sweep is
pre-authorized by this protocol.

## Execution and safeguards

Use tmux and the existing approved online W&B project `AI-Astro/superNISP-corrected`.
Each run snapshots source/config/data/PSF hashes and saves atomic epoch-boundary
model, optimizer, scheduler and RNG state. Preflight verifies original cached
predictions and source/outside metrics, finite gradients, exact state restoration
and numerical next-step equivalence. The CPU controls check flux-preserving
filtering, D4 consistency, PSF geometry, parent-output equivalence, gradients and
optimizer isolation. No model runs begin before online monitoring is initialized.

Files: `code/psf_aware_resnet.py`, `code/prepare_psf_features.py`,
`code/train_psf_redesign.py`, `code/test_psf_aware_resnet.py`.
PSF preparation and associations: `runs/psf-features-v01/`.
New model outputs: `runs/psf-redesign-conditioned-v01/` and
`runs/psf-redesign-deconv-v01/`. The comparison is separate from the manuscript
and existing figure inputs until results have been reviewed.
