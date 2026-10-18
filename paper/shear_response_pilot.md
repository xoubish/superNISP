# PSF audit and controlled shear-response pilot

This is supporting work for the PSF/shear discussion, separate from Figures 7–8.
Those figures compare observed cross-band shapes and are not shear calibration.
No manuscript figure or calibration claim is changed by this experiment.

## Available PSFs

The four NISP stack PSF files in the paired-data package match the source-manifest
SHA256 hashes. Each has a `PRIMARY` header and one `STACKED.PSF` image, 61 × 61,
oversampled by six, at approximately 0.04945 arcsec per PSF pixel. Adaptive Gaussian
widths are 0.2075–0.2081 arcsec; these are moment widths, not measured FWHMs.
The products contain a single stack PSF, with `POLNAXIS=0`, not a spatial atlas.
The arrays sum to approximately 36.5 and contain a few small negative pixels.
Any use must explicitly normalize the signed array and establish the pixel-response
and resampling conventions first, to avoid applying pixel integration twice.

`code/euclid_psf.py` expects an HDU-2 coordinate table and `STMPSIZE` atlas metadata;
it is incompatible with these files. It was not used or silently patched to infer
missing conventions. No local JWST/F115W PSF was found in the project data or paired
package; the historical filename in `clipping.py` does not supply one.
Machine-readable audit: [psf_audit_v01.json](diagnostics/psf_audit_v01.json).

An appropriate F115W reference would need the COSMOS-Web mosaic PSF, including
its sampling, stacking, position, and relevant color dependence. Even that would
not establish an effective PSF for the nonlinear ResNet output.

## Predeclared pilot

The contract and exact source snapshots are in `runs/shear-response-pilot-v01/`.
The runner is `code/pilot_shear_response.py`; launcher:
`code/launch_shear_response_pilot.sh`. It uses the original frozen epoch-49 legacy
ResNet (checkpoint SHA256 `be747f0cd0b02bb81ff0ab42a9ce8ea81a0e473c28c58d8406a428f758ea6b56`)
and the original training normalization. There are no training updates or
checkpoint writes, and no real galaxy or reserved test images are used.
Execution follows the existing tmux and online-W&B requirement in `HANDOFF_GPU.md`.

- Exponential galaxies with intrinsic distortion magnitude 0.35 and eight equally
  spaced orientations; half-light radii 0.18, 0.35, and 0.65 arcsec.
- Shear branches `(0,0)`, `(±0.01,0)`, and `(0,±0.01)`, applied to the intrinsic
  profile before PSF convolution and pixel integration.
- Analytic NISP-like Gaussian PSFs with FWHM 0.45 and 0.60 arcsec and distortion
  `(0.03,0.02)`. These are deliberately controlled assumptions, not calibrated
  replacements for the supplied empirical PSFs.
- Native 41 × 41 stamps at 0.29672 arcsec/pixel. Gaussian independent pixel noise
  of 0.028 MJy/sr, an illustrative level close to the inspected validation RMS.
  Flux is set once using a circular unsheared object and the existing fixed-weight
  S/N definition, for S/N 20, 50, and 100. It is then held fixed across orientations
  and shear branches. Thus the labels specify nominal S/N, not exact S/N for every
  rotated/sheared galaxy. Four independent noise realizations per orientation;
  each realization is reused across shear branches to reduce difference noise.
- Six additional noiseless size/PSF cells at the S/N-50 flux. There are 24 cells,
  624 base objects, and 3,120 sheared/native realizations in total.
- A noiseless high-resolution control uses a circular Gaussian PSF of FWHM
  0.08 arcsec, at native pixel scale / 5. It assumes the same intrinsic profile
  and surface-brightness flux across bands. It is not a JWST simulation validated
  against the real mosaics and does not include target noise.

GalSim measures adaptive observed distortion on the full stamps. Native images
also receive `EstimateShear(..., shear_est='REGAUSS')` using the exact known
pixel-integrated simulation PSF. No PSF correction is applied to ResNet outputs.
These full-stamp measurements are a pilot estimator, not a reproduction of all
Figure 7 crop, mask, and centroid-selection rules. All status codes and successful
corrected distortions, including extreme values, are retained without clipping.

For each cell we measure the complete 2 × 2 response matrix
`R_ij = (<e_i> at +g_j − <e_i> at −g_j) / 0.02`.
The primary comparison uses exactly the same objects successful in all methods
and all five branches. The plotted ratio is `trace(R) / trace(R_intrinsic)`, with
the analytic pre-PSF response measured on those same objects. One means that the
mean trace response matches this idealized intrinsic population. It is **not** an
estimate of multiplicative shear bias, and a closer response alone does not prove
better total shear accuracy or precision. The full matrices retain off-diagonal
responses; zero-shear mean distortions and per-branch success counts are saved.
Separately selected branch means are retained as a selection-sensitivity check,
not as a correction for selection bias.

## Completed results

All 24 cells / 3,120 model inputs completed successfully in tmux (exit 0):
[W&B run](https://wandb.ai/AI-Astro/superNISP-corrected/runs/0r3dbbqc).
The selected checkpoint is unchanged. Three scientific controls passed, and an
independent reader reproduced every available primary response matrix and all
success counts from the frozen per-object measurements. The primary all-method
intersection has 312/624 base objects; the observed-only intersection has 622/624.

Review [the plot](diagnostics/shear_response_review_v01.png),
[PDF](diagnostics/shear_response_review_v01.pdf), and
[verified numerical results](diagnostics/shear_response_review_v01.json).
The original predeclared plot/report are preserved in the run directory.

The 0.45-arcsec PSF is too poorly sampled for native-grid GalSim HSM in this
setup: **all 1,560 REGAUSS attempts at that PSF fail**, so the primary four-method
intersection is empty there. A separate check isolates the PSF adaptive fit and
reproduces the singular-moment/too-small error for initial sigma guesses 0.6,
0.8, 1, 2, and 3 native pixels. The 0.60-arcsec PSF succeeds for all five guesses.
See [sampling check](diagnostics/psf_sampling_probe_v01.json). Changing the initial
guess is not a solution; a properly validated treatment of undersampling or a
different native estimator is needed. This failure is not evidence that PSF
correction is scientifically unnecessary or that ResNet outperforms it.

To avoid hiding valid observed-shape measurements, a clearly labeled **secondary**
analysis compares native, ResNet, and high-resolution observed shapes on their
joint successful sample. The review plot uses that sample and includes REGAUSS
only where it succeeds for the entire identical sample. It does not replace the
failed primary comparison with a different successful population. Only two base
objects fail the observed comparison, both in the smallest-size / S/N-20 /
0.45-arcsec-PSF cell; their failures remain in the denominators.

Selected noiseless response ratios, relative to intrinsic response:

| PSF FWHM | Half-light radius | Native observed | ResNet observed | Native + known PSF correction |
| --- | --- | --- | --- | --- |
| 0.45 arcsec | 0.18 arcsec | 0.409 | 0.858 | failed |
| 0.45 arcsec | 0.35 arcsec | 0.635 | 0.918 | failed |
| 0.45 arcsec | 0.65 arcsec | 0.825 | 0.959 | failed |
| 0.60 arcsec | 0.18 arcsec | 0.290 | 0.526 | 1.044 |
| 0.60 arcsec | 0.35 arcsec | 0.540 | 0.684 | 1.005 |
| 0.60 arcsec | 0.65 arcsec | 0.757 | 0.858 | 1.001 |

ResNet restores some response lost through blurring, but its response remains
dependent on size, noise level, and PSF. It is not uniformly better: at nominal
S/N 20 and radius 0.65 arcsec, the ratios are 0.704 versus native 0.794 for the
narrower PSF, and 0.652 versus native 0.768 for the broader PSF. At the broader
PSF, native REGAUSS is closer to the intrinsic trace response than uncorrected
ResNet throughout this toy grid. The high-resolution control also has residual
PSF/pixel attenuation (ratios approximately 0.907, 0.971, 0.991 with increasing
radius); even a high-resolution observed shape is not the intrinsic shape.

Recommendation: retain Figures 7–8 as observed-shape diagnostics. This pilot
supports pursuing explicit response calibration if shear is a future objective,
but supplies neither calibrated ResNet shears nor a weak-lensing performance
claim for the paper. Resolve empirical PSF conventions and native undersampling
before extending the comparison to realistic simulations.

## Validation and scope

The three controls in `code/test_shear_response.py` test recovery of known sheared
Gaussian shapes after PSF correction, the analytic shear composition and
surface-brightness/pixel-area convention, and consistent object selection when
one method fails on one shear branch. All pass before model execution.

This pilot does not include blends, correlated noise, a realistic morphology
distribution, varying pixel phase, spatial/chromatic PSFs, color gradients, or
detection/deblending selection. Four noise rings per noisy cell are too few to
claim precision calibration; no uncertainty bars or significance claims are made.
Common-success comparisons are conditional, and successful HSM convergence alone
does not guarantee an unbiased measurement. Native PSF correction is an essential
comparison: observed-shape improvement over uncorrected native data would not by
itself establish an advantage over an appropriate native shear estimator.

For a real calibration experiment, first document the empirical PSF conventions
and obtain appropriate mosaic PSFs; then simulate a sufficiently representative
population with independent calibration and validation ensembles. Shear must
enter before the observing process, and the whole detection, measurement,
selection, and weighting pipeline must be rerun. Calibrate the full response and
test residual multiplicative/additive bias on held-out simulations. Artificially
shearing a final ResNet image alone would miss the network's response to changes
in its inputs. The reserved real-data test split remains reserved.

References: [GalSim HSM documentation](https://galsim-developers.github.io/GalSim/_build/html/hsm.html)
for observed moments versus PSF-corrected distortion;
[Sheldon & Huff (2017)](https://arxiv.org/abs/1702.02601) for response and selection
calibration; [Sheldon et al. (2020)](https://arxiv.org/abs/1911.02505) for the role
of shear-dependent detection.
