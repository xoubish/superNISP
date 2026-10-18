# Figure 9: supplied NISP effective-PSF response experiment

Frozen before model execution. This replaces illustrative Gaussian native PSFs
with all four signed, six-times-oversampled NISP Y stack PSF profiles supplied
with the paired data. The original epoch-49 ResNet and normalization are unchanged.
No training, real-galaxy inference, or reserved test-image access.

## Rendering convention and evidence

The supplied products identify PSFTYPE=Inflight, EXTNAME=STACKED.PSF,
OVERSAMP=6, and PSFEx-style model/fit metadata (POLNAXIS=0, PSFAXIS*, LOADED,
ACCEPTED, CHI2). They are empirical effective profiles, not optical-only PSFs.
PSFEx reconstructs stellar pixel values by interpolation of its sampled model,
and explicitly distinguishes the recovered pixel-convolved profile from the
optical profile. See [PSFEx model definition, section 5.3](https://psfex.readthedocs.io/en/latest/Working.html).
GalSim's [PSFEx documentation](https://galsim-developers.github.io/GalSim/_build/html/des.html)
requires `method='no_pixel'` for a galaxy convolved with such an effective PSF,
to avoid a second integration over the detector pixel.

Interpret these supplied stack profiles using that convention. Normalize the
signed array to unit integrated flux; retain negative samples; interpolate with
Lanczos11 and pad factor 4. Center on the adaptive centroid. Convert PSF-grid
coordinates into native-image angular coordinates using the two supplied CD
matrices: `native_scale * inverse(native_CD) * PSF_CD`, with native_scale from
the determinant of the native CD matrix. Use the native scale separately for
each stack and the output scale divided by five. Do not add another native Pixel,
depixelize, or replace the measured profile with a Gaussian fitted width.
This header-based orientation convention and the effective-profile interpretation
are explicit; they are not an independent validation against stack stars or a
spatially varying PSF atlas. The finite 61x61 profile truncates extended wings.

Controls: reconstruct original PSF samples; Gaussian-plus-pixel analytic control
including surface-brightness flux and a deliberately double-blurred countercheck;
rotation/recentering; tightened convolution tolerances; selection failure injected
in one of four PSFs. All must pass before model execution. A direct rotated-profile interpolation
probe showed Lanczos5 peak errors of 0.12%; Lanczos11 lowers these below 0.03%
and passes the fixed tolerance. Bare native-grid PSF adaptive fits fail because
of undersampling, as in the old pilot; require convergence on the oversampled
PSF and convolved galaxy instead, and retain the native PSF failure status.

## Simulation and comparisons

Retain the original pilot's exponential profiles, distortion magnitude 0.35,
eight orientations, shears (0,0), (+/-0.01,0), (0,+/-0.01), native 41x41 and
output 205x205 pixels. Evaluate radii 0.18, 0.35, 0.65 arcsec at S/N 20, 50, 100
and noiseless: 48 PSF/size/SNR cells, 1,248 object-PSF realizations and 6,240
sheared native inputs. Four noise realizations per orientation in noisy cells;
one orientation ring for noiseless cells. Independent Gaussian native pixel
noise sigma=0.028 MJy/sr. Fix seed 20261017. Reuse the same realization across
five shear branches AND four PSFs at fixed size/SNR/rotation/repeat.

Flux is fixed from the circular unsheared profile with the original weighted
S/N definition (Gaussian sigma 2 pixels truncated at radius 5), separately for
each PSF. No orientation/shear renormalization. Noiseless uses S/N-50 flux.
Target remains the original noiseless 0.08-arcsec circular Gaussian PSF control,
with high-resolution pixel integration; it is not a validated JWST simulation.

Measure full-stamp GalSim observed adaptive distortions. Auxiliary native REGAUSS
results/failures are saved but not displayed and do not select the observed sample.
For each size/SNR cell, require the same objects to succeed in native, ResNet,
and high-resolution shapes across all five branches and all four PSFs. Compute
R_ij by central finite difference and normalize trace(R) to the analytic intrinsic
trace for exactly that common population. Average the four per-PSF response ratios
with equal weight. Save all individual matrices, success counts, and PSF ranges.
The four PSFs are not four independent noise realizations. No uncertainty or
calibrated shear claim is made from this small pilot.

Main Figure 9: two panels, radii 0.18 and 0.35 arcsec, S/N 20/50/100, three methods.
These radii were selected in the earlier Gaussian pilot for their larger gains;
the omitted radius and noiseless cells remain fully reported. Use equal PSF
weights to avoid selecting a favorable stack; this is not weighting by the
observed galaxy abundance. All PSF files and source headers are hash checked.

Run in persistent tmux with online W&B at the previously approved destination
AI-Astro/superNISP-corrected, logging configuration and aggregate synthetic metrics
only. No real galaxy images or PSF image arrays are uploaded. The local run keeps
source snapshots, contract, raw per-object measurements, hashes, and a report.
