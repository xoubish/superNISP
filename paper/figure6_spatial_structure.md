# Figure 6 companion: spatial power and reference correlation

Local exploratory comparison, 2026-10-16. User approved testing the visual
sharpness of diffusion with power spectra and Fourier cross-correlation.
No new inference, training, W&B initialization or external upload was performed.
The main Figure 6 source-error/speed plot remains unchanged.

## Finding

Diffusion produces power-spectrum amplitudes closer to **observed** JWST than
the residual networks on intermediate and small scales, but its signed Fourier
correlation with JWST is lower. This supports a measurable trade-off between
spectral-amplitude agreement and correspondence with individual reference features.
It does not establish better recovery of the locations or shapes of fine features.

The qualitative pattern survives changes in outer taper, a common smooth source
weight, the inclusion of incomplete cutouts with tapered validity, and
reference-only selection for stronger reference/background power contrast.
The finest-scale results are particularly noise-sensitive: observed JWST spectra
include noise, and the background proxy is not an unbiased estimate of noise PSD.
The numerical choices were saved in the run contract before measuring results.

## Population, windows and statistics

The cached v03 reconstructions cover all 2,118 validation objects. Primary curves
use the **1,363** cutouts with complete common-valid coverage in the central
105×105 field. The remaining 755 are excluded only for geometry, not model
success, morphology or errors. All 2,118 remain in a separate validity-taper
check. Each method uses the exact same object IDs and window as its reference.
The background diagnostic has adequate window area for **1,356** primary objects.

For each image, subtract its own window-weighted mean, multiply by a separable
Tukey window with alpha=0.5, and divide the FFT by sqrt(sum(window²)). Removing
the weighted mean is specific to this structure diagnostic; the Figure 6 pixel
metrics remain unchanged. The raw signed physical intensities are used without
clipping, flux fitting, PSF deconvolution or reference-noise subtraction.

For each radial frequency bin, compute P_model=mean(|F_model|²),
P_reference=mean(|F_reference|²), and signed cross power
C=mean(Re(F_model F_reference*)). The plotted power ratio is
10**mean(log10(P_model/P_reference)): a geometric mean of per-object ratios.
The plotted correlation is the arithmetic mean of per-object
r=C/sqrt(P_model P_reference). This is signed Fourier correlation, not
magnitude-squared coherence. Power near unity indicates amplitude agreement;
r near unity indicates correlated Fourier structure. A uniformly dimmed image
can retain high r, so these measurements must be interpreted together.

Objects have equal weight in each stated statistic. Intervals use 2,000 paired
spatial-block resamples (seed 20261017); the same draws apply to all models.
Paired differences in **per-object absolute log-power error** are also saved,
so the power comparison is not merely cancellation of over- and underprediction.

Frequencies are in cycles per target pixel. Radial bins exclude DC and the
Fourier corners above radius 0.5. Separate broad bands were fixed at wavelengths
15–30, 6–15, 3–6 and 2–3 target pixels. Angular labels use the measured sample
median target scale 0.05934367 arcsec/pixel; its range is 0.0593416–0.0593454.
The NISP pixel Nyquist marker is 10 target pixels (~0.593 arcsec), not an
estimated effective resolution or a PSF correction.

## Noise and edge checks

The outside-source window is the complement of the frozen 2-sigma source mask,
with a 3-pixel cosine taper at source/invalid boundaries and the outer Tukey
window. It is usable when window energy exceeds 16 pixels and effective area
(sum(w)²/sum(w²)) is at least 64 pixels. Energy normalization makes the white-noise
power scale comparable, but source selection, residual faint light, correlated
noise, different windows and mode mixing preclude treating it as exact noise PSD.
We do not subtract it or claim a noise-debiased resolution measurement.

Window checks use Tukey alpha=0.25, Hann, and a shared source weight made by
Gaussian-smoothing the source mask with sigma=3 pixels. The all-object check
smoothly tapers to invalid regions over 5 pixels. The complete-object window
comparisons hold the population fixed. The all-object check changes population
and is labelled separately.

Reference-contrast subsets are defined independently in each fixed scale band:
complete cutout, usable background, and P_reference/P_background >3. These are
reference-only sensitivity subsets, not calibrated S/N selections. They retain
1,312, 944, 421 and 203 objects from coarse to fine; their varying populations
must not be combined into a single all-object spectrum.

## Quantitative example: 6–15 target pixels (~0.36–0.89 arcsec)

For the **944** reference-contrast-selected objects in this band:

| Method | Observed-JWST power ratio (ideal 1) | Fourier correlation (ideal 1) |
| --- | ---: | ---: |
| Bilinear | 0.056 | 0.349 |
| Bicubic | 0.097 | 0.368 |
| Diffusion (1 sample) | 0.884 | 0.643 |
| Diffusion (mean of 3) | 0.687 | 0.722 |
| ResNet | 0.444 | 0.782 |
| ResNet-D5 | 0.443 | 0.783 |

Relative to ResNet-D5, single-sample diffusion has lower absolute log-power
error by 0.166 dex (paired 95% interval for diffusion minus ResNet-D5:
−0.189 to −0.141 dex), but lower Fourier correlation by 0.141
(interval −0.150 to −0.132). This is an amplitude advantage paired with weaker
reference correspondence. Averaging three diffusion samples reduces power and
increases correlation, moving toward the smoother reconstruction behaviour.

For transparency, the same band's **unselected 1,363-object primary population**
gives:

| Method | Observed-JWST power ratio | Fourier correlation |
| --- | ---: | ---: |
| Bilinear | 0.087 | 0.270 |
| Bicubic | 0.152 | 0.285 |
| Diffusion (1 sample) | 0.904 | 0.501 |
| Diffusion (mean of 3) | 0.611 | 0.577 |
| ResNet | 0.270 | 0.648 |
| ResNet-D5 | 0.270 | 0.649 |

## Background control

In the outside-source region, single-sample diffusion has a power ratio of
0.792 but correlation of only 0.005 in the same 6–15-pixel band. The corresponding
ResNet-D5 values are 0.006 and 0.029. Thus diffusion's extra power is also present
as largely uncorrelated background texture. The reference-contrast subset and
soft-source check show that the amplitude/correlation trade-off is not confined
to a full-field background average, but do not isolate a noise-free galaxy signal.

## Suggested paper wording

“Diffusion reconstructions retain power-spectrum amplitudes closer to the
observed JWST reference on intermediate and small spatial scales, whereas the
residual networks attain higher Fourier cross-correlation and lower pixel errors.
This indicates a trade-off between spectral-amplitude agreement and per-object
reconstruction fidelity. Because the reference contains noise and diffusion also
produces uncorrelated background power, the additional small-scale power alone
cannot establish improved recovery of individual morphological features.”

Keep both power and correlation panels together. Include the background and
window checks alongside any claim about sharper detail. Treat this as exploratory
validation evidence, with the existing fixed checkpoints and one training seed.

## Artifacts and verification

- Notebook: `paper_figures/06b_spatial_structure.ipynb`.
- Main diagnostic: `paper_figures/exports/fig06_spatial_structure_v01.pdf` / `.png`.
- Window checks: `paper_figures/exports/fig06_spectral_window_checks_v01.pdf` / `.png`.
- Background control: `paper_figures/exports/fig06_spectral_background_v01.pdf` / `.png`.
- Summary/provenance: `paper_figures/inputs/spatial_spectra_v01.json`.
- Local per-object arrays: `paper_figures/inputs/spatial_spectra_v01.npz`.
- Completed CPU run: `runs/spectra-v01`; console: `runs/spectra-v01-console`.
- Code: `code/spatial_spectra.py` and `code/analyze_spatial_spectra.py`.

Four analytical tests cover identity, amplitude changes, sign, equal power with
wrong phase, invalid-pixel exclusion, and paired block weighting. An independent
FFT implementation reproduced 15,984 raw power/cross-power values across fixed
random and validity-edge examples. Array identities, signed-correlation bounds,
reference-only subsets and primary bootstrap intervals were verified. Inputs,
source snapshots, notebook and figure hashes were checked. Plots were visually
reviewed. Verification and source snapshots are retained with the run.

The power/correlation distinction also appears in the related, different setting
of cosmological-field diffusion super-resolution:
[Rouhiainen et al.](https://arxiv.org/abs/2311.05217).
The taper follows [SciPy's Tukey window definition](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.windows.tukey.html).
