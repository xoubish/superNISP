# Figure 6: strict source-only Fourier comparison (local v01)

Current presentation: [consolidated Figure 6 notebook](../paper_figures/06_model_comparison.ipynb)
combines source/outside RMSE, power, and correlation in v04. The analysis and
original exports documented below remain unchanged.

Requested repeat of the whole-cutout spatial-structure plot, using only source
pixels. No new inference, training, W&B upload, or manuscript installation.
The main source-error/speed Figure 6 and whole-cutout Fourier exports are preserved.

## Artifacts

- [Source-only plot](../paper_figures/exports/fig06_source_spatial_structure_v01.png)
- [Vector PDF](../paper_figures/exports/fig06_source_spatial_structure_v01.pdf)
- [Boundary sensitivity](../paper_figures/exports/fig06_source_window_checks_v01.png)
- [Notebook](../paper_figures/revisions/06c_source_spatial_structure_v01_before_consolidation.ipynb)
- Summary: `paper_figures/inputs/spatial_spectra_source_v01.json`; local per-object
  arrays: same stem `.npz`.
- Run and frozen contract/source snapshots: `runs/spectra-source-v01`.
- Analysis: `code/analyze_source_spatial_spectra.py`; window helper:
  `code/source_spatial_spectra.py`.

## Fixed protocol

Use the frozen v03 predictions and 2σ JWST detection masks from the source-error
experiment. Include all detected neighbouring sources. The window is **exactly
zero outside source ∩ common-valid pixels**, unlike the earlier Gaussian soft
source weight, which has tails outside the source region. Inside the support,
use sin²[(π/2) min(distance-to-exterior / 3 pixels, 1)] multiplied by the original
separable outer Tukey window (alpha 0.5). Distances are Euclidean distances to
excluded pixel centres. Each image has its own window-weighted mean removed,
and FFTs are normalized by sqrt(sum(window²)). Identical windows are applied to
all reconstructions and JWST. No image smoothing, flux fitting, noise subtraction,
or PSF correction is performed.

Keep the same 1,363 completely valid cutouts as the whole-cutout plot, without
model-dependent exclusions. Each object has equal statistical weight. Power
ratios are geometric means; signed annular cross-correlations are arithmetic
means. Use the original radial bins and 2,000 paired draws of 18 spatial blocks.
The third panel uses source-window JWST power divided by outside-source JWST
power on the 1,356 fields with sufficient background support. Different windows
and possible faint sources make this a diagnostic proxy, not calibrated S/N.

Predeclared checks change the inward width to 1 pixel (effectively a hard source
boundary at this sampling) or 5 pixels, recompute the original whole-cutout
window, and include all 2,118 fields with their valid source support. The
contract and code were saved before measuring these results.

## Results on the same 1,363 fields

At 6–15 target pixels (0.36–0.89 arcsec), ideal power ratio and correlation are
both 1:

| Method | Whole-cutout power ratio | Source-only power ratio | Whole-cutout correlation | Source-only correlation |
|---|---:|---:|---:|---:|
| Diffusion, one sample | 0.904 | 0.912 | 0.501 | 0.604 |
| Diffusion, mean of 3 | 0.611 | 0.720 | 0.577 | 0.684 |
| ResNet | 0.270 | 0.417 | 0.648 | 0.737 |
| ResNet-D5 | 0.270 | 0.419 | 0.649 | 0.738 |

For single diffusion minus ResNet-D5 in this band, the paired mean difference
in absolute per-object log-power mismatch is −0.139 dex (95% block interval
−0.162 to −0.114): diffusion is closer in power. The signed-correlation difference
is −0.135 (−0.144 to −0.126): ResNet-D5 aligns better with the observed reference.
Thus the whole-cutout trade-off persists within detected sources. Do not infer
that a geometric-mean power ratio close to one means every object has accurate
power; at coarse scales the paired absolute-error diagnostic need not favour
diffusion even when its aggregate ratio is close to one.

At 3–6 target pixels (0.18–0.36 arcsec), source-only single diffusion has power
ratio 0.688 and correlation 0.256, versus 0.097 and 0.464 for ResNet-D5.
The intermediate-scale ordering persists with 1- and 5-pixel boundary tapers:
diffusion correlations 0.596/0.614 versus ResNet-D5 0.739/0.746. The all-2,118
source check gives power 0.920/0.426 and correlation 0.611/0.740, respectively.

## Interpretation and limits

Removing outside-source pixels raises correlation for all learned models, but
it does not reverse the ordering. Single-sample diffusion preserves more observed
small-scale power, while the residual networks more accurately align with the
reference under this statistic. This supports a difference in amount versus
agreement of spatial variation; it does not establish recovered fine morphology.

Noise within the source mask remains. Masks are selected from the same noisy
JWST reference, so this is a reference-defined measurement. The common irregular
window mixes modes and can create agreement through shared geometry. In
particular, the finest-scale correlations change materially with taper width:
single diffusion 0.216/0.086/0.069 and ResNet-D5 0.423/0.244/0.190 for widths
1/3/5. Avoid interpreting the finest masked bins as an intrinsic resolution
measurement. The main conclusion is unchanged by these boundary checks.

The more defensible paper wording is: “Within reference-defined source regions,
diffusion retains more small-scale power than the residual networks, while the
residual networks show higher signed Fourier correlation with the observed JWST
reference.” Keep the power and correlation panels together and explicitly state
that no noise or window deconvolution has been performed.

## Verification

Six analytical tests cover amplitude/phase distinction, block weighting, invalid
pixels, source-mask holes and neighbours, taper values, and exact invariance to
arbitrary outside-source pixels. Independent cached-image FFT calculations,
array identities, block intervals, subset IDs, source/input hashes and export
hashes are checked by `runs/spectra-source-v01/verify_source_spatial_spectra.py`.
The recomputed whole-cutout and background controls match the previous per-object
arrays exactly. Final verification is recorded in that run directory.
