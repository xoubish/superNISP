# Figure 7: measured ellipticity against JWST, native NISP versus models

The active executed notebook is
[07_model_shapes.ipynb](../paper_figures/07_model_shapes.ipynb).
[Main Figure 7](../paper_figures/exports/fig07_shape_hexbin_v05.png) is now a
hexbin density view with 1:1 and robust best-fit lines; the
[paired gain bars](../paper_figures/exports/fig07_shape_improvement_v05.png) and
[companion summary](../paper_figures/exports/fig07_model_shapes_v05.png) are
re-exported unchanged as companions. All are available as PDF, SVG and PNG.
Previous layouts and executed notebooks are preserved (v04 notebook archived as
`revisions/07_model_shapes_v04_before_scatter_view.ipynb`).
The frozen v02 measurements and every numerical result below are unchanged.
The hexbin PDF is installed in the local manuscript as `paper/figs/shape_hexbin_v05.pdf`
(Figure 7, label `fig:ellipticity_comp`, Shape Preservation section) with a caption
in the Figure 6 style and one results paragraph quoting the slopes and RMSE
reductions above. The legacy paragraphs that quote ODR slopes from the earlier
galsim measurement remain and are flagged with a TODO comment for revision.
No training, inference, test-set access or upload was performed.

## Current figure: hexbin density against JWST with 1:1 and best-fit lines

Two rows (fixed weighted moments, adaptive moments) by four columns (native NISP,
diffusion one sample, ResNet, ResNet-D5). Each panel shows the density of measured
ellipticity against the JWST F115W measurement as hexagonal bins (34 across the
[-1,1] range, one colour ramp per method, logarithmic count scale so the tails
stay visible), pooling e1 and e2 for every galaxy in the estimator's common
sample (1,446 fixed, 1,498 adaptive). A dashed line marks 1:1.
The solid line is a robust Theil-Sen (median pairwise) slope of measured on JWST
ellipticity, fitted only to galaxies with |e_JWST| < 0.5 and drawn over that
range; the dotted line in the model panels repeats the native-NISP fit over the
same range. An ordinary least-squares fit on the full sample was tried first and
sits visibly below the ridge of the cloud (slopes 0.61 native, 0.67 to 0.71
ResNet): noise in the JWST axis and the tails pull it down, so the core robust
fit is used. Annotations give the core fit slope and the full-sample
combined-component RMSE about 1:1 (identical to the frozen summary values). Axes
span [-1,1]; no measurement lies outside. Hexagons are vector patches in the
PDF/SVG.

Slope intervals below come from a 500-draw spatial-block bootstrap with the same
blocks and seed (20261017) as the gain intervals, computed in the notebook; the
RMSE intervals there use the same 500 draws, while the gain intervals keep their
frozen 2,000 draws.

| Estimator | Method | Core slope (95% interval) | RMSE about 1:1 (95% interval) |
|---|---|---:|---:|
| Fixed | Native NISP | 0.678 [0.657, 0.700] | 0.201 [0.194, 0.207] |
| Fixed | Diffusion (one sample) | 0.828 [0.795, 0.861] | 0.229 [0.219, 0.238] |
| Fixed | ResNet | 0.800 [0.774, 0.827] | 0.162 [0.155, 0.168] |
| Fixed | ResNet-D5 | 0.802 [0.775, 0.829] | 0.162 [0.155, 0.169] |
| Adaptive | Native NISP | 0.607 [0.583, 0.638] | 0.242 [0.231, 0.253] |
| Adaptive | Diffusion (one sample) | 0.706 [0.670, 0.743] | 0.297 [0.288, 0.304] |
| Adaptive | ResNet | 0.731 [0.711, 0.753] | 0.217 [0.209, 0.224] |
| Adaptive | ResNet-D5 | 0.734 [0.710, 0.758] | 0.217 [0.209, 0.224] |

ResNet and ResNet-D5 steepen the core slope towards 1:1 and lower the RMSE for
both estimators. Diffusion (one sample) steepens the slope at least as much but
with larger scatter, so its RMSE exceeds native. Note that the native-NISP core is
tight (its normalised median absolute deviation about the fit is smaller than
ResNet's for fixed moments); the ResNet gain is in the slope and the tails, not
the core width. The notebook asserts that the plotted RMSE values equal the
frozen v02 summary.

## Layout alternatives, not selected (all v05, same data and fits)

| Variant | Export | Rendering |
|---|---|---|
| A | `fig07_shape_scatter_v05` | points only |
| B | `fig07_shape_contours_v05` | faint points plus 68% and 95% KDE highest-density contours |
| C | `fig07_shape_regions_v05` | 68% and 95% KDE regions only, filled |
| D | `fig07_shape_hexbin_v05` | hexbin with log colour scale, 34 cells across (chosen main figure) |

The KDE uses a Gaussian kernel with Scott bandwidth on a 161-by-161 grid over
[-1,1]^2; regions are joint highest-density regions, not conditional intervals.
The hexbin (D) was chosen on 2026-10-17; A, B and C remain in the notebook
below the main figure for reference.

## Archived v04 gain view

Fixed and adaptive moments occupy two panels with the same horizontal scale.
Rows show diffusion (one sample), ResNet and ResNet-D5. Zero represents native
NISP's RMSE, with reductions to the right and increases to the left. Points show
100*(1 - RMSE_model/RMSE_native), combining both ellipticity components. Bars run
from zero to each point, and whiskers show conditional 95% intervals on the gain
from the paired spatial-block bootstrap. These intervals account for the paired
model/native comparison rather than treating their uncertainties independently.
The companion diagnostic retains all baselines and measurement-success counts.

## Measurement and selection

`code/add_native_shape_comparison.py` extends the immutable v01 cached shapes
with fresh native NISP measurements. Original per-object JWST/model arrays
remain exactly unchanged. Benchmark validation IDs and row order are asserted,
source and data metadata hashes checked, and native input values fingerprinted.
The contract and source snapshots were saved before measuring native shapes in
`runs/cached-shapes-native-v02/`; outputs are `cached_shapes_v02.json/.npz`.
The NPZ is git-ignored: copy it with its JSON sidecar when moving machines.

Native images are signed background-subtracted physical MJy/sr. The central
21 native pixels cover the same angular field as 105 target pixels. Fixed
moments use Gaussian sigma 2 native pixels and aperture radius 5 native pixels,
equivalent to sigma 10 and radius 25 on the target grid. Pixel-center quadrature
still differs between native and resampled images. HSM uses the same angular
crop and consistently scaled initial size and centroid bounds. Native retains
its own validity mask; cached images retain their common-valid mask. Each
measurement requires its central aperture valid on its own grid. There is no
source-segmentation clipping, image smoothing or PSF matching.

All 2,118 validation fields are attempted. Comparison samples intersect
successful JWST, native and all six cached methods: 1,446 objects for fixed
moments, 1,498 for adaptive moments, and 1,297 common to both estimators.
These samples differ from v01 because native failures now enter the selection.
Uncertainties condition on these successes; failure counts remain on the full
attempted denominator. These are observed cross-band ellipticities, not
PSF-corrected morphology or calibrated gravitational shear.

## Archived v03 contour view and unchanged error-reduction definition

For each method/component, a Gaussian KDE with Scott bandwidth describes the
joint distribution (JWST ellipticity, method ellipticity minus JWST ellipticity)
on the same 1,446-object sample. Density is evaluated on a uniform 401-by-401 grid
over [-2,2] in both dimensions. Highest-density boundaries enclose 68% and 95%
of integrated probability separately for each distribution. Model regions are
blue; native NISP outlines are orange. Solid lines denote 68%, dashed lines 95%,
and a dotted horizontal line denotes zero error. The identical native contours
repeat in every column. These are joint probability regions, not confidence bands
or conditional intervals at a given reference ellipticity.

The displayed axes span [-1,1], containing all the plotted contour regions.
All objects enter the KDE and full-sample RMSE: rare residuals beyond the display
are not discarded. Counts outside the residual range (e1/e2) are native 3/3,
diffusion 6/5, ResNet 1/0, ResNet-D5 1/0, out of 1,446 per distribution.
The notebook verifies grid-integrated probability and enclosed contour levels.
The change in coordinates does not alter the underlying measured errors.

Gain means 100*(1 - model RMSE / native RMSE); positive is better, negative worse.
Combined component RMSE is sqrt(mean[(delta_e1^2 + delta_e2^2)/2]). Intervals use
2,000 paired spatial-block draws, seed 20261017, with equal object weighting.

| Method | Fixed RMSE | Gain vs native, % (95% interval) | Adaptive RMSE | Gain vs native, % (95% interval) |
|---|---:|---:|---:|---:|
| Native NISP | 0.2011 | +0.0 [+0.0, +0.0] | 0.2424 | +0.0 [+0.0, +0.0] |
| Bilinear | 0.1952 | +2.9 [+2.1, +3.6] | 0.2516 | -3.8 [-5.2, -2.7] |
| Bicubic | 0.2014 | -0.2 [-0.7, +0.3] | 0.2432 | -0.3 [-0.6, -0.2] |
| Diffusion (one sample) | 0.2290 | -13.8 [-16.0, -11.3] | 0.2966 | -22.4 [-27.8, -18.0] |
| Diffusion (mean-of-three image) | 0.1875 | +6.8 [+3.5, +10.0] | 0.2481 | -2.4 [-5.9, +0.7] |
| ResNet | 0.1624 | +19.3 [+16.6, +21.9] | 0.2167 | +10.6 [+7.3, +13.2] |
| ResNet-D5 | 0.1625 | +19.2 [+16.5, +21.6] | 0.2168 | +10.5 [+7.6, +13.0] |

ResNet and ResNet-D5 improve agreement with observed JWST for both estimators;
single-sample diffusion performs worse than native NISP. This remains a
validation comparison of the frozen checkpoints and cached diffusion realization.

## Full-denominator measurement success

| Image | Fixed successes / 2,118 | Adaptive successes / 2,118 |
|---|---:|---:|
| Observed JWST | 1662 | 1767 |
| Native NISP | 1741 | 1685 |
| Bilinear | 1661 | 1632 |
| Bicubic | 1622 | 1646 |
| Diffusion (one sample) | 1633 | 1678 |
| Diffusion (mean-of-three image) | 1722 | 1724 |
| ResNet | 1786 | 1742 |
| ResNet-D5 | 1786 | 1745 |

## Previous v04 caption (gain view)

Reduction in observed ellipticity error relative to native NISP for diffusion
(one sample), ResNet and ResNet-D5, using fixed weighted moments (left) and
adaptive moments (right). Points show
$100(1-\mathrm{RMSE}_{\mathrm{model}}/\mathrm{RMSE}_{\mathrm{NISP}})$; bars extend
from the native-NISP baseline at zero. Positive values indicate lower shape error
and negative values higher error. Whiskers are conditional 95% intervals from
2,000 paired spatial-block bootstrap draws. Component RMSE combines both
ellipticity components as
$\sqrt{\langle(\Delta e_1^2+\Delta e_2^2)/2\rangle}$ relative to observed JWST F115W.
All model/native comparisons use the identical successful sample within each
estimator: 1,446 objects for fixed moments and 1,498 for adaptive moments, out of
2,118 attempted validation fields. Measurement success and all comparison methods
are retained in the companion diagnostic. Fixed and adaptive moments define
different shape measurements, and their samples differ. These observed cross-band
shapes include instrumental PSF and pixel-sampling differences; they are not
calibrated shear estimates.

## Verification

The v04 notebook executed successfully, with no error cells; both figures were
visually inspected. Plotted gains are checked against model/native RMSE ratios,
and export manifests verify frozen inputs and notebook sources. Previous data
verification remains valid: Gaussian contour probability and positive/negative paired
gain tests passed. Independent NumPy native fixed moments agree for all 1,741
successful native measurements. Original cached arrays are unchanged, all common
IDs and RMSE summaries match their per-object arrays, and enclosed contour masses
match their stated probabilities. Export sidecars record notebook/input hashes.

## Proposed Figure 7 caption (v05 hexbin view)

Density of ellipticity measured on native NISP, a single diffusion sample, ResNet
and ResNet-D5 (columns) against the same measurement on observed JWST F115W,
using fixed weighted moments (top) and adaptive moments (bottom). Hexagonal bins
are shaded by the logarithm of the number of measurements, with both components
$e_1$ and $e_2$ counted for every galaxy in each estimator's common
successful sample (1,446 galaxies for fixed moments, 1,498 for adaptive moments,
out of 2,118 attempted validation fields). The dashed line is the 1:1 relation;
the solid line is a robust Theil-Sen fit of measured against JWST ellipticity
over galaxies with $|e_{\rm JWST}|<0.5$, drawn over that range, and the dotted
line in the model panels repeats the native-NISP fit. Each panel lists the core
fit slope and the full-sample RMSE about the 1:1 line, combining both components as
$\sqrt{\langle(\Delta e_1^2+\Delta e_2^2)/2\rangle}$. ResNet and ResNet-D5
steepen the relation towards 1:1 and reduce the scatter relative to native NISP
for both estimators; a single diffusion sample steepens the slope but increases
the scatter. Paired bootstrap intervals on the RMSE reduction are given in the
companion gain figure. These observed cross-band shapes include instrumental PSF
and pixel-sampling differences; they are not calibrated shear estimates.
