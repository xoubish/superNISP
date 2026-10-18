# Historical v01 cached model shape comparison and proposed results figures

Superseded figure choice: former 7b is now the single Figure 7, with matched
native-NISP contours. See [the current v02 report](figure7_native_shapes.md).
The v01 populations and layout recommendations below are preserved as history.

Local validation experiment requested after the Figure 6 source-only Fourier
comparison. Frozen v03 predictions are used for all six methods; no inference,
training, W&B upload, manuscript installation or Overleaf push was performed.

## Recommended selection from the Figure 6 diagnostics

1. **Main reconstruction benchmark:** source-only MAE/RMSE and inference speed
   (`fig06_model_comparison_v03`). These directly quantify the accuracy/cost trade-off.
2. **Spatial structure:** retain the source-only Fourier power **and signed
   correlation together** (`fig06_source_spatial_structure_v01`, panels a/b).
   This explains why visually sharper diffusion images can still have larger
   reconstruction errors. Place the source/background proxy panel in the appendix.
3. **Observed shapes:** use the new matched shape summary (`fig07_model_shapes_v01`),
   including both estimators and full-denominator measurement success. This tests
   whether extra spatial power produces more accurate observed ellipticity.

Appendix: whole-cutout spectra, outside-source/background controls, source-mask
examples and boundary-taper sensitivity. Error histograms are optional supplementary
material; the main error summary already establishes the ranking. The component
shape-agreement plot is useful as supplementary evidence. If the main text must
be shorter, prioritize the reconstruction benchmark and shape summary, with
Fourier diagnostics in the appendix. Avoid selecting a power-only panel to imply
that diffusion recovers the correct fine morphology.

This is a proposed results layout, not final manuscript figure numbering. The
existing manuscript still contains older comparison/shape plots and claims, which
must be replaced together when these local results are approved for installation.

## Frozen measurement protocol

All 2,118 validation objects are attempted. Same cached common-valid mask for
JWST, bilinear, bicubic, single diffusion, diffusion mean-of-3, ResNet and ResNet-D5.
No source-segmentation clipping is applied to the shapes: imposing the same
reference outline would alter shape measurement. Signed MJy/sr images receive
no additional clipping, smoothing, centering, or sky subtraction.

The primary fixed estimator reuses `shape_diagnostics.diagnose_moments`:
Gaussian weight sigma 10 target pixels, radius-25 central aperture, intensity
centroid, and signed second moments. Ellipticity is e1=(Qxx−Qyy)/(Qxx+Qyy),
e2=2Qxy/(Qxx+Qyy). Positive flux and positive-definite second moments and complete
valid aperture are required. The existing HSM adaptive estimator uses the same
105-pixel crop, initial sigma 10 pixels, bad-pixel mask, valid radius-25 aperture,
and centroid-displacement guard of 10 pixels. It fits its own weight, size and
centroid. These weights define different observed shapes, so absolute errors
should be compared between methods within an estimator, not between estimators.

Each estimator's errors use only the identical all-seven-image successful
intersection; no model gets its own easier sample. Full-population success counts
and overlapping failure reasons are retained separately. Also compare both
estimators on their joint 1,313-object intersection. The RMSE is
sqrt(mean_objects[(delta_e1²+delta_e2²)/2]); bias, component scatter, fraction with
vector error >0.2, diagnostic OLS slopes and adaptive log-size ratios are saved.
Intervals use 2,000 paired draws of the saved spatial blocks, seed 20261017.
The full contract was written before examining measurements.

## Results

Component RMSE, lower is better. Within each column all methods use the same
successful objects; intervals are conditional 95% spatial-block intervals.

| Method | Fixed moments (1,447 objects) | Adaptive moments (1,546 objects) |
|---|---:|---:|
| Bilinear | 0.1952 [0.1892, 0.2004] | 0.2565 [0.2434, 0.2683] |
| Bicubic | 0.2015 [0.1943, 0.2080] | 0.2477 [0.2347, 0.2597] |
| Diffusion, one sample | 0.2292 [0.2187, 0.2380] | 0.3031 [0.2928, 0.3128] |
| Diffusion, mean-of-3 image | 0.1876 [0.1775, 0.1963] | 0.2538 [0.2436, 0.2634] |
| ResNet | 0.1624 [0.1554, 0.1684] | 0.2210 [0.2129, 0.2283] |
| ResNet-D5 | 0.1626 [0.1555, 0.1687] | 0.2214 [0.2130, 0.2289] |

ResNet and ResNet-D5 are indistinguishable under these paired comparisons:
ResNet-D5 minus ResNet RMSE is +0.00014 [−0.00104, +0.00142] for fixed moments,
and +0.00039 [−0.00110, +0.00183] for adaptive moments. Single diffusion has
larger errors than bilinear with both estimators. Averaging three diffusion
images improves shapes substantially, but its RMSE remains above both ResNets.
Its improvement over bilinear is supported for fixed moments (difference
−0.00754 [−0.01388, −0.00117]); the adaptive difference interval crosses zero
(−0.00268 [−0.01113, +0.00681]).

On the same 1,313 objects measurable by both estimators, fixed RMSE values for
single diffusion / mean3 / ResNet / ResNet-D5 are 0.2181 / 0.1777 / 0.1506 /
0.1510, and adaptive values are 0.2652 / 0.2205 / 0.1909 / 0.1917. The ranking
is unchanged by fixing the population across estimators.

Full-denominator successful measurements out of 2,118:

| Image | Fixed moments | Adaptive moments |
|---|---:|---:|
| Observed JWST | 1662 | 1767 |
| Bilinear | 1661 | 1632 |
| Bicubic | 1622 | 1646 |
| Diffusion, one sample | 1633 | 1678 |
| Diffusion, mean-of-3 image | 1722 | 1724 |
| ResNet | 1786 | 1742 |
| ResNet-D5 | 1786 | 1745 |

The common-valid mask leaves 273 objects with incomplete central apertures for
both estimators; failures also arise from invalid signed quadrupoles, failed
adaptive fits and excessive adaptive centroid displacement. Reason flags overlap
and must not be summed as exclusive categories. Success alone is not shape
accuracy: a smooth positive image can be easy to measure while being wrong.
Counts and values differ from the older single-ResNet runs because this analysis
uses a common validity mask and requires every new model to succeed on the
comparison sample. Do not splice the older errors into this table.

## Paper interpretation

“Diffusion reconstructions retain more observed small-scale power, but the
residual networks yield lower observed ellipticity errors under both fixed
weighted and adaptive moment estimators. ResNet and ResNet-D5 have statistically
indistinguishable shape errors in this matched validation comparison. Averaging
three diffusion realizations reduces shape errors but does not close the gap.”

These are cross-band, observed weighted ellipticities; no PSF deconvolution or
shear calibration was performed. Blends can contaminate moments, and the adaptive
centroid guard is not deblending. Selection is conditional on successful
measurement. The mean3 entry measures a mean image, not a mean of three shape
measurements. Only one individual diffusion draw is cached; realization-to-
realization shape scatter is not measured. No model or checkpoint was selected
using these shape results, but checkpoints were already validation-selected.

## Artifacts and verification

- [Shape summary](../paper_figures/exports/fig07_model_shapes_v01.png)
- [Summary PDF](../paper_figures/exports/fig07_model_shapes_v01.pdf)
- [Component agreement](../paper_figures/exports/fig07_model_shape_agreement_v01.png)
- [Archived v01 notebook](../paper_figures/revisions/07b_model_shapes_v01_before_native_contours.ipynb)
- Frozen result JSON: `paper_figures/inputs/cached_shapes_v01.json`.
- Local per-object arrays: same stem `.npz`, including shapes, moments, success
  flags, HSM errors/status/centroids/sizes, IDs, blocks and the global intersection.
- Run, contract and source snapshots: `runs/cached-shapes-v01`.
- Analysis: `code/analyze_cached_shapes.py` and `code/cached_shape_comparison.py`.

Five analytical tests passed (matched selection/bootstrap plus four existing
moment/HSM robustness tests). Independent NumPy moments reproduced all 14,826
fixed measurements and their flags. Direct GalSim calls reproduced 168 fixed
random-image checks. All success intersections, paired block intervals,
input/source/array hashes and notebook/export hashes passed verification.
