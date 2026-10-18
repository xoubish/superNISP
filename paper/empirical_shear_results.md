# Figure 9: response with the supplied NISP PSFs

The supplied-PSF rerun completed all 48 cells / 6,240 sheared native inputs,
with exit 0 and the original ResNet checkpoint unchanged.
[Figure 9 PNG](../paper_figures/exports/fig09_shear_response_v06.png),
[PDF](../paper_figures/exports/fig09_shear_response_v06.pdf),
[executed notebook and caption](../paper_figures/09_shear_response.ipynb).
The main figure has one panel for each radius, 0.18 and 0.35 arcsec, with S/N
20, 50, and 100. The Gaussian-native-PSF figures remain archived as v01–v04.
The approved v06 PDF is installed as Figure 9 in `survey_enhancements.tex`, with methods, results, and limitations in the main text.

## What changed

Use all four supplied NISP Y stack effective PSF profiles instead of the two
illustrative Gaussian native profiles. Preserve the signed arrays, normalize
integrated flux, recenter, and transform through the supplied PSF/native header
coordinate bases. Render without another native-pixel convolution, following the
PSFEx effective-profile convention. See [frozen protocol and sources](empirical_shear_protocol.md).
No assumed single Gaussian FWHM replaces the profile. The high-resolution
control remains an illustrative circular Gaussian of FWHM 0.08 arcsec.

Average per-PSF response ratios with equal weights, using exactly the same
successful objects across all four PSFs, all three observed methods and all
five shear branches. Equal weighting is not galaxy-abundance weighting. Retain
all PSF-specific matrices, ratios and counts. Noise is matched across PSFs and
shear branches; the four PSFs are not independent noise realizations.

## Results

Ratios are trace response divided by analytic intrinsic trace response on the
same common population, not shape accuracy or multiplicative shear calibration.

| Radius (arcsec) | Nominal input S/N | Native observed | ResNet observed | High-resolution control |
| --- | --- | --- | --- | --- |
| 0.18 | 20 | 0.4201 | 0.5716 | 0.9073 |
| 0.18 | 50 | 0.4208 | 0.7506 | 0.9073 |
| 0.18 | 100 | 0.4170 | 0.8001 | 0.9073 |
| 0.35 | 20 | 0.6283 | 0.7039 | 0.9713 |
| 0.35 | 50 | 0.6357 | 0.8565 | 0.9713 |
| 0.35 | 100 | 0.6397 | 0.8943 | 0.9713 |

ResNet retains more shear sensitivity than native NISP in all six displayed
cells, but remains below the intrinsic and high-resolution-control responses.
The range across four PSFs is small: at most 0.00354 in the displayed ResNet
response ratios. This is a PSF range, not an uncertainty interval.
The smallest-radius S/N-20 cell has 31/32 common objects; all other displayed
cells have 32/32. Across the full grid, 1,244/1,248 object-PSF realizations enter
the joint observed comparison (311 unique orientation/noise cases, each at four
PSFs). Their failures remain in the denominators.

The two displayed radii were selected for larger gains during the older Gaussian
pilot. The new run also evaluates 0.65 arcsec: at S/N 20, ResNet response is
0.7153 versus native 0.7743, so the gain is not uniform over size. At S/N 50/100,
the corresponding ResNet responses are 0.8542/0.9209 versus native 0.8096/0.8191.
The new noiseless ResNet responses are 0.8059, 0.9238, and 0.9510 at radii
0.18, 0.35, and 0.65 arcsec; those cells remain in the numerical notebook table.

All 6,240 auxiliary native REGAUSS attempts fail the undersampled native PSF
adaptive fit. These failures are retained and do not select the observed-method
comparison. This does not establish that PSF correction is unnecessary or that
ResNet outperforms a validated PSF-corrected estimator.

## Checks and reproducibility

Four rendering/selection controls passed before model execution, including
an analytic effective-pixel control, flux preservation, rotation/centering,
supplied-sample reconstruction, tightened convolution tolerances, and an injected
cross-PSF selection failure. Source and PSF hashes were frozen before execution.
An independent reader reproduced all per-PSF responses and success counts,
all 36 aggregated observed-response ratios, and the common-object selection;
checkpoint, source, protocol, raw-measurement and report hashes pass.

- [Control results](diagnostics/empirical_psf_controls_v01.json)
- [Verified plotting bundle](../paper_figures/inputs/shear_response_empirical_v01.json)
- [Run report](../runs/shear-response-empirical-v01/report.json)
- [Independent verification](../runs/shear-response-empirical-v01/verification.json)
- [W&B monitoring](https://wandb.ai/AI-Astro/superNISP-corrected/runs/e1ehrb1p)

No training, real-galaxy inference, or held-out test images were used. No model
execution remains queued. This is still a small synthetic pilot: finite PSF
stamps and header geometry, exponential profiles, fixed pixel phase, independent
pixel noise, and an illustrative target PSF. It does not model blending, realistic
morphology distributions, correlated stack noise, chromatic effects, or full
selection response; it does not provide calibrated shear measurements.
