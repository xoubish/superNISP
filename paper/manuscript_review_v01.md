# Whole-manuscript review

Reviewed `survey_enhancements.tex` against the frozen sample records, implementation, current figure records, and completed experiment notes. This is a review only: the manuscript and Overleaf have not been changed. Line numbers refer to the manuscript at review time.

The current Figures 6–9 and their central interpretation are consistent: residual models improve reconstruction and observed-shape agreement, diffusion retains more observed fine-scale power, and the controlled shear experiment measures response rather than calibrated shear accuracy. The abstract reflects these results. The remaining work is principally factual corrections, completing the model-choice narrative, and removing methods that currently have no reported outcome.

## Definite corrections

1. **Observations, line 206: outdated source-image inventory.** The text says 33 Euclid images and 20 JWST images. The frozen extraction manifest lists four NISP Y science stacks and 19 F115W mosaics; accepted objects use four NISP stacks and 18 JWST mosaics. Replace the old counts and distinguish considered inputs from inputs contributing accepted objects. The quoted approximately 10 square arcminutes per JWST image also needs replacement: the supplied mosaics have dimensions 9600 by 12455 pixels at the stated 60 mas sampling. Rectangular image dimensions must not be presented as valid sky coverage without accounting for masks and overlap.

2. **Introduction, line 155: incorrect instrument attribution.** Lee et al. (2018) and Finner et al. (2023) concern HST/CANDELS imaging, not Spitzer/IRAC. Describe their contributions separately where useful: [Lee et al.](https://arxiv.org/abs/1808.05223) compare optical and near-infrared ellipticity measurements, including degraded images; [Finner et al.](https://arxiv.org/abs/2301.07725) study WFC3-IR PSFs and weak-lensing analysis.

3. **Introduction, line 171: residual architecture terminology.** The implementation uses eight single residual dense blocks with a trunk skip. It does not implement the standard nested residual-in-residual dense blocks claimed here. Match the wording already used in the methods and Figure 4. Also soften the implication that direct regression necessarily recovers genuine information while diffusion alone uses a learned prior; both depend on the training distribution.

4. **Bibliography and template cleanup.** A standalone BibTeX run using the active manuscript citations reports two missing entries (`Huang_et_al_2021`, `drizzle`), a duplicate `wang2018`/`Wang2018` entry, and an undefined `july` month string. BibTeX treats the two Wang keys as duplicates despite capitalization. The cited `DR1-TP002` and `DR1-TP003` records contain the placeholder journal text “A&A, hopefully submitted”; align the references with the actual instrument/Q1 data products used. Check the existing Q1 citation TODOs. The running-author field still contains `F. Author et al.`. Do not alter the consortium author list as part of this cleanup.

5. **Discussion, lines 798–800: stale future-work language.** “More cross-attention layers” implies an existing component absent from the current diffusion implementation. “A third round of training” no longer describes the completed follow-up experiments clearly. Replace with specific remaining questions informed by the continuation and PSF-aware results. Drizzle/resampling alone is not an astrometric-registration solution; describe empirical alignment and effective-PSF validation if retaining this suggestion.

## Missing results or unnecessary material

6. **Robustness methods, lines 550–590, lack corresponding results.** The paper devotes a subsection and full-width table to 21 scenarios but does not report their outcomes. The zero-ellipticity control at line 461 and crowding diagnostic likewise need an outcome or removal. Keep the adaptive-moment definitions because Figures 7–8 require them. Prefer a short robustness result and compact methods description over reinstating an appendix. Existing results are in `shape_robustness_and_convergence.md`; their strict common sample contains 1098 galaxies and is different from the current figure samples. Do not silently transfer their absolute errors or populations into Figures 7–8.

7. **Model choice, lines 674–685: add the completed PSF-aware tests.** `psf_redesign_results.md` records the PSF/noise and inverse-feature warm starts. Compared with matched extra training, their additional source-RMSE reductions are only 0.12–0.16%, with paired intervals spanning zero. PSF/noise improves fixed-weight shape RMSE by about 0.98%, but neither branch clearly improves adaptive shapes or overall visual recovery. A short methods description and two or three results sentences would explain why the original model was retained. These checks do not rule out better PSF-aware architectures. Their shape cohorts also differ from Figure 7.

8. **Explain original ResNet versus D5 explicitly.** D5 has slightly lower source RMSE, so avoid suggesting that the original wins every metric. State the actual decision: similar shape performance, independent-seed evidence for the original, and a consistent primary training budget. This makes the choice understandable without adding a figure.

## Scientific scope and remaining evaluation

9. **The held-out test is still reserved.** Lines 211 and 685 correctly disclose that the findings use validation data; line 781 refers to a subsequent evaluation stage. The major remaining scientific completion item is a one-time evaluation of the frozen model and protocol on the 2876 held-out galaxies. This review does not authorize or perform it. If deferred, the final paper must consistently present itself as a validation study and avoid implying that the held-out evaluation appears later in this manuscript.

10. **Distinguish sampling from effective resolution, lines 194–197.** Five-times finer sampling is established. The nominal F115W instrument PSF is not automatically the effective PSF of the 60 mas mosaic and resampled target. The approximately 0.04 arcsec instrument FWHM is supported by [STScI documentation](https://jwst-docs.stsci.edu/jwst-near-infrared-camera/nircam-performance/nircam-point-spread-functions), which also distinguishes resampled-product PSFs. Clarify the NISP width's provenance and soften “must therefore perform an effective deconvolution”; the present analysis does not validate a deconvolved output PSF or a fivefold gain in recovered resolution.

11. **Connect the sample selection to the motivation.** The introduction motivates near-infrared shapes for VIS-faint galaxies, but training and evaluation require F814W < 25, photometric redshift < 3, and one COSMOS field. Add one explicit discussion sentence explaining that the study does not yet demonstrate performance on the redder/optically fainter population motivating that application.

## Editorial finish

- Add a short conclusion that states the image-fidelity, observed-shape, and controlled-response findings positively, followed by the next validation step.
- Add a data/code availability statement with actual release locations or an accurate availability plan; do not invent public links.
- Condense repeated caveats while retaining the distinctions between observed shapes, successful-measurement selection, and shear calibration.
- Soften the introductory blanket statement that cosmological constraining power is now limited by systematics rather than statistics; the balance depends on the analysis.
- Keep Figures 6–9. These findings do not motivate another figure or restoration of the appendix.

## Verification limits

The review included source/implementation and experiment-record comparisons, primary-source checks for the instrument attribution and NIRCam PSF, and an actual standalone BibTeX run. It did not include a new training run, held-out evaluation, or Overleaf build. A full local manuscript build remains blocked by the previously identified missing `etex.sty`; the bibliography errors were checked independently of that dependency.
