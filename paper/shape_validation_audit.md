# Shape-validation audit and next experiment

Audit date: 2026-10-16. Source tree: `main`, commit `5e2fe67`.
This is a working research note, not a replacement for the manuscript.
No inference or retraining was performed during this audit. Existing paper
numbers are unverified until their inputs and evaluation are recovered or rerun.

## Scientific target

The first result to establish is:

> On an independent sample of matched galaxies, superresolution reduces
> observed ellipticity error relative to native NISP and interpolation by X%,
> with a stated confidence interval, measurement success rate, and range of
> galaxy properties over which the improvement holds.

JWST F115W is a higher-resolution reference, not a noiseless, PSF-free truth
image of the NISP Y-band sky. The experiment measures agreement with that
reference. A claim about calibrated gravitational shear needs a further
experiment with known applied shears, PSFs, and selection effects.

## Evidence already in the paper

The supplied `figs/ellipticity_comparison.png` labels bilinear slopes of
0.64/0.67 and ResNet slopes of 0.70/0.74 for e1/e2. These are encouraging
changes toward unity. They do not establish a percentage reduction in
per-galaxy shape error; scatter, offsets, selection, and uncertainties matter.
The figure has JWST on the horizontal axis, consistent with the manuscript.

The code currently on main does not reproduce that specification:
`code/evaluate_models.py:478` uses ordinary least squares (`np.polyfit`),
with JWST as the dependent variable, whereas the manuscript describes ODR
with SR as the dependent variable. Do not reinterpret or invert the existing
slopes without the original measurements.

## Findings that affect the scientific comparison

1. **Observed shape is being called shear.** `get_moments()` returns
   `observed_e1/e2` and `observed_shape.g1/g2` from `FindAdaptiveMom`.
   These are two parameterizations of the same observed shape, not four
   independent observables or a calibrated gravitational-shear measurement.
   The slope of SR shape against JWST shape is not the multiplicative shear
   calibration bias. Rename these quantities and avoid translating their
   slope deficits directly into a distance from cosmological requirements.
   See the [GalSim HSM documentation](https://galsim-developers.github.io/GalSim/_build/html/hsm.html).

2. **The models are evaluated on different representations.** ResNet uses
   205-pixel targets with per-image affine z-score normalization; diffusion
   uses 105-pixel targets with nonlinear asinh normalization. The evaluation
   measures shapes and pixel errors directly in these representations,
   without inverting asinh. The apertures, target images, and bilinear
   baselines therefore differ. Nonlinear intensity transforms can change
   measured shapes. Use the same angular measurement region and a common
   linear intensity representation. Affine normalization after consistent
   background removal does not itself change noiseless moment ellipticity;
   absolute photometric errors still require a justified flux scale.

3. **Diffusion inference needs repair before architectural conclusions.**
   `code/diffusion/model_sr3.py:467` selects nonconsecutive timesteps for
   shortened sampling, but `p_mean_variance()` uses adjacent-step posterior
   coefficients. The destination timestep never enters the update. Turning
   off the added noise does not turn this into a correct DDIM jump.
   Compare the [authors' DDIM implementation](https://github.com/ermongroup/ddim/blob/main/functions/denoising.py),
   which explicitly uses both timesteps. This is a code-level issue; its
   contribution to the published-looking figures is unknown. Also check
   initialization around the LR image, clipping to [-1, 1] despite unclipped
   asinh targets, and random initialization even with `deterministic=True`.

4. **The evaluation subset is used for model selection.** The visible
   training scripts select checkpoints using validation data, and the
   evaluation recreates a validation split. This does not establish an
   untouched test set. The manuscript also acknowledges spatial overlap
   between train and test cutouts. Freeze train/validation/test object IDs
   and use spatial grouping with a buffer to prevent overlapping footprints
   across subsets. Existing checkpoints cannot become independent-test
   models merely by repartitioning galaxies they may already have seen.

5. **Normalizer state is not preserved as an evaluation artifact.** The
   diffusion loader and validation wrapper fit normalizers on full arrays,
   including held-out objects, using random subsets. Train and validation
   instantiate separate fits. Fit on training data only, save the fitted
   parameters, and reuse them without refitting.

6. **Measurement failures change the populations being compared.** Plotting
   uses method-specific finite masks. Record every failure and compare
   residuals on explicitly matched objects, alongside each method's success
   fraction on the full reference sample. Retain bad-pixel masks instead of
   silently treating nonfinite or masked pixels as valid zeros. Validate
   that the moments correspond to the central galaxy rather than a neighbor.

7. **Other claims need narrower wording or new evidence.** An image-to-image
   network has no demonstrated single effective PSF just because its targets
   have a JWST PSF. Neither network is free of learned priors. Better SSIM or
   attractive examples alone do not establish deblending or strong-lens
   classification gains. The current catalog cut is ACS F814W < 25, with
   0.01 < photo-z < 3 and 4 < FLUX_RADIUS < 500 (`clipping.py:94` onward),
   differing from the draft's radius description; this selection also needs
   examination before claiming benefits for optically undetected galaxies.

The evaluation sets NISP's pixel scale to 0.10 rather than the draft's 0.30
arcsec/pixel. **Follow-up header inspection:** local MER mosaics are actually
sampled at 0.10 arcsec, while inspected NIR stacks are approximately 0.297
arcsec. Resolve each array's source product before changing the scale;
0.10 is not universally an error. An isotropic scale correction alone does
not fix dimensionless ellipticity. See `architecture_sample_review.md`.

## Local artifact inventory

Read-only header inspection found:

| Product | Local contents |
| --- | --- |
| Undated NISP/JWST 41/205-pixel NumPy pair | 53,229 image pairs |
| `20251124` NISP/JWST NumPy pair | 34,058 image pairs |
| `Nisp_train_cosmos.hdf5` / `Nircam_train_cosmos.hdf5` | 42,583 train + 10,646 test; 25/125-pixel images |
| `code/splits.json` | 29,809 train + 8,516 validation + 4,258 test indices |
| `matched_cat_COSMOS.csv` | 73,365 rows |
| `matched_cat_cosmos_2_20251124.csv` | 39,024 rows |
| `Nircam_train_nonorm.hdf5` | h5py failed to open: bad object header version number |

The draft describes 22,738 matched galaxies but 18,190 + 4,547 = 22,737
split members. The current evaluation caps inference at 2,000. None of this
identifies which local data generated the figures. Catalog row counts also
differ from cutout counts, so a row-index join is unsafe without tracing
the extraction/rejection ordering. The HDF5 read error is an observed local
failure, not yet a diagnosis of its cause. Header inspection does not verify
the integrity of every image.

Five checkpoint files are present under `code/` and were inspected with
`torch.load(..., weights_only=True)`. `best_sweep_model_tbra2akh.pth` has
ResNet-style weights; `final_best.pth` and `stage1_best.pth` are another
compiled ResNet variant. Other files are older architectures. None has
established provenance linking it to the current figures.

Checked local branches `main`, `Junemod`, `shooby_1diffuse`, cached remote
branches including `emily_old`, and artifact paths in all reachable Git
history. A live `git ls-remote --heads origin` confirmed that the current
remote heads for main, emily_old, and shooby_1diffuse match the inspected
commits. No `rrdb_results.npz`, `diff_results.npz`, `best_model_rrdb.pth`, or
`best_model_diffusion.pth` was found in those artifact paths. The file called
`code/diffusion/best_sweep_model_tbra2akh.pth` on shooby_1diffuse is byte-for-byte
the same Git blob as the ResNet checkpoint. Older diffusion-related weights
exist in history but are not identified as the paper's SR3 model.

Data are present locally, but `.gitignore` excludes data directories,
catalog CSV/FITS files, and generally checkpoints (some checkpoints are
already tracked). Local availability and reproducible repository delivery
are separate tasks. Keep derived results and provenance under this repo;
record checksums and retrieval instructions for large inputs.

## Step-by-step work plan

1. **Freeze the data and baseline.** Recover an object-ID/coordinate mapping
   for one paired dataset, verify registration and masks on a representative
   subset, and establish spatially separate train/validation/test groups.
   Save a manifest with input hashes, selection, coordinates, and split IDs.
   Measure native NISP, bilinear, bicubic, and JWST reference shapes with
   matched angular regions and controlled measurement settings. If exact
   cutout provenance cannot be recovered, regenerate pairs with IDs retained.

2. **Establish the main quantitative result with ResNet.** Use existing
   weights for diagnostic work where compatible, and retrain on the frozen
   split for the independent result. For each baseline B, report
   `100 * (1 - RMSE_e(SR) / RMSE_e(B))`, where
   `RMSE_e = sqrt(mean((delta_e1**2 + delta_e2**2) / 2))` on the same objects.
   Give confidence intervals from paired resampling, preferably spatial
   blocks; report component offsets, scatter, outlier fractions, and shape
   success rates. Include a constant/round-shape control to detect apparent
   gains from shrinkage. Treat regression slopes as supporting diagnostics.
   Define quality cuts and bins on validation data, then report results by
   input S/N, angular size/resolution, magnitude, crowding, and available
   redshift. Do not tune those choices against the final test result.

3. **Improve the identified failure mode.** First assess registration and
   shape-measurement failures. Then test one controlled change at a time,
   such as a shape-aware loss or PSF-aware conditioning, using validation
   data. Repair diffusion sampling and intensity conversion before deciding
   whether the diffusion architecture deserves further training. Preserve
   per-object predictions, flags, metrics, seeds, configs, and checkpoints.

4. **Write the paper around the demonstrated scope.** Generate tables and
   figures from saved measurements. State where gains hold and where they
   fail. If weak-lensing calibration remains a central claim, add controlled
   known-shear simulations through the complete measurement pipeline,
   including selection; otherwise present shear calibration as future work.

The next concrete deliverable is an auditable data manifest and baseline
shape table. The target percentage remains an experimental outcome, not a
number to choose in advance.
