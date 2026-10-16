# Methods revision — 2026-10-16

The Methods section in `survey_enhancements.tex` was reviewed against the current
implementation and recorded full-run configurations. Text before Methods and
from Results onward was compared byte-for-byte with the pre-edit manuscript and
is unchanged. The four missing software bibliography entries used by Methods
were added to `mybib.bib`. No training, inference, or model construction ran.
These Methods edits have not been pushed to Overleaf.

## Evidence used

| Subject | Source |
| --- | --- |
| WCS mapping, masks, coverage, backgrounds | `code/build_training_sample.py`; paired sample `manifest.json` and `BUILD_REPORT.md` |
| Fixed training-only scales and signed intensities | `code/paired_sample.py`; paired sample `normalization.json` |
| Diffusion network, loss, generation and selection | `code/diffusion/model_sr3.py`, `code/diffusion_corrected.py`, `code/train_diffusion_full.py`; `runs/diffusion-full50-s20261016/provenance.json` |
| Residual variants and training | `code/residual_baseline.py`, `code/train_paired.py`; both residual full-run `provenance.json` files |
| Parameter counts | Source-verified architecture notebook export records (Figure 3 v05 and Figure 4 v04) |
| Native/interpolated/fixed shapes, failures, RMSE, bootstrap | `code/shape_diagnostics.py`, `code/evaluate_paired_shapes.py`, `code/compare_shape_runs.py` |
| Adaptive moments and 21 robustness scenarios | `code/robust_shape_measurements.py`, `code/run_shape_robustness.py`; completed robustness report |
| Diffusion sample variation and mean-image definition | `code/diffusion_diagnostics.py`, `code/diffusion_full_diagnostics.py` |

All recorded training source hashes matched the current files for the residual,
direct-5x and diffusion full runs. No model weights were loaded for this review.
The training table specifies the budget; it does not assert that ongoing runs
have finished or that the models have converged. Full diffusion evaluation and
its three-draw mean are described as the evaluation protocol.

## Changes

- Replaced the old rotation-only extraction description with explicit WCS
  resampling onto the exact five-times-finer NISP grid.
- Documented independent masks, coverage cuts, background subtraction, physical
  units, fixed scales, and the diffusion transform ordering.
- Removed the claim that output PSFs are known to be close to the JWST PSF and
  the implication that only the diffusion network uses a learned prior.
- Added compact diffusion/loss equations and consolidated shared training and
  model-specific settings into one comparison table.
- Distinguished full residual validation from the fixed 256-object diffusion
  monitor, and physical MAE from differently normalized training objectives.
- Added the observed-shape measurement protocol: fixed signed moments, native
  angular matching, interpolation masks, common successful IDs, failure counts,
  RMSE definition and paired spatial-block intervals.
- Added the independent adaptive-moment and perturbation protocol with a table
  of settings. These checks concern residual models; they are not described as
  completed diffusion robustness experiments.
- Kept the actual bootstrap counts explicit: 1000 for dedicated shape comparisons
  and 200 for individual diffusion-generation summaries. This does not silently
  standardize already-generated reports.

## Validation and remaining scope

Methods references, citation keys, unique labels, LaTeX environment balance,
and whitespace checks passed. An isolated two-column `article` preview compiled
successfully and its two tables were visually checked, with no overfull boxes.
This is a syntax/layout check, not a full A&A manuscript build: the installed
TeX environment still lacks `etex.sty`. The temporary preview replaces citation
rendering and does not resolve references outside the extracted Methods.

The Introduction, Observations, Results, Discussion and abstract retain their
previous content. Some still describe historical sample sizes, architecture
labels, PSF assumptions, PSNR/SSIM results and shear interpretations. Their
reconciliation belongs to the later user-authorized revision, rather than being
silently included in this Methods-only pass. Current Methods describes physical
MAE and the residual RMSE diagnostic actually implemented; no new PSNR/SSIM
protocol is claimed.

Software references were checked against primary sources:
[Astropy](https://arxiv.org/abs/2206.14220),
[SciPy](https://arxiv.org/abs/1907.10121),
[GalSim](https://arxiv.org/abs/1407.7676),
[GalSim HSM documentation](https://galsim-developers.github.io/GalSim/_build/html/hsm.html),
and [W&B citation instructions](https://wandb.ai/site/research/).
