# superNISP GPU handoff

Updated 2026-10-17. Read this first when continuing on `ipacgpu3`.
This records completed work and the next experiment; it does not certify
the old paper results or claim a trained model is ready.

## Latest update: all nine figures installed in manuscript — 2026-10-17

`paper/survey_enhancements.tex` now includes all nine approved Matisse-styled
figures in order. Added `paper/figs/shape_dependence_v04.pdf` as Figure 8 and
`paper/figs/shear_response_v06.pdf` as Figure 9; the historical residual-plane
figure and obsolete ODR/shear-calibration claims were replaced. Figure 8 has
input-S/N/coverage methods and results; Figure 9 has the supplied-effective-PSF
rendering convention, finite-difference response, common selection, full-grid
counterexample, and limitations. Related abstract/introduction/discussion text
is consistent with the observed-shape and controlled-response distinction.
No model/data changes, new inference, or external publishing.

Pre-edit manuscript: `paper/diagnostics/figures_8_9_install_v01/manuscript_before.tex`.
Verification: `paper/diagnostics/figures_8_9_install_v01.json`.
Standalone Figures 8/9 caption proof: `paper/diagnostics/figures_8_9_caption_proof_v01.pdf`.
Full A&A compilation still requires `etex.sty`; no dependency stub was substituted.
The individual approved figure PDFs are unchanged in scientific content.

## Previous update: consistent figure style — 2026-10-17

All nine active figure notebooks are rendered with the user-requested custom
Matisse-inspired palette: cobalt ResNet, green D5, ochre diffusion; pink/red
categorical colors removed. Nimbus Sans fonts, consistent markers, line patterns,
white backgrounds, and coverage hatching. See `paper_figures/STYLE_GUIDE.md`.
No training, inference, metric changes, object reselection, or external uploads.

Current primary export versions, Figures 1–9: v08/v06/v06/v06/v08/v07/v06/v04/v06.
Review: `paper_figures/exports/all_figures_style_v01.pdf` (nine pages), PNG contact
sheet, and Figure 6/8 color-vision/grayscale proof sheets. Styled Figures 1–7 are
installed in `paper/survey_enhancements.tex`; Figure 8/9 remain notebook drafts.
The manuscript's historical residual-plane Figure 8 still awaits scientific
replacement. No full manuscript compilation: missing TeX dependencies persist.

`09_shape_robustness.ipynb` was removed from the active folder and preserved as
`paper_figures/revisions/09_shape_robustness_retired.ipynb`. Pre-style notebooks,
helpers, and manuscript source are in `paper_figures/revisions/style_before_v01/`.
Figure 1 uses exact display rasters from the hash-verified v07 SVG because the
raw mosaic cache is absent here; Figure 2 uses exact verified integer histogram
counts recovered from its v05 SVG, with six selected distributions independently
checked. Details are in the style guide. Figure 7's supplementary bootstrap table
is frozen at its previous displayed precision. Numerical bundles are unchanged.
Verification: `paper/diagnostics/figure_style_v01.json`.

## Previous update: Figure 9 with supplied NISP PSFs completed — 2026-10-17

The user approved rerunning the shear experiment with supplied NISP PSFs.
`runs/shear-response-empirical-v01/` completed 48 cells / 6,240 native shear inputs
with exit 0, original epoch-49 ResNet unchanged. W&B:
https://wandb.ai/AI-Astro/superNISP-corrected/runs/e1ehrb1p . No job remains queued.
Runner `code/empirical_shear_response.py`; launcher `code/launch_empirical_shear.sh`;
protocol `paper/empirical_shear_protocol.md`; results `paper/empirical_shear_results.md`.

All four supplied signed effective PSFs are normalized, centered, and mapped
using supplied PSF/native CD matrices, with Lanczos11 interpolation. Native
rendering uses no_pixel (PSFEx effective profile already includes pixel response).
Four scientific controls passed; `code/verify_empirical_shear.py` independently
reproduced all matrices, ratios, counts, and hashes. Native bare-PSF HSM fits and
all auxiliary native REGAUSS attempts fail from undersampling; failures are kept
and do not select the observed comparison. This remains a synthetic pilot.

Current notebook: `paper_figures/09_shear_response.ipynb`, executed v05.
Exports: `paper_figures/exports/fig09_shear_response_v05.{png,pdf,svg,json}`.
Frozen input: `paper_figures/inputs/shear_response_empirical_v01.json`.
Two panels now identify radii 0.18/0.35 arcsec, with three method curves versus
S/N 20/50/100. Each point equally averages all four PSFs on identical successful
objects (31/32 for smallest/lowest-SNR, otherwise 32/32). ResNet responses at
S/N 100 are 0.8001/0.8943 versus native 0.4170/0.6397. Keep original model.
The larger-radius S/N-20 counterexample remains: ResNet 0.7153 < native 0.7743.
All sizes/noiseless inputs and PSF-specific values are in diagnostics and notebook
tables. Gaussian v01–v04 plots/notebooks are archived. No manuscript installation
or external paper upload. Real galaxy and reserved test images were not read.

## Previous update: PSF-aware redesign completed — 2026-10-17

Both redesigns finished 10 epochs / 22,270 updates, selected epoch 5, and exited
0. The automatic CPU comparison also exited 0. No job remains queued by this
experiment. Read [paper/psf_redesign_results.md](paper/psf_redesign_results.md).
Results: `runs/psf-redesign-comparison-v01/`; notebook:
`paper_figures/diagnostics/psf_redesign.ipynb`; paired-effect plot:
`paper/diagnostics/psf_redesign_paired_v01.png` (also PDF/JSON).

Source RMSE: original 0.0601735, extra-training control 0.0594465, conditioned
0.0593742, inverse-feature 0.0593513 MJy/sr. The additional 0.122%/0.160% gains
over control have paired intervals spanning zero. Conditioned fixed-shape RMSE
improves 0.98% over control (difference CI excludes zero), but neither redesign
clearly improves adaptive shapes or fine-scale correlation over control. Visual
examples remain very similar. User-confirmed decision: retain the original independently
replicated paper ResNet; conditioned is a follow-up candidate, with no demonstrated
overall benefit from the inverse feature. No manuscript/numbered-figure switch.

Three CPU controls and both GPU preflights passed. Independent verification checks
all training streams against control, exact epoch-zero parent weights, inherited
and new optimizer steps, final learning rates, hashes, and 20 paired shape
bootstrap differences. Primary common samples: 1,627 fixed / 1,700 adaptive;
do not directly compare absolute errors to Figure 7's different common sample.
Original checkpoint unchanged; test images unused. Source snapshots and the
protocol remain immutable for reproducibility. Earlier launch details follow.

## Previous update: PSF-aware redesign training — 2026-10-17

User requested a redesign and comparison of RMSE, visuals, sharpness and shapes
before finalizing the paper. After automatic review blocked the initial launch,
the user **explicitly approved uploading these runs' training/validation metrics
and fixed galaxy images to AI-Astro/superNISP-corrected**. Both runs then launched:

| Branch | GPU | tmux | W&B |
| --- | --- | --- | --- |
| Image + PSF + noise + mask | 0 | `supernisp-psf-conditioned-v01` | [on773llj](https://wandb.ai/AI-Astro/superNISP-corrected/runs/on773llj) |
| Same + bounded inverse-filter feature | 1 | `supernisp-psf-deconv-v01` | [hdsb3bfx](https://wandb.ai/AI-Astro/superNISP-corrected/runs/hdsb3bfx) |

Outputs: `runs/psf-redesign-{conditioned,deconv}-v01/`; adjacent `-console`
directories have logs/exit codes. At this update both passed GPU finite-update,
exact checkpoint restoration and numerical next-step tests, and reproduced the
parent's complete validation predictions/metrics before entering epoch 1.
Read live `progress.json`, `live.json`, `status.json` before claiming completion.
Three CPU tests passed, including original-output equivalence, optimizer isolation,
flux-preserving inverse filter, D4 transformations, and PSF sampling geometry.

Frozen design: [paper/psf_redesign_protocol.md](paper/psf_redesign_protocol.md).
Do not edit active training sources or that protocol; resume signatures include
their hashes. The original epoch-49 weights/Adam initialize both; zero context
adapter initially preserves the original. Ten extra epochs / 22,270 updates each,
same masked L1 and batch/augmentation stream as completed control v03, checked
per epoch. New context LR 1e-4→1e-5; original weights 1e-5→1e-6. Select source MAE
with epoch zero eligible; report selected/final. Every new epoch checkpoint saved.
Both current parent and matched extra-training control are required baselines.

PSF associations/proxies: `runs/psf-features-v01/`. Four nearly identical empirical
PSFs; header-based orientation/pixel-response approximation is explicit. No JWST
information in inference inputs, no test images, no physically exact deconvolution
or calibrated shear claim. Original image remains available to both new models.
Evaluation code `code/compare_psf_redesign.py` waits for completion and compares
all selected/final checkpoints, including common-success shapes and failure rates,
source/outside spectra, S/N/stack subgroups, fixed Figure 5 image/residual sheets,
and paired spatial-block intervals. Results go to `runs/psf-redesign-comparison-v01/`.
Current manuscript/figure inputs remain unchanged pending the comparison.

## Previous update: PSF audit and synthetic shear-response pilot — 2026-10-17

User authorized the PSF audit and controlled shear-response pilot. Completed all
24 cells / 3,120 frozen original-ResNet inputs in tmux
`supernisp-shear-response-pilot-v01`, exit 0, with online
[W&B 0r3dbbqc](https://wandb.ai/AI-Astro/superNISP-corrected/runs/0r3dbbqc).
No training, checkpoint writes, or real/test galaxy images. Contract, immutable
source snapshots, per-object shapes/statuses, and primary results are in
`runs/shear-response-pilot-v01/`; launcher `code/launch_shear_response_pilot.sh`.
All three physics/selection controls passed; independent response/count/hash
checks passed. Full report: [paper/shear_response_pilot.md](paper/shear_response_pilot.md).
Review plot: `paper/diagnostics/shear_response_review_v02.png` (also PDF;
v02 JSON records presentation provenance, v01 JSON retains numerical results).
At the user's request, v02 omits the green native PSF-correction curve and failure
annotations. The original three observed-shape curves, samples, and values are
unchanged. Corrected results and failures remain in the report and archived v01.

The four NISP PSFs match manifest hashes but use a different format from the old
`EuclidPSF` loader. Their pixel-response/stacking conventions need confirmation;
no local corresponding JWST PSF or learned-output effective PSF is established.
Pilot uses explicitly analytic PSFs. Native REGAUSS fails for every narrow-PSF
case (0.45 arcsec on 0.29672-arcsec pixels); independently reproduced as a PSF-fit
undersampling failure, not repaired by changing initial guesses. Preserve this
primary failure. The secondary review plot uses the common observed-shape sample
(622/624 objects), and includes correction only where it succeeds on that exact
sample. Broader-PSF correction is more faithful in trace response than uncorrected
ResNet; ResNet usually improves over native observed response but has size/noise/
PSF dependence and some worse low-S/N cases. This is not a shear calibration or
precision claim. Figures 7–8 and manuscript unchanged by this pilot. Next research
steps: validated native sampling treatment, documented empirical/mosaic PSFs,
realistic independent calibration/validation simulations with selection.

## Previous update: Figure 8 updated to current models — 2026-10-17

User approved a compact Figure 8 focused on input S/N and measurement success;
notebook 9 stays supporting analysis, not a standalone paper figure.
`paper_figures/08_shape_dependence.ipynb` is executed v03, with fixed/adaptive
shape gains and common-success coverage in one row. Export: `fig08_shape_dependence_v03`.
At user request, estimator names are on the y-axes; panel titles are removed.
The v02 notebook/export is preserved, with numerical inputs unchanged.
Inputs: `shape_snr_v02.json/.npz`; current `cached_shapes_v02` shape arrays unchanged.
Same model checkpoints as Figures 6/7, no new training/inference. Original S/N bins
and all 2,118 assignments reproduced; fixed common counts 179/441/387/322/117,
adaptive 297/419/363/304/115. Gains recombine to Figure 7 aggregate errors.
Three S/N tests and independent object-level bootstrap checks of all 40 displayed
method/bin combinations passed. Source/contract/verification: `runs/shape-snr-v02/`.
Report/caption: `paper/figure8_shape_snr.md`. Old notebook is archived. This is a
local review draft; the old residual-plane Figure 8 in the manuscript is not yet
replaced. No Overleaf upload. Prior appendix-removal edits remain intact.

## Previous update: appendix removed from paper — 2026-10-17

User requested removing the appendix and retaining essential information in the
main text. `paper/survey_enhancements.tex` now has no appendix or dangling appendix
references. Abstract: one seed-reproducibility sentence. Methods: concise matched
continuation protocol, including its different selection rule. Results: seed RMSE
ranges/0.09% relative sample SD and continuation trade-off supporting original ResNet.
Figures 6 v06 and 7 v05 remain installed. Detailed tables, seed plot and the previous
manuscript are archived in `paper/diagnostics/resnet_validation_archive/`, excluded
from the publication bundle. Manifest: `paper/diagnostics/appendix_removal_v01.json`.
Reference/environment checks and a two-column preview compile passed; full A&A
build still lacks `etex.sty`. Local upload ZIP refreshed; no Overleaf publish.
The unrelated historical shape/Discussion paragraphs still need their own revision.

## Previous update: Figure 7 directly shows improvement — 2026-10-17

User still found contour differences unclear. The active executed
`paper_figures/07_model_shapes.ipynb` now exports `fig07_shape_improvement_v04`:
two panels (fixed/adaptive moments), three model rows, percentage RMSE reduction
relative to native NISP, zero baseline and paired spatial-block 95% intervals.
ResNet: +19.3% fixed, +10.6% adaptive; single diffusion: -13.8%, -22.4%.
Same frozen `cached_shapes_v02` data and matched populations, no new measurements.
Companion export: `fig07_model_shapes_v04`. The v03 notebook is preserved as
`revisions/07_model_shapes_v03_before_gain_view.ipynb`; all older exports remain.
Current caption and checks: `paper/figure7_native_shapes.md`. Local draft only.

## Previous update: Figure 7 residual view — 2026-10-17

User found the native-contour/hexbin overlap difficult to read. The active
`paper_figures/07_model_shapes.ipynb` now exports `fig07_model_shape_residuals_v03`:
2x3 panels of delta-e relative to JWST, with blue model 68%/95% probability regions,
orange native NISP 68%/95% contours, and a zero-error horizontal line. Same frozen
`cached_shapes_v02` measurements and 1,446 common successes; no changed results.
The v02 notebook and agreement exports are archived/preserved. Companion summary
is now exported as `fig07_model_shapes_v03`. Local figure revision only.

## Previous update: consolidated Figure 7 with native NISP contours — 2026-10-17

Active notebook: `paper_figures/07_model_shapes.ipynb` (executed). Former 7 and
7b are archived in `paper_figures/revisions/*_v01_before_native_contours.ipynb`.
Main export `fig07_model_shape_agreement_v02`; companion `fig07_model_shapes_v02`.
CPU-only native measurements extend frozen v01 arrays in `cached_shapes_v02`;
contract/source snapshots in `runs/cached-shapes-native-v02`. No inference.
Same 1,446 fixed-moment successes in all panels, with native 68%/95% KDE contours.
ResNet RMSE reduction versus native: 19.3% fixed, 10.6% adaptive (1,498 objects).
Single diffusion is worse than native: 13.8% fixed and 22.4% adaptive.
Full-denominator success counts, paired block intervals, both-estimator common
sample (1,297), caption and checks: `paper/figure7_native_shapes.md`.
Local draft only; manuscript shape Results/Discussion still await revision.

## Previous update: all runs complete; model choice added to paper — 2026-10-17

Both corrected refinement runs (10 epochs each), both independent baseline
seeds (50 epochs each), and both automatic comparisons finished with exit 0.
The user approved retaining the original epoch-49 ResNet as the main model,
keeping Figure 6's three panels and placing seed reproducibility in the appendix.
Source RMSE across the three selected seeds is 0.0600667–0.0601735 MJy/sr.
No further training experiment is queued.

The local manuscript now includes the approved Figure 6 PDF, current regional
comparison and model-choice prose, a corrected validation-only abstract,
three-seed methods/caption notes, and `paper/resnet_validation_appendix.tex`
with the seed plot and selected/final checkpoint tables. Frozen numerical
provenance: `paper/diagnostics/resnet_model_selection_v01.json`.
The original model and regional analysis choices are retained for the next
held-out evaluation, which has **not yet been run**. Remaining historical
shape Results/Discussion sections still need their own revision; this update
does not validate their old shear claims. The full journal-class build is
blocked by missing `etex.sty`; the changed sections compile in an isolated
two-column preview and were visually checked. No Overleaf push performed.

## Previous update: refinements complete; two new seeds launched — 2026-10-17

Both corrected refinement runs completed 10 epochs / 22,270 updates with exit
0, matching batch/augmentation streams and the original parent unchanged.
The automatic local comparison also exited 0. Balanced-best source RMSE is
2.19% lower, outside RMSE 2.40% higher; neither fixed nor adaptive shapes show
a clear improvement. This is a trade-off, not an overall replacement decision.
Full findings and artifacts: [refinement results](paper/resnet_refinement_results.md).

The user asked for more tests and said “continue.” Two independent baseline
replications now run with the **unchanged original 50-epoch training recipe**:

- GPU 0: `runs/residual-full50-s20261018`, tmux
  `supernisp-residual-full50-s20261018`, [W&B l0kbzuub](https://wandb.ai/AI-Astro/superNISP-corrected/runs/l0kbzuub).
- GPU 1: `runs/residual-full50-s20261019`, tmux
  `supernisp-residual-full50-s20261019`, [W&B 8vh9u6nb](https://wandb.ai/AI-Astro/superNISP-corrected/runs/8vh9u6nb).

Both first epochs passed. Each is planned for 111,350 updates; source/data
hashes match the original. Contract: `runs/resnet-seed-replication-v01/contract.json`.
The follow-up evaluator `code/evaluate_residual_seed_replicates.py --wait`
compares best/last predictions across all three seeds; output will be
`runs/resnet-seed-replication-v01/evaluation/`. No test-set access or new
architecture/loss sweep. Inspect live progress/status rather than assuming
these timestamped notes establish completion.
Evaluator queue: tmux `supernisp-seed-replication-eval-v01`, console
`runs/resnet-seed-replication-eval-v01-console/`. Post-launch configuration,
source/data hash and exact train/validation selection checks passed for both
seeds; both reached epoch 3/50.

## Earlier update: full refinements launched — 2026-10-17

The user authorized full shared execution after the successful probe. Both
10-epoch jobs are now training in persistent tmux sessions, with online W&B:

| Branch | GPU | Run directory / tmux suffix | W&B |
|---|---|---|---|
| Original masked-L1 control | 0 | `resnet-refine-control-v03` / `supernisp-refine-control-v03` | [sap2hiib](https://wandb.ai/AI-Astro/superNISP-corrected/runs/sap2hiib) |
| Equal source/outside L1 | 1 | `resnet-refine-balanced-v03` / `supernisp-refine-balanced-v03` | [609wkepy](https://wandb.ai/AI-Astro/superNISP-corrected/runs/609wkepy) |

Both reproduced the frozen parent validation predictions and metrics exactly.
Both passed finite-gradient and exact checkpoint/RNG/optimizer restoration
checks. Repeated CUDA updates agree within float32 roundoff (maximum parameter
differences 3.73e-9 / 3.73e-9); historical bilinear CUDA backward is not bitwise
deterministic. Initial `v01` launches stopped at that overly strict preflight
assertion before any experiment epochs. The `v02` runs were then stopped after
one epoch because preflight had advanced aliased Adam step counters by three.
Those runs are explicitly marked invalid. Deep-copying the optimizer state and
a new regression test fix this in `v03`; the disk parent remains untouched.
All failed/stopped logs and source snapshots are retained.

Each process has a 1,024 MiB PyTorch allocator cap; initial peak reserved memory
814 MiB, sampled remaining device memory about 972 MiB. The unrelated Ray
workers remain active. Sharing has no guaranteed compute percentage.

Trainer: `code/train_residual_refinement.py`; five CPU tests in
`code/test_residual_refinement.py` passed. Each run saves `progress.json`,
`live.json`, `status.json`, last/best full-state checkpoints, measurements,
predictions and fixed galaxy galleries. Console directories have `-console`
appended and contain the exact `launch.sh`, log and eventual exit status.
Inspect live status rather than treating this timestamped note as completion.
For corrected `v03`, a separate read-back of both saved epoch-zero checkpoints
confirmed every model and Adam state tensor exactly matches the on-disk parent,
including all Adam step counters at 109,123.

The local-only automatic comparison entrypoint is
`code/compare_residual_refinements.py --wait`; it is waiting in tmux
`supernisp-refine-comparison-v02`, with console
`runs/resnet-refinement-comparison-v02-console/`. Planned output is
`runs/resnet-refinement-comparison-v01/report.md` and four PNG/PDF plots.
It checks matched training streams and compares original/best/last predictions,
paired spatial-bootstrap errors and spectra, fixed/adaptive observed shapes,
and the pre-existing Figure 5 examples. See the detailed experiment note below.
Its analytic paired-RMSE check, adaptive-shape smoke check and all four plot
exports passed on cached baseline data in a temporary directory.
The approved Figure 6 and original model remain unchanged.

## Previous update: sharing trial passed — 2026-10-17

User authorized a bounded trial on an occupied GPU. `code/probe_shared_gpu.py`
completed 100 batch-eight ResNet training updates on GPU 0 in tmux with online
W&B. Median steady-state step 0.05955 s; peak sampled process memory 1,308 MiB;
PyTorch allocator cap 1,024 MiB; peak allocated/reserved 593.76/804 MiB;
minimum sampled device-free memory 982 MiB. No OOM/nonfinite results.
After exit, memory returned to the original 43,165 MiB used / 2,296 MiB free.
The original Ray workers remain running; their throughput impact was not measured.

Trial updates were discarded and the parent checkpoint hash is unchanged.
Artifacts: `runs/shared-gpu-probe-v01/`, console `runs/shared-gpu-probe-v01-console/`,
exit 0, tmux `supernisp-shared-gpu-probe-v01`, W&B `24xwfbn4`.
[Measured trial report](paper/shared_gpu_probe.md) gives timing definitions and
limits. One shared run is feasible under the observed load; no percentage is
reserved. At this stage full refinements had not been launched and their trainer
was not implemented. This superseded the earlier assumption that occupied
GPUs categorically require moving machines. The paired experiment plan below remains.

## Earlier next-machine plan — 2026-10-17 (superseded by shared launch above)

The user approved a paired ResNet refinement experiment, then asked to move
machines after both GPUs were found occupied by unrelated Ray workers.
**No full refinement training has started or been queued.** The concrete plan and
transfer list are in [ResNet refinement handoff](paper/resnet_refinement_handoff.md).
Implement a separate trainer for two matched 10-epoch forks of the epoch-49
ResNet: original masked-L1 control versus 50/50 source/outside L1, with inherited
weights/Adam state and a fresh cosine LR schedule (1e-5 to 1e-6). Same parent RNG,
batches, augmentation, budget and source-MAE checkpoint selection for both.
Preserve epoch 0 as a candidate and compare errors, spectral agreement and shapes.
The new refinement trainer/launcher is **not yet implemented**.

CPU-only `code/prepare_refinement_masks.py` completed full-field train/validation
masks at `runs/residual-refinement-masks-v01/`. All 17,816/2,118 objects have both
regions; median source fractions are 13.34%/13.27%. Validation masks exactly
reproduce Figure 6 after cropping/intersection. Completion flags and hash were
verified. Transfer the completed bundle; no regeneration is needed.
Bring the parent checkpoint/data and gitignored caches to the new machine as
listed in the detailed note. Existing tmux + online-W&B training requirements
still apply; current Figure 6 outputs remain local-only.

Current Figure 6 is the single `06_model_comparison.ipynb`, export **v06**:
three panels in one row, source/outside RMSE together (filled circle/open square),
power and correlation (solid/dashed), with no panel titles or footer notes.
The separate PSNR/SSIM exploration remains in the same notebook. Earlier
Figure 6 status sections below are historical.

## All-model observed-shape comparison complete (local v01)

User asked which recent Figure 6 diagnostics belong in the paper and requested
shapes for diffusion, ResNet and ResNet-D5. CPU-only `code/analyze_cached_shapes.py`
used v03 cached predictions, including bilinear/bicubic and diffusion mean3.
Run `runs/cached-shapes-v01`; frozen summary/arrays `cached_shapes_v01.json/.npz`.
Archived notebook `paper_figures/revisions/07b_model_shapes_v01_before_native_contours.ipynb`; exports `fig07_model_shapes_v01`
and `fig07_model_shape_agreement_v01`. No inference, upload or manuscript edits.

Fixed moments: same 1,447 successes, single diffusion/mean3/ResNet/D5 RMSE
.2292/.1876/.1624/.1626. Adaptive moments: same 1,546 successes, RMSE
.3031/.2538/.2210/.2214. ResNets indistinguishable by paired block intervals.
Same 1,313 objects across both estimators preserve the ranking. Full-success
fractions use all 2,118 fields. Common validity, no source-segmentation clipping;
observed cross-band shapes, no PSF/shear calibration. Tests and independent
all-object NumPy moments, direct HSM samples, bootstrap/hash checks passed.

Recommended main results: source pixel errors/speed, source-only Fourier power
and correlation together, and matched shape errors with measurement success.
Background/window controls in appendix; if space tight, prioritize errors/speed
and shapes. Report `paper/model_shape_comparison.md` contains detailed values,
limitations and a proposed results layout; numbering is not yet installed.

## Figure 6 strict source-only Fourier repeat complete (local v01)

User requested the Fourier comparison within sources only. CPU analysis
`code/analyze_source_spatial_spectra.py` uses frozen v03 predictions and 2σ masks,
with exactly zero outside source/common validity. Primary inward taper width 3
pixels, checks 1/5; all include neighbours and use identical model/reference
windows. Same 1,363 complete fields; all 2,118 retained in a separate check.
Run `runs/spectra-source-v01`; bundle `spatial_spectra_source_v01.json`/`.npz`.
Notebook `paper_figures/06c_source_spatial_structure.ipynb`; exports
`fig06_source_spatial_structure_v01` and `fig06_source_window_checks_v01`.

Source-only intermediate-scale (0.36–0.89 arcsec) single diffusion / ResNet-D5
power ratios .912/.419, correlations .604/.738. Higher correlation within
sources does not reverse model ordering. Fine bins depend on mask-edge taper;
do not claim noise-free morphology or intrinsic resolution from these curves.
Six analytic tests plus independent FFT, original-control equality, block
interval and provenance checks. Full report `paper/figure6_source_spatial_structure.md`.
No inference, external upload or manuscript change. Keep all Figure 6 local.

## Figure 6 Fourier companion complete (local v01)

User approved power-spectrum and Fourier cross-correlation tests of diffusion
sharpness. CPU-only `code/analyze_spatial_spectra.py` used the frozen v03 cache;
no model inference, W&B or Overleaf upload. Run `runs/spectra-v01`, tmux
`supernisp-spectra-v01`, console `runs/spectra-v01-console`, exit 0.
Notebook `paper_figures/06b_spatial_structure.ipynb`; main source-error Figure 6
is unchanged. Summary `paper_figures/inputs/spatial_spectra_v01.json`; NPZ arrays
stay local/ignored. Main, window-check and background-control PDF/SVG/PNG exports
are in `paper_figures/exports/fig06_spatial*_v01` / `fig06_spectral*_v01`.

Primary population 1,363 complete common-valid cutouts; all 2,118 included in a
separate tapered-validity check. Primary Tukey alpha .5, alternatives .25/Hann
and shared soft source mask. Each image is weighted-mean-subtracted for this
structure diagnostic. No noise subtraction, flux fitting or PSF deconvolution.
Report geometric-mean power ratios and mean signed cross-correlation, with
2,000 paired block draws. Outside-source proxy has 1,356 usable primary fields.

Finding: diffusion is closer in observed power, while residuals have higher
reference correlation. At 6–15 target pixels (~.36–.89 arcsec), the reference-only
contrast>3 subset (944 objects) gives diffusion P ratio .884 / r .643 versus
ResNet-D5 .443 / .783. Paired absolute log-power-error difference is -.166 dex
(CI -.189,-.141), but correlation difference -.141 (CI -.150,-.132).
The qualitative trade-off survives alternative windows and reference-contrast
subsets. Diffusion also has uncorrelated background power (same-band P ratio
.792, r .005); do not claim recovery of real features from power alone.

Four analytical tests passed. Independent FFT checks reproduced 15,984 raw
power/cross-power values; array identities, selection IDs, primary bootstrap
intervals and provenance verified. Plots visually reviewed. Verification and
source snapshots are saved under `runs/spectra-v01`. Full interpretation and
proposed paper wording: `paper/figure6_spatial_structure.md`.

## Figure 6 source-mask experiment complete (v03)

User approved trying reference-defined source MAE/RMSE with separate background
errors. Keep all Figure 6 artifacts local. Frozen masks were built with
`code/prepare_source_masks.py`, protocol in `code/source_masks.py`, before model
errors were evaluated. Primary 2-sigma masks cover a median 18.85% of common-valid
pixels; sensitivity masks at 1.5/3 sigma cover 31.89%/10.92%. All 2,118 objects
have nonempty source and outside-source regions at every threshold.

Completed tmux `supernisp-figure6-v03-b8`, exit 0; console
`runs/figure6-v03-b8-console`; output `runs/figure6-v03-b8`.
Masks: `runs/figure6-source-masks-v03`, including random and Figure 5 visual
reviews. Original batch-size-two attempt was stopped (exit 143) for throughput;
its partial cache in `runs/figure6-v03` is incomplete and must not be used.
The completed run uses batch eight, immutable Figure 5 v06 checkpoints and
original seeds. It reuses v01 timings and saves central predictions in
`runs/figure6-v03-b8/predictions.h5` for future mask-only revisions.

Active notebook is v03 / benchmark v03; previous v02 archived. Four CPU metric,
mask and bootstrap checks pass. All four rendered figures were visually reviewed.
Full verification is saved in `runs/figure6-v03-b8/verification.json`: all
source/background errors reproduce from cache; paired bootstrap summaries are
exact; all-pixel scores match v01 within 8.89e-16 MJy/sr. Source, checkpoint,
cache and export hashes pass. Verification script/environment retained.
Main plot: source MAE/RMSE + original speed. Companions: source distributions,
all/source/outside-source paired gains, and masks for the five Figure 5 examples.
Report: `paper/figure6_source_mask_experiment.md`. Bundle:
`paper_figures/inputs/model_benchmark_v03.json`.

Source MAE/RMSE improvements vs bilinear: diffusion single 9.01%/26.61%,
mean3 21.18%/37.10%, ResNet 29.74%/45.63%, ResNet-D5 29.95%/46.56%.
Diffusion single has 3.93% worse outside-source MAE and 6.13% worse RMSE.
Learned-model source ordering is stable across all three thresholds. Bicubic's
source MAE comparison changes sign with threshold; do not call it robustly better.
Masks can include positive noise and miss faint light; outside-source is not
necessarily pure sky. Predictions are evaluated unsmoothed with no new offsets.
No Figure 6 upload or manuscript installation was performed.

## Model display name: ResNet-D5

User approved **ResNet-D5** for the direct 5× residual network, with **ResNet**
for the two-stage baseline, across Figures 5/6 and all text. Active architecture
Figure 4 is v05, examples Figure 5 v07, and comparison Figure 6 v02. Main and
companion plots were rerendered from unchanged saved inputs. Figure 5 retains
`BUNDLE_VERSION = 'v06'`; Figure 6 retains `BENCHMARK_VERSION = 'v01'`.
Previous notebooks are archived in `paper_figures/revisions/`. Internal keys,
checkpoint paths and immutable run snapshots retain their original names.
Manuscript methods, training table and captions use the new name; the installed
PDFs are `residual_architecture_v05.pdf` and `galaxy_examples_v07.pdf`.
Figure 6 remains local and is not included in the manuscript upload.
Naming updates published to Overleaf on 2026-10-16, commit `7d851d0`;
all 38 publication files match. Verified exported PDF labels and provenance
hashes; Figure 5/6 input hashes and scientific settings match their prior versions.

## Figure 6 local accuracy/speed benchmark complete

User requested Figure 6 and selected **accuracy and speed together**. After
automatic review required a new W&B upload approval, the user explicitly chose
**Keep Figure 6 local**. This overrides the older online-W&B requirement for
this figure. Do not upload Figure 6 measurements or plots. Exporter `--local`
skips all W&B initialization/API/artifact calls; console/progress/config files
provide local monitoring. Launch also sets `WANDB_MODE=disabled`.

Completed tmux `supernisp-figure6-v01`, exit 0, console `runs/figure6-v01-console`, output
`runs/figure6-v01`, GPU1. `code/export_model_benchmark.py` reuses Figure5 v06
immutable checkpoints (49/49/35). Full 2,118-object validation split, identical
support for all methods (JWST and both interpolation masks), central 105×105
pixels matching Figure5, plus full-field sensitivity check. Bilinear, bicubic,
single diffusion sample, linear mean of three, ResNet and ResNet-D5.
Accuracy: per-object signed physical MAE/RMSE, object-weighted means and paired
spatial-block bootstrap (2,000 draws). Timing: float32 without TF32, batch8,
device-resident inputs, 2 warmups/10 synchronized wall-time repeats; includes
representation conversion and diffusion noise generation, excludes IO/transfers,
metrics and support masks. Timing uses median/min/max; not single-request latency.

Two CPU regression checks passed for signed/masked errors and object-weighted
paired block uncertainty. Notebook `06_model_comparison.ipynb` VERSION v01 is
executed; main 3-panel mean-error/speed comparison and companion error
CDF/paired-gain sheet exported in PDF/SVG/PNG, visually reviewed. Prior
inventory draft archived. Bundle `paper_figures/inputs/model_benchmark_v01.json`.
All source/checkpoint/data/bundle/notebook/export hashes verified; all 2,118 IDs
and common supports checked. Cached Figure5 comparisons: 120 errors agree to
maximum absolute difference 2.11e-8 MJy/sr. Verification saved in
`runs/figure6-v01/verification.json`; source/plot snapshots retained there.

Central-crop reductions vs bilinear in mean MAE / mean per-object RMSE:
diffusion 4.51% / 26.90%; mean3 17.39% / 37.25%; ResNet 26.05% / 46.11%;
ResNet-D5 26.16% / 46.95%. Same learned-model ordering on full fields;
bicubic's small RMSE gain changes sign under the full-field definition.
Median ms/object at batch8: bilinear .00315, bicubic .00578, diffusion 125.31,
mean3 377.99, ResNet .4717, ResNet-D5 .6382. Intervals use 18 spatial blocks;
one training seed per model, validation-selected models, cross-band noisy
reference; not a held-out test, morphology ranking or shear calibration.
See `paper/figure6_reconstruction_benchmark.md` for the review summary.
No W&B initialization/upload, training/test access, manuscript insertion or push.

## Figure 5 v06 published to Overleaf

User requested the push after local insertion. `python scripts/overleaf.py
publish` fetched without remote conflicts and published commit `13ff60c` to
Overleaf project `6ad2883f967dc592692b8809`. Changes: Figure 5 PDF and the
identical `main.tex`/`survey_enhancements.tex` sources. All 36 publication files
match the Overleaf checkout; local HEAD and origin/main agree, checkout clean.
Full Overleaf compilation was not inspected. Main research repo not committed.

## Figure 5 v06 installed in manuscript

User approved v06 and requested adding it to the paper. Exact approved PDF
copied to `paper/figs/galaxy_examples_v06.pdf`, included as the fifth figure
under Results with label `fig:galaxy_examples`. Caption describes all five rows,
six columns, observed cutouts without injections, shared intensity scales,
field size, scale bars, checkpoint epochs and illustrative selection.
The old separate diffusion/ResNet image galleries are replaced by the combined
grid; their citations and directly associated visual claims are updated.
Remaining historical timestep/quantitative results still require revision;
this insertion does not validate them.

Verified exact PDF checksum, fifth-figure position, graphics paths and updated
labels. A standalone article proof of the actual figure/caption compiles on
one page. Full A&A manuscript compilation still stops at missing `etex.sty`.
Regenerated `.overleaf/superNISP-paper.zip` and verified embedded PDF and both
main sources. No Overleaf push in this step.

## Figure 5 v06 — simple isolated source first

User liked v05 and requested one ordinary example instead of its row e, placed
at the top. Restored isolated modest inclined galaxy 504234 (v04 row c), then
retained v05 rows a-d in order. Current IDs: 504234, 749105, 677525, 592591,
744769. The large face-on disk is now c and the crowded field is now e.
All arrays and display scales exactly match their respective source rows;
checkpoint hashes unchanged (49/49/35). This is a rearrangement of previously
viewed examples, documented as such in the notebook/caption/selection.

Active notebook VERSION v06; v05 archived. Reuse
`paper_figures/inputs/galaxy_examples_selection_v06.json`. Main and residual
PDF/SVG/PNG exports complete. Main grid visually checked; input, checkpoint,
source, notebook and export hashes verified. Executed notebook and exporter
snapshots saved in `runs/figure5-v06`. Completed tmux `supernisp-figure5-v06`,
exit 0, with the user's existing explicit W&B upload authorization:
https://wandb.ai/AI-Astro/superNISP-corrected/runs/1kwf3s0z .
No manuscript insertion or Overleaf push.

## Figure 5 v05 — pairs and crowded fields, b retained

User requests stronger examples including multiple galaxies; keep b. Selected
IDs 749105, 677525, 592591, 744769, 997288 from observed NISP/JWST images before
new predictions: close pair, retained face-on disk, overlapping elongated
sources, crowded field, and another close pair. All validation; no claims of
physical association or statistical representativeness. Observation-only
search script `paper_figures/inspect_multiple_sources.py` scanned 2,118 validation
objects; 369 qualifying fields, 313 distinct, top 64 visually reviewed.
Candidate sheets/record: `paper_figures/inputs/figure5_multiple_sources_v05/`.
Selection/scales/seeds and search hashes frozen in
`paper_figures/inputs/galaxy_examples_selection_v05.json`.
Observed-only preview: `paper_figures/exports/fig05_observed_selection_v05.png`
and `.pdf`. Row b keeps its object and display limits. The current diffusion
best.pt hash still matches v04's epoch-35 snapshot; training has now finished.

The user explicitly approved uploading the revised Figure 5 images, prediction
arrays and selection/checkpoint metadata to AI-Astro/superNISP-corrected after
automatic approval review requested specific authorization. Export completed
in tmux `supernisp-figure5-v05`, CUDA 1, output `runs/figure5-v05`, console
`runs/figure5-v05-console`, exit 0. W&B synced the figures and prediction artifact:
https://wandb.ai/AI-Astro/superNISP-corrected/runs/50peiezt .

Active notebook VERSION v05, five rows and same six columns. All b arrays and
display scales exactly match v04; checkpoint hashes unchanged (ResNet49,
direct5 49, diffusion35). Source, selection, bundle, notebook and export hashes
verified; diffusion sample/mean consistency checked; main and residual figures
visually inspected. Exporter and executed notebook snapshots are in the run
folder. V04 notebook preserved in `revisions/05_galaxy_examples_v04.ipynb`.
Reuse selection v05 for future updates. No manuscript insertion or Overleaf push.

## Figure 5 v04 — retain a/b/d, replace c and add e

Current five-row selection: 722361, 677525, 504234, 679385, 1126532.
New c/e chosen from observed NISP/JWST pairs in the existing seeded 40-object
pool before inspecting predictions. C is a modest inclined galaxy; e is an
asymmetric compact galaxy. Illustrative selection, not population statistics.
Use `paper_figures/inputs/galaxy_examples_selection_v04.json` for future updates.
Notebook VERSION v04, N_EXAMPLES=5; v03 notebook archived. Six columns unchanged.
All a/b/d arrays and display scales exactly retained; checkpoint hashes unchanged
(best epochs ResNet49/direct5 49/diffusion35). Main grid visually reviewed;
bundle, source, inputs and export hashes checked. Main/residual v04 exports ready.
Completed tmux `supernisp-figure5-v04`, runs/figure5-v04, exit 0; W&B
https://wandb.ai/AI-Astro/superNISP-corrected/runs/xqwixuq9 . Exporter and executed
notebook frozen in the run folder. No manuscript insertion or Overleaf push.

## Figure 5 v03 — retain b/d, replace a/c

User liked b/d and requested more representative a/c. Current v03 IDs are
722361, 677525, 1033198, 679385. New a/c selected from observed images only in
a seeded 40-object validation pool (22.5<F814W<24, 4.3<FLUX_RADIUS<6); their
magnitudes 22.88/23.11 and catalog radii 5.03/4.84 are nearer sample middle.
Still illustrative, not statistically representative. Selection/scales saved
in `inputs/galaxy_examples_selection_v03.json`; use it for future exports.
Six-column order unchanged. All b/d arrays and display limits are exactly equal
to v02; checkpoint hashes remain 49/49/35. Export/bundle/notebook hashes verified.
Completed tmux `supernisp-figure5-v03`, `runs/figure5-v03/`, exit 0, W&B
https://wandb.ai/AI-Astro/superNISP-corrected/runs/p9068svz . Main and companion
residual exports v03 are ready; main grid visually checked. V02 notebook archived.
No manuscript insertion/push, training changes or test-data access.

## Figure 5 v02 — requested ordering and observed morphology selection

Current main grid: NISP Y, JWST F115W, Bilinear, Diffusion, ResNet, ResNet-D5.
User requested clearer paper examples. Inspected NISP/JWST-only candidate sheets
from the 60 brightest validation galaxies with F814W<23 and 5<FLUX_RADIUS<25;
selected 734405, 677525, 671723, 679385 before new predictions, illustrating
barred spiral, more face-on disk, edge-on disk, compact smooth morphology.
Explicitly illustrative bright examples, not a random population/performance
selection. Freeze in `paper_figures/inputs/galaxy_examples_selection_v02.json`.
Shared display limits use observed JWST only; no per-model stretching.

Exporter supports `--selection`, stores bilinear values in physical NISP units
and a strict interpolation support mask. Main notebook VERSION v02 has six
columns; residual sheet includes bilinear and the three models in matching order.
V01 notebook and artifacts retained. `runs/figure5-v02/` completed, exit 0,
tmux `supernisp-figure5-v02`, W&B
https://wandb.ai/AI-Astro/superNISP-corrected/runs/wh0bth5m . Checkpoints 49/49/35.
Verified new selection hash, bilinear native-center alignment/units, masks,
finite predictions, linear sample mean, notebook/export hashes. Main grid
visually reviewed. No training changes or test data. Not added to paper/pushed.

## Figure 5 provisional model comparison

The user authorized Figure 5 using the best ResNet/direct5 and best-current
diffusion, with later updates. `code/export_paper_examples.py` snapshots stable
`best.pt` copies, verifies recorded code/data hashes, freezes validation IDs and
display limits before inference, requires tmux + online W&B, and exports a
versioned physical-intensity bundle. It then executes only the plotting notebook
and logs the main grid and residual sheet. Future versions reuse selection/scales
and diffusion noise seeds. No training code was changed and no test data used.

Completed `runs/figure5-v01/`, tmux `supernisp-figure5-v01`, exit 0;
W&B https://wandb.ai/AI-Astro/superNISP-corrected/runs/napgcdal . Best snapshots:
ResNet epoch 49, direct5 epoch 49, diffusion epoch 35 (training was ongoing).
Validation IDs 992795, 1024714, 463235, 1106391 were selected by central NISP
brightness ranks 20/45/70/95 percent in a seeded 256-object pool before model
outputs. Four matched rows use ~6.3-arcsec central crops, common fixed per-row
asinh limits and 2-arcsec bars. Separate linear residual review sheet. Three
seeded diffusion samples and a linear mean are stored; main grid uses one draw.

Updated `paper_figures/05_galaxy_examples.ipynb`; exports
`fig05_galaxy_examples_v01` and `fig05_galaxy_residuals_v01` (PDF/SVG/PNG+JSON).
Bundle `inputs/galaxy_examples_v01.npz` is ignored; metadata and frozen selection
JSON are retained. Source hashes, stable snapshots, bundle hash, sample mean,
notebook source/export hashes and successful execution checked; both figures
visually reviewed. Not inserted into Results or pushed to Overleaf yet.

## Overleaf full main.tex

At the user's request, `scripts/overleaf.py` now copies the complete local
`paper/survey_enhancements.tex` to Overleaf's `main.tex`, retaining an identical
copy under the original filename. Bundle and status comparisons use the same
mapping; every publish refreshes both copies. The local manuscript remains
`paper/survey_enhancements.tex`. Published successfully in commit `8bc9420`,
including the pending Methods revision and software references. Verified exact
main.tex/source byte equality and unique ZIP entries. No one-line wrapper remains.

## Figure notebook numbering corrected

All active `paper_figures/` notebooks now have unique prefixes 01–09.
Residual architecture remains Figure 4; galaxy examples, model comparison,
observed shapes, property dependence and robustness are Figures 5–9.
Renamed notebook headings, export stems and README/input-contract references;
refreshed notebook outputs and the three saved-diagnostic previews. Earlier
notebook versions are archived as `revisions/*_before_renumber.ipynb`, with
historical exports retained. No models ran. Current links and export provenance
checks passed. Manuscript LaTeX already numbers figures automatically and has
no duplicate Figure 4; historical Results figures remain for later revision.

## Methods text revision (published in 8bc9420)

The user requested a complete Methods review, including tables, while leaving
Introduction and Results alone. `paper/survey_enhancements.tex` now documents
corrected WCS extraction, masks/backgrounds/normalization, the two architectures,
a shared training/comparison table, checkpoint selection, fixed observed-shape
measurements, matched failure populations, block-bootstrap uncertainty and the
21-scenario residual robustness protocol with its own table. Added four missing
software references in `paper/mybib.bib`. Training source hashes match all three
recorded full runs. No models ran. All manuscript text outside Methods is
byte-for-byte unchanged from before this revision. Methods-only two-column
preview compiled with no overfull boxes; tables visually checked. Full A&A
build remains blocked by missing `etex.sty`. See
`paper/methods_revision_notes.md` for evidence and limitations. Published in `8bc9420`.

## Paper figure notebooks started

Current architecture exports: Figure 3 v05 and Figure 4 v04. The user requested
consistent color meanings across both diagrams. `paper_figures/architecture_style.py`
is the shared source: blue learned operations, teal NISP/conditioning, yellow
JWST/F115W images, purple diffusion quantities/updates, gray neutral/fixed
operations. Both upsampling variants are blue; all F115W estimates are yellow.
Both notebooks rendered and manuscript PDFs/provenance were verified; installed
`diffusion_architecture_v05.pdf` and `residual_architecture_v04.pdf`.
Published to Overleaf at user request in commit `f16fbf5`, including both
architecture captions, rewritten method subsections, and bibliography updates.
Full local TeX compilation remains blocked by the previously recorded missing
TeX dependencies; figure rendering and export provenance checks passed.

Latest architecture colors: Figure 3 v04 and Figure 4 v03 use stronger blue,
teal, purple and amber fills, matching borders and darker text/arrows. Layouts
and scientific content are unchanged. Both notebooks executed successfully;
PDF/SVG/PNG exports and source provenance were refreshed, and the PDFs are
installed in the manuscript as `diffusion_architecture_v04.pdf` and
`residual_architecture_v03.pdf`. Prior pale versions are archived. Not pushed.

Figure 4 now has a simplified ResNet schematic in
`paper_figures/04_residual_architecture.ipynb`, exported as
`fig04_residual_architecture_v02` (PDF/SVG/PNG). It shows the common eight-block
architecture, separate two-stage and direct-5x output paths, and a compact dense
block inset. Model/trainer/loader hashes match both recorded full runs; no model
ran. Installed `paper/figs/residual_architecture_v02.pdf` under `fig:resnet_model`.
Rewrote its caption and residual subsection in plain scientific language; corrected
the old RRDB, two-stage-training, SSIM-loss, hyperparameter and sweep descriptions.
Training table now reflects 50 epochs, batch 8, Adam/OneCycle max LR 1e-4, masked
L1 and full-validation checkpoint selection. Added ESRGAN and pixel-shuffle
citations. The existing diffusion-example figure was preserved and moved under
the diffusion-results heading without changing figure order. Source/export hashes,
notebook execution, citation keys, labels and training settings were checked.
Not pushed to Overleaf. Historical results remain for later revision.

The simplified Figure 3 v03 is approved for inclusion in the paper. Its caption
now uses plain scientific language. The diffusion subsection and training table
have been rewritten against the current implementation: velocity prediction is
explained, the asinh transform and validity maps are described, and generation
and validation selection are separated from training. The old VGG-loss,
500-epoch, sweep and convergence claims were removed from this subsection.
Added the DDIM reference and corrected the SR3 bibliography entry against arXiv.
Citation keys, table settings, notebook provenance and installed figure hashes
were checked. Full local TeX compilation still lacks `etex.sty`. These changes
have not been pushed to Overleaf; other historical methods/results prose remains.

Figure 3 v03 is now a simplified review draft in `paper_figures/03_diffusion_architecture.ipynb`.
It redraws training, the source-verified conditional U-Net and 50-step generation,
including all three skip levels and an inset for the FiLM residual block (14
blocks total). Source hashes match `diffusion-full50-s20261016`; no model ran.
PDF/SVG/PNG: `paper_figures/exports/fig03_diffusion_architecture_v03.*`.
The PDF is installed as `paper/figs/diffusion_architecture_v03.pdf`, replacing
the historical diffusion chart under the existing `fig:diff_model` label, with
a new caption. It has not been pushed to Overleaf. The previous notebook is
archived at `paper_figures/revisions/02_diffusion_architecture_v01.ipynb`.
The detailed v02 notebook is also archived in `revisions/03_diffusion_architecture_v02.ipynb`.
v03 removes equations and implementation annotations from the plot, retaining
channel widths, the conditioning inset and 50-step generation loop.
Historical methods/results prose remains to be revised separately.

Figure 2 parent comparison v05 is rendered in `paper_figures/02_sample_properties.ipynb`.
At the user's request, the parent now uses only valid ACS F814W magnitude < 25
from the full COSMOS2020 catalog: 233,784 objects. No classification, footprint,
redshift, size, quality, or buffer cut is applied. Invalid redshifts are omitted
only from the redshift histogram (225,567 valid measurements), not the parent.
Gray histograms sit behind the selected splits. Counts, tails, and catalog
checksum are in `inputs/parent_sample_v05.json`; all 22,810 selected IDs are
in the parent. PDF/SVG/PNG, the executed notebook, and the manuscript PDF/caption
are updated. v04's footprint/classification comparison and notebook/helper
snapshots are preserved. No images or model outputs are used.

The user subsequently approved adding the latest overview v07 to the paper.
It shows a 24.3-arcsec validation field centred on COSMOS2020 734405 in block
3:2, with 5-arcsec scale bars. This supersedes the earlier requested 49-arcsec
layout described below. The exact v07 PDF is copied to
`paper/figs/sample_overview_v07.pdf` and included first in
`paper/survey_enhancements.tex`, label `fig:sample_overview`, with a caption
and an updated adjoining paragraph describing the corrected sample and splits.
The rest of the historical manuscript still requires revision. Full local
compilation was attempted: latexmk lacks Perl Time::HiRes, and direct pdflatex
stops at missing etex.sty before processing manuscript content. The asset hash,
figure ordering, unique label and whitespace checks pass.

Sample properties is now Figure 2, with notebook `02_sample_properties.ipynb`
and export `fig02_sample_properties_v05`. It uses one legend in the left panel
and teal/orange/dashed-purple training/validation/test curves. The PDF is
included second in the manuscript as `paper/figs/sample_properties_v05.pdf`,
label `fig:sample_properties`; later manuscript figures shift automatically.
The v02 notebook is archived under `revisions/`.
The histograms include test magnitude/redshift distributions, explicitly
requested by the user, with split counts and the original bins.
Only catalog metadata is used; test images/predictions remain reserved. The
notebook documents that validation supports development and checkpoint/model
selection, while final test performance must wait for frozen model/measurement
choices. Previous train/validation-only v01 is archived under `revisions/`.

Figure 1 review update: the user wants paired NISP/JWST observations instead
of the accounting panel, a closer field with less whitespace, no
figure/panel titles, and the two original rows split into separate figures.
`01_sample.ipynb` now contains footprint + paired field in one row;
`02_sample_properties.ipynb` contains magnitude/redshift as Figure 2.
The overview has now been rendered in the
local mosaic workspace; other machines can set `MOSAIC_ROOT` or two explicit paths.
`paper_figures/sample_region.py` extracts native 82-pixel NISP and aligned
410-pixel JWST observations centred on galaxy 734405 in validation block 3:2, with masks,
physical units, shared footprint, and cached input/source hashes. No models run.
The v07 PDF/SVG/PNG and executed notebook now show the real 24.3-arcsec field
around a larger spiral, selected from observed images to illustrate resolution;
source checksums match the sample manifest. Copy the local, git-ignored
`inputs/sample_region_v06.npz` and its JSON sidecar to render on the GPU copy
without mosaics. Metadata lookup supports both the sibling package and local
`data/ready/` layout. Missing mosaics/cache still produce a clear waiting state
and skip overview export. v01/v02 combined layouts and notebook snapshots are
preserved, as are the previous v03 validation field and notebook/extractor
snapshots and the v04 training field. v05 moves the footprint legend to the
upper right and removes the field-size text below the images. The separate
properties figure is ready. v06 halves the image field width and uses three
adjacent axes to reduce whitespace, retaining the v05 shared stretch and
native NISP pixels. v07 removes the block outline and circled arrow, showing
only the true WCS footprint with two lines to corresponding corners of the
NISP view. It reuses the v06 image cache; v05/v06 exports and notebook snapshots
are preserved.

The user requested a figure-first paper workflow while experiments continue.
See [paper_figures/README.md](paper_figures/README.md): nine editable notebooks,
one per proposed figure, with captions/review notes and PDF/SVG/PNG exports plus
input/source hashes. The plot code lives in each notebook; do not regenerate
notebooks over later manual edits. Manuscript integration is recorded above.

Figures 1–3 show the sample metadata and the current diffusion/residual
architectures. Figures 4–5 wait for completed, matched prediction and benchmark
exports; their contracts are in `paper_figures/inputs/README.md`. Figures 6–8
are explicitly labeled previews from the frozen legacy epoch-49 validation
analyses, not final model comparisons or calibrated-shear results. Test images
remain unopened; test catalog positions/counts and magnitude/redshift appear
in the sample description figures.

These notebooks use metadata, observed images and cached diagnostics: no model imports,
checkpoint loading, inference, training, or W&B calls. Future model prediction
exports must still run in tmux with online W&B initialized first. Keep active
pipeline source hashes unchanged. Latest status inspected during this work:
direct-5x training completed epoch 50 / 111,350 updates and its pipeline entered
robustness; full diffusion remains in the full50 stage. Read live status files
before making scheduling decisions.

## Full corrected diffusion run launched on GPU 1

The user authorized full diffusion training on the other GPU. New persistent
session: `supernisp-diffusion-full50`; console `runs/diffusion-full50-console/`;
pipeline status/protocol/snapshots `runs/diffusion-full50-pipeline/`.
Read [full diffusion protocol](paper/full_diffusion_training.md).

Fresh weights, all 17,816 training objects, 50 epochs, batch 8 and 111,350
updates. The validated diffusion core is unchanged. Monitor 256 fixed
validation galaxies with 50-step NISP-only generation each epoch, select by
physical MAE, then evaluate all 2,118 validation galaxies with three seeds and
their linear-intensity mean image. Eight fixed galaxy/residual panels appear
in online W&B; the training link is saved under
`runs/diffusion-full50-s20261016/wandb_url.txt`. Test data remain untouched.

Full diffusion W&B: [gepl2iv2](https://wandb.ai/AI-Astro/superNISP-corrected/runs/gepl2iv2).
The batch-8 preflight passed two regression checks and finite GPU
forward/backward/sampling at 3.13 GiB peak tensor allocation. Saved training
rows exactly match the baseline. Initial galaxy panels have been rendered.

The diffusion wrapper automatically yields GPU 1 at batch boundaries during
the already queued direct-5x analyses, then resumes. Active direct-5x code and
commands were not modified. Do not duplicate either run or edit its queued
sources. Final diffusion comparison is queued after training, with common
object IDs, failures and spatial-block intervals against the frozen residual
and the completed direct-5x result when available. Equal update counts do not
match representation, loss, capacity, validation-selection population or compute.

## Corrected diffusion pilot completed; direct-5x continues

The user authorized repairing and piloting diffusion. Historical diffusion
files are preserved; new code is `diffusion_corrected.py`,
`diffusion_diagnostics.py`, `test_diffusion_corrected.py`,
`check_diffusion_corrected.py`, `train_diffusion_corrected.py` and
`run_diffusion_pilot.py` under `code/`. No active direct-5x source was changed.
Read [corrected diffusion protocol](paper/corrected_diffusion_pilot.md).

Five mathematical/integration regressions and actual-pair CUDA checks passed
in tmux with [W&B gokjvp87](https://wandb.ai/AI-Astro/superNISP-corrected/runs/gokjvp87).
The new path has valid jump-aware DDIM, exact zero-terminal signal and stable
v prediction, no clamp, saved train-only asinh scaling with inverse physical
metrics, NISP-mask conditioning, masked loss, and NISP-only model selection.
It requires fresh weights; it is not a sampler-only historical-checkpoint fix.

GPU 1 completed tmux `supernisp-diffusion-pilot`: 32-pair overfit/resume check
(40 epochs / 320 updates, require >=10% generated MAE reduction), then a fresh
512-train / 64-validation pilot (5 epochs / 640 updates), with 50-step sampling
and three final noise seeds at the best checkpoint. All models initialize only
after online W&B; fixed galaxy/residual and repeated-sample images are logged.
Inspect `runs/diffusion-pilot-pipeline/status.json` and
`runs/diffusion-pilot-console/console.log`; the latter directory receives the
overall exit status. All three diffusion check/pilot/assessment console exit
statuses are 0; sessions now contain idle shells. Do not relaunch completed jobs.
Full diffusion training was subsequently authorized and launched above.
Test data remain untouched.

[Pilot 6rqj99e8](https://wandb.ai/AI-Astro/superNISP-corrected/runs/6rqj99e8)
completed five epochs / 640 updates; best epoch 5. Generated validation MAE
fell 71.52% from random initialization, to 0.0203956 MJy/sr. The overfit/resume
gate passed with 73.38% reduction. Seven regression tests plus actual-pair CUDA
checks passed; the best checkpoint reproduced its fixed-seed selection metric.

[Assessment d28ha2le](https://wandb.ai/AI-Astro/superNISP-corrected/runs/d28ha2le)
uses the same 36 validation objects for three diffusion samples and baselines:
shape component RMSE 0.214–0.226 diffusion, 0.18772 native, 0.18615 bilinear,
0.15956 frozen full50 residual. Diffusion fails on 7/12/11 of all 64 objects
across the three seeds; common-population sample scatter is 0.15711. This pilot
learns but does not establish a shape gain; 640 updates cannot fairly rank it
against a 111,350-update residual model. Image panels expose variable/noisy
structure. Tracked report: `paper/diagnostics/diffusion_corrected_pilot.json`.
Next possible diffusion check: mean generated images in linear intensity before
shape measurement, with fixed IDs/failures; now queued at the end of the newly
authorized full run. No completed full-diffusion result exists yet.

## Direct-5x comparison launched

The user authorized the controlled head comparison. It is running in persistent
tmux session `supernisp-direct5`; inspect `runs/direct5-comparison/status.json`
and `runs/direct5-console/console.log` before launching anything on GPU 0.
The sequence is overfit learning check, short pilot, fresh full 50-epoch run,
best-checkpoint fixed-shape analysis/comparison, then the same 21 robustness
scenarios and paired architecture comparisons, including HSM. Every model
process requires online W&B before constructing a model. Full-run galaxy images
are logged at initialization and every epoch. Test data remain untouched.

Read [direct-5x experiment protocol](paper/direct5_head_comparison.md).
The exact protocol, commands and source snapshots are saved in
`runs/direct5-comparison/`. All queued source hashes are checked before each
stage: do not edit them during this pipeline. Full training is under
`runs/direct5-full50-s20261016/`; its W&B URL appears in `wandb_url.txt` once
the learning checks pass and training starts. Overall exit status appears in
`runs/direct5-console/exit_status`. Do not duplicate the active job.

Startup verified: both learning gates passed (overfit MAE reduced 22.10%;
pilot validation MAE reduced 24.65% from initialization). Full training is now
running: [W&B ilaim0sg](https://wandb.ai/AI-Astro/superNISP-corrected/runs/ilaim0sg).
Training/validation selections and fixed galaxy-gallery selection exactly match
the baseline. Source/data hashes match; only `head` and output directory differ
in the full-run configuration. Direct5 has 2,959,953 parameters versus legacy
2,332,177. These are startup checks, not evidence of architecture improvement.

## Epoch-49 robustness and convergence completed

The user authorized the next observed-shape robustness step and asked whether
more training was needed. The 50-epoch run plateaued under its completed
OneCycle schedule: validation MAE improved 0.572% from epoch 40 to best 49,
0.154% from 45 to 49, and only 0.00524% from 48 to 49. Epoch 50 was slightly
worse. Its final LR is ~4e-10. Keep epoch 49 as the frozen baseline; this is
practical convergence under that schedule, not proof that another schedule
cannot improve it. Any continuation needs a separate recorded schedule/budget.

[Robustness run lb3m3r9k](https://wandb.ai/AI-Astro/superNISP-corrected/runs/lb3m3r9k)
completed 21 predeclared scenarios on all 2,118 validation objects in tmux
session `supernisp-robustness`, exit status 0. Four regression checks passed;
the prior nominal per-object results were reproduced. New code:
`robust_shape_measurements.py`, `run_shape_robustness.py`,
`test_shape_robustness.py` under `code/`. Training code is unchanged.

Independent GalSim 2.7.1 adaptive moments find 12.20% lower SR component RMSE
than native (conditional block interval 9.02–15.10%) and 16.45% lower than
bilinear on 1,351 objects shared with the nominal comparison. The direction
of improvement survives weight/aperture/centering/background perturbations,
HSM crop/initial-guess variations and three input-noise trials, including on
the same 1,098 galaxies measurable in all scenarios. Background changes and
HSM initialization materially affect effect sizes and/or measurement success;
retain those limitations. These remain conditional validation results.

Results: `runs/shape-robustness-best49/`; console: `runs/robustness-console/`.
Read [robustness and convergence](paper/shape_robustness_and_convergence.md).
Tracked JSON: `paper/diagnostics/shape_robustness_best49.json` and
`paper/diagnostics/full50_convergence.json`. No test images were used.

Next: controlled direct-5x head comparison with the same trunk and 50-epoch
update budget, retaining fixed moments as reference and HSM as cross-check.
That comparison is now launched as described above. No continuation training
was started for the legacy baseline.

## Full run and queued analysis completed

Both `runs/full50-console/exit_status` and
`runs/final-analysis-console/exit_status` are 0. All 50 training epochs and
111,350 updates completed. Best validation MAE is 0.9645708 at **epoch 49**.
The final validation analysis and fixed-population comparison also completed:

- [Best-checkpoint analysis 56s99iwd](https://wandb.ai/AI-Astro/superNISP-corrected/runs/56s99iwd).
- [Epoch-11 versus best-49 comparison su0iypr0](https://wandb.ai/AI-Astro/superNISP-corrected/runs/su0iypr0).
- On the identical 1,484 common objects, SR component RMSE decreased a further
  6.80% from epoch 11 (conditional 95% spatial-block interval 5.04–8.59%).
- Among the fixed 1,527 reference/baseline eligible objects, SR failures fell
  from 43 to 4: 39 recovered measurements, none lost. This remains a validation
  diagnostic, not final test evaluation or calibrated shear.
- Tracked final reports: `paper/diagnostics/validation_best49_shapes.json` and
  `paper/diagnostics/full50_shape_comparison.json`.

The active/queued descriptions below are historical. The sessions now contain
idle shells; do not relaunch the completed baseline or queued analysis.

## Agreed paper scope — observed-shape recovery

The user explicitly agreed to **improved observed-shape recovery** as the
paper target. Establish whether NISP-to-F115W SR improves agreement with
observed JWST shapes, for which galaxies, with matched measurement populations,
failure rates and spatial uncertainty estimates. Do not present this as
calibrated gravitational shear or a demonstrated weak-lensing cosmology gain.
Keep the Y/F115W band difference and instrumental PSFs explicit.

Next priority after the current baseline: independent adaptive-moment
cross-check and sensitivity to weight/aperture, centering, background and
neighbors on validation data. Freeze a reproducible measurement/selection
contract before final test evaluation. The fixed-moment diagnostic remains
useful but its RMSE improvement alone does not certify unbiased ellipticity.
Known-shear calibration is future work for a separate weak-lensing claim;
it is not required to finish the agreed observed-shape-recovery study.

## Validation analysis and queued completion workflow

While full50 training continues, the user asked to work on the other steps.
Implemented standalone validation analysis without changing any active training
source (all four training source hashes were rechecked and still match).

- `code/shape_diagnostics.py`: the existing signed moment definition with
  explicit failure reasons; native/bilinear/bicubic baselines in matched angular
  regions; common-success comparisons, round-shape control, paired spatial-block
  bootstrap intervals and fixed property bins.
- `code/evaluate_paired_shapes.py`: immutable checkpoint copy, verified code/data
  hashes, full validation only, row/ID catalog joins, per-object CSV/NPZ, report,
  plot and W&B failure table. Runs on GPU 1 in tmux with online W&B.
- Four metric/geometry/mask/bootstrap regression checks passed. Two comparison
  regression checks passed (IDs/reordering/unchanged baselines). Real-data
  same-checkpoint comparison correctly returned zero change and zero lost IDs.
- [Epoch-11 analysis od7fm46d](https://wandb.ai/AI-Astro/superNISP-corrected/runs/od7fm46d)
  completed on all 2,118 validation objects. The immutable snapshot and detailed
  results are in `runs/validation-shapes-full50-snapshot/`.
- 1,527 objects pass JWST and every baseline; 43 fail SR, leaving 1,484 common
  successes. On those same objects, component ellipticity RMSE is native
  0.20698, bilinear 0.20074, bicubic 0.20741, SR 0.17759, round control 0.28872.
  These are intermediate validation diagnostics, not final test/shear results.
  All 43 SR failures on baseline/reference eligible objects have input S/N proxy
  1.33–3.75 and ACS F814W magnitude 24.15–24.99. Failure flags retain negative
  flux/indefinite moments; no clipping or new sample cut was applied.

**A final-analysis job is queued in `supernisp-final-analysis`.** It waits for
the existing training to exit successfully with 50 completed epochs, then:

1. Evaluates `best.pt` on validation using GPU 1, into
   `runs/validation-shapes-full50-best/` with a separate online W&B run.
2. Uses `code/compare_shape_runs.py` to compare it with epoch 11 on identical
   common object IDs, saving recovered/lost measurements and paired uncertainty
   estimates into `runs/full50-shape-comparison/` with another W&B run.

Follow `runs/final-analysis-console/console.log` or attach with
`tmux attach -t supernisp-final-analysis`. Check its `exit_status` when done.
Do not duplicate this pending job. The watcher aborts after a failed/incomplete
training exit or a 24-hour wait. It does not touch test data or train models.

Read [validation diagnostics](paper/validation_shape_diagnostics.md) for the
measurement contract, limitations, commands and preliminary results. The tracked
machine-readable snapshot is `paper/diagnostics/validation_epoch11_shapes.json`.
Next: inspect the queued best-checkpoint analysis and fixed-population comparison
when training completes, then choose one controlled architecture ablation.

## Active full-split training and image monitoring

The user requested a longer run and galaxy images in W&B after the pilot.
**A 50-epoch run is now launched; inspect its status before starting anything
else on CUDA device 0.**

- W&B: [residual-full50-s20261016 / nifpxes6](https://wandb.ai/AI-Astro/superNISP-corrected/runs/nifpxes6).
  Remote API verification confirmed state `running`, 1,120 optimizer updates,
  and all four epoch-0 image panels. This is a startup snapshot, not completion;
  current status is in W&B and the run directory.
- Persistent session: `tmux attach -t supernisp-full50`.
- Console: `runs/full50-console/console.log`; `exit_status` appears on exit.
- Artifacts: `runs/residual-full50-s20261016/`, including per-epoch
  `progress.json`, `last.pt`, `best.pt`, and final `status.json` on completion.
- Uses all 17,816 train / 2,118 validation examples, 50 epochs / 111,350
  optimizer updates, batch 8, seed 20261016, max LR 1e-4. It starts from scratch
  with a matching 50-epoch schedule and the unchanged legacy residual model.
  Best model selection remains validation MAE. Test data remain reserved.
- Eight fixed evaluation galaxies appear under W&B `galaxies/full_1`,
  `full_2`, `central_1`, and `central_2`, at initialization and every epoch.
  Panels show NISP native sampling, bilinear NISP, SR, JWST and signed residual.
  Per-object physical-intensity/asinh display scales stay fixed across epochs;
  central crops, object IDs, masks and moment-check failures are visible.
- `code/galaxy_visuals.py` is included in source snapshots and hashes.
  Rendering preserves Python/NumPy/Torch/CUDA RNG state, so it does not alter
  training/resume random streams. The trainer supports `--gallery-count`,
  `--image-every`, and `--mode train`.

[Visual checks sfgwa5ku](https://wandb.ai/AI-Astro/superNISP-corrected/runs/sfgwa5ku)
passed in tmux and synced eight image files. Locally inspected panels show
the pilot predictions and a separate set of failed-moment examples. All 20
pilot prediction failures were reproduced; 13 have passing JWST moments.
This does not yet diagnose their quantitative cause. The longer run retains
the existing metric and failure-count logging.

Next: inspect full-run learning curves and fixed galaxy panels, confirm best
checkpoint selection and shape failure rates, then plan architecture comparisons.
See [the training guide](paper/corrected_training.md) for image conventions and
resume instructions. Older completed pilot results are recorded below.

## GPU continuation — 2026-10-16

The extracted sibling package is present and all 46 `SHA256SUMS` entries pass.
No rebuild or transfer is needed. CUDA works outside the Codex execution
sandbox: PyTorch 2.6.0+cu124, NVIDIA driver 590.48.01, two L40S GPUs.
An in-sandbox `nvidia-smi` failure does not mean the host driver is broken.

**User requirement:** run models inside tmux/screen so disconnects do not
terminate work, and attach online W&B monitoring before running models.
The user approved project `superNISP-corrected` under the existing account;
it resolves to [AI-Astro/superNISP-corrected](https://wandb.ai/AI-Astro/superNISP-corrected).
Do not fall back silently to offline monitoring or foreground training.

Implemented `code/residual_baseline.py`, `code/train_paired.py`,
`code/launch_paired.py`, `code/check_paired_training.py`, and
`code/run_paired_pilot.py`. See [training contract and commands](paper/corrected_training.md).
The new baseline matches the original eight-block, 64-feature model exactly.
The trainer uses the corrected loader, masked loss, update-level OneCycleLR,
validation-only pilot selection, provenance, and resumable epoch checkpoints.
The direct-5x head is an explicit option but has not been trained or ranked.

Integration checks passed in persistent session `supernisp-check`, tracked at
[W&B lkg2d7mu](https://wandb.ai/AI-Astro/superNISP-corrected/runs/lkg2d7mu).
Checks cover both loader representations, worker loading/flux round trips,
legacy output equivalence, finite CUDA gradients for both heads, masked-pixel
gradients, paired augmentation, fixed moment diagnostics, and exact optimizer
update restoration from checkpoint including scheduler and RNG state.

The gated initial experiment was launched in tmux session `supernisp-pilot`:

```bash
tmux attach -t supernisp-pilot
tail -f runs/pilot-console/console.log
```

Sequence: 32 fully valid training pairs, 40 epochs / 160 updates; deliberately
stop after epoch 2 and resume; require a training-loss decrease; then a fresh
3-epoch pilot on 1,024 training / 256 validation objects (384 updates).
Both experiments completed successfully; the console exit status is 0 and
W&B finished syncing. The tmux session remains open at an idle shell.

- [Overfit emr5ag36](https://wandb.ai/AI-Astro/superNISP-corrected/runs/emr5ag36):
  all 40 epochs / 160 updates, planned resume succeeded, masked training MAE
  1.1754255 at initialization to 0.9242856; about 22 seconds of measured
  train/evaluation time, excluding process setup and W&B startup/sync.
- [Pilot 9c47c2of](https://wandb.ai/AI-Astro/superNISP-corrected/runs/9c47c2of):
  all 3 epochs / 384 updates, validation MAE 1.4428717 at initialization to
  1.1082243, physical-intensity RMSE 0.4275776 to 0.3729545 MJy/sr.
  Best checkpoint is epoch 3; about 71.5 seconds train/evaluation wall time,
  peak allocated CUDA memory 0.577 GiB.
- Prediction moment-diagnostic failures rose from 0 to 20/256; reference
  failures remained 25/256, and paired successes fell from 231 to 218.
  The apparent shape RMSE decrease cannot establish shape improvement with
  this changing population. No test data were used and no scientific
  SR-improvement percentage was measured.

Checkpoints and provenance are in `runs/residual-{overfit,pilot}-s20261016/`.
Tracked metrics, checkpoint/source hashes, and checks are in
`paper/diagnostics/corrected_training_pilot.json`. Runs and weights are gitignored.

Next: inspect pilot predictions and the 20 moment failures, add fixed matched
interpolation/native comparison diagnostics, then choose a full-split baseline
budget before architecture comparisons. The original startup and implementation
checklist below describes the state before this continuation.

Pre-existing local edits in `code/test_model.ipynb` and untracked
`code/older/{losses_olddiff,model_olddiff,train_olddiff}.py` were left untouched.

## Objective and agreed order

Finalize the Euclid NISP / JWST superresolution paper with a reproducible
answer to whether SR improves galaxy shapes, by how much, and for which
galaxies. The user asked to review architecture, code, and sample **before**
improving downstream measurements. They then asked to prepare the correct
sample locally so only a compact package needs transferring to the GPU.

The sample rebuild is complete. Next: integrate the new loader into a
correct training pipeline, establish a residual baseline, and compare a
small set of architectures under controlled conditions. Shape-estimator
development and final paper numbers come later. The improvement percentage
is an experimental outcome, not a target to force.

## Locations and repository state

Expected GPU layout (transfer/extraction has not been confirmed in this chat):

```text
/home/shemmati/Work/Projects/
├── superNISP/                         # https://github.com/xoubish/superNISP.git
└── supernisp_nir_y_f115w_v1/           # extracted portable sample
```

Local source workspace: `/Users/shemmati/Dropbox/superNISP`.
Local package: `data/ready/supernisp_nir_y_f115w_v1/`.
Archive: `data/ready/supernisp_nir_y_f115w_v1.tar`, 3,585,710,080 bytes.
Archive SHA256:

```text
91cb6b09cd1cfd25cf99cb68441edcdd062bb73b82fa76eb2a272e955430bf33
```

The audit and sample code are committed in `61705a3` (`paper1`), following
the original audited baseline `5e2fe67`. At handoff preparation, local main
and its cached origin/main both pointed to `61705a3`; no live remote check
was needed for this document. This handoff is a subsequent change.
Earlier chat statements that the sample code was uncommitted are superseded.
Data are gitignored: cloning does not download the prepared sample.
The tar includes copies of the sample code, but active development should
use the repository's `code/` files and record the code revision per run.
Keep run outputs under the repo in run-specific directories, with large
artifacts gitignored and their hashes/provenance recorded in tracked reports.

## Completed checks — do not restart these without new evidence

- Inspected local branches and reachable history, plus live remote branch
  refs at the time of the audit. The exact checkpoints and result arrays
  behind the paper figures were not recovered. A diffusion-named checkpoint
  in another branch was actually the same blob as a residual checkpoint.
  Do not repeat this hunt unless a new location or artifact becomes available.
- Reviewed residual and diffusion source, loaded compatible old residual
  checkpoints, and ran finite forward/backward checks. These were execution
  checks, not architecture rankings or scientific training results.
- Found old extraction defects: RMS mistaken for DQ, reflected mask transfer,
  padding/cropping issues, and lost accepted-object IDs. Old array versions
  largely overlap and must not be treated as independent populations.
- Rebuilt from original NIR-Y stacks and COSMOS-Web F115W mosaics using named
  SCI/RMS/DQ layers, explicit WCS mapping, independent masks, and retained IDs.
- Verified all stored stamps for finite values, invalid fill, coverage,
  central validity, IDs, and split consistency. Confirmed no cross-split
  object centers within 20 arcsec.
- Passed five synthetic extraction regression tests and a loader smoke test
  with two spawned workers, physical-intensity round trips, masked loss,
  and one tiny CPU optimizer step in both linear and asinh representations.
- Checked WCS/image alignment on a seeded subset: 86 usable bright fits from
  512 sampled objects; median shifts essentially zero, largest fitted
  component 0.341 NISP pixel. No empirical shifts were applied. This is an
  approximate diagnostic, not precision astrometric calibration for all objects.
- Verified all 46 archived regular files against their package checksums.
  Transfer integrity and GPU execution still need checking on the destination.

Evidence and details:

- [Corrected sample build](paper/corrected_sample_build.md) — completed rebuild.
- [Architecture/sample review](paper/architecture_sample_review.md) — findings
  and proposed comparisons. Its original final extraction step is now complete.
- [Shape-validation audit](paper/shape_validation_audit.md) — paper/evaluation
  issues and the eventual quantitative science test.
- [Machine-readable original audit](paper/diagnostics/training_pipeline_audit.json).
- Package `README.md`, `manifest.json`, `validation.json`, `BUILD_REPORT.md`.

## Frozen sample and training contract

**One band pair:** Euclid NISP Y input, JWST NIRCam F115W target.
41×41 native NISP pixels → 205×205 target pixels on exactly the same WCS
footprint. Fivefold pixel sampling is not a demonstrated fivefold gain in
resolved detail. NISP sampling is about 0.297 arcsec; target about 0.0593 arcsec.

| Split | Galaxies |
| --- | ---: |
| Train | 17,816 |
| Validation | 2,118 |
| Test | 2,876 |
| Total | 22,810 |

39,024 candidates verified against COSMOS2020. Selection: `lp_type == 0`,
ACS F814W < 25, 0.01 < photo-z < 3, and 4 < FLUX_RADIUS < 500 in catalog
units. Rejections: 8,229 spatial buffer; 7,043 coverage; 723 central invalid
region; 219 partial NISP cutouts. This is an optically selected COSMOS sample.

Spatial splits use 180-arcsec blocks, seed 20261016, nominal 80/10/10
assignments and 10-arcsec border exclusion. Use saved indices; do not randomly
re-split. At least 95% valid coverage in each instrument and a fully valid
central 5×5 NISP / 25×25 JWST region. No additional S/N cut was imposed.

Use `code/paired_sample.py`:

```python
from paired_sample import PairedSample, masked_l1

train = PairedSample(data_dir, split='train', representation='linear')
validation = PairedSample(data_dir, split='validation', representation='linear')
# batch['nisp']: B,1,41,41; batch['jwst']: B,1,205,205
# loss = masked_l1(model(batch['nisp']), batch['jwst'], batch['jwst_valid'])
```

The import works in scripts inside `code/`; otherwise add `code/` to the
Python module search path. Existing training loaders expect a different
schema and cannot simply be pointed at this HDF5 file.

- HDF5 stores NISP science/RMS in microJy per native pixel, JWST in MJy/sr,
  independent validity masks, NISP DQ, IDs, splits, and calibration metadata.
- Loader converts NISP to MJy/sr, subtracts per-image outer-region background,
  and divides by fixed, instrument-specific **training-only** scales.
  Do not refit normalization or use per-image target variance scaling.
- `representation='asinh'` applies `asinh(z/3)`; it is unbounded. Neither
  representation may be clamped to [-1,1]. Start with linear as the reference;
  compare asinh separately if needed.
- Invalid pixels are zero filled but are not zero-flux observations. Apply
  `jwst_valid` to the loss. Input mask/noise conditioning can be an explicit
  ablation; the current loader exposes both. RMS always refers to linear space.
- `jwst_to_mjy_sr()` inverts normalization. Default background zero gives
  sky-subtracted predictions. The reference's `jwst_background` is available
  for round-trip checks, not as an input for NISP-only deployment.
- Do not manually flip or rotate to align the images. Augment training pairs
  consistently, including masks and any PSF information; do not augment validation.
- Retrain from scratch. Old checkpoints may have seen the new test objects.

## GPU startup: verify transfer and environment only

If the tar is still present, verify before extracting:

```bash
cd /home/shemmati/Work/Projects
sha256sum -c supernisp_nir_y_f115w_v1.tar.sha256
# Only after OK, and if not already extracted:
tar -xf supernisp_nir_y_f115w_v1.tar
```

With the extracted directory:

```bash
cd /home/shemmati/Work/Projects/supernisp_nir_y_f115w_v1
sha256sum -c SHA256SUMS
cd /home/shemmati/Work/Projects/superNISP
git status -sb
git log -1 --oneline
nvidia-smi
python -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No CUDA device")'
python code/smoke_test_sample.py ../supernisp_nir_y_f115w_v1
```

Inspect the existing Python environment first. Loader dependencies are NumPy,
h5py, and a GPU-compatible PyTorch. Package `environment-used.json` records
the Mac verification environment, not a CUDA installation prescription.
The existing smoke test is **CPU-only** even on a GPU host; separately verify
a CUDA forward/backward step when integrating the real training entrypoint.
Do not download original mosaics or rerun the full audit/build just to start.
The optional full validator regenerates `validation.json` and `preview.png`;
verify checksums first if rerunning it for a concrete reason.

## Next implementation work (not yet done)

1. Create a portable training entrypoint using `PairedSample`, explicit data
   and output paths, masked loss, validation-only model selection, and saved
   configs/seeds. Avoid import-time sweeps or hardcoded NERSC paths in old files.
   Record manifest/normalization hashes, code revision, environment, optimizer
   and scheduler state, and best-checkpoint selection in each run directory.
2. Establish a corrected current residual baseline. `claude_model_NIR_2.py`
   used eight single residual dense blocks, 64 features, two 2× pixel-shuffle
   stages then resize to 205 (2,332,177 parameters). These are not nested ESRGAN
   RRDBs. `claudemodel.py` has an existing direct-5× design with a different
   trunk; comparing those two unchanged does not isolate the upsampling head.
   Fix OneCycleLR stepping to each successful optimizer update if using it;
   the old HDF5 path steps it once per epoch.
3. Run a small training-only overfit experiment (about 32 clean pairs), finite
   CUDA gradients, masked-pixel checks, and checkpoint reload/resume checks.
   Then run a short train/validation pilot before committing to a long run.
4. Compare the current head with direct-5× using the same trunk, sample,
   normalization, augmentation, loss, seeds, and optimizer-update budget.
   Test residual scaling or removing the edge enhancer separately. Consider
   a similarly sized residual UNet after this baseline. Report GPU time and
   memory as well as quality; repeat finalists with at least three seeds.
5. Keep a fixed, simple linear-intensity shape diagnostic on validation
   objects during comparisons, alongside reconstruction errors. Do not use
   sharper appearance alone to pick a model. Reserve final test evaluation
   until model and analysis choices are frozen.

Diffusion is a later comparator: its old shortened sampler uses adjacent-step
coefficients while skipping timesteps; its output clamp conflicts with asinh;
normalizers refit on held-out data; and validation uses noisy HR reconstruction
instead of full LR-only generation. Repair and test these before judging its
architecture. Production training code has not yet been repaired by this handoff.

## Scientific boundaries and eventual result

JWST SCI-only coverage masks use finite/nonzero support; no full JWST RMS/DQ
was supplied. Bilinear resampling affects PSF and noise correlations. Supplied
NISP PSFs retain their original sampling headers, which need inspection before
use. Y and F115W differ: do not impose naive blur/downsample equality as a
physical constraint without a cross-band model.

The old paper figures/numbers remain unverified. GalSim `observed_shape.g`
is a shape parameterization, not calibrated gravitational shear. Measure
agreement with the JWST reference in linear intensity, with matched angular
regions, object populations, and explicit measurement failure rates.
Ultimately compare native NISP and interpolation with SR using ellipticity
RMSE, e.g. `100 * (1 - RMSE_e(SR) / RMSE_e(baseline))`, with paired spatial
uncertainty estimates and bins in galaxy properties. Regression slopes alone
do not establish that percentage. No shape-improvement result has yet been
measured on this rebuilt sample.

Update this document with run IDs, changes, checks, and the next concrete
step as work proceeds, so another session can continue without restarting.
