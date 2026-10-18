# Figure 6 and ResNet model-choice revision

Updated 2026-10-17 following the user's approval to retain the original ResNet
and add the decision and reproducibility evidence to the paper.

## Current revision: appendix removed — 2026-10-17

The user requested a shorter paper without this appendix. The abstract now keeps
one sentence about seed reproducibility. Essential continuation protocol details
have moved into Methods, including the shared parent/optimizer state, matched
batches/augmentations, schedule, equal-region loss and distinct checkpoint rule.
Results retain the seed RMSE ranges and 0.09% relative sample standard deviation,
the continuation trade-off, and the decision to keep the original ResNet.
Appendix/seed-figure references are removed or redirected to the Results subsection.
Figures 6 v06 and 7 v05 remain in the main manuscript.

The former appendix, seed plot and pre-removal manuscript are preserved in
`diagnostics/resnet_validation_archive/`, outside the publication-file selection.
The new manifest is `diagnostics/appendix_removal_v01.json`; the v01 model-selection
manifest below describes the historical installation, whose paths are now archived.
Structural/reference checks passed. The changed sections compiled in a two-column
article preview without unresolved references or overfull boxes. A full A&A build
still stops at the existing missing `etex.sty` dependency. The local upload ZIP
was refreshed; no remote publish was performed.

## Historical manuscript changes

- Installed the approved, unchanged Figure 6 v06 PDF as
  `figs/model_comparison_v06.pdf`. It is the sixth figure in the manuscript.
- Replaced the historical diffusion-step and pixel-metric Results passage
  with the current source/outside RMSE and Fourier power/correlation comparison.
  The old figure files remain available; the previous prose remains in Git history.
- Added regional masks, aggregation and spectral-window definitions. The
  mean-per-object regional RMSE is distinguished from the training monitor's
  root of the mean per-object MSE.
- Updated the training-seed Methods paragraph, Figure 6 caption and model-choice
  Results subsection. Original seed 20261016, epoch 49 remains the main model;
  ResNet-D5 remains a comparator. This preserves the primary 50-epoch budget.
- Added `resnet_validation_appendix.tex`, containing the three-seed plot,
  selected/final checkpoint tables and the matched continuation protocol and
  trade-off. Spatial-bootstrap intervals and between-seed variation are separate.
- Corrected the abstract's historical sample counts and test-set claim, replacing
  unsupported old numerical shape claims there with the verified validation
  findings. Updated the opening Discussion paragraph to match the power versus
  correlation trade-off.
- Synchronized the Figure 6 notebook's caption notes. Its code cells, stored
  outputs, original PDFs and underlying inference results are unchanged. The
  pre-install notebook is archived in `paper_figures/revisions/`.

## Numerical sources and checks

`diagnostics/resnet_model_selection_v01.json` records the exact source-report
hashes, selected checkpoint, installed PDF hashes and manuscript hashes, with
the numerical seed and continuation summaries. All table entries were checked
against the completed analysis JSON files. Main Figure 6 values were checked
against the frozen benchmark and source-spectrum reports.

The two copied PDFs match their source hashes. Figure ordering, unique labels,
references, LaTeX environments, notebook validity and unchanged notebook code/
outputs were checked. `git diff --check` passed.

Direct `pdflatex` compilation of the complete A&A manuscript stops because this
environment lacks `etex.sty`, before reading the changed content. An isolated
two-column article preview of the changed regional Methods, Results and appendix
compiled successfully. Both figures and both tables were visually checked; no
overfull boxes were reported. The preview does not replace journal-class layout
verification and its Figure 5 cross-reference is unresolved because that section
was not included. The manuscript uses the A&A class's `appendix` environment.

## Remaining scope

The held-out test has not been evaluated. All new empirical statements describe
validation results, and no further training sweep is queued. Older Introduction,
shape Results and later Discussion passages still require revision against the
corrected shape analyses; this scoped change does not validate their historical
shear interpretation or regression numbers.

This is a local manuscript update. The upload archive was regenerated with
`scripts/overleaf.py bundle`; no remote Overleaf edits or push were performed.
