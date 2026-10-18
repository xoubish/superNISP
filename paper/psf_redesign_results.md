# PSF-aware redesign: completed results

Both ten-epoch / 22,270-update experiments and the automatic comparison completed
with exit 0. Both select epoch 5 under the predeclared source-MAE rule. The original
checkpoint is unchanged. Training and validation metrics plus fixed galaxy images
were uploaded with the user's explicit authorization to the existing project:
[PSF + noise](https://wandb.ai/AI-Astro/superNISP-corrected/runs/on773llj) and
[PSF + noise + inverse feature](https://wandb.ai/AI-Astro/superNISP-corrected/runs/hdsb3bfx).

**Decision (confirmed by the user): keep the original, independently replicated ResNet as the paper
model.** This redesign does not establish a useful improvement across RMSE,
visual structure, sharpness, and both shape estimators. The PSF/noise-only variant
is the more promising follow-up, owing to its fixed-weight shape result, but the
inverse-filter feature has no demonstrated overall advantage. This conclusion
applies to this warm-start experiment, not every possible PSF-aware architecture.
It does not certify that no better network exists.

## What improved, and what did not

All reconstruction numbers below use the same 2,118 validation objects. Source
RMSE is the mean of per-object source-region RMSE, in MJy/sr. Best-checkpoint
shape errors below use the primary common success sample across the reference
and **all selected/final checkpoints**, so their absolute values should not be
compared directly with Figure 7's different common population.

| Model | Source RMSE | Reduction vs original | Additional reduction vs matched control | Fixed shape RMSE | Adaptive shape RMSE |
| --- | ---: | ---: | ---: | ---: | ---: |
| Original ResNet | 0.0601735 | — | — | 0.189929 | 0.238957 |
| Extra-training control | 0.0594465 | 1.208% | — | 0.189830 | 0.237933 |
| PSF + noise | 0.0593742 | 1.328% | 0.122% | 0.187964 | 0.238144 |
| PSF + noise + inverse feature | 0.0593513 | 1.366% | 0.160% | 0.188354 | 0.238792 |

Most of the source-RMSE improvement over the original is reproduced by extra
training alone. Relative to that matched control, source-RMSE differences are
−0.0000723 [−0.0001908, +0.0000671] for PSF/noise and
−0.0000952 [−0.0002811, +0.0001109] MJy/sr for the inverse-feature branch.
Both 95% paired spatial-block intervals include zero. Outside-source RMSE also
has no clear incremental improvement over the matched control.

The **fixed-weight shape** result is the strongest positive sign: PSF/noise
reduces component RMSE by 0.98% relative to the matched control, with absolute
difference −0.001867 [−0.003509, −0.000251]. The inverse-feature branch's difference
is −0.001476 [−0.003017, +0.000018], which includes zero. The primary fixed-shape
sample contains 1,627 galaxies. Individual fixed-shape successes are 1,785 for
the control, 1,796 for PSF/noise, and 1,793 for the inverse-feature branch, out of
2,118 attempts; the target succeeds for 1,662.

Neither branch clearly improves **adaptive shapes**. Relative to the control,
differences are +0.000211 [−0.001192, +0.002059] and
+0.000859 [−0.001479, +0.004539], respectively. Their point estimates are slightly
worse, with intervals spanning improvement and deterioration. The common sample
contains 1,700 objects; individual adaptive successes are 1,740 / 1,749 / 1,747 for
control / PSF-noise / inverse-feature, with 1,767 successful targets. Pairwise-only
success comparisons and every failure count are retained in the numerical report.

**Sharpness and structure:** fine-scale source correlation (3–6 target pixels)
is 0.464736 for the original, 0.467212 for the control, 0.467861 for PSF/noise,
and 0.467831 for the inverse-feature branch. The extra correlation differences
over control have intervals crossing zero. Power fidelity improves relative to
the control, but not clearly relative to the original at this scale. This does
not establish more recovered resolution. Outside-source structure remains weak
and similar across methods; no independent reference-noise correction is available.

**Visuals:** the five pre-existing Figure 5 examples, displayed with identical
fixed limits, are very similar across all four selected models. In particular,
the redesign does not visibly restore the disk/spiral detail missing from the
original prediction. The residual sheet also shows no conspicuous new recovery.
This is a qualitative inspection of fixed examples, not a blinded visual study.

Subgroup checks are mixed rather than uniformly favorable. Relative to control,
PSF/noise source RMSE is 0.29–0.33% lower in the two lowest S/N bins but 0.05–0.08%
higher in the next two. Across the three represented validation PSF stacks it is
0.32% higher, 0.31% lower, and 0.27% lower. These descriptive subgroup means are
not separately selected models or significance claims.

## Plots and reproducibility

- [Paired differences versus matched control](diagnostics/psf_redesign_paired_v01.png)
  ([PDF](diagnostics/psf_redesign_paired_v01.pdf)). Zero means no difference.
- [Diagnostic notebook](../paper_figures/diagnostics/psf_redesign.ipynb): editable
  RMSE/shape summary and all saved plots.
- [Image comparison](../runs/psf-redesign-comparison-v01/fixed_images.png) and
  [residual comparison](../runs/psf-redesign-comparison-v01/fixed_residuals.png).
- [Sharpness/structure](../runs/psf-redesign-comparison-v01/sharpness_structure.png),
  [RMSE](../runs/psf-redesign-comparison-v01/rmse.png),
  [shapes](../runs/psf-redesign-comparison-v01/shapes.png), and
  [learning curves](../runs/psf-redesign-comparison-v01/learning_curves.png).
- [Full numerical comparison](../runs/psf-redesign-comparison-v01/comparison.json),
  [independent verification](../runs/psf-redesign-comparison-v01/verification.json),
  and [frozen protocol](psf_redesign_protocol.md).

Three CPU controls and both GPU preflights passed. All ten training streams match
the completed control exactly. Independent verification confirms exact original
weights at epoch zero, correct inherited/new Adam step counts and final learning
rates, input/source/artifact hashes, and 20 paired shape-difference/bootstrap
calculations. The selected and final checkpoints and every intermediate new epoch
checkpoint are retained. No training job remains queued by this experiment.

The bootstrap intervals are exploratory, conditional on validation selection and
one parent seed. Only four very similar stack PSFs are available, and their
native-grid kernel is an explicitly documented proxy. A small improvement cannot
be attributed uniquely to PSF information rather than noise/mask information or
extra capacity. A stronger architecture claim would require matched-capacity
ablations and independent training seeds. Neither observed-shape improvement nor
input PSF conditioning constitutes shear calibration.

The manuscript and numbered figure inputs were not replaced. The reserved test
images remain unused. The original paper model has the stronger independent-seed
evidence; the current experiment supplies a concrete check of the proposed
redesign, not a reason to make a last-minute network substitution.
