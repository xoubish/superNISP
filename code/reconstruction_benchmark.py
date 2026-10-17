"""Matched physical-pixel errors and paired spatial-block uncertainty."""
import numpy as np
import torch


def pixel_errors(prediction, target, valid):
    """Per-object MAE/RMSE on one explicitly shared mask; no intensity clipping."""
    if prediction.shape != target.shape or valid.shape != target.shape:
        raise ValueError('Prediction, target and mask geometry must match')
    valid = valid.bool()
    dimensions = tuple(range(1, target.ndim))
    count = valid.sum(dimensions)
    if torch.any(count == 0):
        raise ValueError('Empty common pixel support')
    if not torch.isfinite(prediction[valid]).all() or not torch.isfinite(target[valid]).all():
        raise ValueError('Nonfinite values on common pixel support')
    # Double accumulation prevents bright-source sums from losing precision.
    difference = torch.where(valid, prediction.double()-target.double(), 0.)
    return dict(mae_mjy_sr=difference.abs().sum(dimensions)/count,
                rmse_mjy_sr=(difference.square().sum(dimensions)/count).sqrt(),
                valid_pixels=count)


def summarize(methods, blocks, repeats=2000, seed=20261016):
    """Object-weighted means and paired gains; resample entire spatial blocks."""
    labels, inverse = np.unique(np.asarray(blocks), return_inverse=True)
    if len(labels) < 2 or repeats < 2:
        raise ValueError('Need multiple spatial blocks and bootstrap draws')
    counts = np.bincount(inverse)
    draws = np.random.default_rng(seed).integers(len(labels), size=(repeats, len(labels)))
    sampled_counts = counts[draws].sum(1)
    report = {}
    for metric in ['mae_mjy_sr', 'rmse_mjy_sr']:
        estimates = {}
        sampled = {}
        for name, values in methods.items():
            error = np.asarray(values[metric], dtype=float)
            if error.shape != inverse.shape or not np.isfinite(error).all() or (error < 0).any():
                raise ValueError('Invalid or unmatched per-object errors')
            estimates[name] = float(error.mean())
            sums = np.bincount(inverse, weights=error, minlength=len(labels))
            sampled[name] = sums[draws].sum(1)/sampled_counts
        baseline = estimates['bilinear']
        if baseline <= 0 or np.any(sampled['bilinear'] <= 0):
            raise ValueError('Bilinear error must be positive for relative comparison')
        for name in methods:
            report.setdefault(name, {})[metric] = dict(
                mean=estimates[name],
                ci95=np.percentile(sampled[name], [2.5, 97.5]).tolist(),
                gain_vs_bilinear_percent=100*(1-estimates[name]/baseline),
                gain_ci95_percent=np.percentile(
                    100*(1-sampled[name]/sampled['bilinear']), [2.5, 97.5]).tolist())
    return dict(methods=report, spatial_blocks=len(labels), replicates=repeats, seed=seed,
                rule='Resample spatial blocks with replacement, retaining all objects and methods paired. '
                     'Equal object weights within the resulting population. Percentile 95% intervals; '
                     'conditional on this field, validation selection and trained checkpoints.')
