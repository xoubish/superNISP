# Figure inputs

No model is executed by the notebooks. Create these artifacts in a separate
persistent, online-W&B export/benchmark job using frozen checkpoints. Figure 5 may use a provisional best-so-far diffusion
checkpoint while training continues; its metadata records that status.

## Figure 5: `galaxy_examples_vNN.npz` and `galaxy_examples_vNN.json`

NPZ arrays, all sharing the first axis and ordering:

- `object_ids`, `rows`: unique catalog IDs and corrected-sample rows.
- `nisp`, `nisp_valid`: `(N, 41, 41)`, sky-subtracted linear MJy/sr and bool mask.
- `bilinear`, `bilinear_valid`: `(N, 205, 205)`, linear-intensity NISP
  interpolation and its strict input-support mask (from v02 onward).
- `jwst`, `jwst_valid`: `(N, 205, 205)`, same physical units and target mask.
- `legacy`, `direct5`, `diffusion`, `diffusion_mean3`: `(N, 205, 205)` predictions
  from frozen selected checkpoints. The main grid shows the three individual
  models; the mean image is available as an optional column.
- `diffusion_samples`: `(N, 3, 205, 205)`, the three seeded diffusion draws.

JSON: `complete: true`, `split: "validation"`, `units: "MJy/sr"`, `object_ids`
matching NPZ order, `checkpoint_sha256` keyed by the four prediction names,
`selection` describing the input-only rule and its seed, `sampling_seeds`,
`nisp_pixel_scale_arcsec` (one value per object, from actual stored pixel area),
`data_hashes` and `npz_sha256`. `checkpoints` records epochs, selection metrics, source hashes, snapshot paths
and whether training is complete. `limits_mjy_sr` stores frozen display limits.
Use fixed shared per-object display scales and
symmetric residual limits; retain negative pixels and invalid support.

`galaxy_examples_selection.json` retains the first input-ranked selection.
The current observed-morphology selection is `galaxy_examples_selection_v04.json`;
pass it with `--selection paper_figures/inputs/galaxy_examples_selection_v04.json`
and reuse it for subsequent checkpoint updates. It fixes validation IDs, observed-image selection, display limits,
and diffusion noise seeds. A new bundle version preserves older predictions.
Run `code/export_paper_examples.py` with `--version vNN --output runs/figure5-vNN`
through `code/launch_paired.py`, after setting the same `VERSION` in the notebook.
The export job checks for tmux/screen and online W&B before loading any model,
snapshots each `best.pt` with a stable-copy check, verifies recorded code/data
hashes, saves predictions, executes the plotting notebook and logs both the
main figure and its companion residual sheet to W&B. The notebook itself only
reads saved arrays. Use a fresh output and console directory for each update.

## Figure 6: `model_benchmark.json`

Top-level: `complete: true`, `split: "validation"`, `units: "MJy/sr"`,
`object_ids` (unique, common to every method), `population_rule`, `data_hashes`,
and `timing_protocol` (GPU, batch size, precision, warmup, repeats, whether
preprocessing and transfers are included, diffusion steps/seeds/ensemble size).

`methods` maps display names to records with `object_ids` in the same order,
`mae_mjy_sr` and `rmse_mjy_sr` (per-object arrays of physical pixel errors on
the same valid target pixels), `inference_ms_per_object` and
`checkpoint_sha256` (or `null` for interpolation). Include legacy, direct5,
diffusion single sample and mean image, and an interpolation baseline. Compute
spatial-block intervals in the benchmark producer; add them to a later figure
revision rather than inventing uncertainty from aggregate values.

These are export contracts, not results. Keep large image bundles local; the
small sidecar/provenance files should accompany any shared rendered figures.
