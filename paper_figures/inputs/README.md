# Pending figure inputs

No model is executed by the notebooks. Create these artifacts in a separate
persistent, online-W&B export/benchmark job after the runs and choices are ready.
They are intentionally absent during initial layout work.

## Figure 5: `galaxy_examples.npz` and `galaxy_examples.json`

NPZ arrays, all sharing the first axis and ordering:

- `object_ids`, `rows`: unique catalog IDs and corrected-sample rows.
- `nisp`, `nisp_valid`: `(N, 41, 41)`, sky-subtracted linear MJy/sr and bool mask.
- `jwst`, `jwst_valid`: `(N, 205, 205)`, same physical units and target mask.
- `legacy`, `direct5`, `diffusion`, `diffusion_mean3`: `(N, 205, 205)` predictions
  from frozen selected checkpoints; all must be present for the initial grid.

JSON: `complete: true`, `split: "validation"`, `units: "MJy/sr"`, `object_ids`
matching NPZ order, `checkpoint_sha256` keyed by the four prediction names,
`selection` describing the input-only rule and its seed, `sampling_seeds`,
`nisp_pixel_scale_arcsec` (one value per object, from actual stored pixel area),
`data_hashes` and `npz_sha256`. Use fixed shared per-object display scales and
symmetric residual limits; retain negative pixels and invalid support.

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
