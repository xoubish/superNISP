# Corrected sample build report — 2026-10-16

Built from the original local NIR-Y stacks and COSMOS-Web F115W mosaics.
No old cutout intensities or checkpoints were used to generate the new images.

- Candidates verified against COSMOS2020: 39,024.
- Accepted: 22,810; train 17,816 / validation 2,118 / test 2,876.
- Rejected: 8,229 spatial buffer; 7,043 insufficient coverage; 723 central
  invalid pixels; 219 incomplete NISP cutouts.
- Every stored stamp passed finite-value, masked-fill, coverage, central-region,
  ID, and split checks. No cross-split object pairs within 20 arcsec.
- Five extraction regression tests passed: grid/footprint, constant surface
  brightness and invalid support, rotated source alignment, named DQ vs RMS,
  and deterministic spatial blocks.
- Two-worker loading, physical-intensity round trips, masked loss, and one
  small CPU optimizer step passed for both linear and asinh representations.
  These are software checks, not scientific model-training results.
- WCS-only alignment diagnostics on a seeded 512-object subset yielded
  86 bright, sufficiently covered fits. All fitted correlations exceeded
  0.7. Median shift (dy,dx): (-0.000003,-0.000021) NISP pixel;
  component 5th–95th percentiles: dy [-0.093,0.037],
  dx [-0.093,0.049]. The largest absolute
  fitted component was 0.341 pixel.
  Gaussian PSF matching was approximate; these diagnostics are not a full
  astrometric or shape calibration. No empirical shifts were applied.

The original extractor treated the extension after SCI as a mask. These
files contain SCI, RMS, DQ, so that selected RMS rather than quality flags.
The new builder uses layer names. This agrees with the official
[NIR stacked-frame product description](https://euclid.esac.esa.int/msp/dpdd/v1.1/nirdpd/dpcards/nir_stackedframe.html).
It also removes the extra reflected mask transfer and uses explicit WCS
mapping instead of fixed rotations and nominal pixel scales.

This is a clean, reproducible training sample with documented selection,
not certification for weak-lensing cosmology. JWST has no supplied RMS/DQ
here; finite nonzero science support is used as a conservative coverage
proxy. Keep these limitations in downstream evaluation and the paper.

Portable package: `data/ready/supernisp_nir_y_f115w_v1/`.
