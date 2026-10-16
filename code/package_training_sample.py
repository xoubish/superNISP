"""Finish a validated sample with documentation, checksums, and one transfer tar."""
from pathlib import Path
import argparse
import importlib.metadata
import json
import shutil
import tarfile
from build_training_sample import sha256


def package(directory):
    root=Path(__file__).resolve().parents[1]
    directory=Path(directory).resolve()
    manifest=json.loads((directory/'manifest.json').read_text())
    validation=json.loads((directory/'validation.json').read_text())
    if manifest['pilot'] or not validation['full_array_validation']:
        raise ValueError('Only a fully validated production sample can be packaged')
    if manifest['accepted']!=validation['accepted']:
        raise ValueError('Validation does not match manifest')
    if not all(row['sha256'] for row in manifest['source_files']):
        raise ValueError('Missing source checksums')
    for name in ['paired_sample.py','validate_training_sample.py','smoke_test_sample.py',
                 'test_training_sample.py','package_training_sample.py']:
        shutil.copy2(root/'code'/name,directory/name)
    versions={name:importlib.metadata.version(name) for name in
              ['numpy','pandas','h5py','astropy','scipy','torch','matplotlib']}
    (directory/'environment-used.json').write_text(json.dumps(versions,indent=2)+'\n')
    (directory/'requirements-loader.txt').write_text('numpy\nh5py\ntorch\n')
    (directory/'requirements-validation.txt').write_text('numpy\nh5py\npandas\nastropy\nscipy\nmatplotlib\n')
    size=(directory/'sample.h5').stat().st_size/1e9
    counts=manifest['split_counts']
    text=f'''# Corrected superNISP NIR-Y / JWST F115W training sample

Built locally on 2026-10-16. {manifest['accepted']:,} paired galaxies from
{manifest['candidates']:,} existing COSMOS candidates. Sample HDF5: {size:.2f} GB.
Train: {counts.get('train',0):,}; validation: {counts.get('validation',0):,}; test: {counts.get('test',0):,}.

Copy this entire directory, or its tar archive, to the GPU machine. You do
not need the large source FITS mosaics, COSMOS FITS catalog, old NumPy arrays,
or local decompression cache. The tiny NISP PSF files are already included.
The builder is retained for provenance; regenerating the sample requires
the original source files and original repository layout.

## After transfer

```bash
tar -xf {directory.name}.tar
cd {directory.name}
sha256sum -c SHA256SUMS
python smoke_test_sample.py .
```

Use a GPU-compatible PyTorch installation on the training machine. The loader
also needs NumPy and h5py (`requirements-loader.txt`). The recorded environment
is the local verification environment, not a required CUDA installation.
The smoke test uses two spawned workers and one tiny CPU training step in
each normalization mode. It does not train a scientific model.

Optional full integrity/geometry/preview check (additional dependencies in
`requirements-validation.txt`):

```bash
python validate_training_sample.py .
```

That command regenerates validation.json and preview.png; run checksum
verification before it, since regenerated diagnostic files may have new hashes.

## Training interface

```python
from paired_sample import PairedSample, masked_l1
from torch.utils.data import DataLoader

train = PairedSample('.', split='train', representation='linear')
validation = PairedSample('.', split='validation', representation='linear')
loader = DataLoader(train, batch_size=8, shuffle=True, num_workers=4)

for batch in loader:
    inputs = batch['nisp'].to(device)       # B,1,41,41
    targets = batch['jwst'].to(device)      # B,1,205,205
    valid = batch['jwst_valid'].to(device)
    predictions = model(inputs)
    loss = masked_l1(predictions, targets, valid)
    # zero_grad, backward, optimizer.step in the training loop
```

On platforms using spawned workers, put the training loop under an
`if __name__ == '__main__':` guard. The loader opens HDF5 separately in each
worker. Existing repository loaders expect another schema: use this loader
instead of pointing their old paths at sample.h5.

The model output must be 205x205 for a 41x41 input. Inputs are in native NISP
orientation; JWST has the same orientation and footprint. No manual flip,
rotation, resizing, normalization refit, or random re-split is needed.
Use paired transforms if augmenting, including both masks and any PSF data.
Use the provided independent validation split for model selection. Retrain
from scratch for independent final tests: older checkpoints may have seen
galaxies now assigned to validation or test.

## Units and normalization

HDF5 nisp/nisp_rms are microJy per native NISP pixel; jwst is MJy/sr. Valid
pixels retain physical intensity, including negative sky fluctuations.
Invalid pixels are filled with zero and identified by separate validity
masks; they are not measurements of zero flux. nisp_dq preserves the flags.

The loader converts NISP to MJy/sr using the recorded local pixel solid angle,
subtracts each image's sigma-clipped outer-region background, then divides
by its instrument's fixed training-only scale. JWST uses the same recipe
without unit conversion. No clipping or per-image variance rescaling occurs.
`representation='asinh'` applies asinh(z/3) after this affine normalization;
it is unbounded and must not be clamped to [-1,1]. The nisp_rms_linear output
always refers to the linear normalized representation.

`train.jwst_to_mjy_sr(predictions, background)` reverses normalization.
The batch jwst_background restores the observed reference's sky level;
that reference background is unavailable for NISP-only deployment. Use zero
background for sky-subtracted predictions. Retain a consistent representation
for losses and convert back to linear intensity for scientific measurements.

## Sample construction

- Existing dated coverage candidate list; IDs and coordinates verified against
  COSMOS2020. Selection: lp_type==0, ACS F814W<25, 0.01<photo-z<3,
  4<FLUX_RADIUS<500 in the catalog's native units. This is an optically selected
  COSMOS sample, not a complete NISP sample.
- NISP layers are selected by name: SCI, RMS, DQ. The old extraction used the
  RMS extension as a mask. Good pixels require DQ==0 and finite science and
  positive finite RMS below the no-data sentinel range.
- NISP 41x41 stamps are not resampled. The 205x205 JWST target grid is exactly
  five times finer in WCS, with identical pixel-edge footprint. JWST is
  bilinearly interpolated in surface brightness. No fixed rotation or mask
  reflection is used. Native NISP sampling is about 0.297 arcsec; the output
  is about 0.0593 arcsec, not an assumed universal 0.06.
- JWST and NISP masks are independent. Valid JWST pixels are never erased
  merely because NISP is missing there. A target sample is valid only if all
  bilinear contributors are finite and nonzero in the native science mosaic.
- Each stamp has at least 95% valid coverage in both instruments. The central
  5x5 NISP pixels / 25x25 target pixels must be fully valid. No S/N cut is
  imposed. Rejected IDs and reasons are included, so selection is auditable.
- Fixed 180-arcsec sky blocks, deterministic seed 20261016, nominal 80/10/10
  assignments, and a 10-arcsec border exclusion on all blocks. Validation
  confirms no accepted cross-split objects within 20 arcsec. Actual split
  counts depend on sky coverage and quality cuts.

## Files and limitations

sample.h5 holds images, masks, RMS, DQ, object IDs, split indices, and scalar
calibration/background metadata. objects.csv maps every HDF5 row to its
catalog ID, coordinates, source files, source-pixel origin, and selection
properties. candidates.csv and rejected.csv preserve the selection history.
manifest.json lists source checksums and policies; source_headers holds the
FITS headers needed to reconstruct each WCS. nisp_psf holds the supplied
stack PSF models; inspect their original sampling headers before use.

JWST inputs here are SCI-only: no JWST RMS or full DQ map was available.
Its mask is therefore a coverage proxy; noise estimates from the stamp border
are diagnostic scalars, not precision per-pixel uncertainties. WCS alignment
passes geometry tests and image checks, but is not a demonstrated subpixel
astrometric calibration for every galaxy. Bilinear resampling adds correlated
noise and affects the effective JWST PSF. The bands differ; no color correction,
PSF deconvolution, or shear calibration is performed. This package fixes the
identified extraction defects; it is not a claim that all scientific
systematics are solved. See validation.json for the checks actually run.
'''
    (directory/'README.md').write_text(text)
    sums=[]
    for p in sorted(directory.rglob('*')):
        if p.is_file() and p.name!='SHA256SUMS' and '__pycache__' not in p.parts:
            sums.append(f'{sha256(p)}  {p.relative_to(directory).as_posix()}')
    (directory/'SHA256SUMS').write_text('\n'.join(sums)+'\n')
    archive=directory.with_suffix('.tar')
    if archive.exists():
        raise FileExistsError(archive)
    # HDF5 and PSF files are already compressed. Avoid an expensive second gzip.
    with tarfile.open(archive,'w') as tar:
        tar.add(directory,arcname=directory.name,filter=lambda info: None if '__pycache__' in info.name else info)
    archive.with_suffix('.tar.sha256').write_text(f'{sha256(archive)}  {archive.name}\n')
    print(f'Transfer: {archive} ({archive.stat().st_size/1e9:.2f} GB)')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory',type=Path)
    package(parser.parse_args().directory)
