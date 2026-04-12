# SuperNISP Documentation

## Project Summary

SuperNISP is a research codebase for astronomical image super-resolution. Its goal is to learn a mapping from low-resolution Euclid NISP near-infrared galaxy cutouts to higher-resolution JWST NIRCam cutouts at comparable wavelengths.

The scientific motivation is weak-lensing shape measurement in the infrared. Euclid weak-lensing work typically relies on the higher-resolution VIS channel, but reliable shape recovery from NISP would expand the useful galaxy sample and improve cosmological constraints. This project explores whether deep learning can recover JWST-like high-resolution morphology from Euclid NISP images.

At a high level, the repository contains:

- Data preparation tools that select galaxies from COSMOS/HUDF catalogs, match them to Euclid and JWST image tiles, and extract paired cutouts.
- Several PyTorch model families for 5x super-resolution, mostly from `41 x 41` Euclid/NISP inputs to `205 x 205` JWST/NIRCam targets.
- Training scripts for residual dense convolutional networks, spatial-transformer variants, and an SR3-style conditional diffusion model.
- Notebooks and older experiments used for data inspection, debugging, and model prototyping.

## Scientific Data

The project uses paired galaxy cutouts from:

- **Low-resolution input:** Euclid NISP Y-band images.
- **High-resolution target:** JWST NIRCam F115W images.
- **Catalogs:** COSMOS2020 and CANDELS/HUDF-style catalogs for source selection and sky coordinates.

The active data products in `data/` include NumPy and HDF5 training arrays such as:

- `data/euclid_NIR_cosmos_41px_Y.npy`
- `data/jwst_cosmos_205px_F115W.npy`
- `data/euclid_NIR_cosmos_41px_Y_20251124.npy`
- `data/jwst_cosmos_205px_F115W_20251124.npy`
- `data/Nisp_train_cosmos.hdf5`
- `data/Nircam_train_cosmos.hdf5`

The `catalog/` directory stores FITS catalogs and matched catalog CSVs, including COSMOS2020 and HUDF/CANDELS-related inputs. The `data/` and `catalog/` directories are very large local research artifacts, not lightweight source files.

## Data Preparation Pipeline

The main data-preparation code is in `code/clipping.py`.

It provides a pipeline for:

1. Reading source catalogs.
2. Applying galaxy selection cuts.
3. Finding which JWST and Euclid FITS tiles contain each source.
4. Matching galaxies that appear in both instruments.
5. Extracting aligned cutouts around each galaxy.
6. Optionally masking, rotating, mirroring, or deconvolving cutouts.
7. Saving paired JWST and NISP arrays as `.npy` files.

Important functions:

- `cut_catalog()` selects COSMOS sources using catalog properties such as magnitude, photometric redshift, galaxy type, and flux radius.
- `cut_catalog2()` performs a similar selection for HUDF/CANDELS-style catalogs.
- `match_catalog()` uses FITS WCS information to determine which sources fall inside an image.
- `clip_images()` extracts paired JWST and NISP `Cutout2D` image stamps.
- `process_all()` orchestrates the full workflow for a given field and Euclid data type.

The `meta` dictionary in `code/clipping.py` defines supported field/instrument configurations. The current configurations cover `cosmos` and `HUDF`, with Euclid variants `NISP-Y` and `NISP-Y_MER`.

## Main Super-Resolution Model

The most direct active training path is:

- `code/train_claude.py`
- `code/claudemodel.py`
- `code/dataset.py`

This path trains `EuclidToJWSTSuperResolution`, a residual dense convolutional model.

### Input and Output

The model expects single-channel astronomical cutouts:

- Input: Euclid/NISP image, usually `1 x 41 x 41`.
- Output: JWST/NIRCam-like image, usually `1 x 205 x 205`.

The scaling factor is 5x.

### Architecture

`EuclidToJWSTSuperResolution` in `code/claudemodel.py` uses:

- An initial convolution to lift the 1-channel image into feature space.
- Residual dense blocks for feature extraction.
- A trunk residual connection.
- A lightweight detail-enhancement block.
- PixelShuffle-based direct 5x upsampling.
- Final convolution layers to reconstruct a single-channel high-resolution image.

The model is inspired by ESRGAN/RRDB-style super-resolution, simplified for single-band astronomical cutouts.

### Training Strategy

`train_two_stage()` in `code/claudemodel.py` performs two training phases:

1. **Stage 1:** Initial reconstruction training using a weighted combination of L1 and MSE loss.
2. **Stage 2:** Fine-tuning with L1, MSE, and SSIM loss.

The script logs metrics and image triplets to Weights & Biases:

- Low-resolution input resized for display.
- Super-resolved model output.
- JWST high-resolution target.

It also computes PSNR and SSIM for visual comparison logging.

## Dataset Loaders

There are two active dataset implementations.

### `code/dataset.py`

This is the HDF5 loader used by the main RRDB-style training path.

It expects HDF5 files with split-specific datasets such as:

- `train_keys`
- `train_img`
- `val_img` or `test_img`, depending on usage

It opens HDF5 files lazily per worker to support PyTorch `DataLoader` multiprocessing.

### `code/diffusion/dataset.py`

This loader supports both HDF5 and NumPy inputs. It adds:

- Optional train/test splitting for NumPy arrays.
- Optional center cropping.
- An invertible global asinh normalization scheme.

The asinh normalization estimates a global sky scale and maps astronomical flux values into a more stable range for diffusion training.

## Alternate Convolutional Model With Alignment

The files:

- `code/claude_model_NIR.py`
- `code/train_claude_NIR.py`

define another super-resolution pathway that includes spatial alignment.

`EuclidJWSTSuperResolution` uses:

- An `ImprovedSpatialTransformer` to learn small translations before super-resolution.
- Multi-scale feature extraction.
- Enhanced residual blocks with squeeze-and-excitation channel attention.
- Progressive PixelShuffle upsampling from `41 x 41` to `164 x 164`.
- Bilinear interpolation to the final `205 x 205` target.

This pathway uses `AstronomicalLoss`, which combines:

- L1 reconstruction loss.
- First-order gradient loss.
- Second-order gradient loss.
- Translation regularization.

`train_claude_NIR.py` also includes normalization options such as adaptive histogram normalization, flux-preserving normalization, percentile normalization, and z-score normalization.

## W&B Sweep Model

The files:

- `code/claude_model_NIR_2.py`
- `code/claude_sweep.py`

define a Weights & Biases hyperparameter sweep workflow for the RRDB-style model.

The sweep varies:

- Number of residual dense blocks.
- Feature count.
- Batch size.
- Stage 1 and Stage 2 learning rates.
- Loss weights.
- Weight decay.
- Gradient clipping.

The sweep objective is a validation loss combining L1 and SSIM-style performance. The best model and configuration are saved under `models/` paths referenced by the script.

## SR3 Diffusion Model

The diffusion branch lives in `code/diffusion/`.

Important files:

- `code/diffusion/model_sr3.py`
- `code/diffusion/dataset.py`
- `code/diffusion/train_sr3.py`
- `code/diffusion/sweep.yaml`

This branch implements an SR3-style conditional diffusion model for super-resolution.

### Model Design

`SR3SuperResolution` wraps a conditional `SR3UNet`.

The UNet:

- Uses sinusoidal timestep embeddings.
- Conditions on the low-resolution image upsampled to the high-resolution grid.
- Uses FiLM-style modulation from both timestep embeddings and condition features.
- Pads images to multiples of 8 internally and crops outputs back to the requested size.

The diffusion wrapper:

- Uses a cosine beta schedule.
- Trains the model to predict noise.
- Reconstructs `x0` for reconstruction and perceptual losses.
- Provides a sampling loop for deterministic or stochastic super-resolution generation.

### Training

`train_sr3.py` trains the diffusion model with:

- Weighted epsilon prediction loss.
- L1 loss on reconstructed `x0`.
- VGG16 perceptual loss.
- Gradient accumulation.
- Reduce-on-plateau learning-rate scheduling.
- Weights & Biases logging.
- Checkpoint artifacts.

The current script crops the standard `41 x 41` to `21 x 21` and `205 x 205` to `105 x 105`, then trains a 5x model on these smaller patches.

## PSF Utilities

`code/euclid_psf.py` defines a small `EuclidPSF` helper for reading Euclid PSF files, locating the nearest PSF stamp to a sky coordinate, and returning a PSF cutout. This supports experiments around PSF-aware preprocessing or deconvolution.

`code/clipping.py` also contains optional PSF-related logic for matching Euclid image files to PSF files and deconvolving NISP cutouts.

## Older Experiments

The `code/older/` directory contains previous model and training attempts. These include:

- Earlier diffusion models.
- Attention-based diffusion variants.
- RRDB/enhanced super-resolution prototypes.
- Older dataset loaders.
- Test scripts and notebooks.

These files are useful historical references but are not the clearest entry point for new work. The active paths are the root-level `code/*.py` files and the `code/diffusion/` branch.

## Notebooks and Visual Artifacts

The repository includes notebooks for exploration and debugging:

- `code/process_data.ipynb`
- `code/data_access.ipynb`
- `code/test_model.ipynb`
- `code/test_model-Aug20.ipynb`
- `code/ellipticity.ipynb`
- `code/diffusion/diffusion_test.ipynb`
- `code/older/SR_GAN.ipynb`
- `code/older/SR_GAN_old.ipynb`
- `code/older/borders.ipynb`
- `code/older/search_coords.ipynb`

There are also output images such as `code/claude_result.png`, `code/older/supernisp.png`, and `code/diffusion/figure_2_ch3.png`.

## Repository Layout

```text
.
|-- README.md
|-- documentation.md
|-- catalog/
|   |-- COSMOS2020 and HUDF/CANDELS catalog files
|   `-- matched catalog CSVs
|-- data/
|   |-- Euclid/NISP FITS files
|   |-- JWST/NIRCam and Euclid cutout arrays
|   |-- HDF5 training datasets
|   `-- PSF files and download scripts
`-- code/
    |-- clipping.py                 # data matching and cutout extraction
    |-- dataset.py                  # HDF5 dataset for RRDB training
    |-- claudemodel.py              # main RRDB-style model and two-stage training
    |-- train_claude.py             # main HDF5 training entry point
    |-- claude_model_NIR.py         # spatial-transformer model variant
    |-- train_claude_NIR.py         # CLI training for spatial-transformer variant
    |-- claude_model_NIR_2.py       # sweep-oriented RRDB variant
    |-- claude_sweep.py             # W&B sweep configuration
    |-- euclid_psf.py               # Euclid PSF helper
    |-- losses.py                   # simple hybrid loss helper
    |-- diffusion/
    |   |-- dataset.py              # diffusion dataset and asinh normalization
    |   |-- model_sr3.py            # SR3 diffusion model
    |   |-- train_sr3.py            # SR3 training script
    |   `-- sweep.yaml              # SR3 sweep config
    `-- older/
        `-- archived experiments and prototype models
```

## Typical Workflows

### Generate Paired Cutouts

Use `code/clipping.py` and call `process_all()` for a configured field and Euclid data type.

Example intent:

```python
from clipping import process_all

jwst_cutouts, nisp_cutouts = process_all(
    field="cosmos",
    euclid_type="NISP-Y",
    save_cat=True,
    save_clips=True,
)
```

This expects access to the configured catalogs, FITS files, and Dropbox token path if using the Dropbox-based download flow.

### Train the Main RRDB Model

Run the main training script from the `code/` directory:

```bash
python train_claude.py
```

It uses W&B project `RRDB-twostage` and the HDF5 paths:

```text
../data/Nisp_train_cosmos.hdf5
../data/Nircam_train_cosmos.hdf5
```

### Train the Spatial-Transformer Variant

Run:

```bash
python train_claude_NIR.py \
  --euclid_data ../data/euclid_NIR_cosmos_41px_Y.npy \
  --jwst_data ../data/jwst_cosmos_205px_F115W.npy
```

This path saves checkpoints, training curves, sample prediction images, and a JSON config into the configured output directory.

### Train the SR3 Diffusion Model

Run from `code/diffusion/`:

```bash
python train_sr3.py
```

The script currently hard-codes dated COSMOS NumPy arrays:

```text
../../data/euclid_NIR_cosmos_41px_Y_20251124.npy
../../data/jwst_cosmos_205px_F115W_20251124.npy
```

It logs training metrics and visualization images to W&B and writes checkpoints to `checkpoints_sr3/`.

## Dependencies

There is no single dependency manifest in the repository. Based on imports, the project uses:

- Python
- PyTorch
- torchvision
- NumPy
- h5py
- OpenCV
- scikit-image
- scipy
- pandas
- matplotlib
- astropy
- photutils
- reproject
- tqdm
- wandb
- dropbox
- scikit-learn

The diffusion perceptual loss downloads or loads VGG16 weights through `torchvision.models`, so that path may require internet access or a pre-populated model cache.

## Current Caveats

- Several scripts contain hard-coded relative paths. They are easiest to run from their own directories.
- `data/` and `catalog/` are large and local; moving the project requires preserving those paths or updating script configuration.
- There are multiple model variants with overlapping names. The clearest current entry points are `train_claude.py`, `train_claude_NIR.py`, and `diffusion/train_sr3.py`.
- Some sweep/config files reference older script names, such as `train.py`, and may need path updates before use.
- There is no centralized package structure or dependency file yet.
- Some older files are duplicated or experimental and should be treated as historical context rather than production code.

## What the Project Does End-to-End

End-to-end, SuperNISP builds supervised training pairs from real astronomical survey data, then trains neural networks to transform Euclid NISP galaxy images into JWST-like higher-resolution images. The intended output is a super-resolved NIR image that better preserves galaxy morphology and fine structure, with the long-term science goal of enabling more accurate infrared shape measurements for weak-lensing analyses.
