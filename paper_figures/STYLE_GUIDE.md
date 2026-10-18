# Shared scientific figure style

The active Figures 1–9 use `matisse-inspired-v01` in `figure_utils.py`.
The palette takes its direction from Matisse's use of saturated pigments:
cobalt, deep green, ochre, and warm earth accents. It is a custom palette,
not a canonical Matisse scheme or a sampling of a particular artwork.
Pink and bright red categorical colors were removed at the user's request.

## Typography and layout

Use Nimbus Sans for labels, ticks, legends, annotations, and mathematical
notation. PDF fonts are embedded; SVG text remains editable. Keep the background
white, axes restrained, grids light, and information-bearing lines strong.
Use consistent sizes within a figure and size each figure for its intended
manuscript width. Avoid decorative cards, shadows, gradients, and tinted panels.
Architecture diagrams use square nodes, semantic colors, and contrasting text.

## Semantic colors

| Meaning | Color | Symbol where applicable |
| --- | --- | --- |
| ResNet | cobalt `#2455A4` | square |
| ResNet-D5 | deep green `#23836F` | upward triangle |
| Diffusion, one realization | ochre `#B58A16` | circle |
| Diffusion, mean of three | violet `#77649B` | filled plus |
| Bilinear | umber `#9C6543` | diamond |
| Bicubic | sky blue `#52A7C1` | downward triangle |
| Native NISP | charcoal `#555555` | circle |
| JWST / high-resolution reference | warm orange `#D9822B` | diamond |

Method identity is repeated in symbols, labels, and/or line patterns; color alone
must not distinguish overlaid curves. Figure 8 uses dotted circles, solid squares,
and dashed triangles. Its fixed/adaptive coverage bars use orange and hatched
charcoal. Figure 6's source/outside symbols remain filled circles/open squares;
its spectral curves repeat method markers and use solid/dashed region lines.
Figures 5 and 7 identify method columns with thin colored header rules.

The sample split palette is cobalt for training, orange for validation, and
green for test; map symbols and histogram line styles differ as well.
In schematics, blue identifies learned operations, green identifies the D5
branch, ochre identifies diffusion quantities, charcoal identifies NISP, and
orange identifies the F115W target/prediction.

## Scientific image and density scales

Keep astronomical images grayscale with the original stretches and masks.
Signed residual images keep their established symmetric `RdBu_r` scale; these
colors encode residual sign rather than categorical model identity. Figure 7
uses one common logarithmic count scale and a `cividis_r` colorbar for every
hexbin panel. Counts, binning, fits, ranges, and scientific populations are
unchanged. Do not independently normalize each panel's density colors.

## Review and reproduction

Run the nine active notebooks, then `python paper_figures/review_visual_style.py`.
The latter creates a nine-page PDF, a contact sheet, and grayscale/protanopia/
deuteranopia/tritanopia proof sheets for Figures 6 and 8. These use the severity-100
[Machado matrices](https://github.com/njsmith/colorspacious/blob/master/colorspacious/cvd.py)
on linear-light sRGB, with out-of-gamut display values clipped. They support
visual review; they do not certify universal accessibility. In particular,
cobalt/green become similar under tritanopia, so marker and line differences
are essential. Figure 6 also keeps each method explicitly labeled in its RMSE panel.

Original executed notebooks and helpers are in `revisions/style_before_v01/`.
The retired second Figure 9 notebook is
`revisions/09_shape_robustness_retired.ipynb`; only `09_shear_response.ipynb`
remains active. Old numerical bundles and experiments are unchanged.

Two portable-input limitations are handled explicitly:

- Figure 1's original mosaics and physical cutout cache are absent here. Its
  notebook reuses the exact embedded raster panels from the hash-verified v07
  SVG, with original orientation and stretch. Labels and footprints are rendered
  afresh. Changing image stretch requires the physical data on the mosaic machine.
- Figure 2's full parent catalog is absent here. `recover_parent_histograms.py`
  recovered exact integer bin counts from the hash-verified v05 SVG. The maximum
  residual before rounding was 0.000177 counts; all totals and six selected-sample
  histograms were independently verified using the local catalog rows. The frozen
  count cache records this provenance. No distribution was estimated or replaced.

Figure 7's supplementary bootstrap table is preserved from the previous executed
notebook, including its displayed precision; the style pass does not rerun the
statistical resampling. Current exports include notebook, helper, and input hashes.
