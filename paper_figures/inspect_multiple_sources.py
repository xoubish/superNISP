"""Observation-only candidate sheets for Figure 5; never loads model outputs.

Scan validation central crops for separated bright peaks to help find fields
with neighbors. Peaks are a browsing aid, not a galaxy/deblending catalogue.
"""
import argparse
import json
from pathlib import Path

import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter, maximum_filter

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT.parent / 'supernisp_nir_y_f115w_v1'
OUT = ROOT / 'paper_figures/inputs/figure5_multiple_sources_v05'


def main():
    OUT.mkdir(exist_ok=True)
    catalog = pd.read_csv(DATA / 'objects.csv').set_index('row')
    candidates = []
    with h5py.File(DATA / 'sample.h5', 'r') as f:
        for index, row in enumerate(f['validation_indices'][:]):
            row = int(row)
            valid = f['jwst_valid'][row, 50:155, 50:155].astype(bool)
            nvalid = f['nisp_valid'][row, 10:31, 10:31].astype(bool)
            if not valid.all() or not nvalid.all():
                continue
            observed = f['jwst'][row, 50:155, 50:155] - f['jwst_background'][row]
            smooth = gaussian_filter(observed, 2)
            peak_mask = smooth == maximum_filter(smooth, size=15)
            peak_mask[:10] = peak_mask[-10:] = False
            peak_mask[:, :10] = peak_mask[:, -10:] = False
            ys, xs = np.where(peak_mask & (smooth > .08))
            ranked = sorted(zip(smooth[ys, xs], ys, xs), reverse=True)
            peaks = []
            for value, y, x in ranked:
                if all(np.hypot(y-p[1], x-p[2]) >= 14 for p in peaks):
                    peaks.append((float(value), int(y), int(x)))
            if len(peaks) < 2 or peaks[1][0] / peaks[0][0] < .12:
                continue
            if not any(np.hypot(p[1]-52, p[2]-52) < 18 for p in peaks):
                continue
            c = catalog.loc[row]
            candidates.append(dict(index=index, row=row, object_id=int(f['object_id'][row]),
                magnitude=float(c.ACS_F814W_MAG), radius=float(c.FLUX_RADIUS), peaks=peaks,
                score=peaks[1][0]))
        candidates.sort(key=lambda c: c['score'], reverse=True)
        # Keep spatially distinct fields to avoid showing the same pair twice.
        distinct = []
        for c in candidates:
            coord = catalog.loc[c['row']]
            if any(np.hypot((coord.ra-catalog.loc[p['row']].ra)*np.cos(np.deg2rad(coord.dec)),
                            coord.dec-catalog.loc[p['row']].dec)*3600 < 5 for p in distinct):
                continue
            distinct.append(c)
        chosen = distinct[:64]
        (OUT / 'candidates.json').write_text(json.dumps(dict(
            split='validation', rule=__doc__, scan_count=len(f['validation_indices']),
            qualifying_fields=len(candidates), distinct_fields=len(distinct), candidates=chosen), indent=2)+'\n')
        for page in range((len(chosen)+15)//16):
            fig, axes = plt.subplots(4, 8, figsize=(16, 9))
            for ax in axes.flat:
                ax.axis('off')
            for position, c in enumerate(chosen[16*page:16*(page+1)]):
                row = c['row']
                target = f['jwst'][row, 50:155, 50:155]-f['jwst_background'][row]
                native = ((f['nisp'][row, 10:31, 10:31]-f['nisp_background'][row])
                          *1e-12/f['nisp_pixel_sr'][row])
                high = max(.05, float(np.percentile(target, 99.8)))
                stretch = max(.02, .08*high)
                for j, arr in enumerate([native, target]):
                    ax = axes[position//4, 2*(position%4)+j]
                    ax.imshow(np.arcsinh(arr/stretch), origin='lower', cmap='gray',
                              interpolation='nearest', vmin=np.arcsinh(-.015/stretch),
                              vmax=np.arcsinh(high/stretch))
                    title = f"{c['object_id']} | {c['magnitude']:.1f}" if j == 0 else f"JWST | {len(c['peaks'])} peaks"
                    ax.set_title(title, fontsize=9)
            fig.tight_layout(pad=.5)
            fig.savefig(OUT / f'candidates_{page+1}.png', dpi=140)
            plt.close(fig)
    print(f'{len(candidates)} qualifying fields; {len(distinct)} distinct; saved {len(chosen)} candidates to {OUT}')


def render_selection(path):
    """Render only observed images for a frozen selection; no model execution."""
    selection = json.loads(path.read_text())
    fig, axes = plt.subplots(len(selection['rows']), 2,
                             figsize=(5.2, 2.5*len(selection['rows'])), squeeze=False)
    with h5py.File(DATA / 'sample.h5', 'r') as f:
        validation_rows = set(f['validation_indices'][:].tolist())
        for i, row in enumerate(selection['rows']):
            assert row in validation_rows
            assert int(f['object_id'][row]) == selection['object_ids'][i]
            limit = selection['limits_mjy_sr'][i]
            target = f['jwst'][row, 50:155, 50:155]-f['jwst_background'][row]
            native = ((f['nisp'][row, 10:31, 10:31]-f['nisp_background'][row])
                      *1e-12/f['nisp_pixel_sr'][row])
            for j, values in enumerate([native, target]):
                ax = axes[i, j]
                ax.imshow(np.arcsinh(values/limit['stretch']), origin='lower', cmap='gray',
                          interpolation='nearest', vmin=np.arcsinh(limit['low']/limit['stretch']),
                          vmax=np.arcsinh(limit['high']/limit['stretch']))
                ax.set_xticks([])
                ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_visible(False)
                if i == 0:
                    ax.set_title(['NISP Y', 'JWST F115W'][j], fontsize=12)
            axes[i, 0].set_ylabel(f'({chr(97+i)})', rotation=0, labelpad=14, fontsize=11)
            axes[i, 1].text(.96, .04, str(selection['object_ids'][i]),
                            transform=axes[i, 1].transAxes, ha='right', va='bottom',
                            color='white', fontsize=8)
    fig.subplots_adjust(left=.07, right=.99, bottom=.01, top=.97, wspace=.025, hspace=.035)
    for extension in ['png', 'pdf']:
        fig.savefig(ROOT / f'paper_figures/exports/fig05_observed_selection_v05.{extension}',
                    bbox_inches='tight', dpi=240)
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--selection', type=Path, help='Render a frozen observed-only selection instead of scanning')
    args = parser.parse_args()
    if args.selection:
        render_selection(args.selection)
    else:
        main()
