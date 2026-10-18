"""Render the observed-shape-only shear pilot from frozen, verified results."""
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'paper/diagnostics'
SOURCE = OUT / 'shear_response_review_v01.json'
SOURCE_SHA256 = '10c06875e67a6c15fe08ac94e4f9323d702d78d4e83c08777e0864dd611579ca'
STYLES = [
    ('native_observed', 'Native observed', '0.4', 'o'),
    ('resnet_observed', 'ResNet observed', '#4477aa', 's'),
    ('target_observed', 'High-resolution control', '#cc8833', 'D'),
]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    assert sha(SOURCE) == SOURCE_SHA256
    report = json.loads(SOURCE.read_text())
    assert report['complete'] and report['original_primary_comparison_reproduced']
    fig, axes = plt.subplots(2, 4, figsize=(13.8, 6.3), sharex=True, sharey=True)
    for i, psf in enumerate([.45, .60]):
        for j, snr in enumerate([None, 20, 50, 100]):
            ax = axes[i, j]
            rows = [r for r in report['cells'] if r['psf_fwhm'] == psf and r['snr'] == snr]
            assert [r['half_light_radius'] for r in rows] == [.18, .35, .65]
            for name, label, color, marker in STYLES:
                assert all(r['methods'][name]['same_sample_available'] for r in rows)
                ax.plot([r['half_light_radius'] for r in rows],
                        [r['methods'][name]['relative_trace_response'] for r in rows],
                        color=color, marker=marker, label=label, lw=1.5, ms=5)
            ax.axhline(1, color='black', lw=.8, ls=':')
            ax.set_title(('Noiseless' if snr is None else f'Nominal S/N {snr}')
                         + f'\nPSF FWHM {psf:.2f} arcsec', fontsize=10)
            ax.text(.98, .04, 'Common: ' + ', '.join(
                f"{r['observed_common']}/{r['attempted']}" for r in rows),
                transform=ax.transAxes, ha='right', fontsize=7)
            ax.set_xticks([.18, .35, .65])
            ax.set_ylim(.2, 1.18)
            ax.grid(alpha=.15)
            if i == 1:
                ax.set_xlabel('Half-light radius [arcsec]')
            if j == 0:
                ax.set_ylabel('Response / intrinsic response')
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=3, frameon=False, fontsize=10)
    fig.text(.5, .012,
             'Synthetic pilot • common observed-shape sample • no calibrated shear or uncertainty claim',
             ha='center', fontsize=10)
    fig.tight_layout(rect=(0, .04, 1, .93))
    exports = {}
    for ext in ['png', 'pdf']:
        path = OUT / f'shear_response_review_v02.{ext}'
        fig.savefig(path, dpi=170, bbox_inches='tight')
        exports[str(path.relative_to(ROOT))] = sha(path)
    plt.close(fig)
    provenance = dict(
        numerical_source=str(SOURCE.relative_to(ROOT)), numerical_source_sha256=sha(SOURCE),
        plotting_source=str(Path(__file__).resolve().relative_to(ROOT)), plotting_source_sha256=sha(__file__),
        displayed_methods=[s[0] for s in STYLES],
        change='Display only: omit native REGAUSS curve, legend entry, and failure annotations. '
               'Keep all numerical results, observed-common samples, scales, and remaining curves unchanged.',
        observed_common=report['observed_common'], exports_sha256=exports,
    )
    (OUT / 'shear_response_review_v02.json').write_text(json.dumps(provenance, indent=2) + '\n')
    print(json.dumps(provenance, indent=2))


if __name__ == '__main__':
    main()
