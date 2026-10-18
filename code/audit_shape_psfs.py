"""Audit local PSF assets without interpreting an empirical ePSF as an optical PSF."""
import hashlib
import json
from pathlib import Path

import galsim
import numpy as np
from astropy.io import fits

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT.parent / 'supernisp_nir_y_f115w_v1'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def audit():
    manifest = json.loads((DATA/'manifest.json').read_text())
    records = []
    for path in sorted((DATA/'nisp_psf').glob('*.fits.gz')):
        expected = next(r for r in manifest['source_files'] if Path(r['path']).name == path.name)
        assert sha(path) == expected['sha256'], path
        with fits.open(path) as hdus:
            a = hdus[1].data.astype(float)
            h, primary = hdus[1].header, hdus[0].header
            scale_xy = [abs(h[f'CD{i}_{i}'])*3600 for i in (1, 2)]
            mom = galsim.hsm.FindAdaptiveMom(galsim.ImageD(a), strict=False)
            records.append(dict(path=str(path), sha256=sha(path),
                hdu_names=[d.name for d in hdus], shape=list(a.shape),
                oversampling=primary['OVERSAMP'], scale_arcsec=scale_xy,
                sum=float(a.sum()), minimum=float(a.min()), negative_pixels=int((a<0).sum()),
                negative_absolute_flux_fraction=float(-a[a<0].sum()/a.sum()),
                header_ree50_arcsec=h['REE50'], header_ree80_arcsec=h['REE80'],
                adaptive_status=int(mom.moments_status),
                adaptive_e=[mom.observed_shape.e1,mom.observed_shape.e2],
                adaptive_sigma_arcsec=float(mom.moments_sigma*np.sqrt(np.prod(scale_xy))),
                manifest_sha256_verified=True))
    jwst = [str(p) for root in [ROOT/'data', DATA] for p in root.rglob('*')
            if p.is_file() and 'psf' in p.name.lower() and
            any(s in p.name.lower() for s in ['jwst','nircam','f115w'])]
    return dict(nisp=records, local_jwst_psf_candidates=jwst,
        searched_roots=[str(ROOT/'data'),str(DATA)],
        findings=[
            'Four NISP stack-level 61x61 PSFs, oversampling 6, approximately 0.04945 arcsec per PSF pixel.',
            'The existing EuclidPSF loader expects HDU 2 plus STMPSIZE and a spatial atlas; these files have only PRIMARY and STACKED.PSF. It cannot load these products.',
            'PSF array sums are approximately 36.5, not one. Preserve signed pixels and explicitly normalize any experimental use.',
            'Header spatial parameter count POLNAXIS=0: no spatially varying PSF supplied by these files.',
            'Pixel-response and stack-resampling conventions require provenance documentation before deciding whether/how to add pixel integration. Do not double-convolve the detector response.',
            'No corresponding local F115W PSF found. A historical filename in clipping.py is not an available calibrated mosaic PSF.',
            'No single effective PSF is established for the nonlinear learned reconstruction.',
        ], source_sha256=sha(__file__), manifest_sha256=sha(DATA/'manifest.json'))


if __name__ == '__main__':
    out = ROOT/'paper/diagnostics/psf_audit_v01.json'
    if out.exists():
        raise FileExistsError(out)
    out.write_text(json.dumps(audit(), indent=2)+'\n')
    print(out)
