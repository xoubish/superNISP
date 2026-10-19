"""Held-out evaluation of the frozen ResNet on a sample split (CPU).

Reproduces the paper's measurement protocol: common validity mask, 2-sigma
JWST source masks, central 105x105 crop, fixed weighted moments, GalSim
adaptive moments, native NISP from the 21x21 central pixels, and paired
spatial-block bootstrap intervals.  Run on 'validation' first to confirm
the published numbers, then on 'test'.
"""
import argparse, csv, json, sys
from pathlib import Path
import numpy as np, torch, h5py
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'code'))
from paired_sample import PairedSample
from residual_baseline import EuclidToJWSTSuperResolution
from shape_diagnostics import interpolation, diagnose_moments
from robust_shape_measurements import adaptive_moments
from source_masks import reference_masks, PROTOCOL
from reconstruction_benchmark import pixel_errors

def ts(x, y):
    i, j = np.triu_indices(len(x), 1); dx = x[j]-x[i]; k = dx != 0
    s = np.median((y[j]-y[i])[k]/dx[k]); return float(s), float(np.median(y-s*x))

def block_boot(values_by_method, blocks, stat, repeats=2000, seed=20261016):
    labels, inv = np.unique(blocks, return_inverse=True)
    rng = np.random.default_rng(seed); out = []
    idx_by = [np.where(inv == b)[0] for b in range(len(labels))]
    for _ in range(repeats):
        idx = np.concatenate([idx_by[b] for b in rng.integers(len(labels), size=len(labels))])
        out.append(stat({m: v[idx] for m, v in values_by_method.items()}))
    return np.percentile(out, [2.5, 97.5], axis=0)

def main(a):
    torch.set_num_threads(a.threads)
    data = Path(a.data); ds = PairedSample(data, a.split, 'linear')
    cat = {int(r['row']): r for r in csv.DictReader((data/'objects.csv').open())}
    ck = torch.load(ROOT/'runs/figure5-v06/legacy-best.pt', map_location='cpu', weights_only=False)
    model = EuclidToJWSTSuperResolution(head=ck['signature']['config']['head']); model.load_state_dict(ck['model']); model.eval()
    n = len(ds); si = PROTOCOL['threshold_sigma'].index(PROTOCOL['primary_threshold_sigma'])
    rec = {k: [] for k in ['ids','blocks','src_rmse_legacy','src_rmse_bilinear','bg_rmse_legacy','bg_rmse_bilinear','src_px','bg_px']}
    crops = {k: np.zeros((n,105,105), 'f4') for k in ['target','legacy']}; common_crop = np.zeros((n,105,105), bool)
    native = {'img': np.zeros((n,21,21)), 'valid': np.zeros((n,21,21), bool)}
    with h5py.File(data/'sample.h5') as f, torch.inference_mode():
        for s in range(0, n, 32):
            items = [ds[i] for i in range(s, min(s+32, n))]
            x = torch.stack([it['nisp'] for it in items]); v = torch.stack([it['nisp_valid'] for it in items])
            tgt = ds.jwst_to_mjy_sr(torch.stack([it['jwst'] for it in items])); jv = torch.stack([it['jwst_valid'] for it in items])
            common = jv.clone(); outs = {}
            for mode in ['bilinear','bicubic']:
                outs[mode], sup = interpolation(x*ds.norm['nisp_scale'], v, mode); common &= sup
            outs['legacy'] = ds.jwst_to_mjy_sr(model(x))
            for b, it in enumerate(items):
                src, _ = reference_masks(tgt[b,0].numpy(), it['jwst_valid'][0].numpy())
                src = torch.from_numpy(src[si][50:155,50:155]) & common[b,0,50:155,50:155]
                bg = common[b,0,50:155,50:155] & ~src
                rec['ids'].append(it['object_id']); rec['blocks'].append(cat[it['row']]['spatial_block'])
                for m in ['legacy','bilinear']:
                    for kind, sup in [('src',src),('bg',bg)]:
                        val = pixel_errors(outs[m][b:b+1,...,50:155,50:155], tgt[b:b+1,...,50:155,50:155], sup[None,None])['rmse_mjy_sr'].item() if sup.any() else np.nan
                        rec[f'{kind}_rmse_{m}'].append(val)
                rec['src_px'].append(int(src.sum())); rec['bg_px'].append(int(bg.sum()))
                j = it['row']; conv = 1e-12/f['nisp_pixel_sr'][j]
                native['img'][s+b] = ((f['nisp'][j]-f['nisp_background'][j])*conv)[10:31,10:31]
                native['valid'][s+b] = f['nisp_valid'][j].astype(bool)[10:31,10:31]
            crops['target'][s:s+len(items)] = tgt[:,0,50:155,50:155].numpy(); crops['legacy'][s:s+len(items)] = outs['legacy'][:,0,50:155,50:155].numpy()
            common_crop[s:s+len(items)] = common[:,0,50:155,50:155].numpy()
            if s % 320 == 0: print(f'{a.split}: {s+len(items)}/{n}', flush=True)
    # shapes
    shapes = {}
    for m, img, msk, step in [('target',crops['target'],common_crop,1.),('legacy',crops['legacy'],common_crop,1.),('native',native['img'],native['valid'],5.)]:
        ti = torch.from_numpy(np.asarray(img, 'f8'))[:,None]; tm = torch.from_numpy(msk)[:,None]
        fx = diagnose_moments(ti, tm, pixel_step=step); ad = adaptive_moments(ti, tm, pixel_step=step)
        shapes[m] = dict(fixed_e=fx['e'].numpy(), fixed_good=fx['good'].numpy(), adaptive_e=np.asarray(ad['e']), adaptive_good=np.asarray(ad['good']))
    blocks = np.array(rec['blocks']); report = dict(split=a.split, objects=n, checkpoint='runs/figure5-v06/legacy-best.pt', regional={}, shapes={})
    for kind in ['src','bg']:
        keep = ~np.isnan(rec[f'{kind}_rmse_legacy'])
        vals = {m: np.array(rec[f'{kind}_rmse_{m}'])[keep] for m in ['legacy','bilinear']}
        ci = block_boot(vals, blocks[keep], lambda d: [d['legacy'].mean(), d['bilinear'].mean()])
        report['regional'][kind] = dict(objects=int(keep.sum()), mean_rmse={m: float(vals[m].mean()) for m in vals}, ci95={m: ci[:,i].tolist() for i,m in enumerate(vals)})
    for case in ['fixed','adaptive']:
        g = {m: shapes[m][f'{case}_good'] for m in shapes}; com = g['target'] & g['native'] & g['legacy']
        T = shapes['target'][f'{case}_e']; E = {m: shapes[m][f'{case}_e'] for m in ['native','legacy']}
        sq = {m: ((E[m]-T)**2).mean(1)[com] for m in E}
        red = lambda d: 100*(1-np.sqrt(d['legacy'].mean())/np.sqrt(d['native'].mean()))
        ci = block_boot(sq, blocks[com], red)
        x = T[com].ravel(); slopes = {}
        for m in E:
            y = E[m][com].ravel(); c = np.abs(x) < .5; slopes[m] = ts(x[c], y[c])
        report['shapes'][case] = dict(success={m: int(g[m].sum()) for m in g}, common=int(com.sum()),
            rmse={m: float(np.sqrt(sq[m].mean())) for m in sq}, reduction_percent=float(red(sq)), reduction_ci95=ci.tolist(),
            theil_sen_slope_intercept=slopes)
    out = Path(a.output); out.mkdir(exist_ok=True)
    np.savez_compressed(out/f'{a.split}_measurements.npz', object_ids=np.array(rec['ids']), blocks=blocks, common=common_crop,
        **{f'{m}__{k}': v for m, d in shapes.items() for k, v in d.items()}, **{k: np.array(v) for k, v in rec.items() if k not in ('ids','blocks')})
    (out/f'{a.split}_report.json').write_text(json.dumps(report, indent=2)+'\n'); print(json.dumps(report, indent=2))

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('split', choices=['validation','test']); p.add_argument('--data', default=str(ROOT.parent/'supernisp_nir_y_f115w_v1'))
    p.add_argument('--output', default=str(Path(__file__).resolve().parent)); p.add_argument('--threads', type=int, default=16)
    main(p.parse_args())
