"""Freeze reference-derived masks before evaluating any model errors (CPU)."""
import argparse
import json
from pathlib import Path
import shutil

import numpy as np
import torch
from paired_sample import PairedSample
from shape_diagnostics import interpolation
from source_masks import reference_masks, PROTOCOL
from train_paired import sha256


def main(args):
    torch.set_num_threads(2)
    args.output.mkdir(parents=True, exist_ok=False)
    dataset = PairedSample(args.data, 'validation', 'linear')
    masks, common_masks, diagnostics, ids = [], [], [], []
    for i in range(len(dataset)):
        item = dataset[i]
        target = dataset.jwst_to_mjy_sr(item['jwst']).numpy()[0]
        source, diagnostic = reference_masks(target, item['jwst_valid'].numpy()[0])
        common = item['jwst_valid'][None].clone()
        for mode in ['bilinear', 'bicubic']:
            _, support = interpolation(item['nisp'][None], item['nisp_valid'][None], mode)
            common &= support
        common = common.numpy()[0, 0, 50:155, 50:155]
        masks.append(source[:, 50:155, 50:155] & common[None])
        common_masks.append(common)
        diagnostics.append(diagnostic)
        ids.append(item['object_id'])
        if (i+1) % 200 == 0:
            print('Masks', i+1, '/', len(dataset), flush=True)
    masks = np.asarray(masks)
    common_masks = np.asarray(common_masks)
    source_counts = masks.sum(axis=(-2, -1))
    common_counts = common_masks.sum(axis=(-2, -1))
    output = args.output/'source_masks.npz'
    np.savez_compressed(output, source=masks, common=common_masks, object_ids=ids,
                        rows=dataset.indices, thresholds=PROTOCOL['threshold_sigma'])
    root = Path(__file__).resolve().parents[1]
    for name in ['source_masks.py', 'prepare_source_masks.py']:
        shutil.copy2(root/'code'/name, args.output/name)
    report = dict(complete=True, split='validation', protocol=PROTOCOL, object_ids=ids,
        rows=dataset.indices.tolist(), diagnostics=diagnostics, npz_sha256=sha256(output),
        source_hashes={n: sha256(args.output/n) for n in ['source_masks.py', 'prepare_source_masks.py']},
        reference_data_hashes={n:sha256(args.data/n) for n in ['sample.h5','normalization.json','objects.csv']},
        coverage=[dict(threshold=t, source_objects=int((source_counts[:,j]>0).sum()),
            background_objects=int((source_counts[:,j]<common_counts).sum()),
            median_source_fraction=float(np.median(source_counts[:,j]/common_counts)),
            fraction_percentiles=np.percentile(source_counts[:,j]/common_counts,[0,5,25,50,75,95,100]).tolist())
            for j,t in enumerate(PROTOCOL['threshold_sigma'])])
    (args.output/'source_masks.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report['coverage'],indent=2),flush=True)
    dataset.close()


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,default=Path(__file__).resolve().parents[2]/'supernisp_nir_y_f115w_v1')
    parser.add_argument('--output',type=Path,required=True)
    main(parser.parse_args())
