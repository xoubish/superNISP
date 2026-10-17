"""Freeze full-field train/validation source masks for the residual-loss ablation."""
import argparse
import json
from pathlib import Path
import shutil
import h5py
import numpy as np
from paired_sample import PairedSample
from source_masks import reference_masks, PROTOCOL
from analyze_psnr_ssim import sha

ROOT = Path(__file__).resolve().parents[1]


def main(args):
    args.output.mkdir(parents=True, exist_ok=False)
    contract = dict(splits=['train', 'validation'], threshold_sigma=2., protocol=PROTOCOL,
        geometry='Full 205x205 target field, target validity only, no cropping for training loss.',
        empty_regions='Renormalize 50/50 weights over present regions for that object; never drop objects.',
        selection='No predictions or reconstruction errors enter mask construction; test split not read.')
    (args.output/'contract.json').write_text(json.dumps(contract, indent=2)+'\n')
    coverage = {}
    target_path = args.output/'source_masks.h5'
    with h5py.File(target_path, 'w') as out:
        out.attrs['complete'] = False
        for split in contract['splits']:
            dataset = PairedSample(args.data, split, 'linear')
            group = out.create_group(split)
            group.create_dataset('rows', data=dataset.indices)
            ids = group.create_dataset('object_ids', (len(dataset),), dtype='i8')
            masks = group.create_dataset('source', (len(dataset), 205, 205), dtype='bool',
                                        chunks=(1, 205, 205), compression='lzf')
            counts, fractions = [], []
            for i in range(len(dataset)):
                item = dataset[i]
                target = dataset.jwst_to_mjy_sr(item['jwst']).numpy()[0]
                valid = item['jwst_valid'].numpy()[0]
                mask, _ = reference_masks(target, valid, thresholds=(2.,))
                masks[i] = mask[0]
                ids[i] = item['object_id']
                counts.append([int(mask.sum()), int(valid.sum()-mask.sum())])
                fractions.append(float(mask.sum()/valid.sum()))
                if (i+1) % 1000 == 0 or i+1 == len(dataset):
                    print(split, i+1, '/', len(dataset), flush=True)
            group.create_dataset('region_counts', data=counts)
            counts = np.asarray(counts)
            coverage[split] = dict(objects=len(dataset), empty_source=int((counts[:,0] == 0).sum()),
                empty_outside=int((counts[:,1] == 0).sum()), median_source_fraction=float(np.median(fractions)))
            dataset.close()
        out.attrs['complete'] = True
    for name in ['prepare_refinement_masks.py', 'source_masks.py', 'paired_sample.py']:
        shutil.copy2(ROOT/'code'/name, args.output/name)
    # The new validation source geometry must reproduce the published crop masks.
    with np.load(ROOT/'runs/figure6-source-masks-v03/source_masks.npz') as saved, h5py.File(target_path) as full:
        index = saved['thresholds'].tolist().index(2.)
        assert full['validation/object_ids'][:].tolist() == saved['object_ids'].tolist()
        np.testing.assert_array_equal(full['validation/source'][:,50:155,50:155] & saved['common'],
                                      saved['source'][:,index])
    record = dict(complete=True, contract=contract, coverage=coverage,
        masks_sha256=sha(target_path), validation_matches_frozen_figure6_masks=True,
        data_hashes={name:sha(args.data/name) for name in ['manifest.json','normalization.json','SHA256SUMS']},
        source_hashes={p.name:sha(p) for p in args.output.glob('*.py')})
    (args.output/'metadata.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(coverage), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    main(p.parse_args())
