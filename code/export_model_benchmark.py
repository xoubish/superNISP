"""Figure 6: matched validation accuracy and measured inference cost.

Use frozen Figure 5 checkpoints. Run in tmux; --local disables all W&B calls.
No training or test-set access. Save per-object errors, timing and provenance.
"""
import argparse
from contextlib import nullcontext
import csv
import json
from pathlib import Path
import shutil
import time
from types import SimpleNamespace

import numpy as np
import h5py
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
import wandb

from train_paired import persistent_session_required, seed_everything, sha256
from paired_sample import PairedSample
from residual_baseline import EuclidToJWSTSuperResolution
from diffusion_corrected import CorrectedDiffusion, fixed_noise
from shape_diagnostics import interpolation
from reconstruction_benchmark import pixel_errors, summarize

ROOT = Path(__file__).resolve().parents[1]
METHODS = ['bilinear','bicubic','diffusion','diffusion_mean3','legacy','direct5']


def main(args):
    persistent_session_required()
    torch.set_num_threads(2)
    seed_everything(20261016)
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    args.output.mkdir(parents=True, exist_ok=False)
    bundle_path = ROOT/'paper_figures/inputs'/f'model_benchmark_{args.version}.json'
    if bundle_path.exists():
        raise FileExistsError('Use a fresh version')
    source_meta = json.loads(args.checkpoints.read_text())
    dataset = PairedSample(args.data, 'validation', 'linear')
    data_hashes = {name:sha256(args.data/name) for name in source_meta['data_hashes']}
    assert data_hashes == source_meta['data_hashes']
    catalog = {int(r['row']):r for r in csv.DictReader((args.data/'objects.csv').open())
               if r['split'] == 'validation'}
    seeds = source_meta['sampling_seeds']
    steps = source_meta['sampling_steps']
    mask_bundle = mask_metadata = None
    if args.source_masks:
        if not args.local:
            raise ValueError('Source-mask experiment is local only')
        mask_metadata = json.loads(args.source_masks.with_suffix('.json').read_text())
        assert mask_metadata['complete']
        assert sha256(args.source_masks) == mask_metadata['npz_sha256']
        for name, digest in mask_metadata['reference_data_hashes'].items():
            assert sha256(args.data/name) == digest
        with np.load(args.source_masks) as saved:
            mask_bundle = {key:saved[key] for key in saved.files}
        assert mask_bundle['rows'].tolist() == dataset.indices.tolist()
    reused = json.loads(args.reuse_timing.read_text()) if args.reuse_timing else None
    if reused:
        assert reused['data_hashes'] == data_hashes
        assert reused['timing_protocol']['seeds'] == seeds
        assert reused['timing_protocol']['sampling_steps'] == steps
        for name in ['legacy','direct5','diffusion']:
            assert reused['checkpoints'][name]['sha256'] == source_meta['checkpoints'][name]['sha256']
    source_names = ['export_model_benchmark.py','reconstruction_benchmark.py',
                    'paired_sample.py','shape_diagnostics.py','diffusion_corrected.py',
                    'residual_baseline.py']
    source_hashes = {name:sha256(ROOT/'code'/name) for name in source_names}
    for name in source_names:
        shutil.copy2(ROOT/'code'/name, args.output/name)
    config = dict(version=args.version, data_hashes=data_hashes, source_hashes=source_hashes,
                  validation_objects=len(dataset), crop_target_pixels=105,
                  sampling_seeds=seeds, sampling_steps=steps, batch_size=args.batch_size,
                  source_masks=str(args.source_masks) if args.source_masks else None,
                  reused_timing=str(args.reuse_timing) if reused else None)
    context = (nullcontext(SimpleNamespace(url=None, log=lambda *a, **kw: None)) if args.local else
               wandb.init(project='superNISP-corrected', entity='AI-Astro', name=args.output.name,
                          mode='online', job_type='paper-reconstruction-benchmark',
                          dir=str(args.output), config=config))
    with context as run:
        if not args.local and (run.settings.mode != 'online' or not run.url):
            raise RuntimeError('Online W&B monitoring required before models')
        if run.url:
            (args.output/'wandb_url.txt').write_text(run.url+'\n')
        (args.output/'config.json').write_text(json.dumps(config|{'local_only':args.local},indent=2)+'\n')
        models, checkpoints = {}, {}
        for name in ['legacy','direct5','diffusion']:
            info = dict(source_meta['checkpoints'][name])
            source = ROOT/info['snapshot']
            assert sha256(source) == info['sha256']
            destination = args.output/f'{name}-best.pt'
            shutil.copy2(source, destination)
            assert sha256(destination) == info['sha256']
            ckpt = torch.load(destination, map_location='cpu', weights_only=False)
            signature = ckpt['signature']
            assert signature['data_hashes'] == data_hashes
            for filename,digest in signature['source_hashes'].items():
                assert sha256(ROOT/'code'/filename) == digest
            training_config = signature['config']
            if name == 'diffusion':
                model = CorrectedDiffusion(training_config['width'],training_config['timesteps'])
            else:
                assert training_config['representation'] == 'linear'
                model = EuclidToJWSTSuperResolution(head=training_config['head'])
            model.load_state_dict(ckpt['model'])
            models[name] = model.to(device).eval()
            info['snapshot'] = str(destination.resolve().relative_to(ROOT))
            checkpoints[name] = info
            del ckpt

        def generated(x, valid, rows, seed):
            sample = models['diffusion'].sample(torch.asinh(x/3), valid,
                        fixed_noise(rows,seed,device), steps=steps)
            return 3*torch.sinh(sample)*dataset.norm['jwst_scale']

        def predict(name, x, valid, rows):
            if name in ['bilinear','bicubic']:
                return F.interpolate(x*dataset.norm['nisp_scale'], size=(205,205),
                                     mode=name, align_corners=False)
            if name == 'diffusion':
                return generated(x,valid,rows,seeds[0])
            if name == 'diffusion_mean3':
                return torch.stack([generated(x,valid,rows,seed) for seed in seeds]).mean(0)
            return dataset.jwst_to_mjy_sr(models[name](x))

        loader = DataLoader(dataset,batch_size=args.batch_size,num_workers=0)
        timing_batch = next(iter(loader))
        tx = timing_batch['nisp'].to(device)
        tv = timing_batch['nisp_valid'].to(device)
        tr = timing_batch['row'].tolist()
        timing = {}
        with torch.inference_mode():
            for name in ([] if reused else METHODS):
                for _ in range(2):
                    result = predict(name,tx,tv,tr)
                torch.cuda.synchronize(device)
                durations = []
                for _ in range(10):
                    torch.cuda.synchronize(device)
                    start = time.perf_counter()
                    result = predict(name,tx,tv,tr)
                    torch.cuda.synchronize(device)
                    durations.append(1000*(time.perf_counter()-start)/len(tr))
                assert torch.isfinite(result).all()
                timing[name] = dict(median_ms_per_object=float(np.median(durations)),
                                    min_ms_per_object=min(durations), max_ms_per_object=max(durations),
                                    repeats_ms_per_object=durations)
                print('Timing',name,timing[name]['median_ms_per_object'],'ms/object',flush=True)
            if reused:
                timing = {name:reused['methods'][name]['timing'] for name in METHODS}
            else:
                del result
            del tx, tv
            records = {name:dict(mae_mjy_sr=[],rmse_mjy_sr=[]) for name in METHODS}
            full = {name:dict(mae_mjy_sr=[],rmse_mjy_sr=[]) for name in METHODS}
            regions = []
            if mask_bundle is not None:
                for threshold in mask_bundle['thresholds']:
                    regions.append(dict(threshold_sigma=float(threshold), **{
                        region:dict(object_ids=[],blocks=[],valid_pixels=[],
                            methods={name:dict(mae_mjy_sr=[],rmse_mjy_sr=[]) for name in METHODS})
                        for region in ['source','background']}))
            cache = None
            if args.cache_predictions:
                cache = h5py.File(args.output/'predictions.h5','w')
                for name in METHODS+['target']:
                    cache.create_dataset(name,(len(dataset),105,105),dtype='f4',chunks=(1,105,105),compression='lzf')
                cache.create_dataset('object_ids',(len(dataset),),dtype='i8')
                cache.attrs['complete'] = False
            rows, ids, blocks, counts, full_counts = [], [], [], [], []
            for batch_number,batch in enumerate(loader):
                x = batch['nisp'].to(device)
                valid = batch['nisp_valid'].to(device)
                batch_rows = batch['row'].tolist()
                target = dataset.jwst_to_mjy_sr(batch['jwst'].to(device))
                common = batch['jwst_valid'].to(device).clone()
                outputs = {}
                for name in ['bilinear','bicubic']:
                    outputs[name], mask = interpolation(x*dataset.norm['nisp_scale'],valid,name)
                    common &= mask
                for name in ['legacy','direct5']:
                    outputs[name] = predict(name,x,valid,batch_rows)
                samples = [generated(x,valid,batch_rows,seed) for seed in seeds]
                outputs['diffusion'] = samples[0]
                outputs['diffusion_mean3'] = torch.stack(samples).mean(0)
                offset = len(ids)
                end = offset+len(batch_rows)
                if mask_bundle is not None:
                    assert mask_bundle['object_ids'][offset:end].tolist() == batch['object_id'].tolist()
                    np.testing.assert_array_equal(mask_bundle['common'][offset:end],common[:,0,50:155,50:155].cpu().numpy())
                    for j,region in enumerate(regions):
                        src = torch.from_numpy(mask_bundle['source'][offset:end,j]).to(device)[:,None]
                        for kind, support in [('source',src),('background',common[...,50:155,50:155] & ~src)]:
                            keep = support.sum((1,2,3)) > 0
                            indexes = keep.cpu().numpy()
                            region[kind]['object_ids'].extend(batch['object_id'].numpy()[indexes].tolist())
                            region[kind]['blocks'].extend(catalog[r]['spatial_block'] for r,k in zip(batch_rows,indexes) if k)
                            region[kind]['valid_pixels'].extend(support[keep].sum((1,2,3)).cpu().tolist())
                            if keep.any():
                                for name in METHODS:
                                    error = pixel_errors(outputs[name][keep,...,50:155,50:155],
                                                         target[keep,...,50:155,50:155],support[keep])
                                    for key in ['mae_mjy_sr','rmse_mjy_sr']:
                                        region[kind]['methods'][name][key].extend(error[key].cpu().tolist())
                if cache is not None:
                    for name in METHODS:
                        cache[name][offset:end] = outputs[name][:,0,50:155,50:155].cpu().numpy()
                    cache['target'][offset:end] = target[:,0,50:155,50:155].cpu().numpy()
                    cache['object_ids'][offset:end] = batch['object_id'].numpy()
                for name in METHODS:
                    measured = pixel_errors(outputs[name][...,50:155,50:155],
                                            target[...,50:155,50:155],common[...,50:155,50:155])
                    wide = pixel_errors(outputs[name],target,common)
                    for key in ['mae_mjy_sr','rmse_mjy_sr']:
                        records[name][key].extend(measured[key].cpu().tolist())
                        full[name][key].extend(wide[key].cpu().tolist())
                counts.extend(measured['valid_pixels'].cpu().tolist())
                full_counts.extend(wide['valid_pixels'].cpu().tolist())
                rows.extend(batch_rows)
                ids.extend(batch['object_id'].tolist())
                blocks.extend(catalog[r]['spatial_block'] for r in batch_rows)
                if batch_number % 8 == 0 or len(ids)==len(dataset):
                    print(f'Evaluated {len(ids)}/{len(dataset)} validation objects',flush=True)
                    run.log({'evaluation/objects':len(ids)})
                    (args.output/'progress.json').write_text(json.dumps(dict(objects=len(ids),total=len(dataset))))
            if cache is not None:
                cache.attrs['complete'] = True
                cache.close()
        assert len(ids) == len(dataset) == len(set(ids))
        assert rows == dataset.indices.tolist()
        assert ids == [int(catalog[row]['id_classic']) for row in rows]
        summary = summarize(records,blocks)
        full_summary = summarize(full,blocks)
        for region in regions:
            for kind in ['source','background']:
                values = region[kind]
                values['summary'] = summarize(values['methods'],values['blocks'])
                values['excluded_empty_object_ids'] = sorted(set(ids)-set(values['object_ids']))
        for name in METHODS:
            records[name].update(object_ids=ids, inference_ms_per_object=timing[name]['median_ms_per_object'],
                timing=timing[name], checkpoint_sha256=(checkpoints['diffusion' if name=='diffusion_mean3' else name]['sha256']
                    if name not in ['bilinear','bicubic'] else None))
        timing_protocol = dict(device=args.device,gpu=torch.cuda.get_device_name(device),
            batch_size=args.batch_size,precision='float32; no autocast or TF32',warmup=2,repeats=10,
            statistic='Median synchronized wall time per batch divided by batch size; range is min/max repeats, not a confidence interval.',
            scope='Device-resident NISP inputs to physical-intensity outputs. Includes representation conversion, diffusion noise creation and ensemble averaging. Excludes disk IO, host/device transfers, metrics, interpolation validity-mask construction and model loading.',
            timing_object_ids=timing_batch['object_id'].tolist(),sampling_steps=steps,seeds=seeds,
            ensemble_size=3,pytorch=torch.__version__,cuda=torch.version.cuda)
        if reused:
            timing_protocol = reused['timing_protocol'] | dict(
                reused_from=str(args.reuse_timing.resolve().relative_to(ROOT)),
                reused_bundle_sha256=sha256(args.reuse_timing))
        report = dict(complete=True,version=args.version,split='validation',units='MJy/sr',
            object_ids=ids,rows=rows,blocks=blocks,valid_pixels=counts,
            population_rule=f'All {len(dataset)} validation objects, no morphology or shape-success cuts. Same per-object mask for every method: intersection of JWST validity and strict bilinear/bicubic input support. Primary central 105x105 target-pixel crop (21 native pixels), matching Figure 5. Signed sky-subtracted physical intensities; no clipping, flux renormalization or band conversion. Each object has equal weight.',
            crop_target_pixels=105, crop_native_pixels=21, data_hashes=data_hashes,
            objects_csv_sha256=sha256(args.data/'objects.csv'),source_hashes=source_hashes,
            checkpoint_source=str(args.checkpoints.resolve().relative_to(ROOT)),
            checkpoint_source_sha256=sha256(args.checkpoints),checkpoints=checkpoints,
            methods=records,summary=summary,timing_protocol=timing_protocol,
            full_field_check=dict(crop_target_pixels=205,valid_pixels=full_counts,methods=full,summary=full_summary),
            wandb_url=run.url,local_only=args.local,
            limitations=['Validation-selected checkpoints, not a held-out test result.',
                        'One training seed per model; block intervals do not measure training-seed variation.',
                        'Cross-band and different-PSF agreement with a noisy observed JWST reference; pixel errors are not morphology or shear calibration.',
                        'Diffusion single draw uses seed 314159; mean3 averages three images in linear intensity.',
                        f'Inference times describe one GPU at batch size {args.batch_size}; amortized time is not single-request latency.'])
        if reused:
            report['limitations'][-1] = f"Inference times reused from the original benchmark at batch size {timing_protocol['batch_size']}; no new speed comparison."
        if mask_bundle is not None:
            report.update(source_regions=regions,source_mask_metadata=mask_metadata,
                source_mask_path=str(args.source_masks.resolve().relative_to(ROOT)),
                source_mask_sha256=sha256(args.source_masks),evaluation_batch_size=args.batch_size)
            report['limitations'].extend([
                'Reference-derived source masks depend on noisy JWST detections; selection can couple to reference noise.',
                'Background means the complement of detected, dilated sources, not guaranteed source-free sky.',
                'Empty source/background regions are excluded per region identically for every method; counts and IDs are saved.'])
        if args.cache_predictions:
            report['prediction_cache'] = dict(path=str((args.output/'predictions.h5').resolve().relative_to(ROOT)),
                                              sha256=sha256(args.output/'predictions.h5'))
        bundle_path.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
        shutil.copy2(bundle_path,args.output/'benchmark.json')
        if not args.local:
            table=wandb.Table(columns=['method','mean_mae_mjy_sr','mean_rmse_mjy_sr','ms_per_object'])
            for name in METHODS:
                table.add_data(name,summary['methods'][name]['mae_mjy_sr']['mean'],
                               summary['methods'][name]['rmse_mjy_sr']['mean'],timing[name]['median_ms_per_object'])
            run.log({'benchmark/summary':table})
        import nbformat
        from nbclient import NotebookClient
        notebook=ROOT/'paper_figures/06_model_comparison.ipynb'
        nb=nbformat.read(notebook,as_version=4)
        assert f"VERSION = '{args.version}'" in '\n'.join(c.source for c in nb.cells)
        NotebookClient(nb,timeout=120,kernel_name='python3',resources={'metadata':{'path':str(notebook.parent)}}).execute()
        nbformat.write(nb,notebook)
        shutil.copy2(notebook,args.output/notebook.name)
        if not args.local:
            artifact=wandb.Artifact('figure6-'+args.version,type='paper-figure')
            artifact.add_file(str(bundle_path))
            for stem in ['fig06_model_comparison','fig06_error_distributions']:
                base=ROOT/'paper_figures/exports'/f'{stem}_{args.version}'
                run.log({stem:wandb.Image(str(base.with_suffix('.png')))})
                for suffix in ['.png','.pdf','.json']:
                    artifact.add_file(str(base.with_suffix(suffix)))
            run.log_artifact(artifact)
        (args.output/'status.json').write_text(json.dumps(dict(complete=True,objects=len(ids),
                        wandb_url=run.url,bundle_sha256=sha256(bundle_path)),indent=2)+'\n')
        dataset.close()
        print('Figure 6 complete:',bundle_path,flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,default=ROOT.parent/'supernisp_nir_y_f115w_v1')
    parser.add_argument('--checkpoints',type=Path,default=ROOT/'paper_figures/inputs/galaxy_examples_v06.json')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--version',default='v01')
    parser.add_argument('--device',default='cuda:1')
    parser.add_argument('--batch-size',type=int,default=8)
    parser.add_argument('--local',action='store_true',help='Local-only run: no W&B initialization, API calls or uploads')
    parser.add_argument('--source-masks',type=Path,help='Frozen reference-only NPZ mask bundle')
    parser.add_argument('--reuse-timing',type=Path,help='Reuse verified timings for unchanged checkpoints and sampler')
    parser.add_argument('--cache-predictions',action='store_true',help='Save central predictions locally for mask revisions')
    main(parser.parse_args())
