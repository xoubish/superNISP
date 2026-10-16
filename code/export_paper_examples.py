"""Frozen validation examples for Figure 5; run inside tmux with online W&B.

Updates reuse the saved galaxy IDs, display limits and diffusion noise seeds.
Only the checkpoint snapshots change. No training or test-set access occurs.
"""
import argparse
import csv
import json
from pathlib import Path
import shutil
import time

import numpy as np
import torch
import wandb

from train_paired import persistent_session_required, seed_everything, sha256
from paired_sample import PairedSample
from residual_baseline import EuclidToJWSTSuperResolution
from diffusion_corrected import CorrectedDiffusion, fixed_noise
from shape_diagnostics import interpolation

ROOT = Path(__file__).resolve().parents[1]
INPUTS = ROOT / 'paper_figures/inputs'
RUNS = {'legacy': 'residual-full50-s20261016', 'direct5': 'direct5-full50-s20261016',
        'diffusion': 'diffusion-full50-s20261016'}


def snapshot_best(source, destination):
    # best.pt is copied nonatomically by the trainers. Accept only a stable,
    # deserializable copy whose bytes match before and after copying.
    for attempt in range(5):
        try:
            before = sha256(source)
            shutil.copy2(source, destination)
            digest = sha256(destination)
            if before != digest or sha256(source) != digest:
                raise RuntimeError('Checkpoint changed during snapshot')
            checkpoint = torch.load(destination, map_location='cpu', weights_only=False)
            return checkpoint, digest
        except (RuntimeError, EOFError, OSError):
            if attempt == 4:
                raise
            time.sleep(1)


def select_examples(dataset, data_hashes, path):
    if path.exists():
        selection = json.loads(path.read_text())
        if selection['data_hashes'] != data_hashes:
            raise ValueError('Frozen selection belongs to different data')
        return selection
    seed = 20261016
    pool = np.random.default_rng(seed).permutation(len(dataset))[:256]
    scored = []
    for index in pool:
        item = dataset[int(index)]
        values = item['nisp'][0, 15:26, 15:26]
        mask = item['nisp_valid'][0, 15:26, 15:26]
        score = float(torch.quantile(values[mask], .9)) if mask.any() else float('-inf')
        scored.append((score, int(index)))
    scored.sort()
    quantiles = [.20, .45, .70, .95]
    indices = [scored[round(q * (len(scored)-1))][1] for q in quantiles]
    items = [dataset[i] for i in indices]
    limits = []
    for item in items:
        target = dataset.jwst_to_mjy_sr(item['jwst'])[0, 50:155, 50:155]
        mask = item['jwst_valid'][0, 50:155, 50:155]
        low, high = np.percentile(target[mask].numpy(), [.5, 99.5])
        low = min(float(low), 0.)
        high = max(float(high), low + 1e-6)
        stretch = max(dataset.norm['jwst_scale'], (high-low)*.03)
        limits.append(dict(low=low, high=high, stretch=stretch,
                           residual=max(stretch, (high-low)*.3)))
    selection = dict(split='validation', seed=seed, pool_size=len(pool),
        rule='20th, 45th, 70th and 95th percentile ranks of central NISP 90th-percentile brightness in a seeded pool of 256 validation galaxies; selected before inference, with no shape-success cuts.',
        quantiles=quantiles, indices=indices, rows=[x['row'] for x in items],
        object_ids=[x['object_id'] for x in items], data_hashes=data_hashes,
        limits_mjy_sr=limits, display_crop_native_pixels=21,
        sampling_seeds=[314159,314160,314161])
    path.write_text(json.dumps(selection, indent=2)+'\n')
    return selection


def main(args):
    persistent_session_required()
    torch.set_num_threads(2)
    seed_everything(20261016)
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    data_hashes = {n:sha256(args.data/n) for n in ['manifest.json','normalization.json','SHA256SUMS']}
    bundle = INPUTS / f'galaxy_examples_{args.version}.npz'
    metadata_path = bundle.with_suffix('.json')
    if bundle.exists() or metadata_path.exists():
        raise FileExistsError('Use a fresh version to preserve the previous figure bundle')
    with wandb.init(project='superNISP-corrected', entity='AI-Astro', name=output.name,
                    job_type='paper-figure-export', mode='online', dir=str(output),
                    config=dict(version=args.version, device=args.device, data_hashes=data_hashes)) as run:
        if run.settings.mode != 'online' or not run.url:
            raise RuntimeError('Online W&B monitoring required')
        (output/'wandb_url.txt').write_text(run.url+'\n')
        dataset = PairedSample(args.data, 'validation', 'linear')
        selection = select_examples(dataset, data_hashes, args.selection)
        (output/'selection.json').write_text(json.dumps(selection, indent=2))
        items = [dataset[i] for i in selection['indices']]
        assert [i['object_id'] for i in items] == selection['object_ids']
        assert [i['row'] for i in items] == selection['rows']
        run.config.update(dict(selection=selection))
        print('Frozen galaxy IDs:', selection['object_ids'], flush=True)
        arrays = dict(object_ids=np.array(selection['object_ids']), rows=np.array(selection['rows']))
        x = torch.stack([i['nisp'] for i in items])
        valid = torch.stack([i['nisp_valid'] for i in items])
        arrays['nisp'] = (x[:,0]*dataset.norm['nisp_scale']).numpy()
        arrays['nisp_valid'] = valid[:,0].numpy()
        bilinear, bilinear_valid = interpolation(x*dataset.norm['nisp_scale'], valid, 'bilinear')
        arrays['bilinear'] = bilinear[:,0].numpy()
        arrays['bilinear_valid'] = bilinear_valid[:,0].numpy()
        arrays['jwst'] = dataset.jwst_to_mjy_sr(torch.stack([i['jwst'] for i in items]))[:,0].numpy()
        arrays['jwst_valid'] = torch.stack([i['jwst_valid'] for i in items])[:,0].numpy()
        checkpoints = {}
        for name, folder in RUNS.items():
            training = ROOT/'runs'/folder
            ckpt, digest = snapshot_best(training/'best.pt', output/f'{name}-best.pt')
            signature = ckpt['signature']
            assert signature['data_hashes'] == data_hashes
            for filename, expected in signature['source_hashes'].items():
                assert sha256(ROOT/'code'/filename) == expected, filename
            config = signature['config']
            key = 'generated/mae_mjy_sr' if name == 'diffusion' else 'validation/mae'
            # Records may lag a just-written best.pt. Verify its metric whenever
            # its epoch is already present; the checkpoint stores the criterion.
            records = [json.loads(line) for line in (training/'metrics.jsonl').read_text().splitlines() if line.strip()]
            through = [r for r in records if r['epoch'] <= ckpt['epoch'] and key in r]
            same = [r for r in through if r['epoch'] == ckpt['epoch']]
            if same:
                assert np.isclose(same[0][key], ckpt['best'])
                assert np.isclose(min(r[key] for r in through), ckpt['best'])
            complete = (training/'status.json').exists() and json.loads((training/'status.json').read_text()).get('complete', False)
            checkpoints[name] = dict(epoch=ckpt['epoch'], sha256=digest, run=folder,
                criterion=key, criterion_value=ckpt['best'], training_complete=bool(complete),
                snapshot=str((output/f'{name}-best.pt').relative_to(ROOT)),
                source_hashes=signature['source_hashes'], training_wandb_id=ckpt['wandb_id'])
            print(f'{name}: best epoch {ckpt["epoch"]}, training complete={complete}', flush=True)
            with torch.inference_mode():
                if name == 'diffusion':
                    model = CorrectedDiffusion(config['width'], config['timesteps']).to(device).eval()
                    model.load_state_dict(ckpt['model'])
                    samples = []
                    for seed in selection['sampling_seeds']:
                        outputs = []
                        for index, row in enumerate(selection['rows']):
                            generated = model.sample(torch.asinh(x[index:index+1].to(device)/3),
                                valid[index:index+1].to(device), fixed_noise([row],seed,device), steps=50)
                            outputs.append((3*torch.sinh(generated)*dataset.norm['jwst_scale']).cpu())
                        samples.append(torch.cat(outputs)[:,0].numpy())
                        run.log({'generation/completed_seeds':len(samples)})
                    arrays['diffusion'] = samples[0]
                    arrays['diffusion_mean3'] = np.mean(samples,axis=0)
                    arrays['diffusion_samples'] = np.stack(samples,axis=1)
                else:
                    assert config['representation'] == 'linear'
                    model = EuclidToJWSTSuperResolution(head=config['head']).to(device).eval()
                    model.load_state_dict(ckpt['model'])
                    arrays[name] = torch.cat([dataset.jwst_to_mjy_sr(model(i[None].to(device))).cpu() for i in x])[:,0].numpy()
            del model, ckpt
            torch.cuda.empty_cache()
            run.log({f'checkpoints/{name}_epoch':checkpoints[name]['epoch']})
        for name in ['legacy','direct5','diffusion','diffusion_mean3']:
            assert arrays[name].shape == arrays['jwst'].shape and np.isfinite(arrays[name]).all()
        catalog = {int(r['row']):r for r in csv.DictReader((args.data/'objects.csv').open())}
        scale = [float(np.sqrt(float(catalog[row]['nisp_pixel_sr']))*180/np.pi*3600) for row in selection['rows']]
        np.savez_compressed(bundle, **arrays)
        hashes = {name:values['sha256'] for name,values in checkpoints.items()}
        hashes['diffusion_mean3'] = hashes['diffusion']
        metadata = dict(complete=True, split='validation', units='MJy/sr', version=args.version,
            object_ids=selection['object_ids'], selection=selection, checkpoint_sha256=hashes,
            checkpoints=checkpoints, sampling_seeds=selection['sampling_seeds'], sampling_steps=50,
            nisp_pixel_scale_arcsec=scale, data_hashes=data_hashes, npz_sha256=sha256(bundle),
            limits_mjy_sr=selection['limits_mjy_sr'], wandb_url=run.url,
            selection_file=str(args.selection.resolve().relative_to(ROOT)),
            selection_sha256=sha256(args.selection),
            interpolation_source_sha256=sha256(ROOT/'code/shape_diagnostics.py'),
            exporter_sha256=sha256(Path(__file__)),
            status='Provisional best-checkpoint visual comparison; diffusion may still be training. Examples selected from inputs, not a performance benchmark.')
        metadata_path.write_text(json.dumps(metadata,indent=2)+'\n')
        shutil.copy2(metadata_path, output/'bundle.json')
        # Render only the plotting notebook, inside the same persistent job.
        import nbformat
        from nbclient import NotebookClient
        notebook = ROOT/'paper_figures/05_galaxy_examples.ipynb'
        nb = nbformat.read(notebook,as_version=4)
        assert f"VERSION = '{args.version}'" in '\n'.join(c.source for c in nb.cells)
        NotebookClient(nb,timeout=120,kernel_name='python3',resources={'metadata':{'path':str(notebook.parent)}}).execute()
        nbformat.write(nb,notebook)
        stem = ROOT/'paper_figures/exports'/f'fig05_galaxy_examples_{args.version}'
        run.log({'figure5/galaxy_comparison':wandb.Image(str(stem.with_suffix('.png')))})
        residual_png = stem.parent/f'fig05_galaxy_residuals_{args.version}.png'
        run.log({'figure5/residuals':wandb.Image(str(residual_png))})
        artifact = wandb.Artifact('figure5-'+args.version,type='paper-figure')
        for path in [bundle,metadata_path,stem.with_suffix('.png'),stem.with_suffix('.pdf'),residual_png]:
            artifact.add_file(str(path))
        run.log_artifact(artifact)
        (output/'status.json').write_text(json.dumps(dict(complete=True,wandb_url=run.url,version=args.version),indent=2))
        dataset.close()
        print('Figure 5 complete:',stem.with_suffix('.png'),flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,default=ROOT.parent/'supernisp_nir_y_f115w_v1')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--version',default='v01')
    parser.add_argument('--device',default='cuda:1')
    parser.add_argument('--selection',type=Path,default=INPUTS/'galaxy_examples_selection.json')
    main(parser.parse_args())
