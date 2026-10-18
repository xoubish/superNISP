"""Loss, source-mask loader, RNG restoration and frozen refinement diagnostics."""
import json
import copy
import os
import random
from pathlib import Path

import h5py
import numpy as np
import torch
from paired_sample import PairedSample
from shape_diagnostics import diagnose_moments
from spatial_spectra import taper, validity_taper, transform, frequency_masks, paired_spectrum, BANDS
from probe_shared_gpu import sha

ROOT = Path(__file__).resolve().parents[1]


def restore_parent(model,optimizer,parent):
    # Optimizer.load_state_dict retains some tensor references (notably CPU
    # Adam step counters even for CUDA parameters). Keep the parent immutable.
    model.load_state_dict(parent['model'])
    optimizer.load_state_dict(copy.deepcopy(parent['optimizer']))
    assert all(float(state['step'])==parent['step'] for state in optimizer.state.values())


class SourceMaskedSample(PairedSample):
    def __init__(self, directory, masks_path):
        super().__init__(directory, 'train', 'linear')
        self.masks_path = str(masks_path)
        self._mask_file = None
        self._mask_pid = None
        with h5py.File(masks_path) as saved:
            assert saved.attrs['complete']
            np.testing.assert_array_equal(saved['train/rows'][:], self.indices)
            self.mask_ids = saved['train/object_ids'][:]

    def __getstate__(self):
        state = super().__getstate__()
        state['_mask_file'] = None
        state['_mask_pid'] = None
        return state

    def __getitem__(self, index):
        item = super().__getitem__(index)
        if self._mask_file is None or self._mask_pid != os.getpid():
            if self._mask_file is not None:
                self._mask_file.close()
            self._mask_file = h5py.File(self.masks_path, 'r')
            self._mask_pid = os.getpid()
        assert item['object_id'] == self.mask_ids[index]
        item['source_mask'] = torch.from_numpy(self._mask_file['train/source'][index][None])
        return item

    def close(self):
        super().close()
        if self._mask_file is not None:
            self._mask_file.close()
            self._mask_file = None


def balanced_l1(prediction, target, valid, source):
    """Equal object weights; equal region weights, renormalized if one is empty."""
    if prediction.shape != target.shape or valid.shape != target.shape or source.shape != target.shape:
        raise ValueError('Loss geometry mismatch')
    valid = valid.bool()
    if not valid.flatten(1).any(1).all():
        raise ValueError('Empty valid object')
    error = torch.where(valid, prediction-target, 0.).abs()
    values, present = [], []
    for mask in [valid & source.bool(), valid & ~source.bool()]:
        counts = mask.flatten(1).sum(1)
        values.append((error*mask).flatten(1).sum(1)/counts.clamp_min(1))
        present.append(counts > 0)
    return ((values[0]+values[1])/(present[0].float()+present[1].float())).mean()


def fresh_schedule(optimizer, total_steps, initial_lr=1e-5, final_lr=1e-6):
    # Keep inherited Adam moments, step counts and betas; discard OneCycle LR metadata.
    for group in optimizer.param_groups:
        for key in ['initial_lr','max_lr','min_lr','base_momentum','max_momentum']:
            group.pop(key, None)
        group['lr'] = initial_lr
    return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=final_lr)


def capture_rng(generator):
    return dict(python=random.getstate(), numpy=np.random.get_state(),
                torch=torch.get_rng_state(), cuda=[torch.cuda.get_rng_state(0)],
                loader=generator.get_state())


def restore_single_gpu_rng(state, generator):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    # The parent trained on physical CUDA 0. Each fork exposes one GPU as local 0.
    torch.cuda.set_rng_state(state['cuda'][0], 0)
    generator.set_state(state['loader'])


def fixed_shapes(images, valid):
    parts = []
    for start in range(0, len(images), 32):
        result = diagnose_moments(torch.from_numpy(images[start:start+32,None].astype(np.float64)),
                                  torch.from_numpy(valid[start:start+32,None]))
        parts.append({key:result[key].numpy() for key in ['e','good','flags']})
    return {key:np.concatenate([p[key] for p in parts]) for key in parts[0]}


class FrozenDiagnostics:
    def __init__(self):
        path = ROOT/'paper_figures/inputs/model_benchmark_v03.json'
        self.benchmark = json.loads(path.read_text())
        masks = ROOT/self.benchmark['source_mask_path']
        assert sha(masks) == self.benchmark['source_mask_sha256']
        with np.load(masks) as saved:
            self.valid = saved['common']
            self.source = saved['source'][:,saved['thresholds'].tolist().index(2.)]
            self.ids = saved['object_ids']
        assert self.ids.tolist() == self.benchmark['object_ids']
        self.blocks = np.asarray(self.benchmark['blocks'])
        self.primary = self.valid.all((1,2))
        cache = ROOT/self.benchmark['prediction_cache']['path']
        assert sha(cache) == self.benchmark['prediction_cache']['sha256']
        with h5py.File(cache) as saved:
            np.testing.assert_array_equal(saved['object_ids'][:],self.ids)
            self.target = saved['target'][:]
            self.baseline = saved['legacy'][:]
        self.reference_shapes = fixed_shapes(self.target,self.valid)
        self.baseline_shapes = fixed_shapes(self.baseline,self.valid)
        outer = taper()
        self.windows = [outer*validity_taper(mask,3) for mask in self.source[self.primary]]
        self.reference_ffts = [transform(image,w) for image,w in zip(self.target[self.primary],self.windows)]
        self.frequency_masks = frequency_masks()
        self.primary_count = int(self.primary.sum())
        assert self.primary_count == 1363

    def measure(self, prediction):
        if prediction.shape != self.target.shape or not np.isfinite(prediction[self.valid]).all():
            raise ValueError('Invalid validation predictions')
        difference = prediction.astype(float)-self.target.astype(float)
        arrays = {}
        metrics = {}
        for name,mask in [('source',self.source),('outside',self.valid & ~self.source)]:
            delta = np.where(mask,difference,0.)
            counts = mask.sum((1,2))
            arrays[name+'_mae'] = np.abs(delta).sum((1,2))/counts
            arrays[name+'_rmse'] = np.sqrt((delta**2).sum((1,2))/counts)
            for metric in ['mae','rmse']:
                metrics[name+'_'+metric] = float(arrays[name+'_'+metric].mean())
        shapes = fixed_shapes(prediction,self.valid)
        arrays.update({f'shape_{key}':value for key,value in shapes.items()})
        original_common = self.reference_shapes['good'] & self.baseline_shapes['good']
        common = original_common & shapes['good']
        # Never compare a changing-population new error with an unpaired old error.
        metrics['shape_successes'] = int(shapes['good'].sum())
        metrics['shape_comparison_objects'] = int(common.sum())
        metrics['shape_lost_from_baseline_common'] = int((original_common & ~shapes['good']).sum())
        for name,e in [('new',shapes['e']),('baseline',self.baseline_shapes['e'])]:
            metrics['shape_'+name+'_rmse_on_same_objects'] = float(np.sqrt(np.mean(
                (e[common]-self.reference_shapes['e'][common])**2)))
        results = [paired_spectrum(transform(im,w),ref,self.frequency_masks)
                   for im,w,ref in zip(prediction[self.primary],self.windows,self.reference_ffts)]
        for key in ['log_power_ratio','correlation']:
            arrays[key] = np.stack([r[key] for r in results])
        for j,band in enumerate(BANDS):
            index = -len(BANDS)+j
            metrics[band+'_power_ratio'] = float(10**arrays['log_power_ratio'][:,index].mean())
            metrics[band+'_correlation'] = float(arrays['correlation'][:,index].mean())
        return metrics, arrays

    def verify_parent(self, prediction, metrics):
        np.testing.assert_allclose(prediction[self.valid],self.baseline[self.valid],rtol=2e-4,atol=2e-6)
        primary = next(r for r in self.benchmark['source_regions'] if r['threshold_sigma']==2.)
        for new,old in [('source','source'),('outside','background')]:
            for metric in ['mae','rmse']:
                expected = primary[old]['summary']['methods']['legacy'][metric+'_mjy_sr']['mean']
                np.testing.assert_allclose(metrics[new+'_'+metric],expected,rtol=2e-6,atol=1e-9)
