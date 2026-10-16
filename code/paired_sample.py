"""Portable PyTorch loader for the corrected superNISP sample.

Never fits normalizers or re-splits. Pass the provided masks to the loss.
Linear normalization preserves signed intensity and is unbounded.
"""
from pathlib import Path
import json
import os
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class PairedSample(Dataset):
    def __init__(self, directory, split='train', representation='linear'):
        self.directory = Path(directory)
        if split not in ['train','validation','test']:
            raise ValueError(split)
        if representation not in ['linear','asinh']:
            raise ValueError(representation)
        self.representation = representation
        self.norm = json.loads((self.directory/'normalization.json').read_text())
        with h5py.File(self.directory/'sample.h5','r') as f:
            self.indices = f[f'{split}_indices'][:]
        self._file = None
        self._pid = None

    def __len__(self):
        return len(self.indices)

    def __getstate__(self):
        state = self.__dict__.copy()
        state['_file'] = None
        state['_pid'] = None
        return state

    def __getitem__(self, i):
        if self._file is None or self._pid != os.getpid():
            if self._file is not None:
                self._file.close()
            self._file = h5py.File(self.directory/'sample.h5','r')
            self._pid = os.getpid()
        f = self._file
        j = int(self.indices[i])
        lm, hm = f['nisp_valid'][j].astype(bool), f['jwst_valid'][j].astype(bool)
        convert = 1.e-12 / f['nisp_pixel_sr'][j]
        lr = (f['nisp'][j]-f['nisp_background'][j])*convert/self.norm['nisp_scale']
        hr = (f['jwst'][j]-f['jwst_background'][j])/self.norm['jwst_scale']
        rms = f['nisp_rms'][j]*convert/self.norm['nisp_scale']
        if self.representation == 'asinh':
            lr, hr = np.arcsinh(lr/3), np.arcsinh(hr/3)
        return {
            'nisp': torch.from_numpy(np.where(lm,lr,0).astype('f4')[None]),
            'jwst': torch.from_numpy(np.where(hm,hr,0).astype('f4')[None]),
            'nisp_valid': torch.from_numpy(lm[None]),
            'jwst_valid': torch.from_numpy(hm[None]),
            # RMS always refers to the linear normalized input, even in asinh mode.
            'nisp_rms_linear': torch.from_numpy(rms.astype('f4')[None]),
            'object_id': int(f['object_id'][j]), 'row': j,
            'jwst_background': float(f['jwst_background'][j]),
        }

    def jwst_to_mjy_sr(self, prediction, background=0.):
        if self.representation == 'asinh':
            prediction = 3*torch.sinh(prediction)
        background=torch.as_tensor(background,dtype=prediction.dtype,device=prediction.device)
        if background.ndim==1 and prediction.ndim==4:
            background=background[:,None,None,None]
        return prediction*self.norm['jwst_scale']+background

    def close(self):
        if self._file is not None:
            self._file.close()
            self._file = None


def masked_l1(prediction, target, valid):
    """Mean of per-object valid-pixel MAE; excludes unknown target pixels."""
    mask = valid.to(dtype=prediction.dtype)
    dims = tuple(range(1, prediction.ndim))
    count = mask.sum(dim=dims)
    if torch.any(count == 0):
        raise ValueError('Empty target validity mask')
    return ((torch.abs(prediction-target)*mask).sum(dim=dims)/count).mean()
