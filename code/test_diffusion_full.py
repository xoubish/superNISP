"""Resource scheduling and physical ensemble averaging regression checks."""
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch
from diffusion_full_runtime import analysis_pending
from diffusion_full_diagnostics import evaluate_ensemble
from shape_diagnostics import diagnose_moments,METHODS


class FullRunTests(unittest.TestCase):
    def test_analysis_gate(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); training=root/'training'; pipeline=root/'pipeline'; exited=root/'exit'
            paths=(training,pipeline,exited)
            self.assertFalse(analysis_pending(*paths))
            pipeline.write_text(json.dumps(dict(complete=False,stage='full50')))
            self.assertFalse(analysis_pending(*paths))
            training.write_text(json.dumps(dict(complete=True)))
            self.assertTrue(analysis_pending(*paths))
            pipeline.write_text(json.dumps(dict(complete=False,stage='robustness')))
            self.assertTrue(analysis_pending(*paths))
            pipeline.write_text('{')
            self.assertTrue(analysis_pending(*paths))
            exited.write_text('0')
            self.assertFalse(analysis_pending(*paths))
            exited.write_text('1')
            self.assertFalse(analysis_pending(*paths))

    def test_mean_in_linear_intensity(self):
        axis=torch.arange(205)-102
        yy,xx=torch.meshgrid(axis,axis,indexing='ij')
        target=(2*torch.exp(-xx.square()/100-yy.square()/180))[None]
        mask=torch.ones_like(target,dtype=torch.bool)
        class Dataset:
            indices=np.array([10])
            def jwst_to_mjy_sr(self,image): return 3*torch.sinh(image)
            def __getitem__(self,index): return dict(jwst=torch.asinh(target/3),jwst_valid=mask)
        predictions=[{10:torch.asinh(target*scale/3)} for scale in [.25,.75,2.]]
        measured=diagnose_moments(target[None],mask[None])
        reference=dict(rows=np.array([10]),object_ids=np.array([100]),blocks=np.array(['a']))
        for method in ['jwst',*METHODS]:
            for key in ['e','good','flags']: reference[method+'_'+key]=measured[key].numpy()
        with tempfile.TemporaryDirectory() as directory:
            report=evaluate_ensemble(Dataset(),predictions,reference,Path(directory))
        self.assertLess(report['mae_mjy_sr'],1e-7)
        self.assertLess(report['shapes']['methods']['sr']['component_rmse'],1e-6)


if __name__=='__main__': unittest.main()
