"""Exercise the portable loader with worker processes and one CPU train step."""
import argparse
from pathlib import Path
import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader
from paired_sample import PairedSample, masked_l1


def main(directory):
    torch.set_num_threads(2)
    torch.manual_seed(42)
    for representation in ['linear','asinh']:
        ds=PairedSample(directory,representation=representation)
        loader=DataLoader(ds,batch_size=2,num_workers=2,multiprocessing_context='spawn')
        batch=next(iter(loader))
        assert torch.isfinite(batch['nisp']).all() and torch.isfinite(batch['jwst']).all()
        restored=ds.jwst_to_mjy_sr(batch['jwst'],batch['jwst_background'])
        with h5py.File(Path(directory)/'sample.h5','r') as f:
            for k,row in enumerate(batch['row'].tolist()):
                mask=batch['jwst_valid'][k,0].numpy()
                np.testing.assert_allclose(restored[k,0].numpy()[mask],f['jwst'][row][mask],rtol=2e-5,atol=1e-5)
        model=torch.nn.Sequential(torch.nn.Conv2d(1,25,3,padding=1),torch.nn.PixelShuffle(5))
        optimizer=torch.optim.Adam(model.parameters(),lr=1e-4)
        loss=masked_l1(model(batch['nisp']),batch['jwst'],batch['jwst_valid'])
        assert torch.isfinite(loss)
        loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
        optimizer.step()
        pred=torch.zeros(2,1,4,4);target=torch.ones_like(pred);valid=torch.ones_like(pred,dtype=torch.bool)
        valid[:,:,0,0]=False;target[:,:,0,0]=1e8
        assert masked_l1(pred,target,valid)==1
        ds.close()
        print(f'{representation}: worker loading, flux roundtrip, masked loss, and train-step smoke check passed')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory',type=Path)
    main(parser.parse_args().directory)
