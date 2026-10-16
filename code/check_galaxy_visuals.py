"""Check real pilot rendering and identify moment failures before the long run."""
import json
from pathlib import Path
import random

import numpy as np
import torch
import wandb

from paired_sample import PairedSample
from residual_baseline import EuclidToJWSTSuperResolution
from galaxy_visuals import prepare_gallery, log_gallery
from train_paired import persistent_session_required, shape_moments, seed_everything


def main():
    persistent_session_required()
    torch.set_num_threads(4)
    seed_everything(42)
    root = Path(__file__).resolve().parents[1]
    output = root/'runs/galaxy-visual-check'
    output.mkdir(parents=True,exist_ok=True)
    with wandb.init(project='superNISP-corrected',name='galaxy-visual-check',job_type='visual-check',
                    mode='online',dir=str(output)) as run:
        assert run.settings.mode=='online' and run.url
        (output/'wandb_url.txt').write_text(run.url+'\n')
        checkpoint = torch.load(root/'runs/residual-pilot-s20261016/best.pt',map_location='cpu',weights_only=False)
        model = EuclidToJWSTSuperResolution().cuda().eval()
        model.load_state_dict(checkpoint['model'])
        dataset = PairedSample(root.parent/'supernisp_nir_y_f115w_v1','validation')
        indices = checkpoint['validation_indices']
        gallery = prepare_gallery(dataset,indices,8,20261018,output)
        before = (random.getstate(),np.random.get_state(),torch.get_rng_state(),torch.cuda.get_rng_state())
        predicted = log_gallery(model,dataset,gallery,'cuda:0',run,output,3,384,shape_moments)
        assert random.getstate()==before[0]
        assert np.array_equal(np.random.get_state()[1],before[1][1])
        assert torch.equal(torch.get_rng_state(),before[2])
        assert torch.equal(torch.cuda.get_rng_state(),before[3])
        original_limits = json.dumps(gallery['metadata']['limits_mjy_sr'],sort_keys=True)
        assert predicted.shape==(8,1,205,205) and torch.isfinite(predicted).all()
        assert json.dumps(gallery['metadata']['limits_mjy_sr'],sort_keys=True)==original_limits
        torch.testing.assert_close(gallery['target'],dataset.jwst_to_mjy_sr(gallery['batch']['jwst']))
        assert len(list((output/'galaxies').glob('*.png')))==4
        failures = []
        with torch.no_grad():
            for start in range(0,len(indices),8):
                chunk = indices[start:start+8]
                items = [dataset[i] for i in chunk]
                x = torch.stack([v['nisp'] for v in items]).cuda()
                mask = torch.stack([v['jwst_valid'] for v in items]).cuda()
                y = torch.stack([v['jwst'] for v in items]).cuda()
                prediction = dataset.jwst_to_mjy_sr(model(x))
                _,ok = shape_moments(prediction,mask)
                _,reference_ok = shape_moments(dataset.jwst_to_mjy_sr(y),mask)
                for k,item in enumerate(items):
                    if not ok[k]:
                        failures.append(dict(index=int(chunk[k]),row=item['row'],object_id=item['object_id'],
                                             reference_pass=bool(reference_ok[k])))
        assert len(failures)==20
        (output/'pilot_moment_failures.json').write_text(json.dumps(failures,indent=2))
        failed_output = output/'pilot_failures'
        failed_gallery = prepare_gallery(dataset,[item['index'] for item in failures],8,20261018,failed_output)
        log_gallery(model,dataset,failed_gallery,'cuda:0',run,failed_output,3,384,shape_moments)
        report=dict(rendering='passed',rng_preserved=True,images=8,pilot_prediction_failures=len(failures),
                    pilot_failure_reference_pass=sum(item['reference_pass'] for item in failures),wandb_url=run.url)
        run.summary.update(report)
        (output/'checks.json').write_text(json.dumps(report,indent=2))
        print(json.dumps(report),flush=True)
        dataset.close()


if __name__=='__main__':
    main()
