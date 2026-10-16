"""Full validation summaries and a predeclared three-sample mean in linear units."""
import json
import numpy as np
import torch
from diffusion_diagnostics import evaluate_generation as evaluate_base
from shape_diagnostics import METHODS,diagnose_moments,paired_summary


def evaluate_generation(model,dataset,indices,device,steps,seed,batch_size,output,label):
    result=evaluate_base(model,dataset,indices,device,steps,seed,batch_size,output,label)
    path=output/f'{label}-summary.json'
    report=json.loads(path.read_text())
    report['scope']='Fixed validation monitor subset' if len(indices)<len(dataset) else 'Full reserved validation split; test remains untouched'
    path.write_text(json.dumps(report,indent=2,allow_nan=False))
    return result


def evaluate_ensemble(dataset,predictions,reference,output):
    """Average in physical linear intensity, then measure; never average shapes."""
    arrays={k:v.copy() for k,v in reference.items()}
    measured=[]; mae=0.
    lookup={int(row):i for i,row in enumerate(dataset.indices)}
    for row in reference['rows']:
        image=torch.stack([dataset.jwst_to_mjy_sr(p[int(row)]) for p in predictions]).mean(0)[None]
        item=dataset[lookup[int(row)]]
        target=dataset.jwst_to_mjy_sr(item['jwst'][None]); valid=item['jwst_valid'][None]
        values=diagnose_moments(image,valid)
        measured.append({k:values[k].numpy() for k in ['e','good','flags']})
        mae+=float(torch.where(valid,(image-target).abs(),0.).sum()/valid.sum())
    for key in ['e','good','flags']: arrays['sr_'+key]=np.concatenate([m[key] for m in measured])
    unpack={m:{k:arrays[m+'_'+k] for k in ['e','good','flags']} for m in ['jwst',*METHODS]}
    summary=paired_summary(unpack,arrays['blocks'],bootstrap=1000)
    report=dict(mae_mjy_sr=mae/len(reference['rows']),shapes=summary,
                definition='Mean of three independently seeded generated images in linear MJy/sr, followed by shape measurement; not a mean of ellipticities.')
    np.savez_compressed(output/'best-ensemble-measurements.npz',**arrays)
    (output/'best-ensemble-summary.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    return report
