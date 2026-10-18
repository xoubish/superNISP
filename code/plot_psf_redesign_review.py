"""Paired effects relative to extra-training control; render frozen statistics."""
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[1]


def main():
    path=ROOT/'runs/psf-redesign-comparison-v01/comparison.json'
    report=json.loads(path.read_text())
    assert report['complete'] and report['training_streams_match']
    names=['conditioned_best','deconv_best']
    labels=['PSF + noise','PSF + noise\n+ inverse feature']
    colors=['#4477aa','#228833']
    fig,axes=plt.subplots(1,4,figsize=(12,3),sharey=True)
    panels=[('source_rmse',1e4,'Source RMSE change\n[$10^{-4}$ MJy/sr]'),
            ('correlation',1e3,'Fine-scale correlation change\n[$10^{-3}$]'),
            ('fixed',1e3,'Fixed-weight shape RMSE change\n[$10^{-3}$]'),
            ('adaptive',1e3,'Adaptive shape RMSE change\n[$10^{-3}$]')]
    for ax,(metric,scale,label) in zip(axes,panels):
        for i,(name,color) in enumerate(zip(names,colors)):
            key=name+'__minus__control_best'
            if metric in ['fixed','adaptive']:
                rec=report['shapes'][metric]['paired_all'][key]
            elif metric=='correlation':
                full=report['paired'][key]['source_correlation_difference']
                rec=dict(mean=full['mean'][-2],ci95=full['ci95'][-2])
            else: rec=report['paired'][key][metric]
            value=rec['mean']*scale;ci=np.array(rec['ci95'])*scale
            ax.errorbar(value,i,xerr=np.abs(ci-value)[:,None],fmt='o',color=color,capsize=4,ms=7)
        ax.axvline(0,color='black',ls=':',lw=1)
        ax.set_yticks(range(2),labels);ax.set_ylim(1.55,-.55)
        ax.set_xlabel(label,fontsize=10);ax.grid(axis='x',alpha=.15)
        ax.set_title('Higher is better' if metric=='correlation' else 'Lower is better',fontsize=10)
    fig.suptitle('Redesign minus matched extra-training control',fontsize=12)
    fig.text(.5,.01,'Paired 95% spatial-block intervals; validation-selected checkpoints, one parent seed',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.06,1,.93))
    stem=ROOT/'paper/diagnostics/psf_redesign_paired_v01'
    for ext in ['png','pdf']:fig.savefig(stem.with_suffix('.'+ext),dpi=180,bbox_inches='tight')
    plt.close(fig)
    def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
    stem.with_suffix('.json').write_text(json.dumps(dict(comparison_sha256=sha(path),
        source_sha256=sha(__file__),outputs={ext:sha(stem.with_suffix('.'+ext)) for ext in ['png','pdf']},
        meaning='Absolute paired differences scaled only for display; no ratio approximation. Fine scale is 3–6 target pixels.'),indent=2)+'\n')
    print(stem)


if __name__=='__main__': main()
