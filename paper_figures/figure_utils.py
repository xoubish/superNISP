"""Read-only figure inputs, consistent styling, and reproducible vector exports.

This module never imports training/model code, reads checkpoints, opens image
datasets, launches inference, or contacts W&B. Notebooks own their plot code.
"""
import hashlib
import json
import os
from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch,FancyArrowPatch
import pandas as pd


def find_root():
    for path in [Path.cwd(),*Path.cwd().parents]:
        if (path/'HANDOFF_GPU.md').exists() and (path/'code').is_dir(): return path
    raise FileNotFoundError('Open the notebook from the superNISP repository or paper_figures folder')


ROOT=find_root()
FIGURES=ROOT/'paper_figures'
DATA=Path(os.environ.get('SUPERNISP_DATA',str(ROOT.parent/'supernisp_nir_y_f115w_v1'))).expanduser().resolve()
COLORS=dict(train='#0072B2',validation='#D55E00',test='#999999',native='#626A73',
            bilinear='#E69F00',bicubic='#56B4E9',legacy='#0072B2',sr='#0072B2',
            direct5='#009E73',diffusion='#8B5DA8',jwst='#252B33')


def style(fontsize=8):
    mpl.rcParams.update({'font.family':'DejaVu Sans','font.size':fontsize,'axes.labelsize':fontsize,
        'axes.titlesize':fontsize+1,'legend.fontsize':fontsize-1,'xtick.labelsize':fontsize-1,
        'ytick.labelsize':fontsize-1,'axes.spines.top':False,'axes.spines.right':False,
        'axes.linewidth':.7,'lines.linewidth':1.5,'savefig.dpi':240,'figure.dpi':120,
        'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none','axes.unicode_minus':True,
        'figure.facecolor':'white','axes.facecolor':'white'})


def read_json(path): return json.loads(Path(path).read_text())
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def notebook_source_hash(path):
    notebook=read_json(path)
    cells=[dict(kind=c['cell_type'],source=c['source']) for c in notebook['cells']]
    return hashlib.sha256(json.dumps(cells,sort_keys=True).encode()).hexdigest()


def export(fig,stem,notebook,inputs,settings,status='layout draft',formats=('pdf','svg','png')):
    directory=FIGURES/'exports'; directory.mkdir(exist_ok=True)
    paths=[]
    for suffix in formats:
        path=directory/f'{stem}.{suffix}'
        fig.savefig(path,bbox_inches='tight',facecolor='white')
        paths.append(path)
    evidence={str(Path(p).resolve().relative_to(ROOT)) if Path(p).resolve().is_relative_to(ROOT) else str(Path(p).resolve()):sha(p)
              for p in inputs}
    provenance=dict(status=status,notebook=notebook,notebook_source_sha256=notebook_source_hash(FIGURES/notebook),
        helper_sha256=sha(__file__),inputs=evidence,settings=settings,
        versions=dict(matplotlib=mpl.__version__,pandas=pd.__version__),
        exports={p.name:sha(p) for p in paths})
    (directory/f'{stem}.json').write_text(json.dumps(provenance,indent=2,allow_nan=False)+'\n')
    print('Exported:',', '.join(p.name for p in paths))
    return paths


def panel(ax,label,title):
    ax.set_title(f'({label})  {title}',loc='left',fontweight='semibold',pad=8)


def box(ax,x,y,w,h,text,color='#EAF2F6',fontsize=8):
    patch=FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.08,rounding_size=0.1',
                        facecolor=color,edgecolor='#64727E',linewidth=.8)
    ax.add_patch(patch); ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=fontsize)
    return (x,y,w,h)


def arrow(ax,start,end,label=None,color='#485460',style='-',bend=0):
    patch=FancyArrowPatch(start,end,arrowstyle='-|>',mutation_scale=9,lw=.9,color=color,
                          linestyle=style,connectionstyle=f'arc3,rad={bend}')
    ax.add_patch(patch)
    if label: ax.text((start[0]+end[0])/2,(start[1]+end[1])/2+.16,label,ha='center',fontsize=7,color=color)


def verified_sources(provenance,names):
    record=read_json(provenance)
    hashes=record.get('source_hashes',record.get('training_signature',{}).get('source_hashes',{}))
    for name in names:
        if hashes.get(name)!=sha(ROOT/'code'/name):
            raise ValueError(f'Source changed since recorded experiment: {name}; review schematic before exporting')


def completed_evaluation(directory):
    directory=Path(directory)
    if not (directory/'status.json').exists() or not read_json(directory/'status.json').get('complete'):
        raise ValueError(f'Analysis is incomplete: {directory}')
    record=read_json(directory/'provenance.json')
    if record['contract']['split']!='validation': raise ValueError('Preview notebooks accept validation analyses only')
    return record


def run_inventory():
    rows=[]
    specs=[('Residual legacy','residual-full50-s20261016',2332177,2118),
           ('Residual direct 5×','direct5-full50-s20261016',2959953,2118),
           ('Corrected diffusion','diffusion-full50-s20261016',4910593,256)]
    for name,directory,parameters,nval in specs:
        path=ROOT/'runs'/directory
        if not (path/'provenance.json').exists():
            rows.append(dict(model=name,state='not started')); continue
        config=read_json(path/'provenance.json')['config']
        complete=(path/'status.json').exists() and read_json(path/'status.json').get('complete',False)
        progress=read_json(path/'progress.json') if (path/'progress.json').exists() else {}
        rows.append(dict(model=name,state='complete' if complete else 'running / inspect status',
            epoch=progress.get('epoch',0),planned_epochs=config['epochs'],batch=config['batch_size'],
            parameters=parameters,validation_monitor=nval,representation=config.get('representation','asinh'),
            selection='generated physical MAE' if 'diffusion' in directory else 'masked normalized MAE',
            wandb=(path/'wandb_url.txt').read_text().strip() if (path/'wandb_url.txt').exists() else ''))
    return pd.DataFrame(rows)
