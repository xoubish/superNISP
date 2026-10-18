"""Publication-style contact sheet, assembled PDFs, and color-vision proof sheets."""
from pathlib import Path
import json
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import fitz
from figure_utils import FIGURES, style, sha

STEMS=['fig01_sample_v08','fig02_sample_properties_v06','fig03_diffusion_architecture_v06',
       'fig04_residual_architecture_v06','fig05_galaxy_examples_v08','fig06_model_comparison_v07',
       'fig07_shape_hexbin_v06','fig08_shape_dependence_v04','fig09_shear_response_v06']
# Machado et al. (2009), severity 100 matrices. Applied to linear-light sRGB.
# Source: https://www.inf.ufrgs.br/~oliveira/pubs_files/CVD_Simulation/CVD_Simulation.html
# Also tabulated in https://github.com/njsmith/colorspacious/blob/master/colorspacious/cvd.py
MATRICES={
    'Protanopia':[[.152286,1.052583,-.204868],[.114503,.786281,.099216],[-.003882,-.048116,1.051998]],
    'Deuteranopia':[[.367322,.860646,-.227968],[.280085,.672501,.047413],[-.011820,.042940,.968881]],
    'Tritanopia':[[1.255528,-.076749,-.178779],[-.078411,.930809,.147602],[.004733,.691367,.303900]],
}

def simulate(rgb,case):
    linear=np.where(rgb<=.04045,rgb/12.92,((rgb+.055)/1.055)**2.4)
    if case=='Grayscale':
        gray=linear@np.array([.2126,.7152,.0722]);transformed=np.repeat(gray[:,:,None],3,axis=2)
    else:transformed=linear@np.array(MATRICES[case]).T
    transformed=np.clip(transformed,0,1)
    return np.where(transformed<=.0031308,12.92*transformed,1.055*transformed**(1/2.4)-.055)


def main():
    style(10)
    out=FIGURES/'exports'
    page=fitz.open()
    fig,axes=plt.subplots(3,3,figsize=(16,14),layout='constrained')
    for i,(ax,stem) in enumerate(zip(axes.flat,STEMS),1):
        with fitz.open(out/(stem+'.pdf')) as original:page.insert_pdf(original)
        ax.imshow(Image.open(out/(stem+'.png')));ax.axis('off')
        ax.set_title(f'Figure {i}',loc='left',fontsize=12,fontweight='bold')
    page.save(out/'all_figures_style_v01.pdf');page.close()
    fig.savefig(out/'all_figures_style_v01.png',dpi=200,bbox_inches='tight');plt.close(fig)
    for stem in [STEMS[5],STEMS[7]]:
        rgb=Image.open(out/(stem+'.png')).convert('RGB')
        rgb.thumbnail((1600,900));rgb=np.array(rgb)/255.
        fig,axes=plt.subplots(5,1,figsize=(14,19),layout='constrained')
        for ax,case in zip(axes,['Original','Protanopia','Deuteranopia','Tritanopia','Grayscale']):
            ax.imshow(rgb if case=='Original' else simulate(rgb,case));ax.axis('off')
            ax.set_title(case,loc='left',fontsize=10)
        fig.savefig(out/(stem+'_vision_proof.png'),dpi=180,bbox_inches='tight');plt.close(fig)
    (out/'visual_style_review_v01.json').write_text(json.dumps(dict(
        figures={s:{ext:sha(out/(s+'.'+ext)) for ext in ['pdf','png','svg']} for s in STEMS},
        source_sha256=sha(__file__),color_simulation='Machado et al. 2009 severity 100 on linear sRGB; out-of-gamut clipped for display',
        matrices=MATRICES,scope='Visual proof, not a universal accessibility certification'),indent=2)+'\n')
    print('Created nine-page PDF, contact sheet, and Figure 6/8 vision proofs.')

if __name__=='__main__':main()
