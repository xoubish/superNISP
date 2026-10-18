"""Verify the presentation-only Figure 1–9 revision and installed PDFs."""
from pathlib import Path
import sys,json,re
import nbformat
import fitz
sys.path.insert(0,'paper_figures')
from figure_utils import ROOT,FIGURES,sha,notebook_source_hash,STYLE_VERSION
from review_visual_style import STEMS
old_versions=['v07','v05','v05','v05','v07','v06','v05','v03','v05']
allowed=[{'frozen_display_reuse'}, {'colors','counts_source','normalization','test_access','histogram_accounting','catalog_sentinels'}, {'colors','borders'}, {'colors','borders'},set(),set(),{'hexbin'},set(),{'version'}]
records=[]
for idx,(stem,oldversion,allowed_changes) in enumerate(zip(STEMS,old_versions,allowed),1):
 new=json.loads((FIGURES/'exports'/f'{stem}.json').read_text())
 oldstem=stem.rsplit('_',1)[0]+'_'+oldversion
 old=json.loads((FIGURES/'exports'/f'{oldstem}.json').read_text())
 notebook=FIGURES/new['notebook'];n=nbformat.read(notebook,4)
 assert new['notebook_source_sha256']==notebook_source_hash(notebook),(stem,'source')
 assert new['helper_sha256']==sha(FIGURES/'figure_utils.py'),(stem,'helper')
 assert new['visual_style']['version']==STYLE_VERSION
 for c in n.cells:
  if c.cell_type=='code':
   assert c.execution_count is not None,(stem,'unexecuted cell')
   assert not any(o.output_type=='error' for o in c.get('outputs',[])),stem
 for key,h in new['inputs'].items():assert sha(ROOT/key)==h,(stem,key)
 for key,h in new['exports'].items():assert sha(FIGURES/'exports'/key)==h,(stem,key)
 changed={k for k in old['settings'].keys()|new['settings'].keys() if old['settings'].get(k)!=new['settings'].get(k)}
 assert changed<=allowed_changes,(stem,changed-allowed_changes)
 for key in old['inputs'].keys()&new['inputs'].keys():
  if key!='paper_figures/architecture_style.py':assert old['inputs'][key]==new['inputs'][key],(stem,key)
 with fitz.open(FIGURES/'exports'/f'{stem}.pdf') as pdf:
  assert len(pdf)==1
  fonts=sorted({f[3] for f in pdf[0].get_fonts()})
  assert all('NimbusSans' in f for f in fonts),(stem,fonts)
  for block in pdf[0].get_text('dict')['blocks']:
   for line in block.get('lines',[]):
    for span in line['spans']:
     assert fitz.Rect(span['bbox']) in pdf[0].rect + (-1,-1,1,1),(stem,span['text'])
 records.append(dict(figure=idx,stem=stem,changed_presentation_settings=sorted(changed),
   input_files_verified=len(new['inputs']),source_hash=new['notebook_source_sha256'],fonts=fonts,
   exports=new['exports']))
# Exact settings/values preserved in the scientific comparison figures.
for i in [4,5,7]:assert records[i]['changed_presentation_settings']==[]
assert len(list(FIGURES.glob('[0-9][0-9]_*.ipynb')))==9
assert not (FIGURES/'09_shape_robustness.ipynb').exists()
assert (FIGURES/'revisions/09_shape_robustness_retired.ipynb').exists()
old2=json.loads((FIGURES/'exports/fig02_sample_properties_v05.json').read_text())
new2=json.loads((FIGURES/'exports/fig02_sample_properties_v06.json').read_text())
assert old2['settings']['parent']==new2['settings']['parent']
cache=json.loads((FIGURES/'inputs/parent_histograms_v05.json').read_text())
assert cache['selected_histograms_independently_reproduced']==6
tex=(ROOT/'paper/survey_enhancements.tex').read_text()
installed=['sample_overview_v08','sample_properties_v06','diffusion_architecture_v06','residual_architecture_v06','galaxy_examples_v08','model_comparison_v07','shape_hexbin_v06']
for stem,target in zip(STEMS,installed):
 assert f'figs/{target}.pdf' in tex
 assert sha(ROOT/'paper/figs'/f'{target}.pdf')==sha(FIGURES/'exports'/f'{stem}.pdf')
oldtex=(FIGURES/'revisions/style_before_v01/survey_enhancements.tex').read_text()
record=dict(complete=True,style=STYLE_VERSION,active_notebooks=9,figures=records,
    all_current_inputs_exports_and_notebook_sources_verified=True,
    all_shared_numerical_input_hashes_unchanged=True,
    figure2_exact_histograms=cache['selected_histograms_independently_reproduced'],
    figure2_max_integer_recovery_residual=cache['max_integer_count_residual'],
    figure1_display='Exact embedded v07 image pixels reused; physical mosaic cache absent; stretch/orientation preserved',
    color_review='Grayscale and Machado 2009 severity-100 vision proofs for Figures 6/8; redundant markers, patterns and labels retained',
    installed_figures=installed,notebook_only_figures=[8,9],
    manuscript_before_sha256=sha(FIGURES/'revisions/style_before_v01/survey_enhancements.tex'),
    manuscript_after_sha256=sha(ROOT/'paper/survey_enhancements.tex'),
    manuscript_compile='Not run: etex.sty absent; standalone figure PDFs checked',
    verifier_source_sha256=sha(__file__))
(ROOT/'paper/diagnostics/figure_style_v01.json').write_text(json.dumps(record,indent=2)+'\n')
print('PASS: nine executed notebooks, inputs and source hashes, exports, fonts/text bounds, numerical settings, seven installed PDFs.')
