"""Verify approved figure installation, manuscript references and quoted results."""
from pathlib import Path
import json,re,sys,hashlib
import fitz
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'paper_figures'))
from figure_utils import sha,notebook_source_hash
from review_visual_style import STEMS
paper=ROOT/'paper';tex=paper/'survey_enhancements.tex';s=tex.read_text()
labels=re.findall(r'\\label\{([^}]+)\}',s)
refs=re.findall(r'\\(?:ref|pageref)\*?\{([^}]+)\}',s)
assert len(labels)==len(set(labels))
assert not set(refs)-set(labels)
figs=re.findall(r'\\begin\{figure\*?\}.*?\\end\{figure\*?\}',s,re.S)
assert len(figs)==9
expected=['sample_overview_v08','sample_properties_v06','diffusion_architecture_v06','residual_architecture_v06','galaxy_examples_v08','model_comparison_v07','shape_hexbin_v06','shape_dependence_v04','shear_response_v06']
installed=[]
for number,(fig,stem,target) in enumerate(zip(figs,STEMS,expected),1):
 paths=re.findall(r'\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}',fig)
 assert paths==[f'figs/{target}.pdf']
 source=ROOT/'paper_figures/exports'/f'{stem}.pdf';dest=paper/paths[0]
 assert sha(source)==sha(dest)
 record=json.loads(source.with_suffix('.json').read_text())
 assert record['notebook_source_sha256']==notebook_source_hash(ROOT/'paper_figures'/record['notebook'])
 for p,h in record['exports'].items():assert sha(source.parent/p)==h
 for p,h in record['inputs'].items():assert sha(ROOT/p)==h
 installed.append(dict(number=number,path=str(dest.relative_to(ROOT)),sha256=sha(dest),
   notebook=record['notebook'],notebook_source_sha256=record['notebook_source_sha256']))
assert all(t not in s for t in ['ellipticity_plane_residuals.png','fig:plane_residuals','Orthogonal Distance Regression','25--35'])
# Structural checks, ignoring comments and escaped braces.
clean=re.sub(r'(?<!\\)%[^\n]*','',s)
stack=[]
for match in re.finditer(r'\\(begin|end)\{([^}]+)\}',clean):
 action,name=match.groups()
 if action=='begin':stack.append(name)
 else:assert stack.pop()==name
assert not stack
level=0
for ch in re.sub(r'\\[{}]','',clean):
 if ch=='{':level+=1
 elif ch=='}':level-=1;assert level>=0
assert level==0
# Reproduce all explicitly quoted Figure 8 endpoint values and intervals.
snr=json.loads((ROOT/'paper_figures/inputs/shape_snr_v02.json').read_text())
checks=[]
for case,index,expected_values in [('fixed',0,['19.9','13.8','25.3']),('adaptive',0,['7.8','3.2','11.4']),('fixed',-1,['63.0','57.3','68.3']),('adaptive',-1,['39.8','29.1','47.6'])]:
 row=snr['summaries'][case][index];r=row['native_comparison']['legacy']
 actual=[f'{v:.1f}' for v in [r['reduction_percent'][0],*r['ci95_percent'][0]]]
 assert actual==expected_values,(case,index,actual)
 assert all(v in s for v in actual)
 checks.append(dict(estimator=case,bin=row['label'],model='ResNet',gain_and_ci=actual))
for case,rows in snr['summaries'].items():
 for row in rows:
  for model in ['legacy','direct5']:assert row['native_comparison'][model]['reduction_percent'][0]>0
 assert sum(row['objects'] for row in rows)==2118
assert [r['common_successes'] for r in snr['summaries']['fixed']]==[179,441,387,322,117]
assert [r['common_successes'] for r in snr['summaries']['adaptive']]==[297,419,363,304,115]
r=snr['summaries']['fixed'][-1]['native_comparison']['diffusion']
assert [f'{v:.1f}' for v in [r['reduction_percent'][0],*r['ci95_percent'][0]]]==['38.5','26.6','47.2']
for row in snr['cross_estimator_common']['fixed']+snr['cross_estimator_common']['adaptive']:
 assert row['native_comparison']['legacy']['reduction_percent'][0]>0
bundle=json.loads((ROOT/'paper_figures/inputs/shear_response_empirical_v01.json').read_text())
cells={(c['half_light_radius'],c['snr']):c for c in bundle['cells']}
quoted=[(.18,100,'resnet_observed','0.800'),(.35,100,'resnet_observed','0.894'),(.18,100,'native_observed','0.417'),(.35,100,'native_observed','0.640'),(.18,100,'target_observed','0.907'),(.35,100,'target_observed','0.971'),(.65,20,'resnet_observed','0.715'),(.65,20,'native_observed','0.774'),(.65,50,'resnet_observed','0.854'),(.65,50,'native_observed','0.810'),(.65,100,'resnet_observed','0.921'),(.65,100,'native_observed','0.819'),(.18,None,'resnet_observed','0.806'),(.35,None,'resnet_observed','0.924'),(.65,None,'resnet_observed','0.951')]
for radius,snr_value,method,formatted in quoted:
 assert f"{cells[radius,snr_value]['methods'][method]['relative_trace_response']:.3f}"==formatted
 assert formatted in s
for radius in [.18,.35]:
 for snr_value in [20,50,100]:
  c=cells[radius,snr_value]
  assert c['common_success']==(31 if (radius,snr_value)==(.18,20) else 32)
  assert c['methods']['resnet_observed']['relative_trace_response']>c['methods']['native_observed']['relative_trace_response']
report=json.loads((ROOT/'runs/shear-response-empirical-v01/report.json').read_text())
assert len(report['cells'])==48
assert sum(c['attempted']*5 for c in report['cells'])==6240
assert sum(sum(c['methods']['native_regauss']['success_by_branch']) for c in report['cells'])==0
proof=paper/'diagnostics/figures_8_9_caption_proof_v01.pdf'
with fitz.open(proof) as pdf:
 assert len(pdf)==2
 for i,page in enumerate(pdf):
  assert f'Figure {i+8}:' in page.get_text()
  for block in page.get_text('dict')['blocks']:
   for line in block.get('lines',[]):
    for span in line['spans']:assert fitz.Rect(span['bbox']) in page.rect
out=dict(complete=True,scope='Local manuscript installation; no external publication',
 manuscript_sha256=sha(tex),manuscript_before_sha256=sha(Path(__file__).with_name('manuscript_before.tex')),
 figures=installed,all_references_resolve=True,latex_braces_and_environments_balanced=True,
 figure8_quoted_endpoints=checks,figure9_quoted_values=quoted,
 numerical_bundles={str(p.relative_to(ROOT)):sha(p) for p in [ROOT/'paper_figures/inputs/shape_snr_v02.json',ROOT/'paper_figures/inputs/shear_response_empirical_v01.json']},
 caption_proof=dict(path=str(proof.relative_to(ROOT)),sha256=sha(proof),pages=2,compile='passed; visual inspection passed'),
 full_manuscript_compile=dict(status='blocked',reason='A&A class requires unavailable etex.sty',log='paper/diagnostics/figures_8_9_install_v01/texcheck/survey_enhancements.log'),
 verifier_sha256=sha(__file__))
(paper/'diagnostics/figures_8_9_install_v01.json').write_text(json.dumps(out,indent=2)+'\n')
print('PASS: nine figures installed in order; all PDF/input/source hashes; references; TeX structure; quoted numbers and counts; two-page caption proof.')
