"""Recover exact integer histogram counts from the frozen Figure 2 vector export.

The original multi-GB catalog is on the mosaic machine. SVG step heights encode
counts with enough precision to recover their unique integer values; all six
selected-sample histograms are independently checked against local catalog rows.
"""
import json
import re
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
from figure_utils import ROOT, FIGURES, DATA, read_json, sha
from parent_sample import valid_values


def main():
    svg=FIGURES/'exports/fig02_sample_properties_v05.svg'
    provenance=FIGURES/'exports/fig02_sample_properties_v05.json'
    old=read_json(provenance)
    assert sha(svg)==old['exports'][svg.name]
    metadata=FIGURES/'inputs/parent_sample_v05.json'
    assert sha(metadata)==old['inputs']['paper_figures/inputs/parent_sample_v05.json']
    for name in ['manifest.json','objects.csv']:
        assert sha(DATA/name)==old['inputs']['data/ready/supernisp_nir_y_f115w_v1/'+name]
    objects=pd.read_csv(DATA/'objects.csv')
    root=ET.parse(svg).getroot();ns={'s':'http://www.w3.org/2000/svg'}
    result={};max_error=0.
    for i,column,bin_key in [(1,'ACS_F814W_MAG','magnitude_bins'),(2,'ez_z_phot','redshift_bins')]:
        axes=root.find(f".//s:g[@id='axes_{i}']",ns)
        ticks=[]
        for g in axes.findall('.//s:g',ns):
            if g.get('id','').startswith('ytick_'):
                label=g.find('.//s:text',ns);use=g.find('.//s:use',ns)
                ticks.append((float(label.text),float(use.get('y'))))
        values,positions=np.array(ticks).T
        slope,intercept=np.polyfit(values,positions,1)
        np.testing.assert_allclose(slope*values+intercept,positions,atol=1.e-6)
        paths=[g.find('s:path',ns) for g in axes.findall('s:g',ns) if g.get('id','').startswith('patch_')]
        steps=[p for p in paths if len(p.get('d',''))>500]
        assert len(steps)==4
        bins=np.array(old['settings'][bin_key]);result[column]=dict(bins=bins.tolist(),populations={})
        for name,path in zip(['parent','train','validation','test'],steps):
            xy=np.array([float(x) for x in re.findall(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?',path.get('d'))]).reshape(-1,2)
            heights=xy[1:2*len(bins)-1:2,1]
            assert len(heights)==len(bins)-1
            accounting=old['settings']['parent']['histogram_accounting'][column][name]
            raw=(heights-intercept)/slope*accounting['valid_measurements']*np.diff(bins)
            counts=np.rint(raw).astype(int)
            error=float(np.max(abs(raw-counts)));max_error=max(max_error,error)
            assert error<.002, (column,name,error)
            assert np.all(counts>=0)
            assert counts.sum()==accounting['valid_measurements']-accounting['outside_display']
            if name!='parent':
                vals=valid_values(objects[objects.split==name],column,(-99,99,-999,999))
                np.testing.assert_array_equal(counts,np.histogram(vals,bins)[0])
            result[column]['populations'][name]=dict(counts=counts.tolist(),**accounting)
    out=FIGURES/'inputs/parent_histograms_v05.json'
    assert not out.exists()
    out.write_text(json.dumps(dict(complete=True,source_svg=str(svg.relative_to(ROOT)),source_svg_sha256=sha(svg),
        source_export_sha256=sha(provenance),source_metadata_sha256=sha(metadata),
        original_catalog_sha256=old['settings']['parent']['catalog_sha256'],
        objects_sha256=sha(DATA/'objects.csv'),recovery_source_sha256=sha(__file__),
        max_integer_count_residual=max_error,selected_histograms_independently_reproduced=6,
        method='Invert SVG linear y-axis transform, then uniquely recover integer counts; all totals and six selected histograms validated.',
        histograms=result),indent=2)+'\n')
    print('Recovered exact integer counts; maximum residual:',max_error)

if __name__=='__main__':main()
