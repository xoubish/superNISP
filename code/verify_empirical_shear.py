"""Reconstruct responses independently from frozen per-object measurements."""
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'runs/shear-response-empirical-v01'
METHODS=['native_observed','native_regauss','resnet_observed','target_observed']


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def matrix(e):
    return np.column_stack(((e[1]-e[2])/.02,(e[3]-e[4])/.02))


def main():
    report=json.loads((RUN/'report.json').read_text())
    status=json.loads((RUN/'status.json').read_text())
    assert (ROOT/'runs/shear-response-empirical-v01-console/exit_status').read_text().strip()=='0'
    assert status['complete'] and report['complete'] and status['cells']==len(report['cells'])==48
    assert sha(RUN/'report.json')==status['report_sha256']
    assert sha(RUN/'measurements.npz')==report['measurements_sha256']
    for name,h in report['contract']['source_hashes'].items(): assert sha(RUN/name)==h
    assert sha(RUN/'protocol.md')==report['contract']['protocol_sha256']
    assert sha(ROOT/'runs/residual-full50-s20261016/best.pt')==report['contract']['checkpoint_sha256']
    assert sha(ROOT/'paper/diagnostics/empirical_psf_controls_v01.json')==report['contract']['controls_sha256']
    attempted=0;matched=0;ratios_checked=0
    with np.load(RUN/'measurements.npz') as a:
        for ri,radius in enumerate([.18,.35,.65]):
            for si,snr in enumerate([None,20,50,100]):
                cell=next(c for c in report['aggregated'] if c['half_light_radius']==radius and c['snr']==snr)
                data=[]
                for pi in range(4):
                    k=f'psf{pi}_radius{ri}_snr{si}'
                    e=a[k+'__e'];good=a[k+'__good'];truth=a[k+'__truth'];codes=a[k+'__status']
                    assert np.isfinite(e[good]).all() and np.isnan(e[~good]).all()
                    assert (codes[good]==0).all()
                    assert good.shape==e.shape[:-1]
                    data.append((e,good,truth))
                    per=next(c for c in report['cells'] if (c['psf_index'],c['half_light_radius'],c['snr'])==(pi,radius,snr))
                    assert len(e)==per['attempted']==(8 if snr is None else 32)
                    local=np.logical_and.reduce([good[:,:,m].all(axis=1) for m in [0,2,3]])
                    assert int(local.sum())==per['common_success']
                    for m,name in enumerate(METHODS):
                        assert good[:,:,m].sum(axis=0).tolist()==per['methods'][name]['success_by_branch']
                        available=bool(good[local,:,m].all())
                        assert available==per['methods'][name]['same_sample_available']
                        if available:
                            rr=matrix(e[local,:,m].mean(axis=0));ref=matrix(truth[local].mean(axis=0))
                            np.testing.assert_allclose(rr,per['methods'][name]['response'],rtol=1e-12,atol=1e-12)
                            np.testing.assert_allclose(np.trace(rr)/np.trace(ref),per['methods'][name]['relative_trace_response'],rtol=1e-12)
                    attempted+=len(e)
                common=np.logical_and.reduce([g[:,:,m].all(axis=1) for _,g,_ in data for m in [0,2,3]])
                assert int(common.sum())==cell['common_success']
                assert np.flatnonzero(common).tolist()==cell['common_indices']
                matched+=int(common.sum())*4
                for m,name in enumerate(METHODS):
                    if all(g[common,:,m].all() for _,g,_ in data):
                        rs=[];mats=[]
                        for pi,(e,g,t) in enumerate(data):
                            rr=matrix(e[common,:,m].mean(axis=0));ref=matrix(t[common].mean(axis=0))
                            mats.append(rr);rs.append(np.trace(rr)/np.trace(ref))
                            np.testing.assert_allclose(rr,cell['by_psf'][pi]['methods'][name]['response'],rtol=1e-12,atol=1e-12)
                        np.testing.assert_allclose(rs,cell['methods'][name]['per_psf_relative_trace_response'],rtol=1e-12)
                        np.testing.assert_allclose(np.mean(rs),cell['methods'][name]['relative_trace_response'],rtol=1e-12)
                        np.testing.assert_allclose(np.mean(mats,axis=0),cell['methods'][name]['response'],rtol=1e-12,atol=1e-12)
                        np.testing.assert_allclose([min(rs),max(rs)],cell['methods'][name]['psf_range'],rtol=1e-12)
                        ratios_checked+=1
                    else:assert name not in cell['methods']
    assert attempted==1248
    verification=dict(passed=True,source_sha256=sha(__file__),report_sha256=sha(RUN/'report.json'),
        measurements_sha256=sha(RUN/'measurements.npz'),attempted_object_psf_realizations=attempted,
        common_object_psf_realizations=matched,aggregate_ratios_checked=ratios_checked,
        source_and_checkpoint_hashes_verified=True,all_per_psf_responses_and_counts_reproduced=True)
    path=RUN/'verification.json'
    assert not path.exists()
    path.write_text(json.dumps(verification,indent=2)+'\n')
    bundle=dict(complete=True,experiment='Supplied NISP effective PSFs; original ResNet',
        report_path=str((RUN/'report.json').relative_to(ROOT)),report_sha256=sha(RUN/'report.json'),
        verification_path=str(path.relative_to(ROOT)),verification_sha256=sha(path),
        protocol_path='paper/empirical_shear_protocol.md',protocol_sha256=report['contract']['protocol_sha256'],
        measurements_sha256=report['measurements_sha256'],psfs=report['contract']['psfs'],
        aggregation=report['contract']['psf_aggregation'],cells=report['aggregated'],
        attempted_object_psf_realizations=attempted,common_object_psf_realizations=matched)
    path=ROOT/'paper_figures/inputs/shear_response_empirical_v01.json'
    assert not path.exists()
    path.write_text(json.dumps(bundle,indent=2,allow_nan=False)+'\n')
    print(json.dumps(verification,indent=2))
    for c in report['aggregated']:
        print(c['half_light_radius'],c['snr'],c['common_success'],'/',c['attempted'],
              {k:round(v['relative_trace_response'],4) for k,v in c['methods'].items()})

if __name__=='__main__':main()
