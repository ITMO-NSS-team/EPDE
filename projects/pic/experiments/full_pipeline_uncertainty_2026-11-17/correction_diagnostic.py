"""Post hoc attribution of missed bounds; oracle parameters are diagnostic only.

The method, pilots, cutoffs and primary results are unchanged. Correcting with
known generating noise and changing normalization without recalibration is NOT
a proposed procedure or a coverage result.
"""
from bridge import *
from full_pipeline_fast import Experiment,score
from scipy.linalg import eigh
import time


def one(path):
    d=read(path);e=Experiment(d['system'],d['seed'],d['profile']);e.model=d['model']
    with np.load(path.with_suffix('.npz')) as z:
        e.pilot={v:z['anchor_pilot_'+v] for v in e.raw}
        assert {v:R.digest(y) for v,y in e.raw.items()}==d['raw_sha256']
        oracle={v:[.05*float(np.std(y)),[e.rho]*y.ndim,e.model[v][2]] for v,y in e.p.data.items()}
        corrected=e.matrices(e.raw,oracle);ocov=e.metric(e.pilot,oracle);rows=[]
        for b in ('local','broad'):
            est=z[b+'_observed']-z[b+'_clean'];other=corrected[b]-z[b+'_clean'];shift=est-other;cv=z[b+'_anchor_covariance'];idx=e.active
            cov=cv[np.ix_(idx,idx)];g=S.gram(est)[np.ix_(idx,idx)];scale=np.sqrt(np.diag(cov))
            cc=cov/scale[:,None]/scale[None,:];gg=g/scale[:,None]/scale[None,:]
            values,vectors=eigh((gg+gg.T)/2,(cc+cc.T)/2);v=np.zeros(len(e.labels));v[idx]=vectors[:,-1]/scale
            primary=float((est@v)@(est@v));other2=float((other@v)@(other@v));shift2=float((shift@v)@(shift@v));cross=float(2*(other@v)@(shift@v))
            np.testing.assert_allclose(primary,d['actual_scores'][b],rtol=3e-7)
            np.testing.assert_allclose(primary,other2+shift2+cross,rtol=1e-10,atol=1e-12)
            mixed=dict(estimated_moments_estimated_covariance=primary,
                       oracle_moments_estimated_covariance=score(e,S.gram(other),cv)[0],
                       estimated_moments_oracle_covariance=score(e,S.gram(est),ocov[b])[0],
                       oracle_moments_oracle_covariance=score(e,S.gram(other),ocov[b])[0])
            rows.append(dict(bank=b,original_cutoffs={m:d['cuts'][m][b] for m in d['cuts']},scores=mixed,
                             original_worst_direction=dict(oracle_error_squared=other2,correction_shift_squared=shift2,cross_term=cross,total=primary)))
        out=dict(status='COMPLETE_DIAGNOSTIC',system=d['system'],profile=d['profile'],seed=d['seed'],rows=rows,
                 generating_noise_model=oracle,estimated_noise_model=e.model,
                 interpretation='Post hoc attribution for missed primary bounds. Oracle noise is never used '
                 'by the tested procedure. Scores with changed correction/covariance share the original '
                 'estimated pilot and are not recalibrated; do not interpret them as oracle coverage tests.',
                 source_sha256=sources('correction_diagnostic.py','full_pipeline_fast.py','exact_metric_pair.py'),
                 input_sha256={str(path):sha(path),str(path.with_suffix('.npz')):sha(path.with_suffix('.npz'))})
        dump(ROOT/'correction_diagnostics'/path.name,out);print(path.stem,rows,flush=True)


def main():
    seen=set();records=[]
    while True:
        for path in sorted((ROOT/'results').glob('*.json')):
            if path.stem in seen:continue
            d=read(path)
            if any(not x for counts in d['coverage'].values() for x in counts.values()):
                one(path);records.append(path.stem)
            seen.add(path.stem)
        if len(seen)==69:break
        if (ROOT/'processes_fast.json').exists():break
        time.sleep(2)
    dump(ROOT/'correction_diagnostic_summary.json',dict(status='COMPLETE' if len(seen)==69 else 'INCOMPLETE',
         observed_datasets=len(seen),diagnosed_records=records,source_sha256=sources('correction_diagnostic.py')))


if __name__=='__main__':main()
