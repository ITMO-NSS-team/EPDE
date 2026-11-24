"""Paired conditional ablation on previously frozen forced-KdV draws."""
import methods as A
S,R,np,ROOT=A.S,A.R,A.np,A.ROOT
import json
from collections import Counter


def main():
    rows=[];inputs={}
    for seed in (6,7,8):
        rp=S.ROOT/'results'/f'kdv_cossin_seed{seed}.json';ap=S.ROOT/'training_arrays'/f'kdv_cossin_seed{seed}.npz'
        d=json.loads(rp.read_text());c=next(c for c in d['candidates'] if c['id']=='e0:reference');inputs.update({str(p):S.sha(p) for p in (rp,ap)})
        with np.load(ap) as z:
            for g in range(3):
                for bank in ('local','broad','mixed'):
                    point,pilot,draws,test=[z[f'g{g}_{bank}_{k}'] for k in ('point','pilot','draws','independent')]
                    w,meta=A.M.whitening(draws,pilot)
                    preparations={'ols':A.prepare(draws,pilot),'gls':None if w is None else A.prepare(draws,pilot,w)}
                    for method in ('ols','gls','corrected_ols','corrected_gls'):
                        prep=preparations['gls' if method.endswith('gls') else 'ols'];corrected=method.startswith('corrected')
                        if prep is None:
                            rows.append(dict(seed=seed,training=g,bank=bank,method=method,status='infeasible_covariance'));continue
                        pc,pi=A.normal(pilot,c,prep,corrected=corrected);oc,oi=A.normal(point,c,prep,corrected=corrected)
                        # Shared noiseless GLS/OLS reference isolates correction bias;
                        # subtracting noise from genuinely clean data is not a reference.
                        reference,ri=A.normal(pilot,c,prep,corrected=False);assert ri['status']=='ok'
                        samples=[]
                        for b,matrix in enumerate(test):
                            coef,info=A.normal(matrix,c,prep,corrected=corrected)
                            samples.append(dict(draw=b,status=info['status'],coefficients=coef))
                        good=np.asarray([r['coefficients'] for r in samples if r['status']=='ok'])
                        row=dict(seed=seed,training=g,bank=bank,method=method,status='ok',observed_coefficients=oc,observed_status=oi['status'],
                            noiseless_reference=reference,corrected_on_clean_coefficients=pc,corrected_on_clean_status=pi['status'],
                            successful_bootstrap=len(good),bootstrap_statuses=dict(Counter(r['status'] for r in samples)),samples=samples)
                        if len(good)>1:row.update(bootstrap_mean=good.mean(axis=0),bootstrap_sd=good.std(axis=0,ddof=1),
                                                 drift=good.mean(axis=0)-reference)
                        rows.append(row)
        print('conditional',seed,'done',flush=True)
    summary=[]
    for bank in ('local','broad','mixed'):
        for method in ('ols','gls','corrected_ols','corrected_gls'):
            rr=[r for r in rows if r['bank']==bank and r['method']==method];good=[r for r in rr if 'drift' in r]
            summary.append(dict(bank=bank,method=method,successful_bootstrap=sum(r.get('successful_bootstrap',0) for r in rr),
                                total_bootstrap=576,complete_batches=sum(r.get('successful_bootstrap',0)==64 for r in rr),
                                median_drift=None if not good else np.median([r['drift'] for r in good],axis=0),
                                median_sd=None if not good else np.median([r['bootstrap_sd'] for r in good],axis=0)))
    R.dump(ROOT/'conditional_results.json',dict(conditional_reused_seeds=[6,7,8],new_core15_selection_unchanged=True,
        independent_draws_per_batch=64,rows=rows,summary=summary,input_sha256=inputs,
        source_sha256={str(p):S.sha(p) for p in (A.__file__,__file__)}))
    print(json.dumps([{**r,'median_drift':None if r['median_drift'] is None else r['median_drift'].tolist(),
                      'median_sd':None if r['median_sd'] is None else r['median_sd'].tolist()} for r in summary],indent=2))


if __name__=='__main__':main()
