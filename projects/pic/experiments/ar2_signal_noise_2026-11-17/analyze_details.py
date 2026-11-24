"""Read-only paired comparisons and numerical diagnostics of archived fits."""
from pathlib import Path
import json,csv,hashlib
import score_fit as F
np=F.np
ROOT=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def dump(name,d):
    (ROOT/name).write_text(json.dumps(d,indent=2),encoding='utf-8')
def write(name,rows):
    with (ROOT/name).open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,list(rows[0]));w.writeheader();w.writerows(rows)

def main():
    sources={'ar1':F.PREVIOUS/'noise_trials_correlated.json','ar2_fd':ROOT/'noise_trials.json','ar2_score':ROOT/'noise_trials_score.json'}
    draws={k:read(p)['rows'] for k,p in sources.items()}
    key=lambda r:(r['n'],r['true_rho'],r['draw'])
    maps={k:{key(r):r for r in rr} for k,rr in draws.items()}
    assert all(len(m)==576 for m in maps.values())
    for k in maps['ar1']:
        assert len({m[k]['field_sha256'] for m in maps.values()})==1
    rows=[]
    for method,rr in draws.items():
        for n in (301,320,1001):
            for rho in (0.,.4,-.4):
                group=[r for r in rr if (r['n'],r['true_rho'])==(n,rho)]
                models=[r['model'] for r in group if r['model'] is not None]
                sigma=np.array([m[0]/.2 for m in models]);error=np.array([m[1][0]-rho for m in models])
                rows.append(dict(method=method,n=n,rho=rho,draws=len(group),available=len(models),incomplete=sum(r['info']['status']!='ok' for r in group),sigma_ratio_median=float(np.median(sigma)),rho_rmse=float(np.sqrt(np.mean(error**2)))))
    write('pure_noise_comparison.csv',rows)
    numeric=[];inputs=list(sources.values())
    for path in sorted((ROOT/'screen').glob('*.json')):
        new=ROOT/'score_screen'/path.name;inputs.extend([path,new])
        before={ (r['variable'],r['mode']):r for r in read(path)['rows'] }
        for r in read(new)['rows']:
            old=before[r['variable'],r['mode']]
            numeric.append(dict(system=r['system'],profile=r['profile'],variable=r['variable'],mode=r['mode'],
                old_status=old['info']['status'],new_status=r['info']['status'],
                old_objective=old['info'].get('objective'),new_objective=r['info'].get('objective'),
                old_sigma=old['info'].get('sigma'),new_sigma=r['info'].get('sigma'),
                old_rhos=old['info'].get('rhos'),new_rhos=r['info'].get('rhos')))
    diagnostics=[]
    for path in sorted((ROOT/'results').glob('*_ar2_*.json')):
        inputs.append(path);d=read(path)
        for v,info in d['fit_info'].items():
            good=[f for f in info['fits'] if f['success'] and np.isfinite(f['value']) and f['value']<1e40]
            best=min(good,key=lambda f:f['value']);theta=np.array(best['theta']);g=np.array(best['gradient']);dim=len(info['rhos'])
            lo=np.array([-1+1e-6]*dim+[-40.]*dim+[-1+1e-6]*(2*dim));hi=np.array([1-1e-6]*dim+[20.]*dim+[1-1e-6]*(2*dim))
            residual=float(np.max(np.abs(theta-np.clip(theta-g,lo,hi))))
            finite=[f['value'] for f in info['fits'] if np.isfinite(f['value']) and f['value']<1e40]
            diagnostics.append(dict(record=path.stem,variable=v,status=info['status'],selected_start=best['start'],termination=best['message'],projected_gradient_mapping=residual,selected_objective=best['value'],selected_minus_best_including_failed=best['value']-min(finite),successful_objective_spread=info['successful_objective_spread']))
    summary=dict(raw_screen_status_before={s:sum(r['old_status']==s and r['mode']=='raw' for r in numeric) for s in ('ok','boundary','optimizer_failure')},
        raw_screen_status_after={s:sum(r['new_status']==s and r['mode']=='raw' for r in numeric) for s in ('ok','boundary','optimizer_failure')},
        primary_variable_fits=len(diagnostics),projected_gradient_quantiles=np.quantile([r['projected_gradient_mapping'] for r in diagnostics],[0,.5,.9,.99,1]).tolist(),
        primary_projected_gradient_above_1e_minus3=sum(r['projected_gradient_mapping']>1e-3 for r in diagnostics),
        primary_selected_gap_above_1e_minus5=sum(r['selected_minus_best_including_failed']>1e-5 for r in diagnostics))
    dump('detailed_analysis.json',dict(status='COMPLETE',summary=summary,pure_noise=rows,screen_numeric=numeric,primary_optimizer=diagnostics,source_sha256={str(Path(__file__)):sha(Path(__file__))},input_sha256={str(p):sha(p) for p in inputs}))
    print(json.dumps(summary,indent=2))
    for r in diagnostics:
        if r['projected_gradient_mapping']>1e-3 or r['selected_minus_best_including_failed']>1e-5:print(r)

if __name__=='__main__':main()
