from common import *
from collections import Counter
import csv

def write(name,rows):
    with (ROOT/name).open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,list(dict.fromkeys(k for r in rows for k in r)));w.writeheader();w.writerows(rows)

def noise_summary(rows):
    sig=np.array([r['model'][0]/.2 for r in rows]);rr=np.array([r['model'][1][0]-r['true_rho'] for r in rows])
    return dict(draws=len(rows),statuses=dict(Counter(r['info']['status'] for r in rows)),sigma_ratio_quantiles=np.quantile(sig,[0,.1,.5,.9,1]),rho_rmse=float(np.sqrt(np.mean(rr**2))),extreme_sigma=int(np.sum((sig<.5)|(sig>2))),rho_gt_095=sum(abs(r['model'][1][0])>.95 for r in rows))

def main():
    files=sorted((ROOT/'time_validation').glob('*.json'));assert len(files)==69;purefiles=sorted((ROOT/'time_noise').glob('*.json'));assert len(purefiles)==9
    rows=[];boots=[]
    for path in files:
        d=read(path);f=read(ROOT/'time_validation_frozen'/path.name);assert sha(ROOT/'time_validation_frozen'/path.name)==d['frozen_sha256'];meta={k:d[k] for k in ('system','profile','seed')}
        for r in d['rows']:
            rows.append(dict(**meta,variable=r['variable'],method=r['method'],sigma_ratio=r['sigma_ratio'],max_rho_error=float(np.max(np.abs(np.array(r['model'][1])-r['true_rho']))),status=r['info']['status']))
        for b in f['bootstrap']:
            for method in ('time','local6'):
                for v,(m,info) in b[method].items():
                    base=f['time'][v][0];ratio=m[0]/base[0];boots.append(dict(**meta,draw=b['draw'],method=method,variable=v,sigma_ratio=ratio,max_rho_error=float(np.max(np.abs(np.array(m[1])-base[1]))),max_abs_rho=max(abs(r) for r in m[1]),extreme_sigma=ratio<.5 or ratio>2,status=info['status']))
    systems=[];bootgroups=[]
    for n,p in JOBS:
        for method in ('time','local6','time_noise_only'):
            rr=[r for r in rows if (r['system'],r['profile'],r['method'])==(n,p,method)]
            systems.append(dict(system=n,profile=p,method=method,variable_fits=len(rr),sigma_ratio_mean=float(np.mean([r['sigma_ratio'] for r in rr])),sigma_ratio_min=min(r['sigma_ratio'] for r in rr),sigma_ratio_max=max(r['sigma_ratio'] for r in rr),max_rho_error=max(r['max_rho_error'] for r in rr),warnings=sum(r['status']!='ok' for r in rr)))
    for n,p in sorted({(r['system'],r['profile']) for r in boots}):
        for method in ('time','local6'):
            rr=[r for r in boots if (r['system'],r['profile'],r['method'])==(n,p,method)];ss=np.array([r['sigma_ratio'] for r in rr]);rho=np.array([r['max_rho_error'] for r in rr]);bootgroups.append(dict(system=n,profile=p,method=method,fits=len(rr),sigma_ratio_min=float(ss.min()),sigma_ratio_median=float(np.median(ss)),sigma_ratio_max=float(ss.max()),rho_error_median=float(np.median(rho)),rho_error_max=float(rho.max()),rho_gt_095=sum(r['max_abs_rho']>.95 for r in rr),extreme_sigma=sum(r['extreme_sigma'] for r in rr),warnings=sum(r['status']!='ok' for r in rr)))
    pure={m:[] for m in ('time','local6','previous_ar2')};pure['time']=[r for p in purefiles for r in read(p)['rows']];pure['local6']=read(ROOT/'noise_trials.json')['rows'];pure['previous_ar2']=read(PREVIOUS/'noise_trials_score.json')['rows'];comparison=[]
    for method,rr in pure.items():
        for n in (301,320,1001):
            for rho in (0.,.4,-.4):comparison.append(dict(method=method,n=n,rho=rho,**noise_summary([r for r in rr if (r['n'],r['true_rho'])==(n,rho)])))
    out=dict(status='COMPLETE',fresh_anchors=69,fresh_variable_fits=len(rows),bootstrap_fields=192,bootstrap_variable_fits=len(boots),fresh_statuses={m:dict(Counter(r['status'] for r in rows if r['method']==m)) for m in ('time','local6','time_noise_only')},systems=systems,bootstrap=bootgroups,pure_noise=comparison,source_sha256=hashes('analyze_time.py'),input_sha256={str(p):sha(p) for p in files+purefiles+[ROOT/'noise_trials.json',PREVIOUS/'noise_trials_score.json']})
    write('time_fresh_parameters.csv',rows);write('time_bootstrap_parameters.csv',boots);write('time_system_parameters.csv',systems);write('time_bootstrap_summary.csv',bootgroups);dump(ROOT/'time_summary.json',out)
    print(json.dumps(dict(fresh_statuses=out['fresh_statuses'],bootstrap=bootgroups,pure_time=[r for r in comparison if r['method']=='time']),indent=2,default=lambda x:x.tolist()))
    plot(systems)

def plot(rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(10,5));x=np.arange(15)
    for method,shift,label in [('local6',-.12,'Local polynomial'),('time',.12,'Temporal likelihood')]:
        rr=[next(r for r in rows if (r['system'],r['profile'],r['method'])==(n,'iid',method)) for n in SYSTEMS];med=np.array([r['sigma_ratio_mean'] for r in rr]);low=np.array([r['sigma_ratio_min'] for r in rr]);high=np.array([r['sigma_ratio_max'] for r in rr]);ax.errorbar(x+shift,med,yerr=np.stack([med-low,high-med]),fmt='o',capsize=3,label=label)
    ax.axhline(1,color='gray',ls='--');ax.set_xticks(x,SYSTEMS,rotation=45,ha='right');ax.set_ylabel('Estimated / injected noise amplitude');ax.set_title('Fresh iid noise estimates: mean and range across anchors45-47 and variables');ax.legend();fig.tight_layout();fig.savefig(ROOT/'time_noise_comparison.png',dpi=180);fig.savefig(ROOT/'time_noise_comparison.pdf');plt.close(fig)
if __name__=='__main__':main()
