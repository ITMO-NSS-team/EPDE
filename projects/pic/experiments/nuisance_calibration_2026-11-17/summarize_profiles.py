from calibration_context import *
from collections import Counter
from scipy.stats import beta

def stats(x):return dict(min=float(min(x)),median=float(np.median(x)),max=float(max(x)),q05=float(np.quantile(x,.05)),q95=float(np.quantile(x,.95)))
def coverage(values):
    n=len(values);k=sum(v is True for v in values);u=sum(v is None for v in values);lo=0. if k==0 else float(beta.ppf(.025,k,n-k+1));hi=1. if k+u==n else float(beta.ppf(.975,k+u+1,n-k-u))
    return dict(n=n,covered=k,missed=n-k-u,unresolved=u,coverage_lower=k/n,coverage_upper=(k+u)/n,binomial95_interval=[lo,hi])

def main():
    assert read(ROOT/'process.json')['status']=='COMPLETE';summary={};noise={};numerics={};failures=[]
    for cohort in ('model_validation','ode'):
        rows=[read(p) for p in (ROOT/cohort).glob('*.json')];assert len(rows)==192;summary[cohort]={};noise[cohort]={};numerics[cohort]={}
        for profile in ('iid','positive','negative'):
            rr=[r for r in rows if r['profile']==profile];summary[cohort][profile]={kind:{method:coverage([r['coverage'][kind][method] for r in rr]) for method in ('lr','wald')} for kind in ('nominal','model_calibrated')};noise[cohort][profile]=dict(sigma_ratio=stats([r['model'][0]/np.exp(r['true_noise'][0]) for r in rr]),rho_error=stats([r['model'][1][0]-np.tanh(r['true_noise'][1]) for r in rr]));numerics[cohort][profile]=dict(fit_status=dict(Counter(r['fit_info']['status'] for r in rr)),lr_status=dict(Counter(r['evaluation']['lr_status'] for r in rr)),wald_status=dict(Counter(r['evaluation']['wald_status'] for r in rr)),near_unit=sum(abs(r['model'][1][0])>.95 for r in rr),guard_triggers=sum(r['fit_info']['guard_triggered'] for r in rr))
        for r in rows:
            if not all(v is True for cc in r['coverage'].values() for v in cc.values()):failures.append(dict(key=r['key'],coverage=r['coverage'],lr=r['evaluation']['lr'],wald=r['evaluation']['wald']))
    dump(ROOT/'profile_summary.json',dict(status='COMPLETE',coverage=summary,noise=noise,numerics=numerics,failures=failures,source_sha256=src('summarize_profiles.py','protocol.txt')));print(json.dumps(dict(coverage=summary,noise=noise,numerics=numerics)),flush=True)

if __name__=='__main__':main()
