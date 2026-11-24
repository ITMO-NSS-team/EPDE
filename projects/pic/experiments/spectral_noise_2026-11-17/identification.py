from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parent
import stable_fit as F
np=F.np
from scipy.optimize import minimize_scalar
sys.path.insert(0,str(ROOT.parent/'joint_moment_regions_2026-11-17'))
import study as S

def oracle(y):
    n=len(y);inner=float(np.sum(y[1:-1]**2));ends=float(y[0]**2+y[-1]**2);lag=float(y[1:]@y[:-1])
    def quadratic(r):return ((1+r*r)*inner+ends-2*r*lag)/((1-r)*(1+r))
    def obj(r):return np.log(quadratic(r)/n)+(n-1)/n*(np.log1p(-r)+np.log1p(r))
    opt=minimize_scalar(obj,bounds=(-1+1e-6,1-1e-6),method='bounded',options={'xatol':1e-10});assert opt.success
    return float(np.sqrt(quadratic(opt.x)/n)),float(opt.x)

def main():
    rows=[]
    for n in (301,320,1001):
        for profile_id,rho in ((0,0.),(1,.4),(2,-.4)):
            for b in range(64):
                rng=np.random.default_rng(np.random.SeedSequence([950106,n,profile_id,b]));y=S.R.WF.ar_noise(rng,(n,),.2,[rho]);model,info=F.estimate(y,'strips');sig,cor=oracle(y)
                rows.append(dict(n=n,true_rho=rho,draw=b,field_sha256=S.R.digest(y),spectral_status=info['status'],spectral_sigma=model[0] if model else None,spectral_rho=model[1][0] if model else None,oracle_sigma=sig,oracle_rho=cor))
            print(n,rho,'DONE',flush=True)
    summary=[]
    for n in (301,320,1001):
        for rho in (0.,.4,-.4):
            rr=[r for r in rows if r['n']==n and r['true_rho']==rho]
            for model in ('spectral','oracle'):
                sigma=np.array([r[model+'_sigma'] for r in rr]);cor=np.array([r[model+'_rho'] for r in rr]);summary.append(dict(n=n,true_rho=rho,model=model,draws=len(rr),boundary=sum(r['spectral_status']!='ok' for r in rr) if model=='spectral' else 0,
                    sigma_ratio_quantiles=np.quantile(sigma/.2,[0,.1,.5,.9,1]),rho_error_rmse=float(np.sqrt(np.mean((cor-rho)**2))),rho_error_quantiles=np.quantile(cor-rho,[0,.1,.5,.9,1])))
    S.R.dump(ROOT/'identification.json',dict(status='COMPLETE',rows=rows,summary=summary,source_sha256={str(p):S.E.sha(p) for p in (Path(__file__),ROOT/'stable_fit.py',ROOT/'noise_fit.py',ROOT/'identification_protocol.txt')}))
    for row in summary:print(row,flush=True)

if __name__=='__main__':main()
