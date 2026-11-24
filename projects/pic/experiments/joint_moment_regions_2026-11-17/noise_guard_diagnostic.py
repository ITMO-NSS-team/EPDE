import study as S
R,np,ROOT=S.R,S.np,S.ROOT
from pathlib import Path
from math import comb
from scipy.optimize import brentq
import json

def fit(y,axis,guard=True):
    # Exact research copy of current axis_fit, with only its variance guard
    # switchable. No production changes. No-cap is an attribution control.
    n=y.shape[axis];d=np.moveaxis(y,axis,0);var0=float(np.mean(np.var(d,axis=0)))
    floor=R.WF.EPS*float(np.sqrt(np.mean(y*y)));best,prev,m=None,None,0;trace=[]
    while n>2*(m+1)+2:
        m+=1;d=np.diff(d,axis=0);r0=float(np.mean(d*d));cm=comb(2*m,m)
        if not np.isfinite(r0) or cm>1e300:break
        sw=np.sqrt(r0/cm)
        if sw<=floor:break
        stop=prev is not None and sw>=prev;prev=sw;ratio=float(np.mean(d[1:]*d[:-1]))/r0
        fun=lambda rho:R.WF._g(m,rho,1)/R.WF._g(m,rho,0)-ratio;lo,hi=-1+1e-9,1-1e-9
        row=dict(order=m,ratio=ratio,white_amplitude=sw)
        if fun(lo)*fun(hi)<0:
            rho=brentq(fun,lo,hi);sig=float(np.sqrt(r0/R.WF._g(m,rho,0)));allowed=sig*sig<=var0
            row.update(rho=rho,sigma=sig,variance_guard_accepts=allowed)
            trace.append(row)
            if allowed or not guard:
                if best is not None and sig>=best[0]:break
                best=(sig,rho,m)
        else:trace.append(row)
        if stop and best is not None:break
    return best,trace,var0

def main():
    rows=[];traces=[]
    for name in ('ac','kdv_cossin','ks','wave'):
        for seed in (30,31,32):
            for profile in ('positive','negative'):
                e=S.Study(name,seed,profile,'ar')
                for v,y in e.raw.items():
                    true=.05*float(np.std(e.p.data[v]));pure=y-e.p.data[v]
                    for mode,field in [('raw',y),('noise_only',pure)]:
                        sig=[];nocap=[]
                        for axis in range(y.ndim):
                            capped,trace,var=fit(field,axis,True);uncapped,other,_=fit(field,axis,False);original=R.WF.axis_fit(field,axis)
                            assert capped==original
                            if capped is not None:sig.append(capped[0])
                            if uncapped is not None:nocap.append(uncapped[0])
                            if name=='ac' and seed==30 and profile=='positive':
                                n=field.shape[axis];lags=np.arange(1,n);factor=1-(n+2*np.sum((n-lags)*e.rho**lags))/(n*n)
                                traces.append(dict(system=name,seed=seed,mode=mode,axis=axis,samples=n,true_sigma=true,axis_variance=var,expected_noise_sample_variance=true**2*factor,
                                    selected=capped,uncapped=uncapped,trace=trace,uncapped_trace=other))
                        rows.append(dict(system=name,seed=seed,profile=profile,mode=mode,variable=v,
                            capped_sigma_ratio=min(sig)/true if sig else 0,uncapped_sigma_ratio=min(nocap)/true if nocap else 0))
    summary=[]
    for profile in ('positive','negative'):
        for mode in ('raw','noise_only'):
            rr=[r for r in rows if r['profile']==profile and r['mode']==mode]
            summary.append(dict(profile=profile,mode=mode,capped_min_median_max=np.quantile([r['capped_sigma_ratio'] for r in rr],[0,.5,1]),uncapped_min_median_max=np.quantile([r['uncapped_sigma_ratio'] for r in rr],[0,.5,1])))
    out=dict(status='COMPLETE',note='Guard removal is a post-hoc attribution ablation; not a promoted replacement and not a coverage test.',rows=rows,summary=summary,traces=traces,
        source_sha256={**S.source_hashes(),str(Path(__file__)):S.E.sha(__file__)})
    R.dump(ROOT/'noise_guard_diagnostic.json',out);print(json.dumps(R.jsonable(summary) if hasattr(R,'jsonable') else summary,default=lambda x:x.tolist(),indent=2))

if __name__=='__main__':main()
