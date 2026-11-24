"""Posthoc comparison after the frozen p64 study; no estimator is retuned."""
from compat_study import *
from scipy.linalg import svd,qr,cholesky,cho_solve

@lru_cache(None)
def geometry(n,p):
    u,s,v=svd(np.diff(G.basis(n,p),axis=0),full_matrices=False);assert s[-1]<1e-12*s[0];return u[:,:-1],float((n-p)*np.log(2)-np.log(n)+2*np.log(s[:-1]).sum())

@lru_cache(None)
def contrast_cov(n,p,rho):
    q=qr(G.basis(n,p),mode='full')[0][:,p:];r=rho**abs(np.arange(n)[:,None]-np.arange(n)[None,:]);l=cholesky(q.T@r@q,lower=True);return q,l,float(2*np.log(np.diag(l)).sum())

def assess(y,sigma,rho,p,fit=None):
    reg=G.Regression(y,p);fit=reg.fit() if fit is None else fit;nll0=reg.df*(fit['objective']+1);value=H.likelihood(reg,sigma,rho);q,l,ld=contrast_cov(len(y),p,rho);z=q.T@y;dense=reg.df*np.log(sigma*sigma)+float(z@cho_solve((l,True),z))/(sigma*sigma)+ld;np.testing.assert_allclose(value,dense,rtol=3e-8,atol=2e-8);u,lg=geometry(len(y),p);dy=np.diff(y);res=dy-u@(u.T@dy);eta=float(res@res)/(2*reg.df);limit=reg.df*(np.log(eta)+1)+lg;lr=value-nll0;tail=limit-nll0
    return dict(p=p,fit_status=fit['status'],sigma_ratio=fit['sigma']/sigma,rho_error=fit['rho']-rho,lr=lr,limit_lr=tail,nominal_covered=lr<=H.CUTOFF,unbounded_certified=tail<H.CUTOFF-1e-6,limit_below_stored_optimum=tail<-1e-5)

def one(key):
    d=read(ROOT/'signal_frozen'/(key+'.json'));e=Point.case(d['name'],d['profile'],d['seed']);y=e.raw['u'];sigma=.05*np.std(e.p.data['u']);rows=[assess(y,sigma,e.rho,p,d['cv_info']['full_fits'][str(p)]) for p in (32,48,64)];old=read(ROOT/'signal_results'/(key+'.json'));np.testing.assert_allclose(rows[-1]['lr'],old['lr'],rtol=1e-8,atol=2e-8);np.testing.assert_allclose(rows[-1]['limit_lr'],old['limit_lr'],rtol=1e-8,atol=2e-8);return dict(key=key,cohort=d['cohort'],name=d['name'],profile=d['profile'],rows=rows)

def ctrl(n,j):
    y=controlled(1,n,3,j);return dict(key=f'control_n{n}_positive_{j}',cohort='control',name=str(n),profile='positive',rows=[assess(y,1.,.4,p) for p in (32,48,64)])

def main():
    assert read(ROOT/'process.json')['status']=='COMPLETE';dump(ROOT/'dimension_launch.json',dict(status='FROZEN',role='Posthoc sensitivity to fixed mean dimension, designed after seeing p64 unboundedness. No adaptive method or primary result is changed; not fresh validation of a replacement.',source_sha256=src('dimension_diagnostic.py')));rows=[]
    with ProcessPoolExecutor(max_workers=8) as pool:
        jobs=[pool.submit(one,p.stem) for p in sorted((ROOT/'signal_frozen').glob('*.json'))]+[pool.submit(ctrl,n,j) for n in (320,1001) for j in range(64)]
        for f in as_completed(jobs):rows.append(f.result())
    dump(ROOT/'dimension_diagnostic.json',dict(status='PASS',role=read(ROOT/'dimension_launch.json')['role'],rows=rows,source_sha256=src('dimension_diagnostic.py'),input_sha256={str(ROOT/'signal_all_frozen.json'):sha(ROOT/'signal_all_frozen.json')}));print('DIMENSION DIAGNOSTIC PASS',len(rows),flush=True)

if __name__=='__main__':main()
