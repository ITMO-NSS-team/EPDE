from compat_context import *
from scipy.stats import chi2
from scipy.linalg import qr,cholesky,cho_solve
from functools import lru_cache

CUTOFF=float(chi2.ppf(.975,2))

def likelihood(reg,sigma,rho):
    v,d=reg.evaluate(float(np.arctanh(rho)),True,True)
    return float(reg.df*np.log(sigma*sigma)+d['quadratic']/(sigma*sigma)+d['logdet_covariance']+d['logdet_mean_gram'])

def select(y,info):
    reg=G.Regression(y,64);base=info['full_fits']['64'];nll0=reg.df*(base['objective']+1);rows=[]
    for p in G.SIZES:
        f=info['full_fits'][str(p)];value=likelihood(reg,f['sigma'],f['rho']);lr=value-nll0;score=next(r['score'] for r in info['selection'] if r['p']==p);rows.append(dict(p=p,lr=lr,cv=score,compatible=lr<=CUTOFF,status=f['status']))
    assert rows[-1]['compatible'];best=min([r for r in rows if r['compatible']],key=lambda r:(r['cv'],r['p']));fit=info['full_fits'][str(best['p'])];status=fit['status']
    if min(r['lr'] for r in rows)<-1e-6:status='unresolved_rich_optimum'
    return [fit['sigma'],[fit['rho']],[2]],dict(selected_p=best['p'],status=status,cutoff=CUTOFF,rows=rows)

@lru_cache(None)
def contrasts(n):return qr(G.basis(n,64),mode='full',check_finite=False)[0][:,64:]

@lru_cache(None)
def limit_geometry(n):
    q=contrasts(n);distance=abs(np.arange(n)[:,None]-np.arange(n)[None,:]);k=-q.T@distance@q;k=(k+k.T)/2;l=cholesky(k,lower=True);return l,float(2*np.log(np.diag(l)).sum())

def boundary_limit(y):
    q=contrasts(len(y));l,ld=limit_geometry(len(y));z=q.T@y;quadratic=float(z@cho_solve((l,True),z));df=q.shape[1];eta=quadratic/df;return dict(eta=eta,nll=float(df*(np.log(eta)+1)+ld),df=df)

def dense_likelihood(y,sigma,rho):
    q=contrasts(len(y));z=q.T@y;r=rho**abs(np.arange(len(y))[:,None]-np.arange(len(y))[None,:]);v=q.T@r@q;l=cholesky((v+v.T)/2,lower=True);quad=float(z@cho_solve((l,True),z));return float(len(z)*np.log(sigma*sigma)+quad/(sigma*sigma)+2*np.log(np.diag(l)).sum())
