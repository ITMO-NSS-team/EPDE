from adequacy_context import *
import adequacy_fit as A
from scipy.linalg import cholesky,solve_triangular,qr
from scipy.stats import f

def dense_f(y,p,rho):
    n=len(y);r=rho**abs(np.arange(n)[:,None]-np.arange(n)[None,:]);l=cholesky(r,lower=True);yw=solve_triangular(l,y,lower=True);x=solve_triangular(l,G.basis(n,p),lower=True);b=solve_triangular(l,A.union_basis(n,p),lower=True);q0=qr(x,mode='economic')[0];q1=qr(b,mode='economic')[0];e0=yw-q0@(q0.T@yw);e1=yw-q1@(q1.T@yw);v0=float(e0@e0);v1=float(e1@e1);df1=b.shape[1]-p;df2=n-b.shape[1];stat=(v0-v1)/df1/(v1/df2);return dict(q0=v0,q1=v1,stat=stat,pvalue=float(f.sf(stat,df1,df2)))

def dense_limit(y,p):
    n=len(y);q=qr(G.basis(n,p),mode='full')[0][:,p:];k=-q.T@abs(np.arange(n)[:,None]-np.arange(n)[None,:])@q;l=cholesky((k+k.T)/2,lower=True);z=solve_triangular(l,q.T@y,lower=True);eta=float(z@z)/(n-p);return (n-p)*(np.log(eta)+1)+2*np.log(np.diag(l)).sum()

def main():
    rows=[];rng=np.random.default_rng(991100)
    for n in (192,320):
        y=rng.normal(size=n)+np.sin(np.linspace(0,20,n))
        for p in G.SIZES:
            reg=G.Regression(y,p)
            for rho in (-.8,0.,.4,.95,.999999):
                v,d=reg.evaluate(np.arctanh(rho),True,True);fit=dict(**d,objective=v,transformed_rho=np.arctanh(rho),status='boundary' if rho>.99999 else 'ok');s=A.screen(y,p,fit);ref=dense_f(y,p,rho)
                for a,b in [(s['small_quadratic'],ref['q0']),(s['large_quadratic'],ref['q1']),(s['f_statistic'],ref['stat']),(s['pvalue'],ref['pvalue'])]:np.testing.assert_allclose(a,b,rtol=3e-7,atol=1e-8)
                rows.append(dict(n=n,p=p,rho=rho,stat_difference=s['f_statistic']-ref['stat']))
            np.testing.assert_allclose(A.boundary(y,p)['nll'],dense_limit(y,p),rtol=3e-9,atol=2e-8)
    dump(ROOT/'checks.json',dict(status='PASS',dense_cases=rows,source_sha256=src('adequacy_fit.py','adjacent_fit.py','check_adequacy.py','protocol.txt')));print('CHECKS PASS',len(rows),flush=True)

if __name__=='__main__':main()
