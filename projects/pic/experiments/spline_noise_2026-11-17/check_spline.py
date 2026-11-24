from spline_context import *
import spline_fit as G
from scipy.linalg import cholesky,cho_solve,qr
from scipy.stats import multivariate_t


def dense(reg,t):
    rho=np.tanh(t);cov=rho**abs(reg.idx[:,None]-reg.idx[None,:]);qq=qr(reg.x,mode='full')[0][:,reg.p:];d=qq.T@reg.y;c=qq.T@cov@qq;l=cholesky(c,lower=True);q=float(d@cho_solve((l,True),d));ld=2*np.log(np.diag(l)).sum();normalizer=np.linalg.slogdet(reg.x.T@reg.x)[1];return float(np.log(q/reg.df)+(ld+normalizer)/reg.df),np.sqrt(q/reg.df)


def main():
    rng=np.random.default_rng(990001);rows=[]
    for n in (64,96):
        y=rng.normal(size=n)+np.sin(np.linspace(0,6,n));idxs=[np.arange(n),np.setdiff1d(np.arange(n),np.arange(16,24))]
        for p in (8,16):
            for ix in idxs:
                reg=G.Regression(y,p,ix)
                for rho in (-.8,0.,.4,.95):
                    t=np.arctanh(rho);v,info=reg.evaluate(t,False,True);vqr,qri=reg.evaluate(t,True,True);vd,sd=dense(reg,t);np.testing.assert_allclose(v,vd,rtol=2e-9,atol=1e-10);np.testing.assert_allclose(vqr,vd,rtol=2e-9,atol=1e-10);np.testing.assert_allclose(info['sigma'],sd,rtol=1e-9);rows.append(dict(n=n,p=p,training=len(ix),rho=rho,value_difference=abs(v-vd),sigma_difference=abs(info['sigma']-sd)))
                if len(ix)<n:
                    hold=np.setdiff1d(np.arange(n),ix);score,fit=G.predictive(reg,y,hold);r=fit['rho'];c=r**abs(np.arange(n)[:,None]-np.arange(n)[None,:]);ct=c[np.ix_(ix,ix)];ch=cholesky(ct,lower=True);cross=c[np.ix_(hold,ix)];x=G.basis(n,p);invx=cho_solve((ch,True),x[ix]);covbeta=np.linalg.inv(x[ix].T@invx);mu=x[hold]@fit['beta']+cross@cho_solve((ch,True),y[ix]-x[ix]@fit['beta']);diff=x[hold]-cross@invx;pred=c[np.ix_(hold,hold)]-cross@cho_solve((ch,True),cross.T)+diff@covbeta@diff.T;direct=-multivariate_t.logpdf(y[hold],loc=mu,shape=fit['sigma']**2*pred,df=fit['df']);np.testing.assert_allclose(score,direct,rtol=1e-8,atol=1e-8)
    comparisons=[]
    for key in ('ode_iid_seed60','ode_positive_seed57','ode_negative_seed60'):
        e,d,a=Old.Old.load_case(key);start=time.monotonic();model,info,pilot=G.estimate(e.raw['u'])
        for p in G.SIZES:
            fine=G.Regression(e.raw['u'],p).fit(np.linspace(-G.BOUND,G.BOUND,129));coarse=info['full_fits'][str(p)];assert abs(fine['objective']-coarse['objective'])<=1e-7;np.testing.assert_allclose(fine['sigma'],coarse['sigma'],rtol=1e-5)
        comparisons.append(dict(key=key,model=model,selected_p=info['selected_p'],seconds=time.monotonic()-start,selection=info['selection']))
    dump(ROOT/'checks.json',dict(status='PASS',dense_cases=rows,grid_comparisons=comparisons,source_sha256=sources('spline_fit.py','check_spline.py','protocol.txt')));print('CHECKS PASS',len(rows),'dense cases',comparisons,flush=True)


if __name__=='__main__':main()
