"""Global cubic spline means and stationary AR1 restricted likelihood."""
from spline_context import *
from functools import lru_cache
from scipy.interpolate import BSpline
from scipy.linalg import cholesky,cho_solve,solve_triangular,qr
from scipy.optimize import minimize_scalar
from scipy.special import gammaln

SIZES=(8,16,24,32,48,64)
BOUND=float(np.arctanh(1-1e-6))
GRID=np.linspace(-BOUND,BOUND,65)


@lru_cache(None)
def basis(n,p):
    x=np.linspace(0,1,n);knots=np.r_[np.zeros(4),np.linspace(0,1,p-2)[1:-1],np.ones(4)];b=BSpline.design_matrix(x,knots,3).toarray();assert b.shape==(n,p);return qr(b,mode='economic',check_finite=False)[0]


def whiten(a,idx,rho):
    phi=rho**np.diff(idx);den=np.sqrt(1-phi*phi);b=np.array(a,copy=True)
    if a.ndim==1:b[1:]=(a[1:]-phi*a[:-1])/den
    else:b[1:]=(a[1:]-phi[:,None]*a[:-1])/den[:,None]
    return b


@lru_cache(None)
def geometry(n,p,indices):
    ix=np.array(indices);x=basis(n,p)[ix];gaps=np.diff(ix);groups=[]
    for gap in np.unique(gaps):
        where=np.flatnonzero(gaps==gap);a=x[where+1];b=x[where];groups.append((int(gap),where,a.T@a,b.T@b,a.T@b+b.T@a))
    return x,groups


class Regression:
    def __init__(self,y,p,indices=None):
        self.n=len(y);self.p=p;self.idx=np.arange(self.n) if indices is None else np.array(indices);self.y=np.array(y)[self.idx];self.x,self.groups=geometry(self.n,p,tuple(self.idx));self.df=len(self.idx)-p;assert self.df>0
        self.h=[]
        for gap,j,*_ in self.groups:
            a=self.x[j+1];b=self.x[j];ya=self.y[j+1];yb=self.y[j];self.h.append((a.T@ya,b.T@yb,a.T@yb+b.T@ya))
    def evaluate(self,t,exact=False,details=False):
        rho=float(np.tanh(t));phi=rho**np.diff(self.idx);den=1-phi*phi;ld=float(np.log(den).sum())
        if exact:
            xw=whiten(self.x,self.idx,rho);yw=whiten(self.y,self.idx,rho);qq,rr=qr(xw,mode='economic',check_finite=False);beta=solve_triangular(rr,qq.T@yw,lower=False,check_finite=False);lg=2*np.log(abs(np.diag(rr))).sum();g=xw.T@xw
        else:
            g=np.outer(self.x[0],self.x[0]);h=self.x[0]*self.y[0]
            for (gap,j,a,b,c),(ha,hb,hc) in zip(self.groups,self.h):
                ph=rho**gap;de=1-ph*ph;g=g+(a+ph*ph*b-ph*c)/de;h=h+(ha+ph*ph*hb-ph*hc)/de
            chol=cholesky(g,lower=True,check_finite=False);beta=cho_solve((chol,True),h,check_finite=False);lg=2*np.log(np.diag(chol)).sum()
        residual=self.y-self.x@beta;w=whiten(residual,self.idx,rho);q=float(w@w);sigma=float(np.sqrt(q/self.df));value=float(np.log(q/self.df)+(ld+lg)/self.df)
        return (value,dict(rho=rho,sigma=sigma,beta=beta,gram=g,quadratic=q,logdet_covariance=ld,logdet_mean_gram=float(lg),df=self.df)) if details else value
    def fit(self,grid=GRID):
        def fun(t):
            try:return self.evaluate(t)
            except (np.linalg.LinAlgError,FloatingPointError):return float('inf')
        values=np.array([fun(t) for t in grid]);choices=[dict(t=float(grid[j]),value=float(values[j]),kind='endpoint') for j in (0,len(grid)-1)]
        for j in range(1,len(grid)-1):
            if values[j]<=values[j-1] and values[j]<=values[j+1]:
                opt=minimize_scalar(fun,bounds=(grid[j-1],grid[j+1]),method='bounded',options=dict(xatol=1e-8));choices.append(dict(t=float(opt.x),value=float(opt.fun),kind='interior'))
        best=min(choices,key=lambda r:r['value']);v,details=self.evaluate(best['t'],True,True);error=abs(v-best['value']);status='ok' if error<=1e-7 else 'unresolved_arithmetic'
        if abs(details['rho'])>1-1e-5:status='boundary'
        return dict(**details,transformed_rho=best['t'],objective=v,status=status,qr_objective_difference=error,grid_values=values,choices=choices,p=self.p)


def predictive(train,y,hold):
    fit=train.fit();rho=fit['rho'];sigma=fit['sigma'];x=basis(len(y),train.p);idx=train.idx;test=np.array(hold);rth=rho**abs(test[:,None]-idx[None,:]);rhh=rho**abs(test[:,None]-test[None,:]);wx=whiten(train.x,idx,rho);wr=whiten(rth.T,idx,rho);res=train.y-train.x@fit['beta'];mu=x[test]@fit['beta']+wr.T@whiten(res,idx,rho);diff=x[test]-wr.T@wx;chol=cholesky(fit['gram'],lower=True,check_finite=False);s=rhh-wr.T@wr+diff@cho_solve((chol,True),diff.T,check_finite=False);s=(s+s.T)/2;cc=cholesky(s,lower=True,check_finite=False);err=y[test]-mu;quad=float(err@cho_solve((cc,True),err,check_finite=False))/(sigma*sigma);m=len(test);df=fit['df'];score=float(gammaln(df/2)-gammaln((df+m)/2)+m/2*np.log(df*np.pi)+m*np.log(sigma)+np.log(np.diag(cc)).sum()+(df+m)/2*np.log1p(quad/df))
    return score,fit


def folds(n):
    blocks=[np.arange(a,min(a+8,n-8)) for a in range(8,n-8,8)];return [np.concatenate(blocks[k::4]) for k in range(4)]


def estimate(y):
    y=np.asarray(y);assert y.ndim==1;rows=[];full={};allidx=np.arange(len(y))
    for p in SIZES:
        fit=Regression(y,p).fit();full[str(p)]=fit;cv=[]
        for hold in folds(len(y)):
            train=np.setdiff1d(allidx,hold);score,ff=predictive(Regression(y,p,train),y,hold);cv.append(dict(score=score,status=ff['status'],sigma=ff['sigma'],rho=ff['rho'],p=p,qr_objective_difference=ff['qr_objective_difference']))
        rows.append(dict(p=p,score=sum(r['score'] for r in cv),folds=cv))
    selected=min(rows,key=lambda r:(r['score'],r['p']))['p'];fit=full[str(selected)];pilot=basis(len(y),selected)@fit['beta'];return [fit['sigma'],[fit['rho']],[2]],dict(status=fit['status'],selected_p=selected,selection=rows,full_fits=full),pilot
