from adequacy_context import *
from functools import lru_cache
from scipy.linalg import svd,qr
from scipy.stats import f

METHODS=('cv_a01','smallest_a01','cv_a05','smallest_a05')
NEXT=dict(zip(G.SIZES,(16,24,32,48,64,96)))

@lru_cache(None)
def union_basis(n,p):
    q=NEXT[p];b=np.column_stack([G.basis(n,p),G.basis(n,q)]);u,s,v=svd(b,full_matrices=False);rank=p+q-4;assert rank<n;assert s[rank-1]>1e-8*s[0] and (rank==len(s) or s[rank]<1e-10*s[0]);return u[:,:rank]

@lru_cache(None)
def boundary_geometry(n,p):
    u,s,v=svd(np.diff(G.basis(n,p),axis=0),full_matrices=False);assert s[-1]<1e-12*s[0];ld=(n-p)*np.log(2)-np.log(n)+2*np.log(s[:-1]).sum();return u[:,:-1],float(ld)

def boundary(y,p):
    u,ld=boundary_geometry(len(y),p);dy=np.diff(y);res=dy-u@(u.T@dy);df=len(y)-p;eta=float(res@res)/(2*df);return dict(eta=eta,nll=float(df*(np.log(eta)+1)+ld),df=df)

def screen(y,p,fit):
    n=len(y);idx=np.arange(n);v,base=G.Regression(y,p).evaluate(fit['transformed_rho'],True,True);b=union_basis(n,p);wb=G.whiten(b,idx,fit['rho']);wy=G.whiten(y,idx,fit['rho']);u,rr=qr(wb,mode='economic');res=wy-u@(u.T@wy);q1=float(res@res);q0=base['quadratic'];improvement=q0-q1;assert improvement>=-1e-9*max(q0,1e-300);df1=b.shape[1]-p;df2=n-b.shape[1];stat=max(improvement,0)/df1/(q1/df2);pvalue=float(f.sf(stat,df1,df2));limit=boundary(y,p);nll0=(n-p)*(fit['objective']+1);tail=limit['nll']-nll0
    return dict(p=p,next_p=NEXT[p],union_rank=b.shape[1],small_quadratic=q0,large_quadratic=q1,improvement=improvement,df1=df1,df2=df2,f_statistic=stat,pvalue=pvalue,limit=limit,nll0=nll0,limit_lr=tail,boundary_excluded=tail>H.CUTOFF+1e-6,fit_status=fit['status'])

def select(info,rows):
    out={};cv={r['p']:r['score'] for r in info['selection']}
    for method in METHODS:
        alpha=.01 if method.endswith('01') else .05;allowed=[r['p'] for r in rows if r['fit_status']=='ok' and r['pvalue']>=alpha and r['boundary_excluded']];p=min(allowed,key=lambda p:(cv[p],p)) if allowed and method.startswith('cv') else min(allowed) if allowed else None
        out[method]=dict(selected_p=p,status='ok' if p is not None else 'unresolved_no_candidate',allowed=allowed,alpha=alpha,ordering='cv' if method.startswith('cv') else 'smallest')
    return out

def analyze(y,info):
    rows=[screen(y,p,info['full_fits'][str(p)]) for p in G.SIZES];return rows,select(info,rows)
