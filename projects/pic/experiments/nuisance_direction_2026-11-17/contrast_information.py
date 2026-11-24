from context import *
from scipy.linalg import block_diag,cholesky,solve_triangular

def main():
    # Exact Gaussian experiment for noise contrasts; cross-block correlations
    # retained. Signal assumed to lie in the removed polynomial subspace.
    n=320;lag=abs(np.arange(n)[:,None]-np.arange(n)[None,:]);edges=list(range(0,n,12))+[n]
    if edges[-1]-edges[-2]<8:edges.pop(-2)
    rows=[]
    for degree in (-1,0,1,3,5):
        qq=[]
        for a,b in zip(edges[:-1],edges[1:]):
            q=np.eye(b-a) if degree==-1 else np.linalg.qr(np.vander(np.linspace(-1,1,b-a),degree+1,increasing=True),mode='complete')[0][:,degree+1:]
            qq.append(q)
        q=block_diag(*qq);m=q.shape[1]
        for rho in (-.8,-.4,0.,.4,.8,.95):
            c=rho**lag;dc=np.zeros_like(c);mask=lag>0;dc[mask]=lag[mask]*rho**(lag[mask]-1);v=q.T@c@q;dv=q.T@dc@q;l=cholesky(v,lower=True)
            left=solve_triangular(l,dv,lower=True);w=solve_triangular(l,left.T,lower=True).T;w=(w+w.T)/2
            fisher=np.array([[2*m,np.trace(w)],[np.trace(w),np.sum(w*w)/2]]);cov=np.linalg.inv(fisher);info=dict(degree_removed=degree,contrasts=m,rho=rho,fisher=fisher,local_cramer_rao_sigma_log=float(np.sqrt(cov[0,0])),local_cramer_rao_rho=float(np.sqrt(cov[1,1])),local_parameter_correlation=float(cov[0,1]/np.sqrt(cov[0,0]*cov[1,1])))
            if degree==5 and rho in (-.4,.4):
                # Independent Monte Carlo score covariance verifies the formula.
                rng=np.random.default_rng(985771);scores=[];ev=np.linalg.eigvalsh(w)
                for j in range(20):
                    z=rng.normal(size=(1000,m));zz=z*z;scores.append(np.stack([zz.sum(axis=1)-m,.5*(zz@ev-np.sum(ev))],axis=-1))
                empirical=np.cov(np.concatenate(scores).T);error=float(np.max(abs(empirical-fisher)/np.sqrt(np.diag(fisher)[:,None]*np.diag(fisher)[None,:])));assert error<.06;info['monte_carlo_information_relative_error']=error
            rows.append(info)
    dump(ROOT/'contrast_information.json',dict(status='COMPLETE',rows=rows,interpretation='Local information under Gaussian AR1 noise and exactly eliminated polynomial signal. Cross-block covariance retained. Cramer-Rao quantities are local bounds, not confidence intervals or predictions for biased/boundary fits.',source_sha256=sources('contrast_information.py')))
    for r in rows:
        if r['degree_removed'] in (-1,5) and r['rho'] in (-.4,0.,.4):print(r,flush=True)

if __name__=='__main__':main()
