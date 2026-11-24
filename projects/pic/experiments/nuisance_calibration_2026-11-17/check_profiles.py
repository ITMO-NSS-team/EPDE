from run_profiles import *
from scipy.linalg import cho_factor,cho_solve


def dense(data,z):
    n=len(data);sigma=np.exp(z[0]);rho=np.tanh(z[1]);k=np.exp(z[2]);p1,p2=np.tanh(z[3:]);lag=np.abs(np.arange(n+2)[:,None]-np.arange(n+2)[None,:]);diff=np.diff(np.eye(n+2),n=2,axis=0);corr=np.zeros(n);corr[:2]=[1.,p1]
    for i in range(2,n):corr[i]=p1*(1-p2)*corr[i-1]+p2*corr[i-2]
    cov=sigma*sigma*(diff@(rho**lag)@diff.T+k*corr[abs(np.arange(n)[:,None]-np.arange(n)[None,:])]);chol=cho_factor(cov,lower=True);q=(data.T@cho_solve(chol,data)).item();return float((np.log(np.diag(chol[0])).sum()+.5*q)/n)


def main():
    rows=[];rng=np.random.default_rng(989200)
    for n in (24,318):
        data=rng.normal(size=(n,1))
        for pars in ((.1,0.,.01,.95,-.8),(.2,.4,.001,.99,-.95),(.1,-.4,.1,.6,.2)):
            sigma,r,k,p1,p2=pars;z=np.array([np.log(sigma),np.arctanh(r),np.log(k),np.arctanh(p1),np.arctanh(p2)]);v,g=Q.likelihood(data,z);vd=dense(data,z);gd=[]
            for j in range(5):
                dz=np.zeros(5);dz[j]=2e-5;gd.append((dense(data,z+dz)-dense(data,z-dz))/(4e-5))
            np.testing.assert_allclose(v,vd,rtol=1e-8,atol=1e-8);np.testing.assert_allclose(g,gd,rtol=2e-5,atol=2e-4);rows.append(dict(n=n,parameters=pars,value_error=abs(v-vd),gradient_max_error=float(max(abs(g-np.array(gd))))))
    for profile,key in ANCHORS.items():
        e,d,a=Old.load_case(key);m=d['model']['u'];i=d['fit_info']['u'];reg=Q.region_fit(e.raw['u'],m,i);ev=Q.evaluate(e.raw['u'],m,i,reg,np.array([np.log(m[0]),np.arctanh(m[1][0])]))
        assert ev['lr_status']=='ok' and abs(ev['raw_lr'])<=1e-4;assert ev['wald']==0.;print('preflight',key,reg['status'],ev['raw_lr'],flush=True)
    for p in PROFILES:
        y,theta,d=generate('calibration',p,999);y2,t2,d2=generate('calibration',p,999);assert np.array_equal(y,y2);assert d==d2
    dump(ROOT/'profile_checks.json',dict(status='PASS',rows=rows,source_sha256=src('profile_region.py','run_profiles.py','check_profiles.py','protocol.txt')));print('PROFILE CHECKS PASS',flush=True)


if __name__=='__main__':main()
