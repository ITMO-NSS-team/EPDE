from compat_context import *
import compat_fit as H

def main():
    rows=[];rng=np.random.default_rng(990500)
    for n in (96,320,1001):
        y=rng.normal(size=n)+np.sin(np.linspace(0,20,n));reg=G.Regression(y,64);limit=H.boundary_limit(y)
        for rho in (-.8,0.,.4,.95):
            for sigma in (.7,1.3):
                a=H.likelihood(reg,sigma,rho);b=H.dense_likelihood(y,sigma,rho);np.testing.assert_allclose(a,b,rtol=2e-9,atol=2e-8);rows.append(dict(n=n,rho=rho,sigma=sigma,difference=a-b))
        seq=[]
        for delta in (1e-3,1e-4,1e-5,1e-6):
            rho=1-delta;sigma=np.sqrt(limit['eta']/delta);v=H.dense_likelihood(y,sigma,rho);seq.append(dict(delta=delta,nll=v,error=v-limit['nll']))
        assert abs(seq[-1]['error'])<.002;rows.append(dict(n=n,limit=limit,sequence=seq))
    dump(ROOT/'checks.json',dict(status='PASS',rows=rows,source_sha256=src('compat_fit.py','check_compat.py','protocol.txt')));print('CHECKS PASS',len(rows),flush=True)

if __name__=='__main__':main()
