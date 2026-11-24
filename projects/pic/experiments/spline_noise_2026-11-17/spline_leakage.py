"""Posthoc explanation only: deterministic mean error under fixed true rho."""
from spline_context import *
import spline_fit as G
from scipy.linalg import qr,cholesky,cho_solve


def main():
    assert read(ROOT/'point_audit.json')['status']=='PASS';rows=[]
    for name in ('ode','vdp','duffing'):
        e,_,_=C.primary(name,'iid',51);y=e.p.data['u'];sigma=.05*np.std(y)
        for p in G.SIZES:
            # Independent orthogonal contrast on raw data, not the fast AR1
            # whitening calculation used in fitting.
            reg=G.Regression(y,p);qperp=qr(reg.x,mode='full')[0][:,p:];contrast=qperp.T@y
            for profile,rho in [('iid',0.),('positive',.4),('negative',-.4)]:
                corr=rho**abs(np.arange(len(y))[:,None]-np.arange(len(y))[None,:]);cov=qperp.T@corr@qperp;l=cholesky(cov,lower=True);q=float(contrast@cho_solve((l,True),contrast));v,d=reg.evaluate(np.arctanh(rho),True,True);np.testing.assert_allclose(q,d['quadratic'],rtol=2e-7,atol=1e-15);extra=q/(reg.df*sigma*sigma)
                rows.append(dict(system=name,profile=profile,p=p,n=len(y),true_sigma=sigma,signal_quadratic=q,df=reg.df,extra_expected_variance_over_true_variance=extra,root_expected_sigma_squared_over_true_sigma=float(np.sqrt(1+extra)),clean_sha256=R.digest(y)))
    dump(ROOT/'leakage_diagnostic.json',dict(status='PASS',role='Posthoc deterministic approximation diagnostic; not used for selection or uncertainty construction.',formula='E[sigma_hat^2]/sigma_true^2 = 1 + q_signal/((n-p)*sigma_true^2), with rho fixed at truth and fixed p.',rows=rows,source_sha256=sources('spline_leakage.py','spline_fit.py'),input_sha256={str(ROOT/'point_all_frozen.json'):sha(ROOT/'point_all_frozen.json')}));print('LEAKAGE PASS',[(r['system'],r['profile'],r['p'],round(r['root_expected_sigma_squared_over_true_sigma'],4)) for r in rows if r['p'] in (16,24,32)],flush=True)


if __name__=='__main__':main()
