from bridge import *
import compressed_time as F
import native_inverse as C
from scipy.linalg import cholesky_banded
import time


def main():
    namespace = F.A.profile_score.__globals__
    original = namespace['selected_inverse']
    configs = [(0., 1e-5, .99, -.8), (.4, 1e-3, .95, -.5),
               (-.4, 1e-8, .9, 0.), (.95, 1., .8, -.6),
               (-.95, 1e-10, .99, -.99), (.1, 10., -.3, .3)]
    inverses, objectives, fits = [], [], []
    rng = np.random.default_rng(984911)
    for n in (7, 32, 301, 1001):
        for pars in configs:
            band = F.A.covariance_band(n, *pars)
            chol = cholesky_banded(band, lower=True, check_finite=False)
            t = time.perf_counter(); old = original(chol); oldtime = time.perf_counter()-t
            t = time.perf_counter(); new = C.selected_inverse(chol); newtime = time.perf_counter()-t
            relative = float(np.max(abs(old-new))/np.max(abs(old)))
            agrees=bool(np.allclose(new, old, rtol=2e-12, atol=2e-13*np.max(abs(old))))
            independent = None
            if n <= 32:
                dense = np.zeros((n,n))
                for lag in range(len(chol)):
                    for i in range(n-lag):dense[i+lag,i] = chol[lag,i]
                inverse = np.linalg.inv(dense@dense.T)
                direct = np.zeros_like(chol)
                for lag in range(len(chol)):
                    for i in range(n-lag):direct[lag,i] = inverse[i+lag,i]
                independent = float(np.max(abs(direct-new))/np.max(abs(direct)))
            inverses.append(dict(n=n, pars=pars, relative_error=relative, dense_relative_error=independent,
                                 strict_agreement=agrees, python_seconds=oldtime, native_seconds=newtime))
        cc = F.A.Contrasts(rng.normal(size=(n+2,5))); cc.fields=[cc.fields[0]]; cc.count=cc.fields[0].size
        for pars in configs:
            args=([pars[0]],[pars[1]],[pars[2]],[pars[3]],F.A.covariance_band)
            namespace['selected_inverse']=original
            a,ga=F.A.profile_score(cc,*args)
            namespace['selected_inverse']=C.selected_inverse
            b,gb=F.A.profile_score(cc,*args)
            agrees=bool(np.allclose([a,*ga],[b,*gb],rtol=2e-10,atol=2e-11))
            objectives.append(dict(n=n,pars=pars,strict_agreement=agrees,objective_difference=b-a,max_gradient_difference=float(max(abs(ga-gb)))))
    namespace['selected_inverse']=original
    for i,name in enumerate(('duffing','lv','lorenz','pde_compound')):
        p=R.datasets.load(name); y=next(iter(p.data.values())); noise=R.WF.ar_noise(np.random.default_rng(np.random.SeedSequence([984912,i])),y.shape,.05*np.std(y),[0.]*y.ndim); field=y+noise
        t=time.perf_counter(); old,oi=F.estimate(field); oldtime=time.perf_counter()-t
        namespace['selected_inverse']=C.selected_inverse
        try:
            t=time.perf_counter(); new,ni=F.estimate(field); newtime=time.perf_counter()-t
        finally:namespace['selected_inverse']=original
        row=dict(system=name,shape=y.shape,raw_sha256=R.digest(field),python_model=old,native_model=new,
                 python_status=oi['status'],native_status=ni['status'],python_seconds=oldtime,native_seconds=newtime,
                 relative_sigma_difference=abs(new[0]/old[0]-1),max_rho_difference=float(np.max(abs(np.array(new[1])-old[1]))),
                 python_projected_gradient=oi['projected_gradient'],native_projected_gradient=ni['projected_gradient'])
        row['model_agreement']=row['relative_sigma_difference']<1e-4 and row['max_rho_difference']<1e-4 and oi['status']==ni['status']=='ok'
        fits.append(row);print(name,row,flush=True)
    passed=all(r['model_agreement'] for r in fits) and all(r['strict_agreement'] for r in inverses+objectives)
    out=dict(status='PASS' if passed else 'COMPLETE_WITH_NUMERICAL_DIFFERENCES',
             inverses=inverses,objectives=objectives,fits=fits,
             interpretation='Optional runtime probe, not used in the primary 69-dataset experiment. '
             'Timings measured under concurrent workload; numerical agreement does not imply bit-identical fits.',
             source_sha256=sources('native_checks.py','native_inverse.py','compressed_time.py',
                                   'native_probe/selected_inverse.c','native_probe/selected_inverse.dll',
                                   'native_probe/build.cmd','native_probe/protocol.txt'))
    dump(ROOT/'native_checks.json',out);print(out['status'],flush=True)


if __name__=='__main__':main()
