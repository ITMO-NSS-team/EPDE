"""Check native quadrature preserves the user's first-derivative identities."""
import native_quadrature as N
R,O,E,np=N.R,N.O,N.E,N.np
import ode_simple_constants as D
from types import SimpleNamespace
from pathlib import Path
import hashlib


def main():
    t=np.linspace(0,4*np.pi,200);u=np.sin(t)+1.3*np.cos(t)
    p=SimpleNamespace(shape=u.shape,data={'u':u},grids=[t],named_arrays={})
    cs,labels=D.panel();terms,opaque=R.symbols(p,labels);ci=labels.index(())
    # A direct native-grid control: no dense field resampling is performed.
    spec=N.common_spec([[t]],terms,[1],k=2);bank=N.Bank([t],terms,opaque,spec)
    basis=N.M.RealBasis(2,1);sigma=.05*np.std(u);cov=O.Covariance([t],np.eye(len(t)),{'u':(sigma,[0.],[])})
    rng=np.random.default_rng(8503);draws=[]
    for _ in range(512):draws.append(basis.convert(O.matrix(bank,{'u':u+sigma*rng.standard_normal(len(u))},cov)))
    draws=np.stack(draws);clean=basis.convert(bank.matrix({'u':u}));rows=[]
    for c in cs:
        if c['id'] not in ('energy','phase_sin','phase_cos'):continue
        coef,info=E.irls(clean,draws,c);assert info['status']=='ok'
        np.testing.assert_allclose(coef,c['exact'],rtol=1e-5,atol=1e-5)
        residual=clean[:,c['yi']]-clean[:,c['xi']]@coef
        energy=float(residual@residual/(clean[:,c['yi']]@clean[:,c['yi']]))
        assert energy<1e-8
        missing=dict(c,xi=[i for i in c['xi'] if i!=ci]);bad,_=E.ordinary(clean,missing)
        rr=clean[:,c['yi']]-clean[:,missing['xi']]@bad
        bad_energy=float(rr@rr/(clean[:,c['yi']]@clean[:,c['yi']]))
        assert bad_energy>energy
        rows.append(dict(identity=c['id'],coefficients=coef,constant=coef[c['xi'].index(ci)],
                         clean_energy=energy,without_constant_energy=bad_energy,fit=info))
    R.dump(N.ROOT/'invariant_checks.json',dict(control_only=True,rows=rows,all_constants_preserved=True,
           maximum_derivative_order=1,phase_derivative_power=1,energy_derivative_power=2,
           source_sha256={str(Path(p)):hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in (__file__,N.__file__,E.__file__,D.__file__)}))
    print([(r['identity'],r['constant'],r['clean_energy']) for r in rows])


if __name__=='__main__':main()
