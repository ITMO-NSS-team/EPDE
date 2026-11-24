"""Population normal-equation correction and preserved ODE_simple identities."""
import methods as A
np,R,N,O,M,ROOT=A.np,A.R,A.N,A.O,A.M,A.ROOT
import ode_simple_constants as D
from types import SimpleNamespace


def main():
    rng=np.random.default_rng(427);x=np.column_stack([np.ones(9),np.linspace(-1,1,9)])
    point=np.column_stack([x,x@np.array([1.3,-.6])]);c=dict(xi=[0,1],yi=2)
    draws=np.broadcast_to(point,(512,*point.shape)).copy();ex=rng.normal(size=(512,9))*.3
    draws[:,:,1]+=ex;draws[:,:,2]+=.7*ex+rng.normal(size=(512,9))*.1
    prep=A.prepare(draws,point);delta=draws-draws.mean(axis=0)
    population=np.vstack([point,delta.reshape(-1,3)/np.sqrt(511)])
    actual,info=A.normal(population,c,prep);assert info['status']=='ok'
    np.testing.assert_allclose(actual,[1.3,-.6],atol=2e-13)
    omitted=dict(prep,noise_gram=prep['noise_gram'].copy());omitted['noise_gram'][1,2]=omitted['noise_gram'][2,1]=0.
    wrong,_=A.normal(population,c,omitted);assert np.linalg.norm(wrong-actual)>.01
    t=np.linspace(0,4*np.pi,200);u=np.sin(t)+1.3*np.cos(t)
    p=SimpleNamespace(shape=u.shape,data={'u':u},grids=[t],named_arrays={});cs,labels=D.panel();terms,opaque=R.symbols(p,labels)
    spec=N.common_spec([[t]],terms,[1],k=2);bank=N.Bank([t],terms,opaque,spec);basis=M.RealBasis(2,1)
    clean=basis.convert(bank.matrix({'u':u}));rows=[];ci=labels.index(())
    # Zero-noise identity controls: no subtraction of hypothetical noise from clean data.
    prep=A.prepare(np.broadcast_to(clean,(8,*clean.shape)),clean)
    for c in cs:
        if c['id'] not in ('energy','phase_sin','phase_cos'):continue
        coef,info=A.normal(clean,c,prep);assert info['status']=='ok'
        np.testing.assert_allclose(coef,c['exact'],rtol=1e-5,atol=1e-5)
        v=A.S.vector(c,coef,len(labels));energy=float(np.sum((clean@v)**2)/np.sum(clean[:,c['yi']]**2))
        assert energy<1e-8
        rows.append(dict(identity=c['id'],coefficients=coef,constant=coef[c['xi'].index(ci)],energy=energy))
    R.dump(ROOT/'extra_checks.json',dict(population_joint_correction_exact=True,omitted_cross_term_detected=True,
        ode_simple_constants_preserved=True,identity_rows=rows,zero_noise_control_only=True,
        source_sha256={str(p):A.S.sha(p) for p in (A.__file__,__file__,D.__file__)}))
    print('Population correction and ODE_simple constant identities: PASS')


if __name__=='__main__':main()
