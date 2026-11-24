import experiment as E
R,O,N,np,ROOT=E.R,E.O,E.N,E.np,E.ROOT
from types import SimpleNamespace
import ode_simple_constants as D


def main():
    rng=np.random.default_rng(171124)
    t=np.linspace(0,2,81);terms=[R.WF.WeakTerm([('u',(0,))]*p,(0,)) for p in range(4)]
    spec=dict(q=[3],radii=[.45],centers=[[.5,1.,1.5]],frequencies=[[0.]],interpolation_cut=[0])
    # radii is per axis, centers contains all windows on that axis.
    bank=N.Bank([t],terms,{},spec);mu={'u':.7+.2*t};sig=.3
    cov=O.Covariance([t],np.eye(len(t)),{'u':[sig,[1.],[]]})
    target=bank.matrix(mu);mean=np.zeros_like(target);nodes,weights=np.polynomial.hermite.hermgauss(4)
    for z,w in zip(nodes,weights/np.sqrt(np.pi)):
        mean+=w*O.matrix(bank,{'u':mu['u']+sig*np.sqrt(2)*z},cov)
    np.testing.assert_allclose(mean,target,rtol=2e-13,atol=2e-13)
    raw=bank.matrix({'u':mu['u']+sig});np.testing.assert_allclose(O.matrix(bank,{'u':mu['u']+sig}),raw,rtol=2e-13,atol=2e-13)
    np.testing.assert_allclose(raw[:,0],1.,atol=2e-14)
    # Correlated linear-Gaussian moments, empirical calibration with independent draws.
    mix=rng.normal(size=(7,5));cal=rng.normal(size=(256,5))@mix.T;var=cal.var(axis=0,ddof=1)
    check=rng.normal(size=(4096,5))@mix.T;actual=rng.normal(size=(32768,5))@mix.T
    radius=np.quantile(E.score(check,var),.95);coverage=float(np.mean(E.score(actual,var)<=radius));assert abs(coverage-.95)<.015
    np.testing.assert_allclose(E.score(-7*actual,49*var),E.score(actual,var),rtol=1e-14,atol=1e-14)
    assert E.score(np.zeros(3),np.zeros(3))==0 and np.isinf(E.score(np.ones(3),np.zeros(3)))
    # Additive-constant first-derivative ODE_simple identities survive native quadrature.
    t=np.linspace(0,4*np.pi,1001);u=np.sin(t)+1.3*np.cos(t)
    p=SimpleNamespace(shape=u.shape,data={'u':u},grids=[t],named_arrays={});cs,labels=D.panel();tt,opaque=R.symbols(p,labels)
    spec=dict(q=[3],radii=[2.],centers=[[3.,6.,9.]],frequencies=[[0.]],interpolation_cut=[2])
    b=N.Bank([t],tt,opaque,spec);m=b.matrix({'u':u});identities=[]
    for c in cs:
        if c['id'] not in ('energy','phase_sin','phase_cos'):continue
        assert labels.index(()) in c['xi'];v=E.vector(c,c['exact'],len(labels));err=float(np.max(np.abs(m@v)))
        assert err<2e-7,(c['id'],err);identities.append(dict(candidate=c['id'],constant_present=True,max_weak_residual=err))
    # Every dataset geometry must construct before launching the full experiment.
    names=['ac','burgers','burgers_inviscid','duffing','kdv','kdv_cossin','ks','lorenz','lv','ns','ode','pde_compound','pde_divide','vdp','wave']
    shapes={};replay=0.
    for name in names:
        st=E.Study(name,24);shapes[name]={tag:bank.shape for tag,bank in st.banks.items()}
        for tag,bank in st.banks.items():
            plain=bank.matrix(st.raw);other=O.matrix(bank,st.raw)
            np.testing.assert_allclose(plain,other,rtol=2e-12,atol=2e-12);replay=max(replay,float(np.max(np.abs(plain-other))))
        print(name,'geometry PASS',shapes[name],flush=True)
    R.dump(ROOT/'checks.json',dict(status='PASS',gaussian_polynomial_expectation_max_error=float(np.max(np.abs(mean-target))),
        operator_replay_max_error=replay,correlated_linear_gaussian_coverage=coverage,units_invariant=True,zero_variance_explicit=True,
        identities=identities,geometries=shapes,source_sha256={**E.sources(),__file__:E.sha(__file__),D.__file__:E.sha(D.__file__)}))
    print('All checks PASS',flush=True)


if __name__=='__main__':main()
