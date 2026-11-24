import metric as C
R,np,K,ROOT=C.R,C.np,C.K,C.ROOT
from scipy.stats import chi2
import ode_simple_constants as D
from types import SimpleNamespace


def main():
    rng=np.random.default_rng(211711);K.common_width(1);n=2*R.SV._block_rows(1)
    cs=[dict(id='constant',xi=[0],yi=1)];ext=C.Extractor(cs,(n,));zero=np.column_stack([np.ones(n),np.zeros(n)])
    pilot,_,_=ext.extract(zero);ys=.1*rng.normal(size=(256,n))
    cal=np.stack([ext.extract(np.column_stack([np.ones(n),y]))[0] for y in ys]);model=C.NoiseModel(ext,pilot,cal)
    null=[];nonnull=[];tiny=[];replay=0.
    for y in ys[:128]:
        for shift,scale,out in [(0.,1.,null),(1.3,1.,nonnull),(0.,1e-8,tiny)]:
            mat=np.column_stack([np.ones(n),shift+scale*y]);d,dyad,_=ext.extract(mat)
            expected=R.SV.dyad_scores(mat[:,:1],mat[:,1],np.ones(n),(n,),fit_intercept=False)
            np.testing.assert_allclose(dyad,expected,rtol=2e-9,atol=2e-10)
            replay=max(replay,float(np.max(np.abs(dyad-expected))))
            out.append(model.score(d/scale,bias=False)[0])
    np.testing.assert_allclose(null,nonnull,rtol=2e-10,atol=2e-11);np.testing.assert_allclose(null,tiny,rtol=2e-10,atol=2e-11)
    # Known-variance Gaussian control, independent from the estimated-noise tests.
    err=.1*np.sqrt(2/(n//2))*rng.normal(size=32768);q=err**2/(.1**2*2/(n//2));coverage=float(np.mean(q<=chi2.ppf(.95,1)))
    assert abs(np.mean(q)-1)<.04 and abs(coverage-.95)<.01
    # All-axis coefficient drift, constant preservation and signed feature units.
    K.common_width(8);tt,xx=np.meshgrid(np.linspace(-1,1,15),np.linspace(-1,1,17),indexing='ij')
    x=np.column_stack([np.ones(tt.size),tt.ravel()]);y=(1.3+.7*tt).ravel();drift=y+.8*(tt*xx).ravel()
    cs=[dict(id='linear',xi=[0,1],yi=2)];ext=C.Extractor(cs,tt.shape);mat=np.column_stack([x,y])
    p,_,_=ext.extract(mat);fields=[np.column_stack([x,y+.05*rng.normal(size=len(y))]) for _ in range(128)]
    cc=np.stack([ext.extract(a)[0] for a in fields]);model=C.NoiseModel(ext,p,cc);dd,dyad,_=ext.extract(np.column_stack([x,drift]))
    qdrift=model.score(dd,bias=False);qnull=model.score(p,bias=False);assert qdrift[0]>qnull[0]+1
    units=np.array([.03,-25.,4.]);pp,_,_=ext.extract(mat*units);ss=np.stack([ext.extract(a*units)[0] for a in fields])
    dm,_,_=ext.extract(np.column_stack([x,drift])*units);qm=C.NoiseModel(ext,pp,ss).score(dm)
    np.testing.assert_allclose(qm,model.score(dd),rtol=2e-8,atol=2e-8)
    order=np.arange(len(y)).reshape(tt.shape).T.ravel();other=C.Extractor(cs,tt.T.shape)
    po,_,_=other.extract(mat[order]);co=np.stack([other.extract(a[order])[0] for a in fields]);do,_,_=other.extract(np.column_stack([x,drift])[order])
    np.testing.assert_allclose(C.NoiseModel(other,po,co).score(do),model.score(dd),rtol=2e-8,atol=2e-8)
    # Check actual ODE_simple identity panels, with every additive constant retained.
    t=np.linspace(0,4*np.pi,200);u=np.sin(t)+1.3*np.cos(t);du=np.cos(t)-1.3*np.sin(t)
    assert np.max(np.abs(u*u+du*du-2.69))<3e-15
    assert np.max(np.abs(u*np.sin(t)+du*np.cos(t)-1))<3e-15
    assert np.max(np.abs(u*np.cos(t)-du*np.sin(t)-1.3))<3e-15
    problem=SimpleNamespace(shape=u.shape,data={'u':u},grids=[t],named_arrays={});cs,labels=D.panel();terms,opaque=R.symbols(problem,labels)
    K.common_width(max(len(c['xi']) for c in cs));raw={'u':u+.05*np.std(u)*rng.normal(size=len(u))}
    ff,_=R.bandwidth(raw,[t]);plan=R.local_plan([t],terms,ff['loo'],max(len(c['xi']) for c in cs));bank=R.LocalBank([t],terms,opaque,plan)
    ex=C.Extractor(cs,plan['shape']);cp,_,_=ex.extract(bank.matrix({'u':u}))
    draws=np.stack([ex.extract(bank.matrix({'u':u+.05*np.std(u)*rng.normal(size=len(u))}))[0] for _ in range(128)])
    nm=C.NoiseModel(ex,cp,draws);observed,_,_=ex.extract(bank.matrix(raw));qq=nm.score(observed);identity=[]
    for j,c in enumerate(cs):
        if c['id'] in ('energy','phase_sin','phase_cos'):
            assert labels.index(()) in c['xi'];assert np.isfinite(qq[j]);identity.append(dict(candidate=c['id'],constant_present=True,q_bias=qq[j]))
    assert len(identity)==3
    R.dump(ROOT/'checks.json',dict(status='PASS',production_dyad_replay_error=replay,zero_nonzero_constant_null_same=True,
        noise_scale_invariant_null=True,known_gaussian_null_mean=float(np.mean(q)),known_gaussian_null_coverage=coverage,
        feature_target_units_invariant=True,axis_transposition_invariant=True,drift_response=qdrift,identities=identity,
        source_sha256={str(p):C.E.S.sha(p) for p in [__file__,C.__file__,D.__file__,ROOT/'protocol.txt']}))
    print('PASS: production replay, zero/nonzero null, units, axes, drift, ODE_simple identities')


if __name__=='__main__':main()
