import study as S
R,np,ROOT=S.R,S.np,S.ROOT
from pathlib import Path

def main():
    rng=np.random.default_rng(171130);active=[0,1,2]
    mixing=rng.normal(size=(24,16))
    def draws(n):
        mm=(rng.normal(size=(n,16))@mixing.T).reshape(n,8,3);mm=np.concatenate([mm,np.zeros((n,8,1))],axis=2)
        return np.einsum('bri,brj->bij',mm,mm)
    cal=draws(256);sigma=cal.mean(axis=0);w,info=S.metric(sigma,active);assert info['status']=='ok'
    check=draws(4096);actual=draws(32768);qb=S.operator_scores(check,w,active);qa=S.operator_scores(actual,w,active)
    cut=float(np.quantile(qb,.95));coverage=float(np.mean(qa<=cut));assert abs(coverage-.95)<.015
    units=np.array([.01,-2.,30.,7.]);scaled=cal*units[None,:,None]*units[None,None,:];ww,meta=S.metric(scaled.mean(axis=0),active)
    other=S.operator_scores(actual*units[None,:,None]*units[None,None,:],ww,active)
    np.testing.assert_allclose(other,qa,rtol=2e-12,atol=2e-12)
    for g,q in zip(actual[:100],qa[:100]):
        for _ in range(8):
            v=rng.normal(size=4);assert v@g@v<=q*(v@sigma@v)*(1+1e-12)
    # A known bounded ellipsoid, its projections and empty/unbounded controls.
    a=np.array([[10.,2.,6.],[2.,2.,0.],[6.,0.,3.]]);c=dict(id='toy',xi=[1,2],yi=0);ss=100*np.eye(3)
    region=S.quadratic_region(a,a+ss,ss,c,1.)
    assert region['status']=='ellipsoid';np.testing.assert_allclose(region['center'],[1,2],atol=1e-14)
    np.testing.assert_allclose(region['upper']-region['center'],np.sqrt([2.,4/3]),atol=1e-14)
    empty=a.copy();empty[0,0]=20.;assert S.quadratic_region(empty,empty+ss,ss,c,1.)['status']=='empty'
    unbounded=a.copy();unbounded[1,1]=-2.;assert S.quadratic_region(unbounded,unbounded+ss,ss,c,1.)['status']=='unbounded'
    singular=np.diag([10.,0.,3.]);assert S.quadratic_region(singular,singular+ss,ss,c,1.)['status']=='unresolved_singular'
    for angle in np.linspace(0,2*np.pi,33):
        beta=np.array([1,2])+np.sqrt([2.,4/3])*[np.cos(angle),np.sin(angle)];v=S.E.vector(c,beta,3)
        np.testing.assert_allclose(v@a@v,0.,atol=2e-14)
    # A deterministic additive constant remains part of an accepted law.
    x=np.linspace(-1,1,51);m=np.column_stack([2*x+1.3,x,np.ones(len(x))]);noise=np.diag([.1,.1,0.])
    region=S.quadratic_region(S.gram(m)-3*noise,S.gram(m),noise,c,3.)
    v=np.array([1.,-2.,-1.3]);assert v@(S.gram(m)-3*noise)@v<0
    # Noise-shape normalization and units: no use of true amplitude in Study.
    amplitude=[]
    for rho in (0.,.4,-.4):
        fields=R.WF.ar_noise(rng,(8192,128),.2,[0.,rho]);scale={'u':1.};order={'u':[0,3]}
        estimate=S.amplitude({'u':fields},scale,order,[0.,rho]);assert abs(estimate/.2-1)<.015
        scaled=S.amplitude({'u':-11*fields},{'u':11.},order,[0.,rho]);np.testing.assert_allclose(estimate,scaled,rtol=2e-15)
        amplitude.append(dict(rho=rho,estimate=estimate))
    from run_suite import SYSTEMS
    geometry={}
    for name in SYSTEMS:
        e=S.Study(name,30,'iid','iid');obs=e.matrices(e.raw);pilot=e.matrices(e.pilot,False)
        for tag in obs:assert np.all((obs[tag]-pilot[tag])[:,e.deterministic]==0)
        geometry[name]=dict(columns=len(e.labels),stochastic=len(e.active),rows={k:len(v) for k,v in obs.items()})
        print(name,'geometry/constant PASS',flush=True)
    R.dump(ROOT/'checks.json',dict(status='PASS',gaussian_operator_coverage=coverage,units_invariant=True,
        simultaneous_rayleigh_bound=True,bounded_empty_unbounded_controls=True,constant_preserved=True,
        correlated_amplitude=amplitude,geometry=geometry,source_sha256={**S.source_hashes(),str(Path(__file__)):S.E.sha(__file__)}))
    print('PASS',coverage,flush=True)

if __name__=='__main__':main()
