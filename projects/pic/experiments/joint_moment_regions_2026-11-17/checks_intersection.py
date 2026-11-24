import study as S
R,np,ROOT=S.R,S.np,S.ROOT
from pathlib import Path
import importlib.util
spec=importlib.util.spec_from_file_location('joint_intersection_checks',ROOT/'intersect_banks.py')
J=importlib.util.module_from_spec(spec);spec.loader.exec_module(J)

def main():
    c=dict(id='toy',xi=[1,2],yi=0,label='reference',slot=0,registered=True)
    sigma=[10*np.eye(3),10*np.eye(3)]
    aa=[np.diag([-1.,2.,-1.]),np.diag([-1.,-1.,2.])];gg=[a+s for a,s in zip(aa,sigma)]
    both=J.certificate(gg,sigma,1.,c,0)
    assert both['geometry']=='bounded_outer' and 'feasible_witness' in both
    np.testing.assert_allclose(both['lambda_local'],.5,atol=1e-7)
    np.testing.assert_allclose(both['outer_region']['upper'],[np.sqrt(2)]*2,rtol=2e-7)
    assert all(S.quadratic_region(a,g,s,c,1.)['status']=='unbounded' for a,g,s in zip(aa,gg,sigma))
    empty=[a.copy() for a in aa]
    for a in empty:a[0,0]=1.
    assert J.certificate([a+s for a,s in zip(empty,sigma)],sigma,1.,c,0)['geometry']=='empty_certified'
    neg=[np.diag([1.,-2.,1.]),np.diag([1.,-1.,2.])]
    cert=J.certificate([a+s for a,s in zip(neg,sigma)],sigma,1.,c,0);assert cert['geometry']=='unbounded_certified'
    direction=np.asarray(cert['common_negative_direction'])
    for t in (10,100,1000):
        v=S.E.vector(c,t*direction,3);assert all(v@a@v<0 for a in neg)
    units=np.array([3.,-7.,.02]);scaled=lambda a:a*units[:,None]*units[None,:]
    unit=J.certificate([scaled(g) for g in gg],[scaled(s) for s in sigma],1.,c,0)
    assert unit['geometry']=='bounded_outer'
    np.testing.assert_allclose(np.abs(np.asarray(unit['outer_region']['upper'])*units[1:]/units[0]),both['outer_region']['upper'],rtol=2e-7)
    rng=np.random.default_rng(940106);feasible=0
    for _ in range(4096):
        beta=rng.normal(size=2);v=S.E.vector(c,beta,3)
        if all(v@a@v<=0 for a in aa):
            assert np.dot(beta,beta)<=2+1e-12;feasible+=1
    assert feasible>100
    out=dict(status='PASS',complementary_indefinite_constraints_bound_joint_set=True,empty_certificate=True,
        common_negative_ray=True,column_units=True,mixture_implication_feasible_draws=feasible,
        source_sha256={str(Path(__file__)):S.E.sha(__file__),str(ROOT/'intersect_banks.py'):S.E.sha(ROOT/'intersect_banks.py'),str(ROOT/'study.py'):S.E.sha(ROOT/'study.py')})
    R.dump(ROOT/'checks_intersection.json',out);print(out)

if __name__=='__main__':main()
