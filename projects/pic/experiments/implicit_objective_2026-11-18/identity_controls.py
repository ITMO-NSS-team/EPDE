"""Exact first-derivative phase/energy identities, pivots, units and rank failures."""
import json
import numpy as np
from records import ROOT
from implicit import fit,score,projective_error


def main():
    t=np.linspace(0,4*np.pi,201)
    u=np.sin(t)+1.3*np.cos(t);d=np.cos(t)-1.3*np.sin(t)
    cases={'phase_sin':(d*np.cos(t),u*np.sin(t),1.),
           'phase_cos':(u*np.cos(t),-d*np.sin(t),1.3),
           'energy':(d*d,u*u,2.69)}
    rows=[]
    for label,(a,b,k) in cases.items():
        design=np.column_stack([a,b,np.ones(len(t))])
        # Bin averaging is a linear weak measurement that preserves each identity.
        blocks=[design[i::3][:66].reshape(11,6,3).mean(axis=1) for i in range(3)]
        expected=np.array([1.,1.,-k])
        for i in range(3):
            train=blocks[i];left=blocks[(i+1)%3];right=blocks[(i+2)%3];sc=np.linalg.norm(train,axis=0)
            v,status=fit(train)
            assert status=='ok'
            np.testing.assert_allclose(v/(-v[-1]/k),expected,rtol=1e-12,atol=1e-12)
            assert abs(score(left,right,sc,v))<1e-24
            for target in range(3):
                assert abs(score(left,right,sc,v/v[target]))<1e-24
            reduced,_=fit(train[:,:2])
            gap=score(left[:,:2],right[:,:2],sc[:2],reduced)
            assert gap>1e-5
            rows.append(dict(identity=label,fold=i,constant=k,without_constant_score=gap))
    assert fit(np.ones((12,3)))[1]=='nonunique direction'
    assert fit(np.zeros((12,2)))[1]=='zero training column'
    assert fit(np.ones((12,1)))[1]=='ok'
    (ROOT/'identity_checks.json').write_text(json.dumps(dict(rows=rows,rank_checks='PASS'),indent=2))
    print('All three pivots and folds preserve both phase identities and the energy identity.')


if __name__=='__main__':
    main()
