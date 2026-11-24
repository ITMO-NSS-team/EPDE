"""Check whether a single dispersion coefficient satisfies all 12 profiled regions."""
from fractions import Fraction
from math import inf
import json
import numpy as np
from audit import Bank
from kdv_scalar import intervals, intersection, encode
from study import ROOT


def main():
    rows=json.loads((ROOT/'forcing_profile.json').read_text())['rows']
    rr=[r for r in rows if r['method']=='combined']
    assert len(rr)==12 and all(r['status']=='complete' for r in rr)
    common=[(-inf,inf)];by_profile={};details=[]; verified=0
    common_witness=Fraction(-1,5)
    for r in rr:
        p=r['profile'];seed=r['seed'];record=[(-inf,inf)]
        with np.load(ROOT/'forcing_profile'/f'{p}_{seed}.npz') as z:
            for name in r['readouts']:
                bank=Bank(z[name+'_observed'],z[name+'_grams'][:32].mean(axis=0))
                q=Fraction(r['cutoff'])
                xx,xy,yy=[bank.entry(i,j) for i,j in ((1,1),(1,3),(3,3))]
                poly=(xx[0]-q*xx[1],-2*(xy[0]-q*xy[1]),yy[0]-q*yy[1])
                record=intersection(record,intervals(poly))
                assert (poly[0]*common_witness+poly[1])*common_witness+poly[2]<0
                verified+=1
        assert record
        witness=r['term_tests']['dispersion_only']['coefficients'][0]
        assert any(lo<witness<hi for lo,hi in record)
        common=intersection(common,record)
        by_profile[p]=intersection(by_profile.get(p,[(-inf,inf)]),record)
        details.append(dict(profile=p,seed=seed,intervals=encode(record)))
    result=dict(exploratory=True,common=encode(common),example_witness=str(common_witness),
                exact_inequalities_verified=verified,
                by_profile={k:encode(v) for k,v in by_profile.items()},records=details)
    (ROOT/'shared_dispersion.json').write_text(json.dumps(result,indent=2))
    print(result['common'],result['by_profile'])


if __name__=='__main__':
    main()
