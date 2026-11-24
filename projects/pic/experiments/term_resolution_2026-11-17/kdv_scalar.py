"""Complete one-dimensional KdV deletion profiles and a locked prediction."""
from fractions import Fraction
import json
from math import inf
import mpmath as mp
import numpy as np
from audit import Bank
from study import ROOT, COHORTS, digest

mp.mp.dps = 80

def dec(f):
    return mp.mpf(f.numerator)/f.denominator


def intervals(poly):
    a,b,c = poly
    if not a:
        if not b:
            return [(-inf, inf)] if c <= 0 else []
        root = -dec(c)/dec(b)
        return [(-inf, root)] if b>0 else [(root, inf)]
    disc = b*b-4*a*c
    if disc < 0:
        return [(-inf, inf)] if a<0 else []
    lo,hi = sorted([(-dec(b)+s*mp.sqrt(dec(disc))) / (2*dec(a)) for s in (-1,1)])
    return [(lo,hi)] if a>0 else [(-inf,lo),(hi,inf)]


def intersection(a,b):
    return [(max(l,u),min(h,v)) for l,h in a for u,v in b if max(l,u)<=min(h,v)]


def encode(segments):
    return [[float(x) if mp.isfinite(x) else ('-inf' if x<0 else '+inf')
             for x in interval] for interval in segments]


def main():
    records=json.loads((ROOT/'certificates.json').read_text())['records']
    rows=[]
    for r in records:
        if r['system'] != 'kdv_cossin':
            continue
        path=ROOT.parent/COHORTS[r['cohort']]/'results'/r['file']
        data=json.loads(path.read_text())
        candidate=next(c for c in data['candidates'] if c['label']=='reference')
        idx=next(k for k in candidate['xi'] if 'd^3u/' in str(data['labels'][k]))
        target=candidate['yi']
        with np.load(path.with_suffix('.npz')) as archive:
            for t in r['rows']:
                if t['omitted'] is not None or t['method'] not in ('original','all_three'):
                    continue
                polynomials=[]
                coefficient_num=coefficient_den=0.
                for name in t['banks']:
                    suffix='_anchor_covariance' if r['cohort']=='core' else '_covariance'
                    bank=Bank(archive[name+'_observed'], archive[name+suffix])
                    q=Fraction(t['cutoff'])
                    xx,xy,yy=[bank.entry(i,j) for i,j in [(idx,idx),(idx,target),(target,target)]]
                    poly=(xx[0]-q*xx[1], -2*(xy[0]-q*xy[1]), yy[0]-q*yy[1])
                    polynomials.append(poly)
                    scale=yy[0]+q*yy[1]
                    coefficient_num+=float(xy[0]/scale)
                    coefficient_den+=float(xx[0]/scale)
                domain=[(-inf,inf)]
                for poly in polynomials:
                    domain=intersection(domain,intervals(poly))
                rows.append(dict(cohort=r['cohort'],profile=r['profile'],seed=r['seed'],
                                 method=t['method'],domain=domain,poly=polynomials,
                                 ols=coefficient_num/coefficient_den))
    training=[r for r in rows if r['cohort']=='core' and r['profile']=='iid']
    assert len(training)==3
    locked=float(np.median([r['ols'] for r in training]))
    combined=[(-inf,inf)]
    for r in rows:
        combined=intersection(combined,r['domain'])
        for label, coefficient in [('locked',locked),('minus_one',-1.),('zero',0.)]:
            x=Fraction(coefficient)
            accepted=all((a*x+b)*x+c<=0 for a,b,c in r['poly'])
            assert accepted == any(l <= coefficient <= h for l,h in r['domain'])
            r[label+'_accepted']=accepted
        r['intervals']=encode(r.pop('domain'))
        r['polynomial_exact']=[list(map(str,p)) for p in r.pop('poly')]
    out=dict(protocol_sha256=digest(ROOT/'kdv_followup_protocol.txt'),locked=locked,
             all_records_and_designs_intersection=encode(combined),rows=rows)
    (ROOT/'kdv_scalar.json').write_text(json.dumps(out,allow_nan=False))
    for method in ('original','all_three'):
        rr=[r for r in rows if r['method']==method]
        print(method,len(rr),'nonempty',sum(bool(r['intervals']) for r in rr),
              'locked',sum(r['locked_accepted'] for r in rr),'truth dispersion alone',
              sum(r['minus_one_accepted'] for r in rr))
    print('locked',locked,'joint intersection',encode(combined),'example',rows[0]['intervals'])


if __name__=='__main__':
    main()
