"""Joint term deletion over unrestricted surviving coefficients in the confirmation cohort."""
from collections import Counter
from fractions import Fraction
import json
import numpy as np
from scipy.linalg import eigvalsh
from audit import Bank, pd, quadratic
from study import ROOT, digest
from resolution import solve

READOUTS=[p+b for p in ('whole_','projected_') for b in ('local','broad','selected')]
SUPPORTS={'parent':[0,1,2],'drop_nonlinear':[1,2],'drop_dispersion':[0,2],
          'drop_forcing':[0,1],'dispersion_only':[1]}


def scores(grams,cov):
    # The known forcing column (index 2) has identically zero measurement noise.
    assert np.all(grams[:,2,:]==0) and np.all(grams[:,:,2]==0)
    idx=np.ix_([0,1,3],[0,1,3])
    s=cov[idx]; scale=np.sqrt(np.diag(s));s=s/np.outer(scale,scale)
    if eigvalsh(s)[0] <= 1e-12:
        return None
    return np.array([eigvalsh(g[idx]/np.outer(scale,scale),s)[-1] for g in grams])


def verify(exact,indices,result,cutoff):
    mats=[]
    for bank in exact:
        mats.append([[g - Fraction(cutoff)*s for j in indices for g,s in [bank.entry(i,j)]] for i in indices])
    if result['status']=='excluded':
        weights=[Fraction(w) for w in result['weights']]
        assert min(weights)>=0
        assert pd([[sum(w*m[i][j] for w,m in zip(weights,mats)) for j in range(len(indices))]
                   for i in range(len(indices))])
    elif result['status']=='compatible':
        v=[-Fraction(x) for x in result['coefficients']]+[Fraction(1)]
        assert all(quadratic(m,v)<0 for m in mats)


def main():
    rows=[]
    for path in sorted((ROOT/'forcing_profile').glob('*_47*.npz')):
        profile,seed=path.stem.split('_')
        with np.load(path) as z:
            observed={b:z[b+'_observed'] for b in READOUTS}
            clean={b:z[b+'_clean'] for b in READOUTS}
            grams={b:z[b+'_grams'] for b in READOUTS}
        cov={b:g[:32].mean(axis=0) for b,g in grams.items()}
        checks={b:scores(g[32:],cov[b]) for b,g in grams.items()}
        actual={b:scores(np.array([(m-clean[b]).T@(m-clean[b])]),cov[b]) for b,m in observed.items()}
        exact={b:Bank(observed[b],cov[b]) for b in READOUTS}
        for arm,keys in [('whole',READOUTS[:3]),('projected',READOUTS[3:]),('combined',READOUTS)]:
            group=dict(profile=profile,seed=int(seed),method=arm,readouts=keys,archive=digest(path))
            if any(checks[b] is None for b in keys):
                rows.append(dict(**group,status='unresolved_metric'));continue
            cutoff=float(sorted(np.maximum.reduce([checks[b] for b in keys]))[122])
            fits={}
            for name,free in SUPPORTS.items():
                ids=free+[3]
                fit=solve([observed[b] for b in keys],[cov[b] for b in keys],ids,cutoff)
                verify([exact[b] for b in keys],ids,fit,cutoff)
                fits[name]=fit
            v=np.array([6.,1.,-1.,1.]); control={}
            for tag,vec in [('physical',v),('surrogate',np.array([0.,.25,0.,1.]))]:
                control[tag]=bool(all(np.linalg.norm(observed[b]@vec)**2 <= cutoff*(vec@cov[b]@vec) for b in keys))
            rows.append(dict(**group,status='complete',cutoff=cutoff,term_tests=fits,
                             error_covered=bool(max(actual[b][0] for b in keys)<=cutoff),controls=control))
    assert len(rows)==36
    totals={}
    for method in ('whole','projected','combined'):
        rr=[r for r in rows if r['method']==method and r['status']=='complete']
        totals[method]={name:dict(Counter(r['term_tests'][name]['status'] for r in rr)) for name in SUPPORTS}
        totals[method]['parent_and_deletion_excluded']={name:sum(r['term_tests']['parent']['status']=='compatible'
               and r['term_tests'][name]['status']=='excluded' for r in rr) for name in SUPPORTS if name!='parent'}
        totals[method]['error_covered']=sum(r['error_covered'] for r in rr)
        totals[method]['eligible']=len(rr)
        totals[method]['fixed_passed']={tag:sum(r['controls'][tag] for r in rr) for tag in ('physical','surrogate')}
    (ROOT/'forcing_profile.json').write_text(json.dumps(dict(rows=rows,totals=totals,
        protocol_sha256=digest(ROOT/'forcing_profile_protocol.txt'),source=digest(ROOT/'forcing_profile.py')),indent=2))
    print(json.dumps(totals,indent=2))


if __name__=='__main__':
    main()
