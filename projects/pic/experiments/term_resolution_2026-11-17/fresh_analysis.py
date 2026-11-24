"""Evaluate frozen vectors with joint whole-library and two-equation cutoffs."""
import json
from fractions import Fraction
import numpy as np
from scipy.linalg import eigvalsh
from audit import Bank, pd
from resolution import solve
from study import ROOT, digest


def library_score(gram, covariance, active):
    ix=np.ix_(active,active)
    h,s=gram[ix],covariance[ix]
    scale=np.sqrt(np.diag(s))
    h=h/scale[:,None]/scale[None,:]
    s=s/scale[:,None]/scale[None,:]
    return float(eigvalsh((h+h.T)/2,(s+s.T)/2)[-1])


def check_record(path,locked):
    meta=json.loads(path.read_text())
    fp=ROOT/'fresh'/'frozen'/path.name
    assert digest(fp)==meta['frozen_sha256'] and digest(path.with_suffix('.npz'))==meta['array_sha256']
    candidate=next(c for c in meta['candidates'] if c['label']=='reference')
    assert candidate['exact']==[-6.,-1.,1.]
    assert 'd^3u/' in str(meta['labels'][candidate['xi'][1]])
    banks=('local','broad','selected')
    vectors={}
    for tag,coef in [('physical',candidate['exact']),('surrogate',[0.,-.25,0.]),
                     ('ols',[0.,locked,0.]),('disp_minus_one',[0.,-1.,0.]),('zero',[0.]*3)]:
        v=np.zeros(len(meta['labels']))
        v[candidate['yi']]=1.
        v[candidate['xi']]=-np.asarray(coef)
        vectors[tag]=v
    observed={k:{} for k in vectors}
    operator=[]; focused=[]; actual_library=[]; mats=[]; covs=[]; exact={}
    with np.load(path.with_suffix('.npz')) as z:
        for bank in banks:
            grams=z[bank+'_bootstrap_grams']
            noise=z[bank+'_bootstrap_covariances']
            scores=np.array([library_score(g,s,meta['active']) for g,s in zip(grams,noise)])
            assert len(scores)==128
            np.testing.assert_allclose(scores,z['studentized_'+bank+'_scores'],rtol=2e-8,atol=1e-10)
            operator.append(z['studentized_'+bank+'_scores'])
            obs=z[bank+'_observed']; clean=z[bank+'_clean']; cov=z[bank+'_covariance']
            mats.append(obs);covs.append(cov);exact[bank]=Bank(obs,cov)
            actual_library.append(library_score((obs-clean).T@(obs-clean),cov,meta['active']))
            for tag,v in vectors.items():
                if tag in ('physical','surrogate'):
                    num=np.einsum('i,bij,j->b',v,grams,v)
                    den=np.einsum('i,bij,j->b',v,noise,v)
                    assert np.all(den>0)
                    focused.append(num/den)
                allowance=float(v@cov@v)
                residual=obs@v
                delta=(obs-clean)@v
                target_norm=np.linalg.norm(clean[:,candidate['yi']])
                observed[tag][bank]=dict(stat=float(residual@residual/allowance),
                    error_stat=float(delta@delta/allowance),
                    relative_clean_residual=float(np.linalg.norm(clean@v)/target_norm))
    lib_draw=np.maximum.reduce(operator); focus_draw=np.maximum.reduce(focused)
    assert np.all(focus_draw<=lib_draw*(1+1e-7))
    lib=float(np.sort(lib_draw)[122]);focus=float(np.sort(focus_draw)[122])
    decisions={}
    for tag,stats in observed.items():
        v=vectors[tag]
        ids=np.flatnonzero(v)
        reported={}
        for name,cut in [('library',lib),('focused',focus)]:
            if name=='focused' and tag not in ('physical','surrogate'):
                continue
            margins=[]
            for bank,stat in stats.items():
                value=sum(Fraction(float(v[i]))*Fraction(float(v[j]))*
                          (g-Fraction(cut)*s) for i in ids for j in ids
                          for g,s in [exact[bank].entry(i,j)])
                assert (value<0)==(stat['stat']<cut)
                margins.append(value)
            reported[name]=all(x<=0 for x in margins)
        decisions[tag]=reported
    free=[candidate['xi'][0],candidate['xi'][2],candidate['yi']]
    dropped=solve(mats,covs,free,lib)
    if dropped['status']=='excluded':
        coeff=[Fraction(w) for w in dropped['weights']]
        assert min(coeff)>=0
        combo=[[sum(w*(g-Fraction(lib)*s) for w,b in zip(coeff,banks)
                    for g,s in [exact[b].entry(i,j)]) for j in free] for i in free]
        assert pd(combo)
    return dict(profile=meta['profile'],seed=meta['seed'],warnings=meta['warnings'],
                raw_hashes=meta['raw_sha256'],source_array=meta['array_sha256'],source_json=digest(path),
                design_sha256=digest(ROOT/'fresh'/'design_screen.json'),
                library_cutoff=lib,focused_cutoff=focus,cutoff_ratio=focus/lib,
                full_error_covered=max(actual_library)<=lib,
                focused_error_covered=max(stat['error_stat'] for tag in ('physical','surrogate')
                                          for stat in observed[tag].values())<=focus,
                observed=observed,decisions=decisions,dispersion_deleted=dropped,
                calibration=dict(full_max=lib_draw.tolist(),focused_max=focus_draw.tolist()))


def main():
    paths=sorted((ROOT/'fresh'/'results').glob('*.json'))
    assert len(paths)==6
    locked=json.loads((ROOT/'kdv_scalar.json').read_text())['locked']
    records=[check_record(p,locked) for p in paths]
    out=dict(records=records,sources={p:digest(ROOT/p) for p in
             ('fresh.py','fresh_protocol.txt','focused_protocol.txt','fresh_analysis.py','kdv_scalar.json')})
    (ROOT/'fresh_analysis.json').write_text(json.dumps(out,allow_nan=False))
    for r in records:
        print(r['profile'],r['seed'],round(r['cutoff_ratio'],3),r['decisions'],
              'drop dispersion',r['dispersion_deleted']['status'],'errors',
              r['full_error_covered'],r['focused_error_covered'])


if __name__=='__main__':
    main()
