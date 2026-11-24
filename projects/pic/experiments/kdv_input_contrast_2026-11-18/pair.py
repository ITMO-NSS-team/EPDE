"""Fit shared intrinsic KdV coefficients across recorded forced and unforced trajectories."""
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parent
PREVIOUS = ROOT.parent / 'term_resolution_2026-11-17'
ARCHIVE = ROOT.parent / 'full_pipeline_uncertainty_2026-11-17' / 'results'
sys.path.insert(0, str(PREVIOUS))
from audit import Bank, pd, quadratic
from resolution import solve


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def observed_banks(system, seed):
    path = ARCHIVE / f'{system}_iid_seed{seed}.json'
    meta = json.loads(path.read_text())
    npz = path.with_suffix('.npz')
    assert sha(npz) == meta['array_sha256']
    ref = next(c for c in meta['candidates'] if c['label'] == 'reference')
    indices = [*ref['xi'], ref['yi']]
    labels = [meta['labels'][i] for i in indices]
    forced = system == 'kdv_cossin'
    assert len(labels) == (4 if forced else 3)
    assert ref['exact'] == ([-6, -1, 1] if forced else [-6, -1])
    if forced:
        assert 'cos(t)sin(x)' in str(labels[2])
    columns = [0, 1, 2, 3] if forced else [0, 1, 3]
    banks = []
    with np.load(npz) as z:
        scores = np.maximum(z['refit_pilot_local_scores'], z['refit_pilot_broad_scores'])
        assert scores.shape == (128,)
        cutoff = float(np.sort(scores)[125])
        for name in ['local', 'broad']:
            raw = z[name+'_observed'][:, indices]
            moment = np.zeros((len(raw), 4))
            moment[:, columns] = raw
            cov = np.zeros((4,4))
            cov[np.ix_(columns, columns)] = z[name+'_anchor_covariance'][np.ix_(indices, indices)]
            banks.append((moment, cov, cutoff, Bank(moment, cov)))
    return banks, labels, dict(file=path.name, json_sha256=sha(path), npz_sha256=meta['array_sha256'],
                               raw_sha256=meta['raw_sha256'], cutoff=cutoff)


def exact_forms(banks, ids):
    mats = []
    for _, _, cutoff, bank in banks:
        q = Fraction(cutoff)
        mats.append([[g-q*s for j in ids for g,s in [bank.entry(i,j)]] for i in ids])
    return mats


def classify(banks, free):
    ids = [*free, 3]
    outcome = solve([m for m,_,_,_ in banks], [s*q for _,s,q,_ in banks], ids, 1.)
    mats = exact_forms(banks, ids)
    if outcome['status'] == 'unresolved':
        # Find a finite interior witness when the eigenvector proposals miss it.
        # A failed local fit supplies no evidence for exclusion.
        signs = np.r_[-np.ones(len(free)),1.]
        blocks = []
        designs = []
        for m,s,q,_ in banks:
            x = m[:,ids]*signs
            noise = s[np.ix_(ids,ids)]*np.outer(signs,signs)*q
            gram = x.T@x
            strength = gram[-1,-1]+noise[-1,-1]
            assert strength > 0
            blocks.append((gram-noise)/strength)
            designs.append(x/np.sqrt(strength))
        h = np.stack(blocks)
        def values(x):
            v = np.r_[x[:-1],1.]
            return np.einsum('i,bij,j->b',v,h,v)
        def jac(x):
            v = np.r_[x[:-1],1.]
            return np.column_stack((-2*(h@v)[:,:-1],np.ones(len(h))))
        for x in [np.vstack(designs),*designs]:
            beta=np.linalg.lstsq(x[:,:-1],-x[:,-1],rcond=None)[0]
            initial=np.r_[beta,0.]
            initial[-1]=max(values(initial))
            trial=minimize(lambda w:w[-1],initial,method='SLSQP',
                constraints=[dict(type='ineq',fun=lambda w:w[-1]-values(w),jac=jac)],
                bounds=[(None,None)]*len(free)+[(-1.,None)],options={'maxiter':200,'ftol':1e-12})
            if np.isfinite(trial.x).all():
                v=[-Fraction(float(a)) for a in trial.x[:-1]]+[Fraction(1)]
                if all(quadratic(m,v)<0 for m in mats):
                    outcome=dict(status='compatible',coefficients=trial.x[:-1].tolist(),
                                 method='profiled_interior_witness')
                    break
    if outcome['status'] == 'excluded':
        w = list(map(Fraction, outcome['weights']))
        assert min(w)>=0 and max(w)>0
        assert pd([[sum(a*m[i][j] for a,m in zip(w,mats))
                    for j in range(len(ids))] for i in range(len(ids))])
    if outcome['status'] == 'compatible':
        v = [-Fraction(a) for a in outcome['coefficients']] + [Fraction(1)]
        assert all(quadratic(m,v) < 0 for m in mats)
    return outcome


def fixed(banks, vector):
    mats = exact_forms(banks, list(range(4)))
    residuals = [quadratic(m,vector) for m in mats]
    return dict(passed=all(x<=0 for x in residuals), bank_pass=[x<=0 for x in residuals])


def main():
    rows = []
    for seed in [51,52,53]:
        off, off_labels, off_source = observed_banks('kdv',seed)
        on, on_labels, on_source = observed_banks('kdv_cossin',seed)
        assert off_labels == [on_labels[i] for i in [0,1,3]]
        assert off_source['raw_sha256'] != on_source['raw_sha256']
        for setting, banks in [('off',off),('on',on),('pair',off+on)]:
            features = [0,1] if setting=='off' else [0,1,2]
            supports = {'full':features, 'dispersion_only':[1]}
            for i, name in enumerate(['nonlinear','dispersion','forcing'][:len(features)]):
                supports['drop_'+name] = [j for j in features if j!=i]
            fits = {name:classify(banks, cols) for name,cols in supports.items()}
            rows.append(dict(seed=seed,setting=setting,fits=fits,
                             physical=fixed(banks,list(map(Fraction,[6,1,-1,1]))),
                             rival=fixed(banks,[0,Fraction(1,5),0,1]),
                             inputs=[off_source,on_source]))
            print(seed,setting,{k:x['status'] for k,x in fits.items()},
                  'physical',rows[-1]['physical']['passed'],'rival',rows[-1]['rival']['passed'])
    hashes = {p.name:sha(p) for p in [Path(__file__),ROOT/'protocol.txt',PREVIOUS/'resolution.py',PREVIOUS/'audit.py']}
    (ROOT/'results.json').write_text(json.dumps(dict(rows=rows,source_hashes=hashes),indent=2))


if __name__ == '__main__':
    main()
