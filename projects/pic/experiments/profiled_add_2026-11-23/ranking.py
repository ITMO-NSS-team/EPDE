"""Training-only variable projection for all admissible single-term additions."""
from copy import copy, deepcopy
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'candidate_frequency_2026-11-23'))
from candidate_trial import Trial, live, np, trig_terms, minimize_scalar
from swap_trial import fill_prune
from epde.structure.weak_rows import term_columns

sys.path.append(str(live.operators.HISTORICAL))
import arm_pvguide as guide


def libraries(eq):
    # Enumeration creates factor IDs. This diagnostic must not advance discovery's RNG.
    state, numpy_state = random.getstate(), np.random.get_state()
    cfg = fill_prune.factors_setting(eq)
    try:
        ordinary = fill_prune._enumerate(eq.pool,cfg)
        pool = copy(eq.pool)
        pool.families = list(eq.pool.families)
        for i,family in enumerate(pool.families):
            if family.ftype=='trigonometric':
                clone = copy(family)
                clone.status = dict(family.status,meaningful=True)
                pool.families[i] = clone
        expanded = fill_prune._enumerate(pool,cfg)
        assert ordinary.keys() <= expanded.keys()
        terms = list(ordinary.values()) + [v for k,v in expanded.items() if k not in ordinary]
        for term in terms:
            term.pool = eq.pool
        return terms,set(ordinary)
    finally:
        random.setstate(state)
        np.random.set_state(numpy_state)


def set_frequency(term, value):
    factor,index,_axis = next(trig_terms(term))
    desc = dict(factor.params_description)
    desc[index] = dict(desc[index],grid=(value,))
    factor.params_description = desc
    params = factor.params.copy()
    params[index] = value
    factor.params = params
    term.resetSavedState()


def column(term, slot):
    arrays = term_columns(term,slot)
    value, = arrays.values()
    return np.asarray(value).reshape(-1)


def ordered(rows, field):
    return sorted([i for i,r in enumerate(rows) if r[field]>0], key=lambda i:-rows[i][field])


def rank(eq, partners, ranges, library, labels, profile=True):
    slot = eq.main_var_to_explain
    target = eq.target
    active = [t for i,t in enumerate(eq.structure) if i!=eq.target_idx and eq.weights_final[eq.weight_index(i,eq.target_idx)]]
    intercept = bool(eq.weights_final[-1])
    y = column(target,slot)
    x = np.column_stack([column(t,slot) for t in active]) if active else np.empty((len(y),0))
    basis = guide.SC.gain_basis(y,x,np.ones(len(y)),intercept)
    ranked = []
    proposals = []
    total_trials = 0
    cache_hits = 0
    for original in library[0]:
        if not guide.admissible(original,eq,target,partners) or not guide.keeps_target(original,eq,target,partners):
            continue
        tones = list(trig_terms(original))
        assert len(tones)<=1
        term = deepcopy(original) if tones else original
        def gain():
            z = column(term,slot)
            return float(guide.SC.gains_on_basis(basis,z)[0]) if np.isfinite(z).all() else 0.
        initial_gain = gain()
        row = dict(label=labels.term(original),ordinary=original.factors_labels in library[1],
                   fixed_gain=initial_gain,profiled_gain=initial_gain,fits=1)
        if tones:
            factor,index,axis = tones[0]
            initial = float(factor.params[index])
            lo,hi,width = ranges[axis,initial]
            table = {initial:initial_gain}
            def score(w):
                nonlocal cache_hits
                w = float(w)
                if w in table:
                    cache_hits += 1
                else:
                    set_frequency(term,w)
                    table[w] = gain()
                return table[w]
            if profile and lo<initial<hi:
                score(lo)
                score(hi)
                result = minimize_scalar(lambda w:-score(w),method='bounded',bounds=(lo,hi),
                                         options=dict(maxiter=24,xatol=width*1e-4))
                score(result.x)
            best = max(table,key=lambda w:(table[w],-abs(w-initial)))
            set_frequency(term,best)
            row.update(initial=initial,frequency=best,axis=axis,lower=lo,upper=hi,
                       profiled_gain=table[best],fits=len(table))
        total_trials += row['fits']
        ranked.append(row)
        proposals.append(term)
    # Verify the profiled shortlist by independently fitting all coefficients.
    design = np.column_stack((x,np.ones(len(y)))) if intercept else x
    def rss(m):
        weights = np.linalg.lstsq(m,y,rcond=None)[0]
        residual = y-m@weights
        return residual@residual
    base = rss(design)
    for i in ordered(ranked,'profiled_gain')[:8]:
        direct = base-rss(np.column_stack((design,column(proposals[i],slot))))
        np.testing.assert_allclose(direct,ranked[i]['profiled_gain'],rtol=1e-7,atol=1e-10*max(1.,y@y))
    return ranked,proposals,dict(fixed_fits=1+len(ranked),profiled_fits=1+total_trials,
                                 cache_hits=cache_hits,verification_fits=1+min(8,len(ordered(ranked,'profiled_gain'))))
