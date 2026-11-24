"""After search, replay the historical shortlist from two reference-adjacent parents."""
from concurrent.futures import ProcessPoolExecutor
import csv
import json
import random
import sys
from types import SimpleNamespace
from unittest.mock import patch

from operator_trial import HISTORICAL, PARTS, ROOT, SEEDS, SYSTEM, folder, measurements
from search import U, global_var, np
from epde.operators.common.plain_fit import target_gene


def check(job, pair_probe=None):
    seed,part=job
    sys.path.append(str(HISTORICAL))
    import arm_pvguide as guide
    rec=json.loads((folder('pvra',part)/f'{SYSTEM}_{seed}_wape.json').read_text())
    settings=rec['settings']
    axes,data=measurements(SYSTEM,seed)[part]
    grid=tuple(np.meshgrid(*axes,indexing='ij'))
    random.seed(seed)
    np.random.seed(seed)
    search=U.EpdeSearch(**settings)
    _,domain=search.createDomain(grid,ID=0)
    search.set_preprocessor(default_preprocessor_type='FD',preprocessor_kwargs={'accuracy':4})
    _,trajectory=search.createTrajectory(data,domain,cache_id=0)
    result=[]
    def probe(*args):
        native=global_var.samples_manager.weak.plans[0,'u']
        for key,values in rec['geometry']['native'].items():
            assert tuple(getattr(native,key))==tuple(values)
        factors=settings['equation_factors_max_number']
        wanted=U.truth_forms(SYSTEM)['u'][0]
        lib=guide.SC.library_terms(search.pool,factors)
        keys={U.term_key(t):t for t in lib if U.term_key(t) in [wanted[1],*wanted[2]]}
        target=keys[wanted[1]]
        if pair_probe is not None:
            result.extend(pair_probe(seed,part,search,guide,lib,wanted,factors))
            return
        for present,missing in [wanted[2],wanted[2][::-1]]:
            parent=SimpleNamespace(structure=[target,keys[present]],pool=search.pool,main_var_to_explain='u',
                                   metaparameters={'max_factors_in_term':{'value':factors}},_plain_genes=target_gene(target))
            design,reason=guide.child_design(parent,frozenset())
            assert reason is None and design['target_term'] is target
            cands,gains,top,_=guide.stage1(parent,design,frozenset())
            ix,=[i for i,(t,_) in enumerate(cands) if U.term_key(t)==missing]
            rank=list(np.argsort(-gains,kind='stable')).index(ix)+1
            prob=.9*gains[ix]/np.sum(gains[top]) if ix in top else 0.
            result.append(dict(seed=seed,part=part,present=present,missing=missing,rank=rank,
                               eligible_candidates=len(cands),top_eight=int(ix in top),prob_guided_pick=prob))
    with patch.object(search,'_run_optimization',probe):
        families=[U.TrigonometricTokens(freq_from_data=2),U.GridTokens(labels=['t','x'],max_power=2)]
        try:
            search.fit(data=[trajectory],additional_tokens=families)
        except ValueError as e:
            if 'freq_from_data' not in str(e) or result:
                raise
            search.fit(data=[trajectory],additional_tokens=families[1:])
    assert len(result)==2
    return result


if __name__=='__main__':
    with ProcessPoolExecutor(max_workers=3) as pool:
        rows=[r for group in pool.map(check,[(s,p) for s in SEEDS for p in PARTS]) for r in group]
    with (ROOT/'shortlist.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
