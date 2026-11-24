"""Run native discovery with the historical PVRA chooser at the plain add move."""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent
SPLIT = ROOT.parent/'single_record_archive_2026-11-19'
sys.path.insert(0,str(SPLIT))
from run_split import Archive,measurements

HISTORICAL = Path(os.environ.get('EPDE_HISTORICAL_OPERATOR_DIR',
                  Path.home()/'AppData/Local/Temp/claude/C--Users-NSSLab-PycharmProjects-EPDE'/
                  'c5a71b1a-1530-4755-b8a8-0feb3924d241/scratchpad/auto')).resolve()
SEEDS = tuple(range(923101,923107))
ARMS = ('native','pvra')
PARTS = ('early','late')
SYSTEM = 'pde_compound'


def folder(arm,part):
    return ROOT/'results'/arm/part


def run(arm,seed,part,*,experiment=None):
    from search import run_job,np
    from epde.operators.common import fill_prune
    from epde.operators.common.plain_fit import plain_params
    from epde.operators.multiobjective import plain_moves
    from epde.structure.main_structures import Equation
    sys.path.append(str(HISTORICAL))
    import arm_pvguide as guide
    assert Path(guide.__file__).resolve().parent==HISTORICAL
    if arm=='pvpa':
        from search import U
        import arm_pvpa as pair
        assert Path(pair.__file__).resolve().parent==HISTORICAL and pair.G is guide
    proposals = []
    counters = Counter()
    add,fill = Equation.add_random_term,fill_prune.fill
    dest = folder(arm,part)
    archive = Archive(dest) if experiment is None else None
    if experiment is not None and arm not in ARMS:
        raise ValueError('The external search hook supports native and single-add runs.')

    def counted_fill(eq,target,features,system,params):
        assert params['fill_probability']==.25 and params['fill_levels']=='all'
        success = fill(eq,target,features,system,params)
        counters['fill_path_calls'] += 1
        counters['successful_fills'] += int(success)
        return success

    def local_add(eq,*args,**kwargs):
        system = plain_moves._MUTATING['system']
        if system is None or system.vals[eq.main_var_to_explain] is not eq or args or kwargs:
            return add(eq,*args,**kwargs)
        counters['ordinary_add_opportunities'] += 1
        if arm=='native':
            return add(eq)
        if np.random.random()<guide.EPS:
            counters['exploration_draws'] += 1
            return add(eq)
        if len(eq.structure)>=int(eq.metaparameters['max_terms_number']['value']):
            counters['at_cap'] += 1
            return False
        partners = frozenset(fill_prune.partner_targets(system,eq.main_var_to_explain))
        design,reason = guide.child_design(eq,partners)
        if design is None:
            counters['fallback_'+reason] += 1
            return add(eq)
        choose_pair=False
        if arm=='pvpa':
            cap=guide.SC._config()['cap']
            eligible=(pair._features_count(eq)+2<=cap and len(eq.structure)+2<=int(eq.metaparameters['max_terms_number']['value']))
            counters['pair_eligible']+=int(eligible)
            choose_pair=eligible and np.random.random()<pair.P_PAIR
        if choose_pair:
            counters['pair_attempts']+=1
            terms=pair.pair_choose(eq,design,partners,cap)
        else:
            term=guide.ra_choose(eq,design,partners)
            terms=None if term is None else [term]
        if not terms:
            counters['fallback_empty'] += 1
            return add(eq)
        present=set(eq.terms_labels)
        new_terms=[]
        for term in terms:
            new=deepcopy(term)
            new.pool=eq.pool
            new.resetSavedState()
            assert new.factors_labels not in present
            present.add(new.factors_labels)
            new_terms.append(new)
        if arm=='pvpa':
            proposals.append(dict(after_call=archive.calls,target=U.term_key(design['target_term']),
                                  parents=[U.term_key(t) for t in design['terms']],
                                  added=[U.term_key(t) for t in terms]))
            counters['pairs_installed']+=int(len(terms)==2)
        eq.structure.extend(new_terms)
        eq._invalidate_label_cache()
        counters['guided_additions'] += 1
        return True

    with ExitStack() as patches:
        patches.enter_context(patch.object(Equation,'add_random_term',local_add))
        patches.enter_context(patch.object(fill_prune,'fill',counted_fill))
        if experiment is None:
            run_job(SYSTEM,seed,'wape',archive,dest,measurements(SYSTEM,seed)[part])
        else:
            experiment()
    if experiment is None:
        assert counters['successful_fills'] and counters['ordinary_add_opportunities']
        assert bool(counters['guided_additions'])==(arm in ('pvra','pvpa'))
    info = dict(arm=arm,seed=seed,part=part,counters=counters,guide_counts=guide.COUNTS,
                gain_columns=guide.SC.COUNTS['gain_columns'],chosen_ranks=guide.RANKS,
                fill_probability=plain_params()['fill_probability'],
                chooser_sha256=hashlib.sha256(Path(guide.__file__).read_bytes()).hexdigest())
    if arm=='pvpa':
        assert counters['pairs_installed'] and pair.PA['pair_joint_evals']
        info.update(pair_counts=pair.PA,pair_ranks=pair.PAIR_RANKS,pair_events=proposals,
                    pair_sha256=hashlib.sha256(Path(pair.__file__).read_bytes()).hexdigest())
    if experiment is not None:
        return info
    (dest/f'operator_{seed}.json').write_text(json.dumps(info),encoding='utf8')


def launch(job):
    arm,seed,part=job
    dest=folder(arm,part)
    dest.mkdir(parents=True,exist_ok=True)
    if (dest/f'operator_{seed}.json').exists():
        return 0
    with (dest/f'search_{seed}.log').open('w') as f:
        proc=subprocess.run([sys.executable,'-u',__file__,arm,str(seed),part],stdout=f,stderr=subprocess.STDOUT,
                            env=dict(os.environ,PYTHONHASHSEED='0'))
    print(arm,seed,part,proc.returncode,flush=True)
    return proc.returncode


if __name__=='__main__':
    if len(sys.argv)==4:
        assert os.environ['PYTHONHASHSEED']=='0'
        run(sys.argv[1],int(sys.argv[2]),sys.argv[3])
    else:
        arms=('pvpa',) if sys.argv[1:]==['--pair'] else ARMS
        jobs=[(a,n,p) for n in SEEDS for p in PARTS for a in arms]
        with ThreadPoolExecutor(max_workers=6) as pool:
            raise SystemExit(int(any(list(pool.map(launch,jobs)))))
