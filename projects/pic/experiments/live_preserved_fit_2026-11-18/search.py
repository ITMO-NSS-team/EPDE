"""Run one isolated plain-mode search and replay the primary objective of its full front."""
import argparse
from contextlib import ExitStack
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time
from unittest.mock import patch

for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[key] = '1'
ROOT = Path(__file__).resolve().parent
PRIOR = ROOT.parent / 'live_weak_discrepancy_2026-11-18'
sys.path.insert(0,str(PRIOR))
import run as baseline
from blocked import BlockedScore, fit, np
import epde.globals as global_var
from epde.operators.common.fitness import SolverFreeFitness
from epde.operators.common.objectives import Discrepancy, LOSS_NAN_VAL
from epde.operators.common.sparsity import VWSRSparsity
from scipy import linalg

U = baseline.U
SYSTEMS = ('pde_compound','wave','vdp','ac','ode')
ARMS = ('wape','wls','tls')
SEEDS = (916201,916202,916203)


def full_columns(eq, weak):
    target = eq.target_idx
    entries = []
    for i,t in enumerate(eq.structure):
        if i==target or eq.weights_final[eq.weight_index(i,target)]!=0:
            entries.append((U.term_key(t),weak.columns(t,'u')[0],
                            -1. if i==target else float(eq.weights_final[eq.weight_index(i,target)])))
    if eq.weights_final[-1] != 0:
        entries.append(('1',np.ones(len(entries[0][1])),float(eq.weights_final[-1])))
    return entries


def replay(eq, scorer, arm):
    entries = full_columns(eq,scorer.weak)
    labels, cols, coefs = zip(*entries)
    matrix = np.column_stack(cols)
    pivot = labels.index(U.term_key(eq.structure[eq.target_idx]))
    tls, status = fit(matrix)
    full_tls = None
    if status=='ok' and abs(tls[pivot])*np.linalg.norm(matrix[:,pivot])>1.e-12*np.linalg.norm(tls*np.linalg.norm(matrix,axis=0)):
        full_tls = (-tls/tls[pivot]).tolist()
    if arm=='wape':
        score = float(np.abs(matrix @ coefs).sum()/np.abs(matrix[:,pivot]).sum())
    else:
        ids, target = scorer.layout(eq)
        halves = scorer.matrices(ids)
        values = []
        for g,m in enumerate(halves):
            scale = np.linalg.norm(m,axis=0)
            a = m/scale
            if arm=='tls':
                v = linalg.svd(a,full_matrices=False,lapack_driver='gesvd')[2][-1] / scale
            else:
                p = ids.index(target)
                keep = np.arange(len(ids))!=p
                v = np.ones(len(ids))
                v[keep] = -linalg.lstsq(a[:,keep],a[:,p],lapack_driver='gelsy')[0]
                v /= scale
            residual = halves[1-g] @ v
            values.append(float(np.mean(residual**2)*len(m)/np.dot(scale*v,scale*v)))
        score = float(np.mean(values))
    return dict(labels=list(labels), target=labels[pivot], native=list(coefs), tls=full_tls,
                tls_status=status, replay=score)


def run_job(system,seed,arm,observer=None,output_dir=None,measurements=None):
    start = time.monotonic()
    if measurements is None:
        axes,clean = U.load(system)
        rng = np.random.default_rng([seed,list(U.TRUTH).index(system)])
        raw = {v: y+.05*y.std()*rng.standard_normal(y.shape) for v,y in clean.items()}
    else:
        axes,raw = measurements
    grid = tuple(np.meshgrid(*axes,indexing='ij'))
    settings = json.loads((PRIOR/'plain_results'/'wave_73001_wape.json').read_text())['settings']
    settings.update(training_epochs=10,max_deriv_order=(2,)+(4,)*(len(grid)-1),
                    system_rps_params={'variable_order_seed':seed})
    truths = {frozenset([target,*terms]):kind for kind,target,terms in U.truth_forms(system)['u']}
    scored = set()
    ref_points = {}
    pending = {}
    events = 0
    scorer = None
    original_discrepancy = Discrepancy._compute_wape
    original_apply = SolverFreeFitness.apply
    def discrepancy(filler,eq,ctx):
        nonlocal scorer, events
        assert not ctx.for_rps
        events += 1
        if scorer is None:
            scorer = BlockedScore(global_var.samples_manager.weak,'u')
        value = original_discrepancy(filler,eq,ctx) if arm=='wape' else scorer.evaluate(eq,arm)
        support = baseline.support(eq)
        scored.add(support)
        if support in truths:
            pending[id(eq)] = truths[support]
        return value
    def record_apply(host,objective,*args,**kwargs):
        output = original_apply(host,objective,*args,**kwargs)
        if observer is not None:
            observer.evaluated(objective)
        kind = pending.pop(id(objective),None)
        if kind is not None:
            values = (float(objective.fitness_value),float(objective.coefficients_stability))
            target = U.term_key(objective.structure[objective.target_idx])
            ref_points[kind,target,values] = objective.text_form
        return output
    def build(trig):
        random.seed(seed)
        np.random.seed(seed)
        search = U.EpdeSearch(**settings)
        _,domain = search.createDomain(grid,ID=0)
        search.set_preprocessor(default_preprocessor_type='FD',preprocessor_kwargs={'accuracy':4})
        _,traj = search.createTrajectory(raw,domain,cache_id=0)
        families = [U.TrigonometricTokens(freq_from_data=2)] if trig else []
        families.append(U.GridTokens(labels=['t','x'][:len(axes)],max_power=2))
        search.fit(data=[traj],additional_tokens=families)
        return search
    with ExitStack() as patches:
        patches.enter_context(patch.object(Discrepancy,'_compute_wape',discrepancy))
        patches.enter_context(patch.object(SolverFreeFitness,'apply',record_apply))
        guard = patches.enter_context(patch.object(VWSRSparsity,'apply',side_effect=AssertionError('plain mode called VWSR')))
        try:
            search = build(True)
        except ValueError as err:
            if 'freq_from_data' not in str(err) or events:
                raise
            search = build(False)
    assert not pending and events and not guard.call_count
    front = []
    for solution in search.equations(only_print=False)[0]:
        eq = solution.vals['u']
        key = baseline.support(eq)
        obj = solution.obj_fun.tolist()
        details = replay(eq,scorer,arm)
        if max(obj) < LOSS_NAN_VAL:
            np.testing.assert_allclose(obj[0],details['replay'],rtol=1e-6,atol=1e-10)
        front.append(dict(equation=eq.text_form,terms=sorted(key),objectives=obj,fit=details,hit=truths.get(key),
                          distance=min(len(key^t) for t in truths)))
    assert len(search.optimizer._pareto_history)==settings['training_epochs']
    record = dict(system=system,seed=seed,arm=arm,settings=settings,raw_sha256={v:hashlib.sha256(y.tobytes()).hexdigest()
                  for v,y in raw.items()},geometry=scorer.geometry,calls=events,vwsr_calls=guard.call_count,
                  population=search.optimizer._mu,epochs=len(search.optimizer._pareto_history),
                  evaluated_supports=len(scored),cached_fits=len(scorer.cached),unresolved=scorer.invalid,
                  reference_points=[dict(kind=k,target=t,objectives=list(v),equation=e) for (k,t,v),e in ref_points.items()],
                  front=front,seconds=time.monotonic()-start)
    if observer is not None:
        observer.finish(search,record)
    dest = (ROOT/'results' if output_dir is None else output_dir)/f'{system}_{seed}_{arm}.json'
    dest.parent.mkdir(exist_ok=True)
    tmp = dest.with_suffix('.pending')
    tmp.write_text(json.dumps(record,indent=1,allow_nan=False))
    tmp.replace(dest)
    print(system,seed,arm,'front hit',any(r['hit'] for r in front),'calls',events,flush=True)


if __name__=='__main__':
    p = argparse.ArgumentParser()
    p.add_argument('system',choices=SYSTEMS)
    p.add_argument('seed',type=int)
    p.add_argument('arm',choices=ARMS)
    args = p.parse_args()
    if os.environ.get('PYTHONHASHSEED')!='0':
        raise RuntimeError('Run with PYTHONHASHSEED=0')
    run_job(args.system,args.seed,args.arm)
