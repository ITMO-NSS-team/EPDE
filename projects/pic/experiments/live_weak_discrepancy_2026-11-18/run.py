"""Native domain_refactor U search with the two alternative first objectives."""
import argparse
from contextlib import ExitStack
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent
PIC = ROOT.parents[1]
for name in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS'):
    os.environ[name] = '1'
sys.path[:0] = [str(PIC.parent / 'robot' / 'epde_u'), str(PIC), str(PIC.parent.parent)]
import run_auto as U
import numpy as np
from epde.operators.common.objectives import Discrepancy
from epde.operators.common.sparsity import VWSRSparsity
from epde_bench.metrics import select_compromise


def support(eq):
    labels, intercept = U.active_struct(eq)
    return frozenset(labels) | ({'1'} if intercept != 0 else set())


def run(system, seed, arm):
    axes, clean = U.load(system)
    rng = np.random.default_rng([seed, list(U.TRUTH).index(system)])
    raw = {v: y + .05 * y.std() * rng.standard_normal(y.shape) for v, y in clean.items()}
    grids = np.meshgrid(*axes, indexing='ij')
    orders = (2,) + (4,) * (len(axes) - 1)
    width, epochs = 7, 50
    start = time.perf_counter()
    if arm == 'weak':
        from live_score import LiveScore
        data = SimpleNamespace(data=raw, grids=grids, shape=next(iter(raw.values())).shape, named_arrays={})
        scorer = LiveScore(data, raw, seed, orders, width)
    setup = time.perf_counter() - start
    print(system, seed, arm, 'setup', round(setup, 2), flush=True)

    original = Discrepancy._compute_wape
    scored = {}
    calls = 0
    def primary(filler, eq, ctx):
        nonlocal calls
        assert not ctx.for_rps, 'U must fit the evolved support directly'
        calls += 1
        value = scorer.evaluate(eq, ctx) if arm == 'weak' else original(filler, eq, ctx)
        if np.isfinite(value):
            label = support(eq)
            scored[label] = min(scored.get(label, float('inf')), float(value))
        return value

    settings = dict(use_solver=False, device='cpu', multiobjective_mode=True,
                    system_rps_mode='plain', engine='nsga2', evaluation_form='weak',
                    preprocessor_kwargs={'accuracy': 4}, default_preprocessor_type='FD',
                    discrepancy_metric='wape', instability_metric='dyad',
                    system_rps_params={'variable_order_seed': seed}, training_epochs=epochs,
                    early_stopping_patience=None, max_deriv_order=orders,
                    data_fun_pow=3, deriv_fun_pow=2, equation_terms_max_number=width,
                    equation_factors_max_number={'factors_num': [1, 2], 'probas': [.65, .35]},
                    verbose_params={'show_iter_idx': False, 'candidate_objectives': False},
                    memory_for_cache=2)
    def build(with_trig):
        random.seed(seed)
        np.random.seed(seed)
        search = U.EpdeSearch(**settings)
        _, domain = search.createDomain(tuple(grids), ID=0)
        search.set_preprocessor(default_preprocessor_type='FD', preprocessor_kwargs={'accuracy': 4})
        _, traj = search.createTrajectory(raw, domain, cache_id=0)
        families = [U.TrigonometricTokens(freq_from_data=2)] if with_trig else []
        families.append(U.GridTokens(labels=['t', 'x', 'y'][:len(axes)], max_power=2))
        search.fit(data=[traj], additional_tokens=families)
        return search

    with ExitStack() as patches:
        guard = patches.enter_context(patch.object(VWSRSparsity, 'apply', side_effect=AssertionError('VWSR called in plain mode')))
        patches.enter_context(patch.object(Discrepancy, '_compute_wape', primary))
        try:
            search = build(True)
        except ValueError as err:
            if 'freq_from_data' not in str(err) or calls:
                raise
            search = build(False)
        assert calls and guard.call_count == 0
    assert len(search.optimizer._pareto_history) > 0
    front = []
    for solution in search.equations(only_print=False)[0]:
        eq = solution.vals['u']
        front.append(dict(terms=sorted(support(eq)), equation=eq.text_form, objectives=solution.obj_fun.tolist()))
    truths = [(kind, frozenset([target, *terms])) for kind, target, terms in U.truth_forms(system)['u']]
    def distance(key):
        return min(len(key ^ truth) for _, truth in truths)
    for row in front:
        key = frozenset(row['terms'])
        row['distance'] = distance(key)
        row['hits'] = [kind for kind, law in truths if key == law]
    picked = select_compromise([row['objectives'] for row in front])
    record = dict(system=system, seed=seed, arm=arm, settings=settings, noise=.05,
                  raw_sha256={v: hashlib.sha256(y.tobytes()).hexdigest() for v, y in raw.items()},
                  preparation_seconds=setup, search_seconds=time.perf_counter()-start-setup,
                  epochs_recorded=len(search.optimizer._pareto_history), population=search.optimizer._mu, population_measure='nsga_mu',
                  calls=calls, vwsr_calls=guard.call_count, evaluated_supports=len(scored),
                  evaluated_laws={kind: scored[law] for kind, law in truths if law in scored},
                  front=front, min_distance=min(row['distance'] for row in front),
                  selected=picked, selected_distance=front[picked]['distance'])
    if arm == 'weak':
        record['weak'] = dict(columns=len(scorer.columns), supports=len(scorer.scores), negative=scorer.negative,
                              unresolved=scorer.unresolved, modes=scorer.window.shape, draws=128,
                              unweighted_folds=scorer.unweighted_folds)
    dest = ROOT / 'plain_results' / f'{system}_{seed}_{arm}.json'
    dest.parent.mkdir(exist_ok=True)
    dest.write_text(json.dumps(record, indent=1, allow_nan=False), encoding='utf-8')
    print(system, seed, arm, 'minimum distance', record['min_distance'], flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('system', choices=['ode','vdp','wave','ac'])
    parser.add_argument('seed', type=int)
    parser.add_argument('arm', choices=['wape','weak'])
    args = parser.parse_args()
    if os.environ.get('PYTHONHASHSEED') != '0':
        raise RuntimeError('Set PYTHONHASHSEED=0 before starting Python.')
    from filelock import FileLock
    outputs = ROOT / 'plain_results'
    outputs.mkdir(exist_ok=True)
    tag = f'{args.system}_{args.seed}_{args.arm}'
    locks = ROOT / 'logs'
    locks.mkdir(exist_ok=True)
    with FileLock(str(locks/(tag+'.lock'))):
        if not (outputs/(tag+'.json')).exists():
            run(args.system, args.seed, args.arm)
