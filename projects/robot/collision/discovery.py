"""Fold preparation and native plain-mode NSGA-II discovery on physical vector columns."""
import json
import os
from pathlib import Path
import random
import sys
from unittest.mock import patch

import numpy as np

from physics import FPS, LAG, NAMES, OUT, ROBOT, arena_for, design, fit_self, partition, recordings

sys.path.insert(0, str(ROBOT.parents[1]))
from epde.preprocessing import weak_form as wf


def write_json(path, obj):
    path = Path(path)
    temp = path.with_suffix(f'.{os.getpid()}.tmp')
    temp.write_text(json.dumps(obj, allow_nan=False, indent=1), encoding='utf-8')
    temp.replace(path)


def gram(x, y):
    a = np.column_stack((x.reshape(-1, len(NAMES)), y.ravel())).astype(float)
    return (a.T @ a) / len(a)


def solve(g, names):
    ix = [NAMES.index(name) for name in names]
    beta = np.zeros(len(NAMES))
    if ix:
        beta[ix] = np.linalg.lstsq(g[np.ix_(ix, ix)], g[ix, -1], rcond=1e-10)[0]
    return beta


def mse(g, beta):
    v = np.r_[beta, -1.]
    return max(0., float(v @ g @ v))


def window(pos):
    # Use the first physical robot's x track in each training segment to choose one
    # geometry, then apply it independently to every robot and component in that segment.
    obs = pos[:, 0, 0].astype(float)
    fields = wf.Fields({'u': obs}, [np.arange(len(obs))/FPS])
    library = [wf.WeakTerm([('u', (i,))], (0,)) for i in (0,1)]
    plan = wf.choose_window(fields, library, [1], np.random.default_rng(0), gate=.5,
                            rows_ok=lambda shape: shape[0] >= 32)
    return plan.ms[0], plan.ss[0], plan.q[0]


def weak_average(data, kernel, stride):
    # Sum only the requested window centers. No convolution crosses a time, robot or recording boundary.
    length = len(data) - len(kernel) + 1
    out = np.zeros_like(data[:length:stride], dtype=np.float64)
    for i, weight in enumerate(kernel):
        out += weight * data[i:i+length:stride].astype(np.float64, copy=False)
    return out


def weak_design(pos, x, base, plan):
    half, stride, degree = plan
    value, derivative = wf.window_kernels([half], [1/FPS], [degree], [1])[0]
    # Integration by parts: int velocity*psi = -int position*dpsi.
    y = -weak_average(pos, derivative, stride) - weak_average(base, value, stride)
    return weak_average(x, value, stride), y


def training_sample(x, y, tag_index):
    # Fixed stratified screen for structure proposals. Full rows determine the reported coefficients.
    x, y = x.reshape(-1, len(NAMES)), y.ravel()
    rng = np.random.default_rng(22031 + tag_index)
    edges = np.linspace(0, len(y), min(512, len(y))+1, dtype=int)
    idx = rng.integers(edges[:-1], edges[1:])
    return x[idx], y[idx]


def prepare(held):
    dest = OUT/held
    dest.mkdir(parents=True, exist_ok=True)
    train, valid, _ = partition(recordings(), held)
    arenas = arena_for(train)
    lookup, common = fit_self(train, arenas)
    data, plans = {}, {}
    for i, tr in enumerate(train):
        rows, x, base, y, n = design(tr, lookup, common, arenas, stride=1)
        plans[tr.tag] = plan = window(tr.pos[rows])
        wx, wy = weak_design(tr.pos[rows], x, base, plan)
        for form, matrix, target in [('weak', wx, wy), ('strong', x[::LAG], (y[::LAG]-base[::LAG]))]:
            data.setdefault(form+'_gram', []).append(gram(matrix, target))
            sx, sy = training_sample(matrix, target, i)
            data.setdefault(form+'_X', []).append(sx)
            data.setdefault(form+'_y', []).append(sy)
        # Columns needed for the historical two-step local and banded regressions.
        single = n[::LAG] == 1
        data.setdefault('single_gram', []).append(gram(x[::LAG][single], (y[::LAG]-base[::LAG])[single]))
        del x, wx, wy
    for tr in valid:
        _, x, base, y, n = design(tr, lookup, common, arenas, stride=LAG)
        data.setdefault('valid_gram', []).append(gram(x, y-base))
        if not np.any(n > 0):
            raise ValueError('Validation block contains no contacts')
    meta = dict(held=held, train=[tr.tag for tr in train], windows=plans, common=common.tolist(),
                arenas={k:v.tolist() for k,v in arenas.items()},
                lookup={f'{k[0]}/{k[1]}':v.tolist() for k,v in lookup.items()},
                split_frames={tr.tag:len(tr.pos) for tr in train}, screen_rows_per_recording=512)
    for key, value in data.items():
        data[key] = np.stack(value) if 'gram' in key else np.concatenate(value)
    tmp = dest/'prepared.tmp.npz'
    np.savez_compressed(tmp, **data)
    tmp.replace(dest/'prepared.npz')
    write_json(dest/'training.json', meta)


def native_search(held, form, seed, epochs=50):
    from epde.interface.interface import EpdeSearch
    from epde.interface.prepared_tokens import CacheStoredTokens
    from epde.operators.common import plain_fit

    dest = OUT/held
    with np.load(dest/'prepared.npz') as z:
        x, y = z[form+'_X'], z[form+'_y']
    np.random.seed(seed)
    random.seed(seed)
    search = EpdeSearch(use_solver=False, multiobjective_mode=True, engine='nsga2', system_rps_mode='plain',
                        discrepancy_metric='wape', instability_metric='dyad', device='cpu',
                        system_rps_params={'variable_order_seed':seed, 'target_axes':0},
                        verbose_params={'show_iter_idx':True, 'candidate_objectives':False})
    search.set_moeadd_params(population_size='auto', training_epochs=epochs, early_stopping_patience=None)
    # These are already physical derivative rows. EPDE must not difference adjacent robots
    # or differentiate the weak rows a second time. The constant dummy field has no physical RHS role.
    _, domain = search.createDomain((np.arange(len(y), dtype=float),), boundary_width=0, ID=0)
    _, tr = search.createTrajectory({'u':np.zeros_like(y)}, domain, cache_id=0, derivs=y[:,None])
    tokens = CacheStoredTokens('contact', NAMES, token_tensors=dict(zip(NAMES, x.T)),
                               params_ranges={'power':(1,1)}, meaningful=True)
    evaluated = {}
    native_versions = plain_fit._versions

    def support(eq):
        weights = np.asarray(eq.weights_final)
        terms = [term for i, term in enumerate(eq.structure) if i != eq.target_idx]
        active = sorted((term.structure[0].label, float(coef)) for coef, term in zip(weights[:-1], terms)
                        if abs(coef)>1e-12)
        if any(name not in NAMES for name, _ in active):
            return None
        return active, float(weights[-1])

    def archive(*args, **kwargs):
        versions = native_versions(*args, **kwargs)
        for eq in versions:
            item = support(eq)
            if item is None:
                continue
            terms, intercept = item
            key = (tuple(name for name, _ in terms), abs(intercept)>1e-12)
            if key not in evaluated:
                evaluated[key] = dict(terms=dict(terms), intercept=intercept, calls=0,
                                      objectives=[float(eq.fitness_value), float(eq.coefficients_stability)])
            evaluated[key]['calls'] += 1
        return versions

    with patch.object(plain_fit, '_versions', archive):
        search.fit(data=[tr], max_deriv_order=(1,), data_fun_pow=1, deriv_fun_pow=1,
                   equation_terms_max_number=7, equation_factors_max_number=1, additional_tokens=[tokens])
    front = []
    for system in search.equations(only_print=False)[0]:
        eq = system.vals['u']
        item = support(eq)
        if item is not None and np.isfinite(system.obj_fun).all():
            front.append(dict(terms=dict(item[0]), intercept=item[1], objectives=list(map(float, system.obj_fun))))
    actual = len(search.optimizer._pareto_history)
    if actual != epochs or not front or not evaluated:
        raise RuntimeError(f'Incomplete search: epochs={actual}/{epochs}, front={len(front)}, archive={len(evaluated)}')
    write_json(dest/f'{form}_{seed}.json', dict(form=form, seed=seed, epochs=actual,
               population=int(search.optimizer._mu), screen_rows=len(y), native_front=front,
               evaluated=list(evaluated.values())))
