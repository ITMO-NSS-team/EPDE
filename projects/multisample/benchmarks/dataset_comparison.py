"""Paired whole-trajectory tests. Run with PYTHONHASHSEED=0 from the repository root."""
import argparse
from contextlib import redirect_stdout
import json
import os
from pathlib import Path
import random
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'projects/pic/experiments/complementary_windows_2026-11-24'))
from cpu_limit import cap_worker

cap_worker()
import numpy as np
from scipy.integrate import solve_ivp
from scipy.io import loadmat
import torch

import epde
from epde.interface.results import _equation

PIC = ROOT / 'projects/pic/data'
ROBOTS = ROOT / 'projects/robot/results/collision_transfer/preprocessed'
TARGET = 'du/dx0'
OUTPUT = HERE / 'dataset_runs.json'


def field(t, x, u, name, derivatives=None, orders=None):
    return epde.TrajectoryData({'t': t, 'x': x}, {'u': u}, trajectory_id=name,
                               derivatives={} if derivatives is None else {'u': derivatives},
                               derivative_orders=orders)


def reaction_diffusion():
    def generate(nt, nx, modes, name):
        t, x = np.linspace(0, 2, nt), np.linspace(-np.pi, np.pi, nx)
        T, X = np.meshgrid(t, x, indexing='ij')
        u, ut, ux, uxx = (np.zeros_like(T) for _ in range(4))
        for k, amplitude in modes:
            decay = amplitude * np.exp(-(.2 + .1*k*k)*T)
            wave = decay * np.sin(k*X)
            u += wave
            ut -= (.2 + .1*k*k)*wave
            ux += k*decay*np.cos(k*X)
            uxx -= k*k*wave
        derivatives = np.column_stack([v.ravel() for v in (ut, ux, uxx)])
        record = field(t, x, u, name, derivatives, (1, 2))
        return record, {'u': u, TARGET: ut, 'du/dx1': ux, 'd^2u/dx1^2': uxx}
    train = [generate(31, 41, [(1, 1.)], 'k1')[0], generate(23, 53, [(2, .7)], 'k2')[0]]
    _, test = generate(29, 47, [(1, .8), (3, .4)], 'mixed')
    return train, test, 'mixed', {'u': -.2, 'd^2u/dx1^2': .1}


def kdv():
    m = loadmat(PIC / 'kdv/kdv_sindy.mat')
    t, x = m['t'].ravel(), m['x'].ravel()
    # Keep native spacing for the third spatial derivative in both records.
    keep = np.flatnonzero((x >= -15) & (x <= 15))
    first = field(t[:41], x[keep], m['usol'].real.T[:41, keep], 'kdv_sindy')
    u = np.load(PIC / 'kdv/data_kdv_homogen.npy')
    t, x = np.linspace(0, 1, u.shape[0]), np.linspace(-3, 3, u.shape[1])
    second = field(t[::3], x[::3], u[::3, ::3], 'kdv_homogen')
    t, x = np.linspace(0, 1.4, 39), np.linspace(-4, 6, 151)
    T, X = np.meshgrid(t, x, indexing='ij')
    k = .91
    z = np.tanh(k * (X - 4*k*k*T + .5))
    u = 2*k*k*(1-z*z)
    ux = -2*k*u*z
    test = {'u': u, 'du/dx1': ux, 'd^2u/dx1^2': 4*k*k*u-3*u*u,
            'd^3u/dx1^3': (4*k*k-6*u)*ux, TARGET: -4*k*k*ux}
    np.testing.assert_allclose(-6*u*ux-test['d^3u/dx1^3'], test[TARGET], atol=1e-13)
    return [first, second], test, 'soliton_k0.91', {'du/dx1*u': -6., 'd^3u/dx1^3': -1.}


def forced_profile(frequencies, name):
    """Periodic forced KdV using Fourier derivatives and integrating-factor RK4."""
    t, x = np.linspace(0, 1, 51), np.linspace(-np.pi, np.pi, 96, endpoint=False)
    k = np.fft.fftfreq(len(x))*len(x)
    keep = abs(k) < len(x)/3
    drive = np.fft.fft(np.sin(x))
    initial = .35*np.sin(frequencies[0]*x+.2) + .18*np.cos(frequencies[1]*x-.4)

    def integrate(substeps):
        h = (t[1]-t[0])/substeps
        half = np.exp(.5j*h*k**3)
        full = half**2
        state = np.fft.fft(initial)*keep
        def nonlinear(state, now):
            u = np.fft.ifft(state).real
            return keep*(-3j*k*np.fft.fft(u*u) + np.cos(now)*drive)
        frames = [initial]
        for start in t[:-1]:
            for j in range(substeps):
                now = start+j*h
                a = nonlinear(state, now)
                b = nonlinear(half*(state+h*a/2), now+h/2)
                c = nonlinear(half*state+h*b/2, now+h/2)
                d = nonlinear(full*state+h*half*c, now+h)
                state = full*state+h*(full*a+2*half*(b+c)+d)/6
            frames.append(np.fft.ifft(state).real)
        return np.stack(frames)

    coarse = integrate(16)
    for steps in (32, 64, 128, 256):
        u = integrate(steps)
        gradient_diff = (np.fft.fft(u-coarse)*k**3)
        if (np.max(abs(u-coarse)) < 2e-6 and
                np.linalg.norm(gradient_diff) < .001*np.linalg.norm(np.fft.fft(u)*k**3)):
            break
        coarse = u
    else:
        raise ValueError('Forced KdV time-step refinement failed for field or third derivative')
    T, X = np.meshgrid(t, x, indexing='ij')
    ux, uxxx = [np.fft.ifft(np.fft.fft(u)*(1j*k)**order).real for order in (1, 3)]
    ut = np.fft.ifft(keep*np.fft.fft(-6*u*ux-uxxx+np.cos(T)*np.sin(X))).real
    return field(t, x, u, name), {'u': u, 'du/dx1': ux, 'd^3u/dx1^3': uxxx, TARGET: ut, 't': T, 'x': X}


def forced_kdv():
    archived = np.loadtxt(PIC/'kdv/data.csv', delimiter=',').T
    t, x = [np.linspace(0, 1, n) for n in archived.shape]
    first = field(t, x, archived, 'pic_kdv_cossin')
    second, _ = forced_profile((1, 2), 'new_forced_12')
    _, test = forced_profile((2, 3), 'new_forced_23')
    return [first, second], test, 'new_forced_23', {'du/dx1*u': -6., 'd^3u/dx1^3': -1., 'cos*sin': 1.}


def time_record(name, t, y, inputs):
    record = epde.TrajectoryData({'t': t}, {'u': y}, external_inputs=inputs, trajectory_id=name)
    test = {'t': t[:-1], 'u': y[:-1], TARGET: np.diff(y)/np.diff(t),
            **{k: v[:-1] for k, v in inputs.items()}}
    return record, test


def duffing():
    with np.load(PIC / 'duffing/duffing.npz') as archive:
        p = {k: float(archive[k]) for k in ('alpha', 'beta', 'delta', 'gamma', 'omega')}
        def make(name, t, x, v):
            return time_record(name, t, v, {'disp': x, 'disp2': x*x, 'disp3': x**3})
        _, test = make('pic_duffing', *(archive[k][::2].astype(float) for k in ('t', 'x', 'v')))
    def rhs(t, y):
        x, v = y
        return v, -p['alpha']*x-p['beta']*x**3-p['delta']*v+p['gamma']*np.cos(p['omega']*t)
    train = []
    for name, initial, n in [('ic_a', (.8, -.2), 401), ('ic_b', (-.7, .5), 451)]:
        t = np.linspace(0, 18, n)
        sol = solve_ivp(rhs, (t[0], t[-1]), initial, t_eval=t, rtol=1e-10, atol=1e-12)
        assert sol.success
        train.append(make(name, t, *sol.y)[0])
    return train, test, 'pic_duffing', {'u': -p['delta'], 'disp': -p['alpha'],
                                        'disp3': -p['beta'], 'cos': p['gamma']}


def robot(fold):
    paths = sorted(ROBOTS.glob('N*PWM10*.npz'))
    assert len(paths) == 3
    ids = []
    for path in paths:
        with np.load(path) as a:
            ids.append(set(a['ids'].tolist()))
    robot_id = min(set.intersection(*ids))
    samples = []
    for path in paths:
        with np.load(path) as a:
            j = a['ids'].tolist().index(robot_id)
            x = a['raw_pos'][::5, j, 0].astype(float)
            phi = a['phi'][::5, j].astype(float)
            t = a['frame'][::5].astype(float)/float(a['fps'])
        samples.append(time_record(path.stem, t[1:], x[1:] - x[1],
                                   {'heading_cos': np.cos(phi[1:]), 'heading_sin': np.sin(phi[1:]),
                                    'previous_vx': np.diff(x)/np.diff(t)}))
    return [v[0] for i, v in enumerate(samples) if i != fold], samples[fold][1], paths[fold].stem, {}


def evaluate(eq, test):
    coefs, frequencies = {}, {}
    prediction = np.full_like(test[TARGET], eq['intercept'])
    for term in eq['terms']:
        c = term['coefficient']
        if c == 0:
            continue
        values, labels = 1., []
        for factor in term['factors']:
            label, params = factor['label'], factor['parameters']
            power = int(params['power'])
            labels.append(label if power == 1 else f'{label}^{power}')
            if label in ('cos', 'sin'):
                frequencies[label] = float(params['freq'])
                axis = ('t', 'x')[int(params['dim'])]
                values = values * getattr(np, label)(params['freq']*test[axis])**power
            else:
                values = values * test[label]**power
        coefs['*'.join(sorted(labels))] = float(c)
        prediction += c*values
    return np.linalg.norm(prediction-test[TARGET])/np.linalg.norm(test[TARGET]), coefs, frequencies


def configuration(case, mode):
    form = 'weak' if case == 'forced_kdv' else ('forward' if case in ('robot', 'duffing') else 'strong')
    return dict(engine='nsga2', system_rps_mode='plain', multisample_mode=mode,
                  multisample_weighting='samples', use_solver=False, device='cpu',
                  system_rps_params={'target_axes': 0, 'plain_target': True, 'fill_probability': 0.},
                  evaluation_form=form, default_preprocessor_type='FD',
                  boundary_width=4 if case == 'kdv' else 2, data_fun_pow=1, deriv_fun_pow=1,
                  equation_terms_max_number=5, equation_factors_max_number=2 if case in ('kdv', 'forced_kdv') else 1,
                  population_size=8, training_epochs=3, early_stopping_patience=None,
                  verbose_params={'show_iter_idx': False})


def inputs(case, train):
    tokens = [epde.CacheStoredTokens(key, [key], params_ranges={'power': (1, 1)},
              meaningful=True, available_at_origin=True) for key in train[0].external_inputs]
    if case in ('duffing', 'forced_kdv'):
        tokens.append(epde.TrigonometricTokens(meaningful=True, allow_products=case == 'forced_kdv', fit_frequencies=True))
    orders = (1,) if case in ('robot', 'duffing') else (1, 3 if case in ('kdv', 'forced_kdv') else 2)
    return dict(additional_tokens=tokens, max_deriv_order=orders)


def fixed_support(case, mode, *, records=None, noise=0., noise_seed=None):
    """Fit exactly the registered physical terms on native rows to isolate coefficient fitting."""
    import epde.globals as g
    from epde.operators.common.frequency_fit import fit_support
    from epde.operators.common.multisample import joint_subset_table
    from epde.operators.common.subset_selection import subset_table
    from epde.structure.evaluation_rows import slot_of, term_columns
    from epde.structure.main_structures import Equation, Term

    train, test, held, truth = GENERATORS[case]() if records is None else records
    if noise:
        rng = np.random.default_rng(noise_seed)
        train = [epde.TrajectoryData(d.coordinates, {'u': d.fields['u'] + noise*d.fields['u'].std()
                   * rng.standard_normal(d.fields['u'].shape)}, trajectory_id=d.trajectory_id) for d in train]
    if mode == 'single':
        train = train[:1]
    config = configuration(case, 'mean' if mode == 'single' else mode)
    with open(os.devnull, 'w') as quiet, redirect_stdout(quiet), epde.EpdeMultisample(**config) as s:
        s.create_pool(train, **inputs(case, train))
        s._prepare_evaluation_rows()
        terms = []
        for name in [TARGET, *truth]:
            factors = []
            for label in name.split('*'):
                params = dict(freq=0., dim=int(label == 'sin')) if label in ('cos', 'sin') else {}
                factors.append(s.pool.create(label=label, power=1, **params)[1])
            terms.append(Term(s.pool, passed_term=factors, max_factors_in_term=len(factors)))
        eq = Equation(s.pool, terms, 'u', {'max_terms_number': {'value': len(terms)}, 'max_factors_in_term': {'value': 2}})
        eq.target_idx = 0
        if case in ('duffing', 'forced_kdv'):
            eq = fit_support(eq, range(1, len(terms)), False, g.samples_manager.evaluation_rows)
            if eq is None:
                return dict(case=case, mode=mode, held_out=held, training_noise=noise, noise_seed=noise_seed,
                            identifiable=False)
        slot = slot_of(eq)
        ys, *features = [term_columns(t, slot) for t in eq.structure]
        weights = g.samples_manager.row_weights(slot)
        blocks = [(k, np.column_stack([f[k].ravel() for f in features]), ys[k].ravel(), weights[k].ravel()) for k in ys]
        ranks = [int(np.linalg.matrix_rank(x*np.sqrt(w[:, None]))) for _, x, _, w in blocks]
        mask = [(1 << len(features)) - 1]
        out = dict(case=case, mode=mode, held_out=held, record_ranks=ranks, columns=len(features),
                   train_ids=[d.trajectory_id for d in train], training_noise=noise, noise_seed=noise_seed)
        if mode == 'joint':
            _, _, _, coefs, rank = joint_subset_table(blocks, discrepancy='wape', subsets=mask, weighting='samples')
            if not rank[0]:
                return dict(out, identifiable=False)
            c = coefs[0]
        else:
            if min(ranks) < len(features):
                return dict(out, identifiable=False)
            individual = [subset_table(x, y, w, subsets=mask, return_coefs=True)[-1][0] for _, x, y, w in blocks]
            out['per_record_coefficients'] = {key: dict(zip(truth, v[:-1])) for (key, *_), v in zip(blocks, individual)}
            c = np.mean(individual, axis=0)
        from epde.interface.results import _term
        model = dict(terms=[dict(factors=_term(t), coefficient=float(v)) for t, v in zip(eq.structure[1:], c[:-1])],
                     intercept=float(c[-1]))
        error, coeff, freq = evaluate(model, test)
        return dict(out, identifiable=True, heldout_relative_l2=float(error), coefficients=coeff, frequencies=freq)


def run(case, mode, seed, fold):
    train, test, held, truth = robot(fold) if case == 'robot' else GENERATORS[case]()
    assert held not in {record.trajectory_id for record in train}
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    start = time.monotonic()
    with open(os.devnull, 'w') as quiet, redirect_stdout(quiet), epde.EpdeMultisample(**configuration(case, mode)) as s:
        s.fit(samples=train, **inputs(case, train))
        front = s.equations_by_variable(only_print=False)['u']
        ranked = sorted(front, key=lambda eq: (eq.fitness_value, len(eq.structure), eq.text_form))
        models = []
        for eq in ranked:
            if not np.isfinite(eq.fitness_value):
                continue
            item = _equation(eq)
            assert len(item['target']) == 1 and item['target'][0]['label'] == TARGET
            error, coefs, freq = evaluate(item, test)
            assert np.isfinite(error)
            models.append(dict(train_discrepancy=float(eq.fitness_value), heldout_relative_l2=float(error),
                               coefficients=coefs, frequencies=freq, intercept=item['intercept'], equation=item['text'],
                               exact_support=(set(truth) == set(coefs) and item['intercept'] == 0) if truth else None))
    assert models
    return dict(case=case, mode=mode, seed=seed, fold=fold, train_ids=[d.trajectory_id for d in train],
                train_shapes=[list(d.fields['u'].shape) for d in train], held_out=held, test_shape=list(test[TARGET].shape),
                truth=truth, front=models, wall_seconds=round(time.monotonic()-start, 2),
                persistence_relative_l2=(float(np.linalg.norm(test['previous_vx']-test[TARGET])
                                               / np.linalg.norm(test[TARGET])) if case == 'robot' else None))


GENERATORS = {'reaction_diffusion': reaction_diffusion, 'kdv': kdv, 'duffing': duffing}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cases', nargs='+', default=['reaction_diffusion', 'kdv', 'robot', 'duffing'])
    parser.add_argument('--seeds', nargs='+', type=int, default=[0, 1])
    parser.add_argument('--fixed', action='store_true', help='Fit the supplied physical support on the synthetic datasets.')
    parser.add_argument('--forced-kdv', action='store_true', help='Compare dispersion and fitted frequencies on forced KdV.')
    args = parser.parse_args()
    if os.environ.get('PYTHONHASHSEED') != '0':
        parser.error('Set PYTHONHASHSEED=0 before starting Python for repeatable symbolic search.')
    if args.fixed:
        fixed = []
        for case in args.cases:
            if case != 'robot':
                for mode in ('mean', 'joint'):
                    print('Fixed support', case, mode, flush=True)
                    fixed.append(fixed_support(case, mode))
        (HERE / 'fixed_support.json').write_text(json.dumps(fixed, indent=2, allow_nan=False), encoding='utf-8')
    if args.forced_kdv:
        records = forced_kdv()
        estimates = []
        for noise_seed in (None, 2401, 2402, 2403):
            noise = 0. if noise_seed is None else .05
            for mode in ('single', 'mean', 'joint'):
                print('Forced KdV fixed support', noise, noise_seed, mode, flush=True)
                estimates.append(fixed_support('forced_kdv', mode, records=records, noise=noise, noise_seed=noise_seed))
                (HERE/'forced_kdv_support.json').write_text(json.dumps(estimates, indent=2, allow_nan=False), encoding='utf-8')
    if args.fixed or args.forced_kdv:
        sys.exit()
    saved = json.loads(OUTPUT.read_text()) if OUTPUT.exists() else []
    done = {(r['case'], r['mode'], r['seed'], r['fold']) for r in saved}
    for case in args.cases:
        for fold in (range(3) if case == 'robot' else [0]):
            for seed in args.seeds:
                for mode in ('mean', 'joint'):
                    if (case, mode, seed, fold) in done:
                        continue
                    print('Starting', case, fold, seed, mode, flush=True)
                    result = run(case, mode, seed, fold)
                    saved.append(result)
                    temp = OUTPUT.with_suffix('.tmp')
                    temp.write_text(json.dumps(saved, indent=2, allow_nan=False), encoding='utf-8')
                    temp.replace(OUTPUT)
                    print('Finished', result['wall_seconds'], result['front'][0]['heldout_relative_l2'], flush=True)
