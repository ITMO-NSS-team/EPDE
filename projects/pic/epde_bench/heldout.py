"""Train-only EPDE discovery followed by a chronological ODE forecast.

This is separate from fitted-record reconstruction. Raw data are split before
any differentiation or search; supplied full-record derivatives are rejected.
Candidate orientation and selection use training residuals only. The final
training state initializes integration; future dependent-variable observations
are accessed only to compute scores after the equation and forecast are frozen.
"""
import copy
import hashlib
import json
import time
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from . import env
from .config import load_config, resolve_for_problem
from .datasets import load
from .reconstruction import (_DERIV_RE, _explicit_form, SimulationResult,
                             equation_fields, parse_equation, r2_score, simulate_ode)
from .runner import build_search, pareto_front
from .paths import REPO_ROOT



def discover(train, search_cfg):
    """Own the search lifecycle even when fitting raises before returning it."""
    search, trajectory, families = build_search(train, search_cfg)
    try:
        started = time.perf_counter()
        search.fit(data=[trajectory], additional_tokens=families)
        fit_seconds = time.perf_counter() - started
        texts, objectives = pareto_front(search)
        return search, texts, objectives, fit_seconds
    except BaseException:
        search.close()
        raise


def chronological_split(problem, train_fraction=.7):
    """Copy the leading raw samples; no precomputed derivative may cross split."""
    if problem.dim != 0 or len(problem.variables) != 1:
        raise ValueError('held-out protocol currently supports scalar ODEs only')
    if problem.derivs is not None:
        raise ValueError('full-record supplied derivatives have unknown boundary provenance')
    if problem.extra_tokens is not None or problem.named_arrays:
        raise ValueError('held-out protocol requires analytic time tokens, not supplied extra fields')
    t = np.asarray(problem.grids[0], dtype=float)
    if not 0 < train_fraction < 1 or not np.all(np.diff(t) > 0):
        raise ValueError('train_fraction must be in (0,1) and time strictly increasing')
    split = int(len(t) * train_fraction)
    if split < 8 or len(t) - split < 2:
        raise ValueError('split requires at least 8 training and 2 test samples')
    train = replace(problem, grids=(t[:split].copy(),),
                    data={v: np.asarray(a)[:split].copy() for v, a in problem.data.items()},
                    truth=None, truth_alternatives=[], meta=copy.deepcopy(problem.meta))
    return train, split


def _factor_text(factor):
    name, params = factor
    return name + '{' + ', '.join(f'{k}: {v}' for k, v in sorted(params)) + '}'


def explicit_system(problem, system):
    """Algebraically isolate the highest pure derivative, preserving fitted coefficients."""
    if len(system) != 1 or len(problem.variables) != 1:
        return None
    try:
        terms, constant, target, target_coef = parse_equation(system[0])
        residual = terms + [(-target_coef, target)]
        pure = []
        highest = 0
        for i, (coef, factors) in enumerate(residual):
            for name, params in factors:
                m = _DERIV_RE.match(name)
                if m:
                    if m.group(2) != problem.variables[0] or int(m.group(3)) != 0:
                        return None
                    order = int(m.group(1) or 1)
                    highest = max(highest, order)
                    if len(factors) == 1 and dict(params).get('power', 1) == 1 and abs(coef) > 1e-12:
                        pure.append((order, i))
        targets = [i for order, i in pure if order == highest]
        if highest == 0 or len(targets) != 1:
            return None
        idx = targets[0]
        coef, factors = residual[idx]
        rhs = [f'{-c/coef:.17g} * ' + ' * '.join(map(_factor_text, fs))
               for i, (c, fs) in enumerate(residual) if i != idx and c != 0]
        rhs.append(f'{-constant/coef:.17g}')
        oriented = [' + '.join(rhs) + ' = ' + _factor_text(factors[0])]
        return oriented if _explicit_form(problem, oriented) is not None else None
    except (ValueError, KeyError, OverflowError):
        return None


def evaluate_candidates(train, candidates, search_cfg):
    """Select minimal normalized training derivative MSE; stable index breaks ties."""
    evaluations = []
    for i, system in enumerate(candidates):
        oriented = explicit_system(train, system)
        entry = {'index': i, 'oriented': oriented, 'train_relative_mse': None}
        if oriented:
            try:
                observed, predicted = equation_fields(train, oriented[0], search_cfg)
                denominator = max(float(np.mean(observed**2)), np.finfo(float).eps)
                mse = float(np.mean((observed - predicted)**2) / denominator)
                if np.isfinite(mse):
                    entry['train_relative_mse'] = mse
            except (KeyError, ValueError, OverflowError):
                pass
        evaluations.append(entry)
    eligible = [x for x in evaluations if x['train_relative_mse'] is not None]
    selected = min(eligible, key=lambda x: (x['train_relative_mse'], x['index'])) if eligible else None
    return {'candidates': evaluations, 'selected_index': selected['index'] if selected else None,
            'selected': selected['oriented'] if selected else None,
            'selection_rule': 'minimum relative derivative MSE on training interior; ties by Pareto index'}


def forecast_ode(train, system, search_cfg, test_times):
    """Integrate frozen equation from last training state, with no future values input."""
    from scipy.integrate import solve_ivp
    forms = _explicit_form(train, system)
    if forms is None:
        raise ValueError('selected equation is not an explicit ODE')
    # The entire pipeline sees only the training Problem. Boundary derivatives
    # are obtained from a one-sided local polynomial of raw TRAINING values,
    # avoiding numerical differentiation's unreliable endpoint convention.
    t_train = np.asarray(train.grids[0], dtype=float)
    ts = np.asarray(test_times, dtype=float)
    if len(ts) < 2 or not np.all(np.diff(ts) > 0) or ts[0] <= t_train[-1]:
        raise ValueError('test times must follow the last training sample')
    var = train.variables[0]
    order, terms, constant, target_coef = forms[var]
    if order > 3:
        raise ValueError('held-out boundary estimation supports ODE orders up to three')
    y0 = [float(train.data[var][-1])]
    n_fit = min(9, len(t_train))
    polynomial = np.polynomial.Polynomial.fit(t_train[-n_fit:] - t_train[-1],
                                              train.data[var][-n_fit:], min(4, n_fit - 1)).convert()
    for k in range(1, order):
        y0.append(float(polynomial.deriv(k)(0)))

    def factor_value(name, pairs, t, state):
        params = dict(pairs)
        power = params.get('power', 1)
        if name == var:
            value = state[0]
        elif name in ('sin', 'cos'):
            value = getattr(np, name)(params.get('freq', 1) * t)
        elif name == 'x' or name.startswith('x_'):
            value = t
        else:
            m = _DERIV_RE.match(name)
            if not m or m.group(2) != var or int(m.group(3)) != 0:
                raise ValueError(f'unsupported forecast factor {name}')
            k = int(m.group(1) or 1)
            if k >= order:
                raise ValueError(f'non-explicit forecast factor {name}')
            value = state[k]
        return value ** power

    def rhs(t, state):
        out = np.empty(order)
        out[:-1] = state[1:]
        value = constant
        for coef, factors in terms:
            product = coef
            for name, params in factors:
                product *= factor_value(name, params, t, state)
            value += product
        out[-1] = value / target_coef
        return out

    # Validate at boundary so unsupported factors fail without touching test.
    rhs(float(t_train[-1]), np.asarray(y0))
    solution = solve_ivp(rhs, (t_train[-1], ts[-1]), y0, t_eval=ts,
                         method='LSODA', rtol=1e-8, atol=1e-10)
    predicted = np.full(ts.shape, np.nan)
    count = len(solution.t)
    if count:
        predicted[:count] = solution.y[0]
    success = bool(solution.success and count == len(ts) and np.isfinite(predicted).all())
    return SimulationResult(ts, {var: predicted}, {}, success, str(solution.message),
                            count / len(ts), 'chronological held-out forecast; training-only initial state')


def _finite(value):
    return float(value) if np.isfinite(value) else None


def _json_safe(value):
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (float, np.floating)):
        return _finite(value)
    if isinstance(value, np.integer):
        return int(value)
    return value


def _scores(observed, predicted, complete):
    result = {}
    for var, obs in observed.items():
        pred = predicted[var]
        result[var] = ({'rmse': _finite(np.sqrt(np.mean((obs-pred)**2))),
                        'mae': _finite(np.mean(np.abs(obs-pred))),
                        'r2': _finite(r2_score(obs, pred))} if complete else
                       {'rmse': None, 'mae': None, 'r2': None})
    return result


def run_heldout(dataset='ode', train_fraction=.7, seed=0, overrides=None, output_dir=None):
    """Execute real EPDE, freeze a training-selected ODE, and save forecast evidence.

    Optional output_dir receives record.json, trajectory.png, trajectory.npz.
    Returns a JSON-compatible record; no truth law participates in selection.
    """
    started = time.perf_counter()
    cfg = load_config(dataset, overrides=overrides)
    if cfg.get('method', 'epde') != 'epde':
        raise ValueError('held-out discovery requires the EPDE method')
    env.seed_everything(seed)
    problem = load(dataset, **cfg.get('loader', {}))
    train, split = chronological_split(problem, train_fraction)
    search_cfg = resolve_for_problem(cfg, train)
    record = {'protocol': 'chronological EPDE discovery and frozen-equation forecast',
              'dataset': dataset, 'seed': seed, 'started': datetime.now(timezone.utc).isoformat(),
              'config': cfg, 'search_config': search_cfg,
              'train_fraction_requested': train_fraction,
              'split_index': split, 'total_points': problem.shape[0],
              'train': {'n_points': split, 'time_range': [float(train.grids[0][0]), float(train.grids[0][-1])],
                        'data_sha256': hashlib.sha256(np.asarray(train.data[train.variables[0]]).tobytes()).hexdigest()},
              'test': {'n_points': problem.shape[0] - split,
                       'time_range': [float(problem.grids[0][split]), float(problem.grids[0][-1])]},
              'leakage_controls': ['raw chronological split before preprocessing/build_search',
                                  'training-only EPDE coefficients and candidate selection',
                                  'frozen equation before future observations accessed for scoring',
                                  'initial value and one-sided local-polynomial derivatives from last training samples',
                                  'no test tuning; truth equation excluded from training Problem'],
              'environment': env.environment_info(REPO_ROOT)}
    search = None
    train_sim = None
    forecast = None
    try:
        search, texts, objectives, fit_seconds = discover(train, search_cfg)
        record.update(front=texts, objectives=objectives, fit_seconds=fit_seconds)
        record.update(evaluate_candidates(train, texts, search_cfg))
        system = record['selected']
        if system is None:
            record.update(status='no_explicit_candidate', reason='No integratable scalar ODE in Pareto front')
            record['test'].update(integration_success=False, coverage=0.0, metrics={})
        else:
            observed, predicted = equation_fields(train, system[0], search_cfg)
            record['train']['derivative_r2'] = _finite(r2_score(observed, predicted))
            train_sim = simulate_ode(train, system, search_cfg)
            if train_sim:
                record['train'].update(integration_success=train_sim.success, coverage=train_sim.coverage,
                    metrics=_scores(train_sim.observed, train_sim.simulated, train_sim.success))
            forecast = forecast_ode(train, system, search_cfg, problem.grids[0][split:])
            # This is the first use of dependent-variable test observations.
            future = {v: np.asarray(a)[split:] for v, a in problem.data.items()}
            record['test'].update(integration_success=forecast.success, coverage=forecast.coverage,
                                 message=forecast.message,
                                 metrics=_scores(future, forecast.simulated, forecast.success))
            record['status'] = 'ok' if forecast.success else 'integration_failed'
    except Exception as exc:
        record.update(status='error', error=repr(exc))
        record['test'].update(integration_success=False, coverage=0.0, metrics={})
    finally:
        if search is not None:
            search.close()
    record['total_seconds'] = time.perf_counter() - started
    record = _json_safe(record)
    if output_dir is not None:
        destination = Path(output_dir).resolve()
        destination.mkdir(parents=True, exist_ok=True)
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
        var = train.variables[0]
        axes[0].plot(train.grids[0], train.data[var], 'k', label='training observations')
        if train_sim:
            axes[0].plot(train_sim.t, train_sim.simulated[var], 'C0--', label='training reconstruction')
        axes[1].plot(problem.grids[0][split:], problem.data[var][split:], 'k', label='held-out observations')
        if forecast:
            axes[1].plot(forecast.t, forecast.simulated[var], 'C1--', label='frozen-equation forecast')
        for ax, title in zip(axes, [f'First {split/problem.shape[0]:.0%}: training', f'Last {1-split/problem.shape[0]:.0%}: held-out forecast']):
            ax.set(title=title, xlabel='time', ylabel=var)
            ax.legend(fontsize=8)
            ax.grid(alpha=.2)
        fig.suptitle(f'{dataset}, seed {seed}: {record["status"]}')
        fig.tight_layout()
        fig.savefig(destination / 'trajectory.png', dpi=160)
        plt.close(fig)
        arrays = {'train_t': train.grids[0], 'train_observed': train.data[var],
                  'test_t': problem.grids[0][split:], 'test_observed': problem.data[var][split:]}
        if train_sim:
            arrays.update(train_integration_t=train_sim.t, train_predicted=train_sim.simulated[var])
        if forecast:
            arrays['test_predicted'] = forecast.simulated[var]
        np.savez_compressed(destination / 'trajectory.npz', **arrays)
        record['artifacts'] = {name: str(destination / name) for name in ['record.json', 'trajectory.png', 'trajectory.npz']}
        with (destination / 'record.json').open('w', encoding='utf-8') as f:
            json.dump(record, f, indent=2, allow_nan=False)
    return record
