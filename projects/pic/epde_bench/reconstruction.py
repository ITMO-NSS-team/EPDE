"""What a discovered equation reproduces: side-by-side comparisons with the data.

Two views, both computed from the equation's text form and the same
preprocessing the search used:

* :func:`equation_fields` -- the left-hand side of the equation evaluated on the
  data next to the right-hand side the equation predicts from its terms. Works
  for every problem class, including systems and PDEs.
* :func:`simulate_ode` -- for ODEs and ODE systems written as
  ``d^k u/dt^k = f(t, u, u', ...)``: the equation is integrated from the
  initial state of the data and the trajectory is compared with the record.
  This is the stronger check: errors accumulate instead of being re-anchored
  to the data at every point.

:func:`plot_reconstruction` draws whichever views apply.
"""

import re
from dataclasses import dataclass

import numpy as np

from .library import _interior
from .metrics import _parse_factor, _split_sum, _strip_system_prefix
from .preprocessing import derivative_orders, prepare_data
from .truthcheck import _factor_arrays, _term_array

_DERIV_RE = re.compile(r'^d(?:\^(\d+))?([A-Za-z_][A-Za-z0-9_]*)/dx(\d+)(?:\^\d+)?$')


def parse_equation(eq_text):
    """``'c1 * f{..} * g{..} + ... + c0 = target{..}'`` ->
    ``(terms, constant, target, target_coef)``; every term is
    ``(coef, [factor, ...])`` with factors as parsed by the structural metrics."""
    # systems are printed with a brace drawn by '/', '|' and '\\' before each equation
    lhs_text, rhs_text = _strip_system_prefix(eq_text).split('=', 1)

    def term(text):
        coef, factors = 1.0, []
        for piece in (p.strip() for p in text.split('*')):
            factor = _parse_factor(piece) if piece else None
            if factor is not None:
                factors.append(factor)
            elif piece:
                coef *= float(piece)
        return coef, factors

    terms, constant = [], 0.0
    # split on '+' between terms, not inside '{...}' or exponents like 1e+05
    for chunk in _split_sum(lhs_text):
        coef, factors = term(chunk)
        if factors:
            terms.append((coef, factors))
        else:
            constant += coef
    target_coef, target = term(rhs_text)
    return terms, constant, target, target_coef


def equation_fields(problem, eq_text, search, data=None):
    """Observed left-hand side and the value the equation predicts for it,
    both on the interior used by the search. Returns ``(observed, predicted)``
    with the interior shape."""
    data = problem.data if data is None else data
    method = search.get('_reconstruction_differentiation', 'epde')
    if method == 'epde':
        arrays = _factor_arrays(problem, data, search)
    else:
        from .library import _blocks
        blocks, _ = _blocks(problem, data, search, method=method)
        arrays = {_parse_factor(name): array for name, array, *_ in blocks}
    inner = _interior(problem, search)
    terms, constant, target, target_coef = parse_equation(eq_text)
    observed = target_coef * _term_array(target, arrays)
    predicted = np.full(problem.shape, constant, dtype=float)
    for coef, factors in terms:
        predicted = predicted + coef * _term_array(factors, arrays)
    return observed[inner], predicted[inner]


def r2_score(observed, predicted):
    observed, predicted = np.ravel(observed), np.ravel(predicted)
    var = float(np.var(observed))
    return float('nan') if var == 0 else 1.0 - float(np.mean((observed - predicted) ** 2)) / var


def _explicit_form(problem, system):
    """Each equation as ``d^k v/dt^k = rhs`` or ``None`` if the system is not
    an explicit ODE system the integrator can handle."""
    forms = {}
    for eq in system:
        terms, constant, target, target_coef = parse_equation(eq)
        if len(target) != 1 or abs(target_coef) < 1e-12:
            return None
        name, params = target[0]
        match = _DERIV_RE.match(name)
        params = dict(params)
        if match is None or int(match.group(3)) != 0 or params.get('power', 1.0) != 1.0:
            return None
        order, var = int(match.group(1) or 1), match.group(2)
        if var not in problem.variables or var in forms:
            return None
        forms[var] = (order, terms, constant, target_coef)
    if set(forms) != set(problem.variables):
        return None
    for var, (order, terms, *_) in forms.items():
        for _, factors in terms:
            for name, _params in factors:
                m = _DERIV_RE.match(name)
                if m and m.group(2) in forms and int(m.group(1) or 1) >= forms[m.group(2)][0]:
                    return None          # the right-hand side needs the highest derivative
    return forms


@dataclass
class SimulationResult:
    t: np.ndarray
    simulated: dict
    observed: dict
    success: bool
    message: str
    coverage: float
    validation: str = 'reconstruction of the fitted record'

    def __iter__(self):
        return iter((self.t, self.simulated, self.observed))


def simulate_ode(problem, system, search, data=None, horizon=None, start_fraction=0.0):
    """Integrate an explicit ODE (system) from the first interior point of the
    data. Returns ``(t, {var: simulated}, {var: observed})`` or ``None`` when the
    equations are not of the explicit form ``d^k v/dt^k = f(t, v, v', ...)``.
    ``horizon`` limits the time span (in time units) after the start."""
    from scipy.integrate import solve_ivp
    if problem.dim != 0:
        return None
    forms = _explicit_form(problem, system)
    if forms is None:
        return None
    data = problem.data if data is None else data
    method = search.get('_reconstruction_differentiation', 'epde')
    if method == 'epde':
        fields, stacks = prepare_data(problem, data, search)
    else:
        from .library import _differentiate
        kwargs = search.get('preprocessing', {}).get('preprocessor_kwargs') or {}
        smoother = ({'window_length': int(kwargs['polynomial_window']),
                     'polyorder': int(kwargs.get('poly_order', 3))}
                    if 'polynomial_window' in kwargs else None)
        order = derivative_orders(search, 1)[0]
        fields = data
        stacks = {var: np.column_stack([_differentiate(values, problem.grids, 0, k, method, smoother)
                                        for k in range(1, order + 1)])
                  for var, values in data.items()}
    inner = _interior(problem, search)[0]
    t_all = np.asarray(problem.grids[0], dtype=float)
    if not 0 <= start_fraction < 1:
        raise ValueError('start_fraction must be in [0, 1)')
    interior_indices = np.arange(t_all.size)[inner]
    offset = int(len(interior_indices) * start_fraction)
    interior_indices = interior_indices[offset:]
    t = t_all[interior_indices]
    if horizon is not None:
        t = t[t <= t[0] + horizon]
    if t.size < 2:
        raise ValueError("integration needs at least two time points")
    i0 = int(interior_indices[0])
    max_order = derivative_orders(search, 1)[0]
    extras = {label: np.asarray(array, dtype=float) for label, (array, _) in problem.named_arrays.items()}
    layout, y0 = [], []                      # state: (var, k) for k < order
    for var, (order, *_rest) in forms.items():
        for k in range(order):
            layout.append((var, k))
            if k == 0:
                y0.append(float(fields[var][i0]))
            elif k <= max_order:
                y0.append(float(stacks[var][i0, k - 1]))
            else:
                return None
    index = {key: n for n, key in enumerate(layout)}

    def factor_value(name, params, time, state):
        params = dict(params)
        power = float(params.get('power', 1.0))
        if name in forms:
            return state[index[(name, 0)]] ** power
        m = _DERIV_RE.match(name)
        if m:
            key = (m.group(2), int(m.group(1) or 1))
            if key not in index:
                raise KeyError(name)
            return state[index[key]] ** power
        if name in ('sin', 'cos'):
            return getattr(np, name)(float(params.get('freq', 1.0)) * time) ** power
        if name == 'x' or re.match(r'^x_\d+$', name):
            return time ** power
        if name in extras:
            return np.interp(time, t_all, extras[name]) ** power
        raise KeyError(name)

    def rhs(time, state):
        out = np.empty_like(state)
        for var, (order, terms, constant, target_coef) in forms.items():
            for k in range(order - 1):
                out[index[(var, k)]] = state[index[(var, k + 1)]]
            value = constant
            for coef, factors in terms:
                prod = coef
                for name, params in factors:
                    prod *= factor_value(name, params, time, state)
                value += prod
            out[index[(var, order - 1)]] = value / target_coef
        return out

    try:
        solution = solve_ivp(rhs, (t[0], t[-1]), y0, t_eval=t, method='LSODA', rtol=1e-8, atol=1e-10)
    except (KeyError, ValueError, OverflowError):
        return None
    simulated = {var: np.full(t.shape, np.nan) for var in forms}
    for var in forms:
        values = solution.y[index[(var, 0)]]
        simulated[var][:values.size] = values
    observed = {var: np.asarray(data[var], dtype=float)[interior_indices][:t.size] for var in forms}
    coverage = min(1.0, len(solution.t) / len(t))
    complete = bool(solution.success and coverage == 1.0 and
                    all(np.all(np.isfinite(v)) for v in simulated.values()))
    return SimulationResult(t, simulated, observed, complete, str(solution.message), coverage,
                            'tail reconstruction; equation may have been fitted on this tail' if start_fraction
                            else 'reconstruction of the fitted record')


def simulate_pde(problem, system, search, data=None):
    """Method-of-lines reconstruction for explicit first-time-order 1-D PDEs.

    Spatial derivatives up to order two use the supplied spatial grid. End
    values stay at the initial condition (Dirichlet); future observed fields
    never enter the solver. Unsupported forms return None. This checks a
    fitted equation, and does not claim a held-out training protocol.
    """
    from scipy.integrate import solve_ivp
    if problem.dim != 1 or len(system) != len(problem.variables):
        return None
    data = problem.data if data is None else data
    t = np.asarray(problem.grids[0])[:, 0]
    x = np.asarray(problem.grids[1])[0, :]
    if len(t) < 2:
        raise ValueError('PDE integration needs at least two time points')
    if not (np.all(np.isfinite(t)) and np.all(np.isfinite(x))):
        raise ValueError('PDE coordinates must be finite')
    forms = {}
    for equation in system:
        terms, constant, target, target_coef = parse_equation(equation)
        if len(target) != 1 or target_coef == 0:
            return None
        match = _DERIV_RE.match(target[0][0])
        if not match or int(match.group(1) or 1) != 1 or int(match.group(3)) != 0:
            return None
        var = match.group(2)
        if var not in problem.variables or dict(target[0][1]).get('power', 1) != 1:
            return None
        forms[var] = (terms, constant, target_coef)
    if set(forms) != set(problem.variables) or len(x) < 4:
        return None
    if not (np.all(np.diff(t) > 0) and np.all(np.diff(x) > 0)):
        raise ValueError('PDE coordinates must be strictly increasing')
    variables = problem.variables
    initial = {v: np.asarray(data[v])[0].copy() for v in variables}
    n = len(x) - 2

    def rhs(time, state):
        fields = {v: np.r_[initial[v][0], state[i*n:(i+1)*n], initial[v][-1]]
                  for i, v in enumerate(variables)}
        arrays = dict(fields)
        for v, field in fields.items():
            arrays[f'd{v}/dx1'] = np.gradient(field, x, edge_order=2)
            left, right = np.diff(x)[:-1], np.diff(x)[1:]
            second = np.empty_like(field, dtype=float)
            second[1:-1] = 2 * ((field[2:] - field[1:-1]) / right
                               - (field[1:-1] - field[:-2]) / left) / (left + right)
            second[0], second[-1] = second[1], second[-2]
            arrays[f'd^2{v}/dx1^2'] = second
        outputs = []
        for var in variables:
            terms, constant, coef_target = forms[var]
            value = np.full(len(x), constant, dtype=float)
            for coef, factors in terms:
                product = np.full(len(x), coef, dtype=float)
                for name, pairs in factors:
                    params = dict(pairs)
                    if name in arrays:
                        factor = arrays[name]
                    elif name == 'x' or name.startswith('x_'):
                        factor = time if params.get('dim', 1) == 0 else x
                    elif name in ('sin', 'cos'):
                        coordinate = time if params.get('dim', 1) == 0 else x
                        factor = getattr(np, name)(params.get('freq', 1) * coordinate)
                    else:
                        raise KeyError(name)
                    product *= factor ** params.get('power', 1)
                value += product
            outputs.append(value[1:-1] / coef_target)
        return np.concatenate(outputs)
    y0 = np.concatenate([initial[v][1:-1] for v in variables])
    try:
        rhs(t[0], y0)  # reject unsupported factors before solving
    except KeyError:
        return None
    solution = solve_ivp(rhs, (t[0], t[-1]), y0, t_eval=t, method='BDF', rtol=1e-7, atol=1e-9)
    simulated = {}
    for i, var in enumerate(variables):
        field = np.full(problem.shape, np.nan)
        count = len(solution.t)
        field[:count, 0], field[:count, -1] = initial[var][0], initial[var][-1]
        field[:count, 1:-1] = solution.y[i*n:(i+1)*n].T
        simulated[var] = field
    coverage = len(solution.t) / len(t)
    success = bool(solution.success and coverage == 1 and
                   all(np.all(np.isfinite(v)) for v in simulated.values()))
    return SimulationResult(t, simulated, data, success, str(solution.message), coverage,
                            'fitted-record reconstruction with fixed initial boundary values')


def plot_reconstruction(problem, system, search, data=None, title=None, horizon=None):
    """Side-by-side figure for a discovered system (list of equation texts).

    ODEs: observed vs predicted highest derivative, and, when the equations
    can be integrated, the simulated trajectory over the record.
    1-D PDEs: observed and predicted left-hand side as (t, x) maps plus their
    difference. 2-D and 3-D PDEs: the same at one time moment.
    Returns ``(figure, summary)`` where ``summary`` holds R^2 values."""
    import matplotlib.pyplot as plt
    summary = {}
    fields = [(eq, *equation_fields(problem, eq, search, data)) for eq in system]
    for eq, obs, pred in fields:
        summary[eq.split('=', 1)[1].strip()] = r2_score(obs, pred)
    simulation = simulate_ode(problem, system, search, data, horizon) if problem.dim == 0 else None

    if problem.dim == 0:
        rows = len(fields) + (len(problem.variables) if simulation else 0)
        fig, axes = plt.subplots(rows, 1, figsize=(10, 2.3 * rows), squeeze=False)
        t = np.asarray(problem.grids[0])[_interior(problem, search)[0]]
        for ax, (eq, obs, pred) in zip(axes[:, 0], fields):
            lhs = eq.split('=', 1)[1].strip()
            ax.plot(t, obs, 'k', lw=1.2, label='from data')
            ax.plot(t, pred, 'C1--', lw=1.0, label='predicted by the equation')
            ax.set_ylabel(_short(lhs))
            ax.set_title(f'{_short(lhs)}: data vs equation, R² = {summary[lhs]:.4f}',
                         fontsize=9)
            ax.legend(fontsize=8, loc='upper right')
        if simulation:
            ts, sim, obs = simulation
            for ax, var in zip(axes[len(fields):, 0], problem.variables):
                ax.plot(ts, obs[var], 'k', lw=1.2, label='data')
                ax.plot(ts, sim[var], 'C0--', lw=1.0, label='solution of the equation')
                score = r2_score(obs[var], sim[var]) if simulation.success else float('nan')
                summary[f'trajectory {var}'] = score
                ax.set_ylabel(var)
                ax.set_title(f'{var}: integrated equation, R² = {score:.4f}; coverage {simulation.coverage:.0%}'
                             + ('' if simulation.success else f'; failed: {simulation.message}'), fontsize=9)
                ax.legend(fontsize=8, loc='upper right')
        axes[-1, 0].set_xlabel(problem.axis_names[0])
    else:
        inner = _interior(problem, search)
        coords = [np.unique(np.asarray(g)[inner]) for g in problem.grids]
        fig, axes = plt.subplots(len(fields), 3, figsize=(14, 3.6 * len(fields)), squeeze=False)
        for row, (eq, obs, pred) in enumerate(fields):
            lhs = eq.split('=', 1)[1].strip()
            if obs.ndim > 2:                                  # one time moment, middle of extra axes
                k = obs.shape[0] // 2
                obs, pred = obs[k], pred[k]
                while obs.ndim > 2:
                    obs, pred = obs[..., obs.shape[-1] // 2], pred[..., pred.shape[-1] // 2]
                horizontal, vertical = 2, 1
                moment = f', {problem.axis_names[0]} = {coords[0][k]:.3g}'
            else:
                obs, pred = obs.T, pred.T                     # (x, t): time to the right
                horizontal, vertical = 0, 1
                moment = ''
            extent = [coords[horizontal][0], coords[horizontal][-1],
                      coords[vertical][0], coords[vertical][-1]]
            lim = np.nanpercentile(np.abs(obs), 99) or 1.0
            for ax, field, name in [(axes[row, 0], obs, 'from data'),
                                    (axes[row, 1], pred, 'predicted by the equation'),
                                    (axes[row, 2], pred - obs, 'difference')]:
                im = ax.imshow(field, origin='lower', aspect='auto', cmap='RdBu_r',
                               vmin=-lim, vmax=lim, extent=extent)
                ax.set_title(f'{_short(lhs)}: {name}{moment}', fontsize=9)
                ax.set_xlabel(problem.axis_names[horizontal])
                ax.set_ylabel(problem.axis_names[vertical])
                fig.colorbar(im, ax=ax)
            axes[row, 0].set_title(f'{_short(lhs)}: from data{moment} (R² = {summary[lhs]:.4f})',
                                   fontsize=9)
    fig.suptitle(title or f'{problem.title}: equation against the data')
    fig.tight_layout()
    return fig, summary


def _short(token_text):
    """``'d^2u/dx0^2{power: 1.0}'`` -> ``'d^2u/dx0^2'`` for axis labels."""
    return re.sub(r'\{[^}]*\}', '', token_text).strip()
