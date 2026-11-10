"""Building blocks evaluated on the data: powers, derivatives, coordinates,
sin/cos and the problem's extra fields, named exactly as EPDE prints the
corresponding tokens. Shared by the signal check (truthcheck.py) and the
sparse-regression baseline (baselines.py).
"""

import numpy as np


def _deriv_name(var, axis, order):
    if order == 1:
        return f'd{var}/dx{axis}'
    return f'd^{order}{var}/dx{axis}^{order}'


def _differentiate(u, grids, axis, order, method, smoother_kws=None):
    """``order``-th derivative of ``u`` along ``axis`` on a uniform grid.
    ``smoother_kws`` go to the Savitzky-Golay filter of the smoothed method
    (``window_length``, ``polyorder``)."""
    import pysindy as ps
    coords = np.unique(np.asarray(grids[axis]))
    step = float(coords[1] - coords[0])
    if method == 'smoothed_finite_difference':
        diff = ps.SmoothedFiniteDifference(d=order, axis=axis, smoother_kws=smoother_kws or {})
    else:
        diff = ps.FiniteDifference(order=2, d=order, axis=axis)
    return np.asarray(diff._differentiate(u, step))


def _blocks(problem, data, search, method="epde"):
    """Building blocks: list of (name, array, family, meaningful), plus the
    time derivatives available as left-hand sides."""
    space = search.get('search_space', {})
    from .preprocessing import derivative_orders, prepare_data
    orders = derivative_orders(search, len(problem.grids))
    stacks = None
    if method == 'epde':
        data, stacks = prepare_data(problem, data, search)
    data_pow = int(space.get('data_fun_pow', 1))
    deriv_pow = int(space.get('deriv_fun_pow', 1))
    # Explicit PySINDy methods are retained for independently configured baselines.
    prep = search['preprocessing'].get('preprocessor_kwargs') or {}
    smoother_kws = ({'window_length': int(prep['polynomial_window']),
                     'polyorder': int(prep.get('poly_order', 3))}
                    if 'polynomial_window' in prep else None)
    blocks, lhs = [], {}
    for var, u in data.items():
        for p in range(1, data_pow + 1):
            blocks.append((f'{var}{{power: {p:.1f}}}', u ** p, f'data:{var}', True, 0))
        lhs[var] = []
        column = 0
        for axis, max_order in enumerate(orders):
            for k in range(1, int(max_order) + 1):
                d = (stacks[var][:, column].reshape(u.shape) if stacks is not None
                     else _differentiate(u, problem.grids, axis, k, method, smoother_kws))
                column += 1
                name = _deriv_name(var, axis, k)
                if axis == 0:
                    lhs[var].append((f'{name}{{power: 1.0}}', d, k))
                for p in range(1, deriv_pow + 1):
                    blocks.append((f'{name}{{power: {p:.1f}}}', d ** p, f'deriv:{name}', True,
                                   k if axis == 0 else 0))
    for spec in space.get('tokens', []) or []:
        family = spec.get('family')
        if family == 'grid':
            for axis, grid in enumerate(problem.grids):
                for p in range(1, int(spec.get('max_power', 1)) + 1):
                    blocks.append((f'x{{power: {p:.1f}, dim: {axis:.1f}}}', np.asarray(grid) ** p,
                                   'grid', True, 0))
        elif family == 'fixed_trigonometric':
            f = float(spec.get('freq', 2.0))
            for axis, grid in enumerate(problem.grids):
                for fn, label in ((np.sin, 'sin'), (np.cos, 'cos')):
                    blocks.append((f'{label}{{power: 1.0, freq: {f:.1f}, dim: {axis:.1f}}}',
                                   fn(f * np.asarray(grid)), 'trig', bool(spec.get('meaningful', False)), 0))
    for label, (array, meaningful) in problem.named_arrays.items():
        blocks.append((f'{label}{{power: 1.0}}', array, f'extra:{label}', meaningful, 0))
    return blocks, lhs


def _auto_widths(problem, search):
    """The widths EPDE gives ``boundary_width='auto'`` when it builds the pool
    (``EpdeSearch._resolve_auto_boundaries``): per axis, the widest edge reach
    of the preprocessing over the variables. Supplied derivatives are neither
    smoothed nor differentiated, so they reach nowhere; a pipeline that cannot
    say how far it reaches gets ``EpdeSearch.FALLBACK_BOUNDARY`` per end."""
    from epde.interface.interface import EpdeSearch
    from .preprocessing import derivative_orders, make_preprocessor
    ndim = len(problem.shape)
    if problem.derivs is not None:
        return [0] * ndim
    pipe = make_preprocessor(search)
    orders = derivative_orders(search, ndim)
    reach = [0] * ndim
    for var in problem.variables:
        own = pipe.edge_reach(np.asarray(problem.data[var]), orders)
        if own is None:
            return [EpdeSearch.FALLBACK_BOUNDARY] * ndim
        reach = [max(a, int(b)) for a, b in zip(reach, own)]
    return reach


def _interior(problem, search):
    from epde.interface.search_config import load_search_config
    width = load_search_config(search).domain.boundary_width
    if isinstance(width, str):
        width = _auto_widths(problem, search)
    widths = [width] * len(problem.shape) if np.isscalar(width) else list(width)
    if len(widths) != len(problem.shape) or any(
            int(w) != w or w < 0 or 2*w >= n for w, n in zip(widths, problem.shape)):
        raise ValueError(f'boundary_width leaves no valid interior: {widths} for {problem.shape}')
    return tuple(slice(int(w), n - int(w)) for w, n in zip(widths, problem.shape))
