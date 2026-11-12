"""A system whose equations restate one relation: an equation that follows from the others.

The lv fronts filled with systems whose two equations are one relation twice -- the sum identity
u' + v' = 20 (u - v) beside its own time derivative: each equation a valid identity of the data, the
system the dynamics of u + v alone. Such a system is refused (its following equation is re-rolled); a
system of identities that together still pin down every law is kept.

Equation i follows from the others when its residual ``target - sum_j c_j term_j - c_0`` is a
combination of the other equations' residuals, each multiplied by 1, a field, or a factor of equation i,
and of their first derivatives along every axis -- the identity scorer's test. Every field, every
derivative of a field (mixed ones included) and every coordinate is an independent random value; the
coordinates are drawn over the data's axis ranges, so a factor like cos(7e-5 x), 1 to nine digits on the
data, is not taken for 1. Columns are unit-normalised. The fitted coefficients are inexact, so "is a
combination" is a relative residual below ``REDUNDANCY_RTOL``: a restated relation leaves the
coefficients' error (about 1e-3 on the lv fronts), an independent law leaves one of order 1.

Fields, their derivatives, grid tokens and sin / cos are differentiated here; a system holding any other
kind of token is left as it is, and the test is then off: that, and data with no registered axis ranges,
are warned of once per fit (:func:`reset` re-arms the warning).
"""
import warnings
from typing import List, Optional

import numpy as np

import epde.globals as global_var

#: Relative residual below which an equation is a combination of the others (see the module docstring).
REDUNDANCY_RTOL = 0.1
#: Random points: at least this many, and four per basis column.
_MIN_POINTS = 64
#: The draws are the module's own, so the test leaves the search's random stream untouched.
_SEED = 20261022

_REPORTED = set()      # the reasons the test was found off for since the last reset

_NO_RANGES = 'the data have no registered axis ranges.'
_NOT_DIFFERENTIATED = ('a token of the system is of a kind the test does not differentiate (it differentiates '
                       'fields, their derivatives, grid tokens and sin / cos).')


def reset() -> None:
    """Forget the reasons already reported, so that the next fit reports them again."""
    _REPORTED.clear()


def _off(reason: str) -> list:
    """[], the test's result when it cannot run; the reason is warned of once until :func:`reset`."""
    if reason not in _REPORTED:
        _REPORTED.add(reason)
        warnings.warn('The test for an equation that follows from the other equations of its system is off: '
                      + reason, global_var.EPDEUsageWarning, stacklevel=3)
    return []


def _param(factor, name, default):
    for idx, description in factor.params_description.items():
        if description['name'] == name:
            return float(factor.params[idx])
    return default


class _Jet(object):
    """Independent random values, at ``n`` points, of every field and derivative (keyed by the
    variable and its per-axis derivative orders) and of every coordinate (over its data range)."""

    def __init__(self, n: int, ranges: list, rng):
        self.n, self.ranges, self._rng, self._values = n, ranges, rng, {}

    def field(self, variable, orders: tuple) -> np.ndarray:
        key = ('field', variable, orders)
        if key not in self._values:
            self._values[key] = self._rng.standard_normal(self.n)
        return self._values[key]

    def coordinate(self, axis: int) -> np.ndarray:
        key = ('coordinate', axis)
        if key not in self._values:
            lo, hi = self.ranges[axis]
            self._values[key] = self._rng.uniform(lo, hi, self.n) if hi > lo else np.full(self.n, float(lo))
        return self._values[key]


def _factor(factor, jet: _Jet, ndim: int):
    """``(value, [d value / d x_a for every axis a])`` of ``factor`` on the jet, or None for a kind of
    token this test does not differentiate."""
    power = _param(factor, 'power', 1.0)
    zero = np.zeros(jet.n)
    if factor.deriv_code is not None:
        orders = [0] * ndim
        for axis in factor.deriv_code:
            if axis is not None:
                orders[axis] += 1
        variable = factor.variable
        base = jet.field(variable, tuple(orders))
        grads = [jet.field(variable, tuple(order + (k == a) for k, order in enumerate(orders)))
                 for a in range(ndim)]
    elif factor.ftype == 'grids':
        axis = int(_param(factor, 'dim', 0))
        base = jet.coordinate(axis)
        grads = [np.ones(jet.n) if a == axis else zero for a in range(ndim)]
    elif factor.ftype == 'trigonometric' and factor.label in ('sin', 'cos'):
        axis, freq = int(_param(factor, 'dim', 0)), _param(factor, 'freq', 1.0)
        arg = freq * jet.coordinate(axis)
        base, slope = (np.sin(arg), freq * np.cos(arg)) if factor.label == 'sin' else \
            (np.cos(arg), -freq * np.sin(arg))
        grads = [slope if a == axis else zero for a in range(ndim)]
    else:
        return None
    value = base ** power
    return value, [power * base ** (power - 1) * grad for grad in grads]


def _term(term, jet: _Jet, ndim: int):
    parts = [_factor(factor, jet, ndim) for factor in term.structure]
    if any(part is None for part in parts):
        return None
    value = np.prod([v for v, _ in parts], axis=0)
    grads = []
    for a in range(ndim):
        grad = np.zeros(jet.n)
        for k, (_, part_grads) in enumerate(parts):
            others = [v for l, (v, _) in enumerate(parts) if l != k]
            grad = grad + part_grads[a] * (np.prod(others, axis=0) if others else 1.0)
        grads.append(grad)
    return value, grads


def _residual(equation, jet: _Jet, ndim: int):
    """``(residual, [its derivative along every axis])`` of a fitted equation on the jet, or None."""
    tgt = equation.target_idx
    weights = np.asarray(equation.weights_final, dtype=float)
    target = _term(equation.structure[tgt], jet, ndim)
    if target is None:
        return None
    value, grads = target[0].copy(), [g.copy() for g in target[1]]
    for idx, term in enumerate(equation.structure):
        if idx == tgt:
            continue
        coef = weights[equation.weight_index(idx, tgt)]
        if coef == 0:
            continue
        evaluated = _term(term, jet, ndim)
        if evaluated is None:
            return None
        value -= coef * evaluated[0]
        grads = [g - coef * eg for g, eg in zip(grads, evaluated[1])]
    return value - weights[-1], grads


def _axis_ranges() -> Optional[list]:
    """(min, max) of every axis over the registered trajectories' inner grids: the samples
    manager's memoised ``TrajectoriesManager.axis_ranges``."""
    manager = getattr(global_var, 'samples_manager', None)
    return manager.axis_ranges() if hasattr(manager, 'axis_ranges') else None


def redundant_equations(equations: list, ranges: list = None, order: list = None) -> List[int]:
    """Positions of the fitted ``equations`` (one system) that follow from the others, tested in order:
    an equation found to follow is left out of the others for the ones after it, so a relation stated
    twice loses one statement. [] for one equation, or (with a warning, see the module docstring) when the
    data have no axis ranges or a token cannot be differentiated here.
    ``ranges`` -- (min, max) per axis -- default to the registered data's.

    ``order`` (a permutation of the positions) tests in that order instead. The random points are the
    same for every order: they are drawn by the test in the equations' own order, which runs first, and
    the permuted test reuses them; its positions come back sorted, and the own order's result when the
    two agree."""
    if len(equations) < 2:
        return []
    ranges = ranges if ranges is not None else _axis_ranges()
    if not ranges:
        return _off(_NO_RANGES)
    ndim = len(ranges)
    fields = sorted({eq.main_var_to_explain for eq in equations})
    rng = np.random.default_rng(_SEED)
    # Columns per test: (1 + fields + factors of the equation) multiples of every other equation plus
    # its derivatives; the largest equation bounds the factor count.
    widest = max(sum(len(term.structure) for term in eq.structure) for eq in equations)
    n = max(_MIN_POINTS, 4 * (len(equations) - 1) * (1 + len(fields) + widest + ndim))
    jet = _Jet(n, ranges, rng)
    residuals = [_residual(eq, jet, ndim) for eq in equations]
    if any(r is None for r in residuals):
        return _off(_NOT_DIFFERENTIATED)

    def tested(sequence) -> list:
        redundant = []
        for i in sequence:
            eq = equations[i]
            value = residuals[i][0]
            scale = np.linalg.norm(value)
            if not np.isfinite(scale) or scale == 0:
                continue
            multipliers = [np.ones(n)] + [jet.field(var, (0,) * ndim) for var in fields]
            for term in eq.structure:
                for factor in term.structure:
                    evaluated = _factor(factor, jet, ndim)
                    multipliers.append(evaluated[0])
            columns = []
            for j, (other, grads) in enumerate(residuals):
                if j == i or j in redundant:
                    continue
                columns.extend(m * other for m in multipliers)
                columns.extend(grads)
            if not columns:
                continue
            basis = np.stack(columns, axis=1)
            norms = np.linalg.norm(basis, axis=0)
            keep = np.isfinite(norms) & (norms > 0)
            if not keep.any():
                continue
            basis = basis[:, keep] / norms[keep]
            coef, *_ = np.linalg.lstsq(basis, value / scale, rcond=None)
            if np.linalg.norm(value / scale - basis @ coef) < REDUNDANCY_RTOL:
                redundant.append(i)
        return redundant

    own = tested(range(len(equations)))
    if order is None or list(order) == list(range(len(equations))):
        return own
    permuted = sorted(tested(order))
    return permuted if permuted != sorted(own) else own
