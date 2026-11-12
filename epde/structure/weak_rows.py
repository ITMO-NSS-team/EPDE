"""Weak-form rows of the solver-free search (``domain.evaluation_form='weak'``): the EPDE side of
:mod:`epde.preprocessing.weak_form`.

The pieces:
- :func:`weak_term`: an EPDE term as a symbolic ``WeakTerm``.
  - A ``'poly of v'`` factor of integer power p is p data factors ``(v, zero orders)``.
  - A ``'deriv of v'`` factor is p data factors ``(v, per-axis counts of its deriv_code)``.
  - A ``'grids'`` factor is a coordinate power on its ``dim`` axis.
  - Every other factor (trigonometric, constant, user grid functions) is OPAQUE: a pointwise factor whose
    values come from its evaluator on the full grid (``CustomEvaluator.full_grid``). A term with an opaque
    factor gets no integration by parts.
  - A stored-token factor (``CacheStoredTokens``) is opaque too: its values are the trajectory's full-grid
    tensors (``Trajectory.raw_token``), raised to the factor's power. Weak rows integrate the full grid (no
    domain boundary), so a stored tensor must hold valid values on every grid point. Stored tokens are not
    window-selection columns. ``ExternalDerivativesTokens`` stay refused: their tensors are differentiated
    after the domain mask and no full-grid tensor exists.
- :class:`WeakRows`: per trajectory, the raw fields; per (trajectory, equation slot), the window plan. Term
  columns are cached by (slot, ``Term.factors_labels``).
- :func:`build_weak_rows`: picks each slot's window once per fit (``weak_form.choose_window``).

The selection library is the pool's data, derivative and grid tokens, each as a one-factor term at power 1.
The candidate targets are the slot variable's derivative tokens, and the window score is their median (the
window study's ``tok1 / med`` rule).

The edge cut is the selection library's own, which is none: one-factor power-1 terms integrate by parts
completely. A search term that keeps a pointwise derivative (``u_x^2``, a product with an opaque factor)
takes ``_central4``'s second-order rule on its two end samples per nesting. The test function vanishes at
every window edge, so those samples carry little weight, and only the windows at the grid's ends contain
them. The window study measured this (Nov 4 2026, ``pick_check_cut.py``): cutting the full stencil reach
instead discards samples, and on short grids that cost more than it saved. On wave 81 x 81 at 5%, the
truth's coefficient error rose from 2e-2 to 1.5e-1.
"""
from typing import Dict, Hashable, List, Optional, Sequence, Tuple

import numpy as np

from epde.evaluators import CustomEvaluator, simple_function_evaluator
from epde.preprocessing import weak_form as WF

__all__ = ['WeakRows', 'weak_term', 'selection_library', 'build_weak_rows', 'axes_of', 'active', 'slot_of',
           'term_columns']


def active() -> bool:
    """Whether the current fit evaluates weak rows (a ``WeakRows`` store on the samples manager)."""
    import epde.globals as global_var
    return getattr(getattr(global_var, 'samples_manager', None), 'weak', None) is not None


def slot_of(equation) -> Optional[str]:
    """The equation slot whose rows ``equation`` is evaluated on: its ``main_var_to_explain`` under weak rows,
    ``None`` under strong rows (one set of rows for every slot)."""
    return equation.main_var_to_explain if active() else None


def term_columns(term, slot: Optional[str]):
    """``term.evaluate`` on the rows of ``slot``: the plain call under strong rows (``slot`` None)."""
    return term.evaluate() if slot is None else term.evaluate(slot=slot)


def _param(factor, name: str):
    """The factor's parameter ``name``, or ``None`` when it has none."""
    for idx, descr in factor.params_description.items():
        if descr['name'] == name:
            return factor.params[idx]
    return None


def _integer(value) -> Optional[int]:
    if value is None:
        return None
    value = float(value)
    return int(value) if value.is_integer() and value >= 1 else None


def _orders(deriv_code, D: int) -> Tuple[int, ...]:
    """Per-axis derivative counts of an EPDE ``deriv_code`` (``[None]`` is the field itself)."""
    if deriv_code is None or list(deriv_code) == [None]:
        return (0,) * D
    counts = [0] * D
    for axis in deriv_code:
        axis = int(axis)
        if not 0 <= axis < D:
            raise ValueError(f'derivative code {deriv_code} names axis {axis} of a {D}-axis grid.')
        counts[axis] += 1
    return tuple(counts)


def _is_data_family(ftype, variable, evaluator) -> bool:
    """A family whose tokens are the observed field ``variable`` or its derivatives."""
    return (isinstance(variable, str) and ftype in (f'poly of {variable}', f'deriv of {variable}')
            and evaluator is simple_function_evaluator)


def _inner_evaluator(evaluator):
    return getattr(evaluator, '_evaluator', evaluator)


def _is_stored_family(ftype, variable, evaluator, family_of_derivs) -> bool:
    """A family of tokens whose values are supplied as tensors (``CacheStoredTokens``)."""
    return (evaluator is simple_function_evaluator and not _is_data_family(ftype, variable, evaluator)
            and ftype != 'grids' and not family_of_derivs)


def weak_term(term, D: int, stored=frozenset()):
    """``(WeakTerm, {opaque label: factor})`` of an EPDE term on a ``D``-axis grid.

    ``stored`` is the set of token labels that have full-grid tensors; a factor with one of those labels (and
    no other reading) is opaque like a coordinate function."""
    data, coords, opaque, factors = [], [0] * D, [], {}
    for factor in term.structure:
        power = _integer(_param(factor, 'power'))
        evaluator = _inner_evaluator(factor.evaluator)
        if power is not None and _is_data_family(factor.ftype, factor.variable, evaluator):
            data.extend([(factor.variable, _orders(factor.deriv_code, D))] * power)
        elif power is not None and factor.ftype == 'grids':
            dim = int(_param(factor, 'dim'))
            if not 0 <= dim < D:
                raise ValueError(f'grid token {factor.label} names axis {dim} of a {D}-axis grid.')
            coords[dim] += power
        elif isinstance(evaluator, CustomEvaluator) and not factor.is_deriv:
            label = factor.structural_label
            opaque.append(label)
            factors[label] = factor
        elif (not factor.is_deriv and evaluator is simple_function_evaluator and factor.ftype != 'grids'
              and not _is_data_family(factor.ftype, factor.variable, evaluator) and factor.label in stored):
            label = factor.structural_label
            opaque.append(label)
            factors[label] = factor
        else:
            raise ValueError(f"domain.evaluation_form='weak' cannot evaluate token {factor.label!r} (family "
                             f'{factor.ftype!r}): it is neither the data, a derivative, a grid coordinate nor a '
                             'function of the coordinates.')
    return WF.WeakTerm(data, coords, opaque), factors


def check_families(pool, stored_labels=frozenset()) -> None:
    """Refuse a pool with a family whose values the weak rows cannot compute on the full grid.

    ``stored_labels`` are the token labels with full-grid tensors on every trajectory."""
    for family in pool.families:
        evaluator = _inner_evaluator(getattr(family, '_evaluator', None))
        if _is_data_family(family.ftype, family.variable, evaluator) or family.ftype == 'grids':
            continue
        derivs = getattr(family, 'family_of_derivs', False)
        if isinstance(evaluator, CustomEvaluator) and not derivs:
            continue
        message = (f"domain.evaluation_form='weak' cannot evaluate the token family {family.ftype!r}: "
                   'weak rows integrate the data, its derivatives and functions of the coordinates over '
                   'the full grid, and this family is none of them.')
        if _is_stored_family(family.ftype, family.variable, evaluator, derivs):
            missing = [label for label in family.tokens if label not in stored_labels]
            if not missing:
                continue
            message += (' A stored token family needs its full-grid tensors on every trajectory '
                        '(CacheStoredTokens(token_tensors=...) or createTrajectory(cached_token_tensors=...)); '
                        f'missing for the tokens {missing}.')
        raise ValueError(message)


def selection_library(pool, D: int) -> List[WF.WeakTerm]:
    """The window-selection library: every data, derivative and grid token as a one-factor term at power 1."""
    library, seen = [], set()

    def add(term):
        if term not in seen:
            seen.add(term)
            library.append(term)

    for family in pool.families:
        evaluator = _inner_evaluator(getattr(family, '_evaluator', None))
        if _is_data_family(family.ftype, family.variable, evaluator):
            for label in family.tokens:
                add(WF.WeakTerm([(family.variable, _orders(family.derivs_ords[label], D))], (0,) * D))
        elif family.ftype == 'grids':
            fixed = getattr(family, 'token_fixed_params', {}) or {}
            for label in family.tokens:
                if 'dim' in fixed.get(label, {}):
                    dims = [int(fixed[label]['dim'])]
                else:
                    lo, hi = family.token_params['dim']
                    dims = range(int(lo), int(hi) + 1)
                for d in dims:
                    if 0 <= d < D:
                        add(WF.WeakTerm([], tuple(int(b == d) for b in range(D))))
    return library


def axes_of(grids: Sequence[np.ndarray]) -> List[np.ndarray]:
    """1-D coordinate axes of a full meshgrid (``indexing='ij'``). Weak kernels need uniform steps."""
    D = len(grids)
    axes = []
    for d, grid in enumerate(grids):
        grid = np.asarray(grid)
        axis = grid[tuple(slice(None) if b == d else 0 for b in range(D))].astype(float)
        if axis.size > 2:
            steps = np.diff(axis)
            if not np.allclose(steps, steps[0], rtol=1e-6, atol=0.0):
                raise ValueError(f"domain.evaluation_form='weak' needs a uniform grid; axis {d} is not uniform.")
        axes.append(axis)
    return axes


class WeakRows:
    """The weak rows of one fit: raw fields per trajectory, a window plan per (trajectory, slot) and the term
    columns built on them."""

    def __init__(self, fields: Dict[Hashable, WF.Fields], plans: Dict[Tuple[Hashable, str], WF.WindowPlan],
                 keys: Sequence[Hashable], tokens: Optional[Dict[Hashable, Dict[str, np.ndarray]]] = None):
        self.fields = fields
        self.plans = plans
        self.keys = list(keys)
        #: {trajectory key: {stored token label: full-grid tensor}}
        self.tokens = {key: dict((tokens or {}).get(key, {})) for key in self.keys}
        self._columns: Dict[tuple, Dict[Hashable, np.ndarray]] = {}

    def _plan(self, key, slot) -> WF.WindowPlan:
        try:
            return self.plans[(key, slot)]
        except KeyError:
            raise KeyError(f'no weak-form window for trajectory {key!r}, equation slot {slot!r}; slots with '
                           f'windows: {sorted({s for _, s in self.plans})}.') from None

    def shapes(self, slot) -> Dict[Hashable, np.ndarray]:
        """Each trajectory's row lattice for ``slot``, as an array like ``Trajectory.inner_shape``."""
        return {key: np.array(self._plan(key, slot).shape) for key in self.keys}

    def weights(self, slot) -> Dict[Hashable, np.ndarray]:
        """Unit row weights: the test function already weights each row's window."""
        return {key: np.ones(self._plan(key, slot).rows) for key in self.keys}

    def _opaque_values(self, factor, key) -> np.ndarray:
        """The full-grid values of an opaque factor on trajectory ``key``."""
        evaluator = _inner_evaluator(factor.evaluator)
        shape = self.fields[key].shape
        if isinstance(evaluator, CustomEvaluator):
            return np.broadcast_to(np.asarray(evaluator.full_grid(factor)[key], dtype=float), shape)
        power = _param(factor, 'power')
        values = np.asarray(self.tokens[key][factor.label], dtype=float)
        if values.shape != tuple(shape):
            raise ValueError(f'the stored token {factor.label!r} has shape {values.shape}, but the grid is '
                             f'{tuple(shape)}: stored tensors must be on the full grid.')
        return values ** (1.0 if power is None else float(power))

    def columns(self, term, slot) -> Dict[Hashable, np.ndarray]:
        """The term's weak column on each trajectory's rows for ``slot``."""
        cache_key = (slot, term.factors_labels)
        cached = self._columns.get(cache_key)
        if cached is not None:
            return cached
        out = {}
        for key in self.keys:
            fields = self.fields[key]
            wterm, factors = weak_term(term, fields.D, frozenset(self.tokens.get(key, {})))
            plan = self._plan(key, slot)
            for label, factor in factors.items():
                fields.opaque[label] = self._opaque_values(factor, key)
            try:
                monos = wterm.monomials(psi_cap=plan.q)
                M, _ = WF.weak_matrix([monos], [wterm.opaque], fields, plan.cut, plan.ms, plan.ss, plan.q)
            finally:
                for label in factors:
                    fields.opaque.pop(label, None)
            out[key] = M[:, 0]
        self._columns[cache_key] = out
        return out


def build_weak_rows(pool, manager, row_ok, slots: Optional[Sequence[str]] = None) -> WeakRows:
    """Pick every (trajectory, slot) window of ``pool`` on the trajectories of ``manager``.

    ``row_ok(shape)`` is the instability metric's row floor. The noise draw of the window score is seeded per
    trajectory position, so a fit's windows are reproducible."""
    keys = list(manager.trajecatoryIDs)
    tokens = {key: manager[key].raw_tokens() for key in keys}
    label_sets = [set(stored) for stored in tokens.values()]
    check_families(pool, frozenset.intersection(*map(frozenset, label_sets)) if label_sets else frozenset())
    if slots is None:
        slots = [family.variable for family in pool.families_demand_equation]
    fields, plans = {}, {}
    for idx, key in enumerate(keys):
        trajectory = manager[key]
        axes = axes_of(trajectory.grids('full'))
        D = len(axes)
        library = selection_library(pool, D)
        fields[key] = WF.Fields(manager.raw_fields(key), axes)
        for slot in slots:
            targets = [k for k, t in enumerate(library) if len(t.data) == 1 and not any(t.coords)
                       and t.data[0][0] == slot and sum(t.data[0][1]) > 0]
            plans[(key, slot)] = WF.choose_window(fields[key], library, targets, np.random.default_rng([0, idx]),
                                                  rows_ok=row_ok)
    return WeakRows(fields, plans, keys, tokens)
