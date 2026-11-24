"""Causal time increments with pointwise or weak spatial measurements."""
import numpy as np

from epde.interface.search_config import check_forward_preprocessor, target_axes_of
from epde.preprocessing import weak_form as wf
from epde.preprocessing.deriv_calculators import _central4, _five_point_applies
from epde.structure.weak_rows import (
    WindowRows, _inner_evaluator, _is_data_family, _param, axes_of,
    check_families, choose_slot_window, product_library, selection_library,
)


class ForwardFields(wf.Fields):
    """Origin samples and paired endpoint measurements on one trajectory."""

    def __init__(self, raw, axes, time_axis, lag):
        self.raw = raw
        self.full_axes = axes
        self.time_axis = time_axis
        self.lag = lag
        t = axes[time_axis]
        if len(t) <= lag:
            raise ValueError(f'forward_lag={lag} leaves no complete intervals in {len(t)} samples.')
        self.elapsed = t[lag:] - t[:-lag]
        self.origins = tuple(slice(None, -lag) if d == time_axis else slice(None) for d in range(len(axes)))
        self.ends = tuple(slice(lag, None) if d == time_axis else slice(None) for d in range(len(axes)))
        super().__init__({name: value[self.origins] for name, value in raw.items()},
                         [axis[s] for axis, s in zip(axes, self.origins)])
        self.window_axes = tuple(d for d in range(self.D) if d != time_axis)

    def get(self, var, orders):
        if not orders[self.time_axis]:
            return super().get(var, orders)
        if sum(orders) != 1:
            raise ValueError('A forward target must be a first time increment.')
        key = var, orders
        if key not in self._cache:
            shape = [1] * self.D
            shape[self.time_axis] = -1
            self._cache[key] = (self.raw[var][self.ends] - self.values[var]) / self.elapsed.reshape(shape)
        return self._cache[key]

    @staticmethod
    def spatial_derivative(values, coords, axis):
        if _five_point_applies(coords):
            return _central4(values, coords, axis)
        return np.gradient(values, coords, axis=axis)

    def every_other(self):
        sl = tuple(slice(None, None, 2) if d in self.window_axes else slice(None) for d in range(self.D))
        return ForwardFields({k: v[sl] for k, v in self.raw.items()},
                             [a[s] for a, s in zip(self.full_axes, sl)], self.time_axis, self.lag)

    def noise_draw(self, rng, names):
        # Perturb a raw sample once even when it is an endpoint in one row
        # and an origin in another.
        raw = dict(self.raw)
        for name in names:
            value = self.raw[name]
            sig, rhos, _ = wf.noise_model(value)
            raw[name] = value + wf.ar_noise(rng, value.shape, sig, rhos)
        return ForwardFields(raw, self.full_axes, self.time_axis, self.lag)


class ForwardRows(WindowRows):
    form = 'forward'

    def __init__(self, fields, plans, keys, tokens, domains, target_labels, pool):
        super().__init__(fields, plans, keys, tokens)
        self.domains = domains
        self.time_axis = fields[keys[0]].time_axis
        self.target_labels = target_labels
        self.pool = pool

    def ensure_target(self, equation):
        """Install the unique first increment without sampling unrelated derivatives."""
        from epde.structure.main_structures import Term
        slot = equation.main_var_to_explain
        if any(self.is_target(term, slot) for term in equation.structure):
            return
        target = equation.pool.create(label=self.target_labels[slot], power=1)[1]
        predictors = [term for term in equation.structure if self.is_predictor(term)]
        cap = equation.metaparameters['max_terms_number']['value']
        equation.structure = predictors[:cap-1] + [Term(equation.pool, passed_term=[target])]
        equation._invalidate_label_cache()

    def is_predictor(self, term):
        return all(self.time_axis not in (factor.deriv_code or ()) for factor in term.structure)

    def is_target(self, term, slot):
        if len(term.structure) != 1:
            return False
        factor = term.structure[0]
        return (factor.variable == slot and list(factor.deriv_code or ()) == [self.time_axis]
                and _param(factor, 'power') == 1)

    def validate_candidate(self, equation):
        slot = equation.main_var_to_explain
        if equation.target is not None and not self.is_target(equation.target, slot):
            raise ValueError(f'Supplied forward equation for {slot!r} has an incompatible target.')
        for term in equation.structure:
            if not self.is_target(term, slot) and not self.is_predictor(term):
                raise ValueError(f'Supplied forward equation for {slot!r} uses a future predictor.')
            for factor in term.structure:
                if not (_is_data_family(factor.ftype, factor.variable, _inner_evaluator(factor.evaluator))
                        or factor.ftype == 'grids' or factor.status.get('available_at_origin') is True):
                    raise ValueError(f'Supplied predictor {factor.label!r} must declare available_at_origin=True.')

    def validate_symbolic(self, symbolic, slot):
        time_factors = [(var, orders) for var, orders in symbolic.data if orders[self.time_axis]]
        if time_factors and not (len(symbolic.data) == 1 and time_factors[0][0] == slot
                                 and sum(time_factors[0][1]) == 1 and not any(symbolic.coords)
                                 and not symbolic.opaque):
            raise ValueError('A forward increment can occur only as the target of its own field.')

    def _opaque_values(self, factor, key):
        # Evaluate supplied tensors and coordinate functions on their declared full grid.
        if factor.status.get('available_at_origin') is not True:
            raise ValueError(f'Forward predictor {factor.label!r} must declare available_at_origin=True.')
        shape, region = self.domains[key]
        values = super()._opaque_values(factor, key, shape)
        return values[region][self.fields[key].origins]


def _point_plan(fields, predictors, domain, row_ok):
    orders = [max((o[d] for term in predictors for _, o in term.data), default=0)
              for d in range(fields.D)]
    if domain.boundary_auto:
        cut = tuple(order * (wf.FD_REACH if _five_point_applies(axis) else 1)
                    for axis, order in zip(fields.axes, orders))
    else:
        cut = tuple(0 if d == fields.time_axis else width
                    for d, width in enumerate(domain.boundary_width_per_axis))
    shape = tuple(n - 2*c for n, c in zip(fields.shape, cut))
    if min(shape) < 1 or not row_ok(shape):
        raise ValueError(f'Insufficient forward rows on lattice {shape}.')
    return wf.WindowPlan(0, (0,) * fields.D, (0,) * fields.D, (1,) * fields.D, cut, shape, 0.)


def _trajectory_fields(trajectory, raw, cfg, axis):
    domain = trajectory._domain
    grids = trajectory.grids('full')
    if not 0 <= axis < len(grids) or axis != domain._time_axis:
        raise ValueError('Forward trajectories must agree on a valid time axis.')
    pipe = trajectory._preprocessor_pipeline
    if pipe is not None:
        check_forward_preprocessor(None, pipe.deriv_calculator_kwargs)
    pointwise = (axis,) if cfg.forward_space == 'weak' else range(len(grids))
    axes = axes_of(grids, pointwise_axes=pointwise)
    for d, coordinates in enumerate(axes):
        if not np.all(np.isfinite(coordinates)) or np.any(np.diff(coordinates) <= 0):
            raise ValueError(f'Forward rows need finite, strictly increasing coordinates on axis {d}.')
        broadcast = [1] * len(axes)
        broadcast[d] = -1
        if not np.all(grids[d] == coordinates.reshape(broadcast)):
            raise ValueError(f'Forward rows need a rectilinear coordinate grid on axis {d}.')
    shape = tuple(map(len, axes))
    cut = [0] * len(axes) if domain.boundary_auto else domain.boundary_width_per_axis
    # Pointwise spatial stencils may read the margin. Crop their completed
    # columns in _point_plan. Both time endpoints must be inside the margin.
    region = tuple(slice(c, n-c) if d == axis or cfg.forward_space == 'weak' else slice(None)
                   for d, (c, n) in enumerate(zip(cut, shape)))
    fields = ForwardFields({name: value[region] for name, value in raw.items()},
                           [a[s] for a, s in zip(axes, region)], axis, cfg.forward_lag)
    return fields, (shape, region)


def build_forward_rows(pool, manager, cfg, plain, row_ok):
    keys = list(manager.trajecatoryIDs)
    if not keys:
        raise ValueError('Forward fitting requires a trajectory.')
    time_axis = manager[keys[0]]._domain._time_axis
    slots = [f.variable for f in pool.families_demand_equation]
    tokens = {key: manager[key].raw_tokens() for key in keys}
    check_families(pool, set.intersection(*(set(v) for v in tokens.values())))
    target_labels = {}
    for family in pool.families:
        if _is_data_family(family.ftype, family.variable, _inner_evaluator(family._evaluator)):
            for label in family.tokens:
                if list(family.derivs_ords[label]) == [time_axis]:
                    target_labels[family.variable] = label
        elif family.ftype != 'grids' and family.status.get('available_at_origin') is not True:
            raise ValueError(f'Forward predictor family {family.ftype!r} must declare available_at_origin=True.')
    for slot in slots:
        allowed = target_axes_of(plain, slot)
        if allowed is not None and allowed != {time_axis}:
            raise ValueError(f'Forward target_axes for {slot!r} must select time axis {time_axis}.')
        if slot not in target_labels:
            raise ValueError(f'Forward slot {slot!r} needs its first time derivative in the token pool.')

    def is_predictor(symbolic):
        return all(not orders[time_axis] for _, orders in symbolic.data)

    fields, plans, domains = {}, {}, {}
    for index, key in enumerate(keys):
        trajectory = manager[key]
        f, domains[key] = _trajectory_fields(trajectory, manager.raw_fields(key), cfg, time_axis)
        fields[key] = f
        predictors = [term for term in selection_library(pool, f.D) if is_predictor(term)]
        if cfg.forward_space == 'pointwise' or not f.window_axes:
            plan = _point_plan(f, predictors, trajectory._domain, row_ok)
            plans.update({(key, slot): plan for slot in slots})
            continue

        options = {}
        if cfg.weak_window_products == 'pairs':
            options.update(products=[term for term in product_library(pool, f.D) if is_predictor(term)],
                           product_cache={})
        orders = tuple(int(d == time_axis) for d in range(f.D))
        for slot in slots:
            target = wf.WeakTerm([(slot, orders)], (0,) * f.D)
            plans[key, slot] = choose_slot_window(f, predictors + [target], [len(predictors)], key, slot,
                                                  index, row_ok, cfg.weak_window_gate, **options)
    rows = ForwardRows(fields, plans, keys, tokens, domains, target_labels, pool)
    rows.fit_frequencies = any(f.status.get('fit_frequencies') for f in pool.families)
    return rows
