"""Fit a shared angular frequency on each encountered single-tone support."""
from copy import deepcopy
from itertools import groupby

import numpy as np
from scipy.optimize import minimize_scalar

from epde.operators.common import plain_fit
from epde.operators.common.frequency_fit import ToneColumn, _FrequencyBlock


def tones(term):
    return [factor for factor in term.structure if factor.label in ('sin', 'cos')]


def term_key(term):
    factors = []
    for factor in term.structure:
        moving = factor.label in ('sin', 'cos')
        params = tuple((desc['name'], float(factor.params[i])) for i, desc in factor.params_description.items()
                       if not (moving and desc['name'] == 'freq'))
        factors.append((factor.ftype, factor.label, params))
    return tuple(sorted(factors))


def bounds_minimum(loss, seed, bracket):
    """Reproduce the single bounded solve used by the per_support baseline."""
    lo, hi, width = bracket
    evaluated = {}
    def score(value):
        value = float(value)
        if value not in evaluated:
            evaluated[value] = loss(value)
        return evaluated[value]
    for value in (seed, lo, hi):
        score(value)
    optimum = minimize_scalar(score, bounds=(lo, hi), method='bounded',
                              options={'xatol': width*1e-4, 'maxiter': 24})
    score(optimum.x)
    valid = [w for w, cost in evaluated.items() if np.isfinite(cost)]
    best = min(valid, key=lambda w: (evaluated[w], abs(w-seed))) if valid else None
    return best, len(evaluated)


def band_minima(loss, seed, bracket):
    """Refine each minimum resolved on the spectral band's mesh, including edges."""
    lo, hi, width = bracket
    grid = np.unique(np.r_[np.linspace(lo, hi, 17), seed])
    evaluated = {}

    def score(omega):
        omega = float(omega)
        if omega not in evaluated:
            value = loss(omega)
            evaluated[omega] = float(value) if np.isfinite(value) else np.inf
        return evaluated[omega]

    costs = [score(w) for w in grid]
    minima = []
    for value, group in groupby(range(len(grid)), key=costs.__getitem__):
        indices = list(group)
        left, right = indices[0], indices[-1]
        before = costs[left-1] if left else np.inf
        after = costs[right+1] if right+1 < len(grid) else np.inf
        if not np.isfinite(value) or value > min(before, after):
            continue
        candidates = [grid[i] for i in indices]
        if len(indices) != len(grid):
            bounds = (grid[max(left-1, 0)], grid[min(right+1, len(grid)-1)])
            fit = minimize_scalar(score, bounds=bounds, method='bounded', options={'xatol': width*1e-6, 'maxiter': 32})
            candidates.append(fit.x)
        minima.append(float(min(candidates, key=lambda w: (score(w), abs(w-seed)))))
    return minima, len(evaluated)


def fitted_versions(eq, omega, args, intercept):
    variant = deepcopy(eq)
    for term in variant.structure:
        for factor in tones(term):
            # Inherited factors must retain the fitted frequency when regenerated.
            index = next(i for i, desc in factor.params_description.items() if desc['name'] == 'freq')
            desc = dict(factor.params_description)
            desc[index] = dict(desc[index], grid=(omega,))
            factor.params_description = desc
            factor.set_param(omega, name='freq')
        term.resetSavedState()
    variant.reset_state(reset_right_part=False)
    variant._invalidate_label_cache()
    if len(variant.terms_labels) != len(variant.structure):
        return []
    return plain_fit._linear_versions(variant, *args, intercept=intercept)


class SupportProfiles:
    """Fit every observed band once per slot, target, predictor support and intercept."""
    def __init__(self, ranges, point, *, multistart=False):
        self.ranges = ranges
        self.point = point
        self.multistart = multistart
        self.profiles = []
        self.fitted = {}
        self.columns = {}
        self.epoch = 0
        self.cache_hits = 0

    def loss(self, eq, ordered, intercept):
        rows = plain_fit.current_rows()
        slot = eq.main_var_to_explain
        target = rows.columns(eq.target, slot)
        samples, weighted = [], []
        fixed = [None if tones(term) else rows.columns(term, slot) for term in ordered]
        for record in sorted(target):
            y = target[record].ravel()
            weights = np.full(len(y), 1 / np.sqrt(len(y)))
            columns = []
            for term, static in zip(ordered, fixed):
                if static is not None:
                    columns.append((static[record], None))
                    continue
                key = (slot, record, term_key(term))
                if key not in self.columns:
                    factors = tones(term)
                    column = ToneColumn(rows, record, slot, term, tones=factors)
                    self.columns[key] = (column, np.zeros(len(factors), dtype=int))
                columns.append(self.columns[key])
            samples.append((y, weights, columns))
            weighted.append(y * weights)
        block = _FrequencyBlock(samples, np.concatenate(weighted), intercept, lambda: None)
        def score(omega):
            residual, full_rank = block.residual(np.array([omega]))
            return float(residual @ residual) if full_rank else np.inf
        return score

    def fit_bands(self, eq, ordered, intercept, axis, args, source):
        loss = self.loss(eq, ordered, intercept)
        fits = []
        for (dimension, seed), bracket in self.ranges.items():
            if dimension != axis:
                continue
            if self.multistart:
                frequencies, evaluations = band_minima(loss, seed, bracket)
                details = dict(proposals=frequencies)
            else:
                omega, evaluations = bounds_minimum(loss, seed, bracket)
                frequencies = [] if omega is None else [omega]
                details = dict(proposed=omega)
            children = []
            for frequency in frequencies:
                for variant in fitted_versions(eq, frequency, args, intercept):
                    pid = self.point(variant)
                    children.append(pid)
                    fits.append((pid, variant))
            self.profiles.append(dict(slot=eq.main_var_to_explain, point=source, epoch=self.epoch,
                                      axis=axis, initial=seed, lower=bracket[0], upper=bracket[1],
                                      fits=evaluations, fitted_points=children, **details))
        return fits

    def refine(self, seeds, args):
        # Retain the original fits as well as all valid refinements. The plain
        # fitter assembles their actual system combinations before Pareto selection.
        sources = set()
        extra = {}
        for eq in seeds:
            if tones(eq.target):
                continue
            predictors = [t for i, t in enumerate(eq.structure) if i != eq.target_idx]
            coordinates = {(int(f.param('dim')), float(f.param('freq'))) for term in predictors for f in tones(term)}
            if len(coordinates) != 1:
                continue
            axis, _ = next(iter(coordinates))
            ordered = sorted(predictors, key=term_key)
            intercept = bool(eq.weights_final[-1])
            key = (eq.main_var_to_explain, term_key(eq.target), tuple(map(term_key, ordered)), intercept)
            source = self.point(eq)
            sources.add(source)
            if key not in self.fitted:
                self.fitted[key] = self.fit_bands(eq, ordered, intercept, axis, args, source)
            else:
                self.cache_hits += 1
            for pid, stored in self.fitted[key]:
                if pid not in sources and pid not in extra:
                    variant = deepcopy(stored)
                    variant.metaparameters = deepcopy(eq.metaparameters)
                    variant._plain_genes = deepcopy(eq._plain_genes)
                    extra[pid] = variant
        return [*seeds, *(variant for pid, variant in extra.items() if pid not in sources)]
