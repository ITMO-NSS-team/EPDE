"""Fold-local weak measurements for terms proposed during an EPDE search."""
from functools import lru_cache
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent / 'variance_reduction_2026-11-16'),
                str(HERE.parent / 'implicit_objective_2026-11-18')]
import moments as M
from implicit import score
from epde.operators.common.objectives import LOSS_NAN_VAL

R, O, Q = M.R, M.O, M.Q
DRAWS = 128


def term_key(term):
    factors = []
    for f in term.structure:
        name = 'x' if f.ftype == 'grids' else f.label
        params = tuple(sorted((info['name'], float(f.params[i]))
                              for i, info in f.params_description.items()))
        factors.append((name, params))
    return tuple(sorted(factors))


class Window:
    def __init__(self, axes, orders, width):
        self.axes = axes
        self.q = [3] + [5] * (len(axes) - 1)
        k = 1
        while (2 * k + 1) ** len(axes) < width + 1:
            k += 1
        self.basis = M.RealBasis(k, len(axes))
        self.shape = (2 * k + 1,) * len(axes)
        self.kernels = []
        for i, (axis, order, q) in enumerate(zip(axes, orders, self.q)):
            cut = int(order) * R.WF.FD_REACH
            radius = (len(axis) - 2 * cut - 1) // 2
            if radius < q:
                raise ValueError(f'Insufficient safe quadrature nodes on axis {i}')
            freq = np.r_[np.arange(k + 1), -np.arange(1, k + 1)] * np.pi / (radius * (axis[1] - axis[0]))
            kernels, _ = Q.axis_weights(axis, cut, [cut + radius], radius, q, freq, q, 3 if i == 0 else 5)
            self.kernels.append(kernels)

    def integrate(self, arr, orders):
        # An optional leading dimension denotes independent bootstrap fields.
        offset = arr.ndim - len(self.axes)
        for i in sorted(range(len(self.axes)), key=lambda j: self.shape[j] / len(self.axes[j])):
            axis = offset + i
            arr = np.moveaxis(np.tensordot(self.kernels[i][orders[i]], arr, axes=(1, axis)), 0, axis)
        return arr.reshape((*arr.shape[:offset], -1))

    def real(self, moments):
        return self.basis.convert(moments[..., None])[..., 0]


class Measurements:
    def __init__(self, window, values, covariance=None):
        self.window = window
        self.values = values
        self.covariance = covariance
        self.offset = next(iter(values.values())).ndim - len(window.axes)
        self.derivatives = {}

    def field(self, variable, orders):
        key = variable, orders
        if key not in self.derivatives:
            if not any(orders):
                self.derivatives[key] = self.values[variable]
            else:
                i = max(i for i, p in enumerate(orders) if p)
                lower = list(orders)
                lower[i] -= 1
                self.derivatives[key] = R.WF._central4(self.field(variable, tuple(lower)),
                                                       self.window.axes[i], self.offset + i)
        return self.derivatives[key]

    def column(self, term, opaque):
        @lru_cache(None)
        def product(data):
            if not data:
                return 1.
            head, rest = data[0], data[1:]
            point = self.field(*head) * product(rest)
            if self.covariance is not None:
                for j, factor in enumerate(rest):
                    point = point - self.covariance.pair(head, factor) * product(rest[:j] + rest[j+1:])
            return point

        result = 0.
        for (data, coords, orders), weight in term.monomials(psi_cap=self.window.q).items():
            arr = product(data)
            for i, power in enumerate(coords):
                if power:
                    shape = [1] * len(coords)
                    shape[i] = -1
                    arr = arr * self.window.axes[i].reshape(shape) ** power
            for token in term.opaque:
                arr = arr * opaque[token]
            arr = np.broadcast_to(arr, next(iter(self.values.values())).shape)
            result = result + weight * self.window.integrate(arr, orders)
        return self.window.real(result)


class LiveScore:
    def __init__(self, problem, raw, seed, orders, width):
        self.problem = problem
        axes = R.axes_of(problem)
        indices = [np.arange(g, len(axes[0]), 3) for g in range(3)]
        self.segment = slice(max(idx[0] for idx in indices), min(idx[-1] for idx in indices) + 1)
        common = [axes[0][self.segment], *axes[1:]]
        self.window = Window(common, orders, width)
        self.folds = []
        self.unweighted_folds = []
        for g, idx in enumerate(indices):
            observed = {v: y[idx] for v, y in raw.items()}
            model = {v: R.WF.noise_model(y) for v, y in observed.items()}
            pilot, _ = R.reconstruct(observed, axes, idx)
            mapping = O.interpolation(axes[0], idx)[self.segment]
            cov = O.Covariance(common, mapping, model)
            mapped = O.map_values(mapping, observed)
            rng = np.random.default_rng(np.random.SeedSequence([1171, seed, g]))
            fields = {v: [y[self.segment]] for v, y in pilot.items()}
            noise_moments = []
            for _ in range(DRAWS):
                draw = {v: y[idx] + R.WF.ar_noise(rng, observed[v].shape, *model[v][:2]) for v, y in pilot.items()}
                fitted, _ = R.reconstruct(draw, axes, idx)
                for v, y in fitted.items():
                    fields[v].append(y[self.segment])
                noise_moments.append(self.field_moments(O.map_values(mapping, draw)))
            train = Measurements(self.window, {v: np.stack(ys) for v, ys in fields.items()})
            white, info = M.whitening(np.stack(noise_moments), self.field_moments({v: y[0] for v, y in train.values.items()}))
            if white is None:
                if any(m[0] != 0 for m in model.values()):
                    raise ValueError(f'Training fold {g} cannot whiten observations: {info}')
                # With zero estimated noise, the model supplies no metric on
                # moment space. Its numerical roundoff covariance is meaningless.
                white = np.eye(self.window.basis.size)
                self.unweighted_folds.append(g)
            self.folds.append((train, Measurements(self.window, mapped, cov), white))
        self.calls = [0, 0]
        self.columns = {}
        self.scores = {}
        self.unresolved = 0
        self.negative = 0

    def field_moments(self, values):
        zero = (0,) * len(self.window.axes)
        return np.column_stack([self.window.real(self.window.integrate(y, zero)) for y in values.values()])

    def column(self, label):
        if label not in self.columns:
            (term,), opaque = R.symbols(self.problem, [label])
            opaque = {k: y[self.segment] for k, y in opaque.items()}
            self.columns[label] = [(train.column(term, opaque), held.column(term, opaque))
                                   for train, held, _ in self.folds]
        return self.columns[label]

    def fold_fits(self, labels):
        columns = [self.column(label) for label in labels]
        for g, (_, _, white) in enumerate(self.folds):
            train = np.stack([col[g][0] for col in columns], axis=-1)
            pilot, draws = train[0], train[1:]
            norm = np.linalg.norm(pilot, axis=0)
            if np.any(norm == 0):
                return
            mean = draws.mean(axis=0)
            corrected = (2 * pilot - mean) / norm
            deviations = ((draws - mean) / norm).reshape(-1, len(columns))
            gram = corrected.T @ corrected - (1 + 1/DRAWS) / (DRAWS-1) * (deviations.T @ deviations)
            eig, vec = np.linalg.eigh((gram + gram.T) / 2)
            if eig[1] - eig[0] <= 64 * np.finfo(float).eps * len(eig) * max(abs(eig)):
                return
            left, right = [white @ np.column_stack([col[h][1] for col in columns]) for h in range(3) if h != g]
            yield left, right, np.linalg.norm(white @ pilot, axis=0), vec[:, 0] / norm

    def support_score(self, labels):
        scores = []
        for left, right, scales, vector in self.fold_fits(labels):
            value = score(left, right, scales, vector)
            if value is None or not np.isfinite(value):
                return None
            scores.append(value)
        return float(np.mean(scores)) if len(scores) == 3 else None

    def evaluate(self, equation, ctx):
        self.calls[int(ctx.for_rps)] += 1
        active = iter(equation.weights_final[:-1])
        labels = [term_key(t) for i, t in enumerate(equation.structure)
                  if i == equation.target_idx or next(active) != 0]
        if equation.weights_final[-1] != 0:
            labels.append(())
        key = tuple(sorted(labels))
        if len(key) < 2 or len(set(key)) != len(key):
            return LOSS_NAN_VAL
        if key not in self.scores:
            value = self.support_score(key)
            self.unresolved += int(value is None)
            self.negative += int(value is not None and value < 0)
            self.scores[key] = LOSS_NAN_VAL if value is None else value
        return self.scores[key]
