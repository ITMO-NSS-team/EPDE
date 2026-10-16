"""Chronological validation of an idealised ball-and-beam hypothesis.

This fixes the structure y'' = a * u_in in advance, fits a on training data,
and checks acceleration and open-loop forced position on the later interval.
It is not a claim that discovery on the full data has a held-out test.
"""
import numpy as np
from .problem import Problem, mesh
from .preprocessing import prepare_data


def ballbeam_validation(problem=None, train_fraction=.7, boundary=8):
    """Return arrays and metrics; no search or random split is performed.

    Differentiation is run separately on each segment (native EPDE FD), and
    boundary samples are discarded. Initial test position and velocity are
    measured locally at the test start; no coefficient uses test observations.
    The trajectory uses the measured test input, so it is a conditional forced
    response, not an autonomous or future-input forecast.
    """
    if problem is None:
        from .datasets import load
        problem = load('ballbeam')
    t = np.asarray(problem.grids[0]); y = np.asarray(problem.data['y'])
    u = np.asarray(problem.named_arrays['u_in'][0])
    boundary = int(boundary)
    if boundary < 2 or not 0 < train_fraction < 1 or t.ndim != 1 or not np.all(np.diff(t) > 0):
        raise ValueError('use increasing 1-D time, a split in (0,1), and boundary >= 2')
    split = int(t.size * train_fraction)
    if min(split, t.size-split) <= 2*boundary+2:
        raise ValueError('each chronological segment needs an interior after boundary exclusion')
    search = {'preprocessing': {'default_preprocessor_type': 'FD', 'max_deriv_order': [2]}}
    segments = []
    for lo, hi in [(0, split), (split, t.size)]:
        part = Problem('ballbeam_segment', problem.title, 'ode', problem.source,
                       mesh(t[lo:hi]), {'y': y[lo:hi]}, ('t',))
        _, stacks = prepare_data(part, part.data, search)
        inside = slice(boundary, hi-lo-boundary)
        segments.append((t[lo:hi][inside], y[lo:hi][inside], u[lo:hi][inside],
                         stacks['y'][inside, 0], stacks['y'][inside, 1]))
    tt, yy, uu, velocity, acceleration = segments[0]
    norm = float(uu @ uu)
    if norm <= np.finfo(float).eps:
        raise ValueError('training input has no excitation')
    coefficient = float(uu @ acceleration / norm)
    tt, yy, uu, velocity, acceleration = segments[1]
    predicted_acceleration = coefficient * uu
    # Trapezoidal forced integration with an observed initial state.
    predicted_velocity = np.r_[velocity[0], velocity[0] + np.cumsum(
        .5*(predicted_acceleration[:-1]+predicted_acceleration[1:])*np.diff(tt))]
    predicted_y = np.r_[yy[0], yy[0] + np.cumsum(
        .5*(predicted_velocity[:-1]+predicted_velocity[1:])*np.diff(tt))]
    def score(actual, predicted):
        error = actual-predicted
        denom = float(np.sum((actual-actual.mean())**2))
        return dict(rmse=float(np.sqrt(np.mean(error**2))),
                    r2=float(1-np.sum(error**2)/denom) if denom else None)
    return dict(coefficient=coefficient, split_index=split, train_stop=float(t[split-1]),
                test_start=float(t[split]), boundary=boundary, train_samples=len(segments[0][0]),
                test_samples=len(tt), acceleration=score(acceleration, predicted_acceleration),
                trajectory=score(yy, predicted_y), time=tt, observed_y=yy, predicted_y=predicted_y,
                observed_acceleration=acceleration, predicted_acceleration=predicted_acceleration)
