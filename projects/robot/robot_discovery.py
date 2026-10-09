"""EPDE discovery on the video-tracked cylindrical vibrobot (N=1, PWM=10, robot 1, exp 1).

Data: 25 fps track, x/y in image pixels (1-px quantised), heading in degrees
(wrapped to [0, 360)).

What the data supports (OLS signal check before any search):
  * translation is a random walk -- x', y' on cos/sin(phi) gives R^2 <= 0.02,
    flat velocity spectrum, MSD slope ~1.1. With no deterministic right part
    every x/y candidate collapses to ``x' = const`` and is flagged degenerate,
    so the initial population never fills. x and y are left out of the search.
  * the heading spins at a constant rate, phi' = omega ~ 3.43 rad/s. That law
    is ``target = intercept`` and is degenerate by construction too, so the
    search is posed on the heading vector (c, s) = (cos phi, sin phi), where
    constant spin is the linear rotation  c' = -omega s,  s' = omega c
    (OLS R^2 0.93 at the 15-frame window used here).
"""
import sys
import os
import glob

# insert, not append: a stale pip-installed epde would otherwise shadow the repo
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import numpy as np
from scipy.signal import savgol_filter

from epde.interface.interface import EpdeSearch
from epde import CacheStoredTokens

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_LEAD_IN = 7  # the robot sits still for the first frames of the video


def load(tag: str) -> np.ndarray:
    fname = glob.glob(os.path.join(DATA_DIR, f"data_{tag}_*.csv"))[0]
    raw = open(fname).read().strip().strip(',')
    return np.array([float(v) for v in raw.split(',') if v.strip() != ''])


def load_track():
    t, x, y, ang = (load(k) for k in ("time", "x", "y", "ang"))
    phi = np.unwrap(np.deg2rad(ang))
    sl = slice(STATIC_LEAD_IN, None)
    return t[sl] - t[STATIC_LEAD_IN], x[sl], y[sl], phi[sl]


def robot_discovery(popsize: int = 16, epochs: int = 20, window: int = 30,
                    max_deriv_order: int = 4, data_fun_pow: int = 3, terms: int = 10,
                    factors: dict = None, translation_tokens: bool = False):
    """``translation_tokens`` offers the smoothed lab-frame velocities x', y'
    as factor-only tokens, so heading equations can express a spin-translation
    coupling (phi' = omega + k v_par  =>  c' = -omega s - k s (c x' + s y'))
    without x and y becoming equations of their own."""
    t, x, y, phi = load_track()
    print(f"{len(t)} frames, {t[-1]:.1f} s; mean spin {np.polyfit(t, phi, 1)[0]:.3f} rad/s")

    search = EpdeSearch(multiobjective_mode=True, use_solver=False,
                        boundary_width=window,
                        default_preprocessor_type='poly',
                        preprocessor_kwargs={'polynomial_window': window, 'poly_order': 3},
                        population_size=popsize, training_epochs=epochs,
                        verbose_params={'show_iter_idx': True})
    _, domain = search.createDomain((t,), ID=0)
    extra_tokens = []
    if translation_tokens:
        # Built after EpdeSearch, which sets up the sample manager.
        dt = t[1] - t[0]
        vel = {'x_dot': savgol_filter(x, window, 3, deriv=1, delta=dt),
               'y_dot': savgol_filter(y, window, 3, deriv=1, delta=dt)}
        extra_tokens.append(CacheStoredTokens(token_type='translation', token_labels=list(vel),
                                              token_tensors=vel, params_ranges={'power': (1, 1)},
                                              params_equality_ranges=None, meaningful=False))
    _, trajectory = search.createTrajectory({'c': np.cos(phi), 's': np.sin(phi)}, domain, cache_id=0)
    # factors_num [1] by default: one factor per term. Factor pairs converged
    # to composite junk on every real dataset so far.
    if factors is None:
        factors = {'factors_num': [1, 2, 3], 'probas': [0.5, 0.3, 0.2]}
    search.fit(data=[trajectory], max_deriv_order=(max_deriv_order,), data_fun_pow=data_fun_pow,
               equation_terms_max_number=terms, equation_factors_max_number=factors,
               additional_tokens=extra_tokens)

    search.equations(only_print=True, num=1)
    return search


if __name__ == "__main__":
    robot_discovery()
