"""EPDE discovery of the REAL double-pendulum equations of motion.

Data: HardwareX low-cost fixed-pivot double pendulum (Mendeley 7yd2ntbh3w), Trial1
video-tracked link angles (DEGREES -> radians), 500 Hz. The OLS signal-check
(dp_real_signal_check.py) confirmed the classic Lagrangian EOM is recoverable here at
R^2~0.95-0.99 (unlike SST), so this is a genuine target.

Target coupled EOM (absolute angles, D = th1 - th2):
  th1'' = -A cos(D) th2'' - A sin(D) th2'^2 - (g/l1) sin th1
  th2'' = -B cos(D) th1'' + B sin(D) th1'^2 - (g/l2) sin th2

The exotic factors are injected as CacheStoredTokens (precomputed feature arrays):
  - trig-of-state  {sin th1, sin th2}     (meaningful -> can stand alone, the gravity term)
  - coupling       {cos D, sin D}         (factor-only -> must multiply a derivative)
With max_deriv_order=(2,) (th', th''), deriv_fun_pow=2 (enables th'^2), and
factors_num=[1,2], EPDE can assemble cos(D)*th2'' and sin(D)*th2'^2.

NOTE on ordering: EpdeSearch.__init__ populates global_var.grid_cache.g_func (via
set_domain_properties), and CacheStoredTokens filters its arrays by that mask at
construction -> build the search object FIRST, then the cache tokens, then fit.
Poly (Chebyshev) preprocessor window ~0.1s = 51 pts (signal-check sweet spot; the
EPDE default of 9 is far too short at 500 Hz).
"""
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import numpy as np

from epde.interface.interface import EpdeSearch
from epde import CacheStoredTokens

BASE = os.path.join(os.path.dirname(__file__), 'Video_Tracking_Data', 'Video_Tracking_Data')


def load_angles(trial='Trial1', start=2000, n=8000):
    """Video-tracked link angles -> (t, theta1, theta2) in RADIANS (raw are degrees)."""
    d = os.path.join(BASE, trial)
    rb0 = np.load(os.path.join(d, 'DPmean_data_RB0.npy'))   # [t, phi1 upper]
    rb1 = np.load(os.path.join(d, 'DPmean_data_RB1.npy'))   # [t, phi2 lower]
    sl = slice(start, start + n)
    t = rb0[0][sl].astype(np.float64)
    th1 = np.deg2rad(rb0[1][sl].astype(np.float64))
    th2 = np.deg2rad(rb1[1][sl].astype(np.float64))
    t = t - t[0]
    return t, th1, th2


def dp_discovery(trial='Trial1', start=2000, n=8000, poly_window=51, poly_order=4,
                 boundary=60, pop=40, epochs=20, terms=10):
    import torch
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    t, th1, th2 = load_angles(trial, start, n)
    dt = float(np.median(np.diff(t)))
    print(f"[dp discovery] {trial} pts={t.size} dt={dt:.4f}s (fs~{1/dt:.0f}Hz) "
          f"th1[{th1.min():.2f},{th1.max():.2f}] th2[{th2.min():.2f},{th2.max():.2f}] rad")
    dim = th1.ndim - 1                       # 0 for a 1-D ODE time series
    D = th1 - th2

    # Build the search object FIRST so grid_cache.g_func exists for the cache tokens.
    epde = EpdeSearch(use_solver=False, multiobjective_mode=True, use_pic=True,
                      boundary=boundary, coordinate_tensors=(t,),
                      verbose_params={'show_iter_idx': True}, device=device)
    epde.set_preprocessor(default_preprocessor_type='poly',
                          preprocessor_kwargs={'polynomial_window': poly_window,
                                               'poly_order': poly_order})

    # sin th1, sin th2: gravity terms, allowed to stand alone (meaningful=True).
    trig_state = CacheStoredTokens(token_type='trig_state',
                                   token_labels=['sin_th1', 'sin_th2'],
                                   token_tensors={'sin_th1': np.sin(th1), 'sin_th2': np.sin(th2)},
                                   params_ranges={'power': (1, 1)}, params_equality_ranges=None,
                                   dimensionality=dim, meaningful=True)
    # cos D, sin D: coupling pre-factors, factor-only (meaningful=False) so they only
    # appear multiplied with a derivative -> cos(D)*th2'', sin(D)*th2'^2.
    coupling = CacheStoredTokens(token_type='coupling',
                                 token_labels=['cos_delta', 'sin_delta'],
                                 token_tensors={'cos_delta': np.cos(D), 'sin_delta': np.sin(D)},
                                 params_ranges={'power': (1, 1)}, params_equality_ranges=None,
                                 dimensionality=dim, meaningful=False)

    epde.set_moeadd_params(population_size=pop, training_epochs=epochs)
    factors_max_number = {'factors_num': [1, 2], 'probas': [0.5, 0.5]}

    epde.fit(data=[th1, th2], variable_names=['theta1', 'theta2'],
             max_deriv_order=(2,), equation_terms_max_number=terms,
             data_fun_pow=1, deriv_fun_pow=2,
             additional_tokens=[trig_state, coupling],
             equation_factors_max_number=factors_max_number,
             eq_sparsity_interval=(1e-8, 1e-0))

    print("\n================ discovered system ================")
    epde.equations(only_print=True, num=1)
    return epde


if __name__ == "__main__":
    import torch
    print('cuda', torch.cuda.is_available())
    dp_discovery()
