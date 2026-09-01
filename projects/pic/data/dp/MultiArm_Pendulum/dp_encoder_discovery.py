"""EPDE discovery of the REAL double-pendulum EOM from the HIGH-SNR encoder data
(Brunton/HardwareX MultiArm-Pendulum, DoubleDataFreeSwing_1, optical encoders, 1 kHz).

Drop-in higher-SNR replacement for dp/dp_real_discovery.py (video markers). The OLS
signal-check (MultiArm_Pendulum/signal_check.py) confirmed the coupled Lagrangian EOM is
recoverable here at R^2=0.998 with the structural tells crisply satisfied -- a much cleaner
target than the video data (~0.95). Angles are already in RADIANS (no deg2rad), and the
encoder logs angular velocity directly, so derivative estimation only needs th'' (1 deriv).

Target coupled EOM (absolute angles, D = th1 - th2):
  th1'' = -A cos(D) th2'' - A sin(D) th2'^2 - (g/l1) sin th1
  th2'' = -B cos(D) th1'' + B sin(D) th1'^2 - (g/l2) sin th2

Exotic factors injected as CacheStoredTokens (same scaffold as dp_real_discovery.py):
  - trig-of-state {sin th1, sin th2}  (meaningful=True  -> gravity term, may stand alone)
  - coupling      {cos D, sin D}      (meaningful=False -> must multiply a derivative)
With max_deriv_order=(2,), deriv_fun_pow=2 (th'^2), factors_num=[1,2] EPDE can assemble
cos(D)*th2'' and sin(D)*th2'^2 -- all three describe the token pool, so they are stated on
fit(), not on the trajectory. Register the domain with createDomain FIRST: the cache tokens
need the grid cache populated, which is what createDomain does.
"""
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

import numpy as np
import scipy.io as sio

from epde.interface.interface import EpdeSearch
from epde import CacheStoredTokens

HERE = os.path.dirname(__file__)


def load_angles(matfile='Double_FreeSwing_1.mat', start=4000, n=6000, step=4):
    """Encoder link angles -> (t, theta1, theta2) in RADIANS. Decimate by ``step`` (the
    1 kHz raw is heavily oversampled for a ~1-2 Hz pendulum); take ``n`` decimated points
    from ``start`` (skip the release transient)."""
    m = sio.loadmat(os.path.join(HERE, matfile))
    th1 = m['Theta1'].ravel().astype(np.float64)
    th2 = m['Theta2'].ravel().astype(np.float64)
    t = m['Time'].ravel().astype(np.float64)
    sl = slice(start, start + n * step, step)
    t, th1, th2 = t[sl], th1[sl], th2[sl]
    t = t - t[0]
    return t, th1, th2


def dp_discovery(start=4000, n=6000, step=4, poly_window=51, poly_order=4,
                 boundary=60, pop=30, epochs=20, terms=10):
    import torch
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    t, th1, th2 = load_angles(start=start, n=n, step=step)
    dt = float(np.median(np.diff(t)))
    print(f"[dp encoder] pts={t.size} dt={dt:.4f}s (fs~{1/dt:.0f}Hz) "
          f"th1[{th1.min():.2f},{th1.max():.2f}] th2[{th2.min():.2f},{th2.max():.2f}] rad")
    dim = th1.ndim - 1
    D = th1 - th2

    epde = EpdeSearch(use_solver=False, multiobjective_mode=True,
                      verbose_params={'show_iter_idx': True}, device=device)
    _, domain = epde.createDomain((t,), boundary_width=boundary, ID=0)
    epde.set_preprocessor(default_preprocessor_type='poly',
                          preprocessor_kwargs={'polynomial_window': poly_window,
                                               'poly_order': poly_order})

    trig_state = CacheStoredTokens(token_type='trig_state',
                                   token_labels=['sin_th1', 'sin_th2'],
                                   token_tensors={'sin_th1': np.sin(th1), 'sin_th2': np.sin(th2)},
                                   params_ranges={'power': (1, 1)}, params_equality_ranges=None,
                                   dimensionality=dim, meaningful=True)
    coupling = CacheStoredTokens(token_type='coupling',
                                 token_labels=['cos_delta', 'sin_delta'],
                                 token_tensors={'cos_delta': np.cos(D), 'sin_delta': np.sin(D)},
                                 params_ranges={'power': (1, 1)}, params_equality_ranges=None,
                                 dimensionality=dim, meaningful=False)

    epde.set_moeadd_params(population_size=pop, training_epochs=epochs)
    factors_max_number = {'factors_num': [1, 2], 'probas': [0.5, 0.5]}

    _, trajectory = epde.createTrajectory({'theta1': th1, 'theta2': th2}, domain, cache_id=0)
    epde.fit(data=[trajectory], max_deriv_order=(2,), data_fun_pow=1, deriv_fun_pow=2, equation_terms_max_number=terms,
             additional_tokens=[trig_state, coupling],
             equation_factors_max_number=factors_max_number)

    print("\n================ discovered system (encoder DP) ================")
    epde.equations(only_print=True, num=1)
    return epde


if __name__ == "__main__":
    import torch
    print('cuda', torch.cuda.is_available())
    # Parsimony: the true coupled EOM is 4 terms (target d^2th1 + cos(D)th2'' +
    # sin(D)th2'^2 + sin th1). terms=10 left 6 spurious slots for bare-angle /
    # constant / composite-target overfits, so cap tight.
    terms = int(os.environ.get('DP_TERMS', '5'))
    epochs = int(os.environ.get('DP_EPOCHS', '25'))
    dp_discovery(terms=terms, epochs=epochs)
