"""Data adapter for the DaISy flexible robot arm (KU Leuven system-ID
database, mechanical/robot_arm.dat: u = reaction torque of the structure,
y = accelerometer output, N = 1024). See configs/robot_arm.yaml.

UNKNOWN-TRUTH config: the official model is a ~5th-order flexible-modes
transfer function, but a 2nd-order fit y'' ~ {y, y', u} already reaches
R^2 = 0.991 -- what reduced ODE the search settles on is the question.
The absolute sampling period is not distributed with the file; dt = 0.01 s
is assumed (coefficients rescale with dt, structure does not).

Forcing u(t) enters as a CacheStoredTokens term (meaningful=True), hence
``TOKENS_NEED_SEARCH = True``.
"""

import os

import numpy as np

_DAT = os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..', '..', 'pic', 'data', 'daisy',
    'robot_arm.dat'
))

_LAST = {}

TOKENS_NEED_SEARCH = True


def load_data(dt=0.01):
    arr = np.loadtxt(_DAT)
    u, y = arr[:, 0].astype(np.float64), arr[:, 1].astype(np.float64)
    t = np.arange(y.size, dtype=np.float64) * float(dt)
    _LAST.update(u=u)
    return (t,), [y], ['y'], 0


def build_extra_tokens(coords, dim):
    from epde import CacheStoredTokens

    return [CacheStoredTokens(
        token_type='forcing',
        token_labels=['u_in'],
        token_tensors={'u_in': _LAST['u']},
        params_ranges={'power': (1, 1)}, params_equality_ranges=None,
        dimensionality=dim, meaningful=True)]
