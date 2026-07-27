"""Data adapter for the REAL single pendulum (HardwareX MultiArm rig,
optical encoder, Simulink-struct logging @10 kHz). See configs/pend_single.yaml.

Small-amplitude free swing: the raw angle sits at theta ~ pi +/- 0.07
(hanging equilibrium). The loader CENTERS the angle (phi = theta - pi):
the uncentered theta is a quasi-constant token (+/-2% around pi), and any
product term X*theta ~ pi*X manufactures a near-copy of the target -- the
coordinate-modulation degeneracy fired live on the first run (discovered
``0.3185*theta''*theta + ... = theta''``, i.e. theta''=theta'' in
disguise, instability exactly 0). Centering removes the bait.

At this amplitude sin(phi) == -phi to 7e-5 relative, so no trig-of-state
token is injected (it would only add a perfectly collinear twin of phi);
the honest target is the damped LINEAR oscillator
``phi'' = a*phi + b*phi'`` with a ~ -64.8 (=-g/l), b ~ -0.65 -- the
signal-check ceiling is R^2 = 0.996.
"""

import os

import numpy as np
import scipy.io as sio

_MAT = os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..', '..', 'pic', 'data', 'dp',
    'MultiArm_Pendulum', 'Single_FreeSwing_1.mat'
))


def load_data(start=4000, n=5000, step=40):
    m = sio.loadmat(_MAT, squeeze_me=True, struct_as_record=False)
    s = m['Theta1']
    t = np.asarray(s.time, float).ravel()
    th = np.asarray(s.signals.values, float).ravel()
    sl = slice(start, start + n * step, step)
    t, th = t[sl], th[sl]
    t = t - t[0]
    phi = th - np.pi          # deviation from the hanging equilibrium
    return (t,), [phi], ['theta'], 0
