"""Data adapter for the REAL double pendulum (unknown-truth config).

See configs/dp_real.yaml. Two variants, both wrapping the established
loaders in ``projects/pic/data/dp/`` rather than duplicating them:

* ``variant='encoder'`` (default): HardwareX MultiArm-Pendulum optical
  encoders @1 kHz, angles already in radians -- the high-SNR winner
  (signal-check R^2=0.998). Loader:
  ``MultiArm_Pendulum/dp_encoder_discovery.load_angles(matfile, start, n, step)``.
* ``variant='video'``: Trial video-tracked markers @500 Hz, DEGREES ->
  radians inside the loader. Loader:
  ``dp_real_discovery.load_angles(trial, start, n)``.

``build_extra_tokens`` injects the trig-of-state / coupling factors as
``CacheStoredTokens`` (the established dp scaffold): sin(th1), sin(th2)
as standalone-capable gravity terms and cos(D), sin(D) (D = th1 - th2)
as factor-only coupling pre-factors. CacheStoredTokens filters its
arrays by ``grid_cache.g_func`` at construction, so the EpdeSearch
object must exist first -- hence ``TOKENS_NEED_SEARCH = True`` (see
``thesis_runner.build_search``).
"""

import importlib.util
import os

import numpy as np

_PIC_DP = os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..', '..', 'pic', 'data', 'dp'
))

# build_extra_tokens needs the SAME angle arrays load_data returned (it
# receives only (coords, dim)); load_data stashes them here. build_search
# always calls load_data first, so the stash is never stale.
_LAST = {}

TOKENS_NEED_SEARCH = True


def _load_pic_module(rel_path, name):
    spec = importlib.util.spec_from_file_location(
        name, os.path.join(_PIC_DP, rel_path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_data(variant='encoder', start=None, n=None, step=4):
    if variant == 'encoder':
        loader = _load_pic_module(
            os.path.join('MultiArm_Pendulum', 'dp_encoder_discovery.py'),
            'pic_dp_encoder')
        kwargs = {'step': step}
        if start is not None:
            kwargs['start'] = start
        if n is not None:
            kwargs['n'] = n
        t, th1, th2 = loader.load_angles(**kwargs)
    elif variant == 'video':
        loader = _load_pic_module('dp_real_discovery.py', 'pic_dp_video')
        kwargs = {}
        if start is not None:
            kwargs['start'] = start
        if n is not None:
            kwargs['n'] = n
        t, th1, th2 = loader.load_angles(**kwargs)
    else:
        raise ValueError(f"variant must be 'encoder' or 'video', got {variant!r}")
    _LAST.update(t=t, th1=th1, th2=th2)
    return (t,), [th1, th2], ['theta1', 'theta2'], 0


def build_extra_tokens(coords, dim):
    from epde import CacheStoredTokens

    th1, th2 = _LAST['th1'], _LAST['th2']
    delta = th1 - th2
    # sin th1, sin th2: gravity terms, allowed to stand alone.
    trig_state = CacheStoredTokens(
        token_type='trig_state',
        token_labels=['sin_th1', 'sin_th2'],
        token_tensors={'sin_th1': np.sin(th1), 'sin_th2': np.sin(th2)},
        params_ranges={'power': (1, 1)}, params_equality_ranges=None,
        dimensionality=dim, meaningful=True)
    # cos D, sin D: coupling pre-factors, factor-only (meaningful=False)
    # so they only appear multiplied with a derivative.
    coupling = CacheStoredTokens(
        token_type='coupling',
        token_labels=['cos_delta', 'sin_delta'],
        token_tensors={'cos_delta': np.cos(delta), 'sin_delta': np.sin(delta)},
        params_ranges={'power': (1, 1)}, params_equality_ranges=None,
        dimensionality=dim, meaningful=False)
    return [trig_state, coupling]
