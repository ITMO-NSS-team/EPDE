"""Data adapter for the REAL Silverbox benchmark (electronic Duffing
oscillator; nonlinearbenchmark.org, fetched/cached via the official
``nonlinear_benchmarks`` package). See configs/silverbox.yaml.

Forced system: m*y'' + d*y' + a*y + b*y^3 = g*u(t). The input voltage u(t)
is injected as a CacheStoredTokens forcing term (meaningful=True), hence
``TOKENS_NEED_SEARCH = True``. Signal-check on SG(7,4) derivatives:
R^2 = 0.987 (train multisine) / 0.984 (arrow); the cubic term's marginal
contribution is dR^2 ~ 4e-4 -- REAL (coefficient stable across records)
but below the derivative-noise floor (fs is only ~10x the resonance), so
recovering y^3 is the hard part of this target.

records: 'train' (multisine, 65062 pts) or 'arrow' (growing amplitude,
32000 pts).
"""

import numpy as np

_LAST = {}

TOKENS_NEED_SEARCH = True


def load_data(record='train', start=2000, n=40000, step=1):
    import nonlinear_benchmarks as nb
    train, test = nb.Silverbox()
    rec = train if record == 'train' else test[2]
    u = np.asarray(rec.u, float).ravel()
    y = np.asarray(rec.y, float).ravel()
    dt = float(rec.sampling_time)
    sl = slice(start, min(start + n * step, y.size), step)
    u, y = u[sl], y[sl]
    t = np.arange(y.size, dtype=np.float64) * (dt * step)
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
