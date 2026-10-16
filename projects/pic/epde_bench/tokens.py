"""Token families that cannot be declared in a YAML config.

Declarative families (trigonometric, grid, ...) live in ``configs/*.yaml``
under ``search.search_space.tokens``. The ones here need an array or a
callable: arrays come from ``Problem.token_groups`` (:func:`stored_groups`),
callables from a loader's ``extra_tokens`` (:func:`cos_t_sin_x`). Both are
built after ``createDomain`` (CacheStoredTokens requires an initialised
search).
"""

from typing import Dict

import numpy as np


def _cos_t_sin_x(*grids, **kwargs):
    return (np.cos(grids[0]) * np.sin(grids[1])) ** kwargs['power']


def cos_t_sin_x(problem) -> list:
    """The single product token ``cos(t)sin(x)``: the source term of the
    forced KdV data set (``kdv/data.csv``)."""
    from epde.interface.prepared_tokens import CustomEvaluator, CustomTokens
    evaluator = CustomEvaluator({'cos(t)sin(x)': _cos_t_sin_x},
                                eval_fun_params_labels=['power'])
    return [CustomTokens(token_type='trigonometric', token_labels=['cos(t)sin(x)'],
                         evaluator=evaluator, params_ranges={'power': (1, 1)},
                         params_equality_ranges={}, meaningful=True,
                         unique_token_type=False, dimensionality=problem.dim)]


def stored_groups(problem) -> list:
    """One ``CacheStoredTokens`` family per ``problem.token_groups`` entry."""
    from epde import CacheStoredTokens
    return [CacheStoredTokens(token_type=token_type, token_labels=list(tensors),
                              token_tensors=dict(tensors), params_ranges={'power': (1, 1)},
                              params_equality_ranges=None, dimensionality=problem.dim,
                              meaningful=meaningful)
            for token_type, tensors, meaningful in problem.token_groups]


# ---------------------------------------------------------------------------
# sin/cos at ONE fixed frequency
# ---------------------------------------------------------------------------
#
# Why not TrigonometricTokens(freq=(2 - 1e-8, 2 + 1e-8)), as every script in
# data/ does? The library's frequency-equality tolerance is 5 %
# of the declared interval (prepared_tokens.py, "freq_equality_fraction").
# For a deliberately narrow interval that splits ONE function into 20
# "different" tokens -- sin(2.000000004 t) and sin(1.999999999 t) -- whose
# columns are numerically identical, so the search assembles exact identities
# such as  a*u'sin(2t) + b*u'sin(2t) = u'sin(2t)  with zero error and zero
# instability, and they dominate the front (seen on `ode`, seed 0). Pinning
# the frequency to a single value keeps one bucket and the same text form
# (``sin{power: 1.0, freq: 2.0, dim: 0.0}``), so truths and metrics are
# unchanged. The library behaviour is kept measurable as the
# ``trig_native`` variant.

def fixed_trigonometric(freq: float = 2.0, dimensionality: int = 0, meaningful: bool = False):
    from collections import OrderedDict
    from epde.evaluators import trigonometric_evaluator
    from epde.interface.prepared_tokens import PreparedTokens
    from epde.interface.token_family import TokenFamily

    class FixedTrigonometricTokens(PreparedTokens):
        def __init__(self):
            self._token_family = TokenFamily(token_type='trigonometric')
            self._token_family.set_status(unique_specific_token=True, unique_token_type=True,
                                          meaningful=meaningful)
            params = OrderedDict([('power', (1, 1)), ('freq', (float(freq), float(freq))),
                                  ('dim', (0, dimensionality))])
            # any positive tolerance puts the single admissible value in bucket 0
            self._token_family.set_params(['sin', 'cos'], params,
                                          {'power': 0, 'freq': 1e-3, 'dim': 0})
            self._token_family.set_evaluator(trigonometric_evaluator)

    return FixedTrigonometricTokens()


#: Token families declared in YAML that EPDE's own registry does not know;
#: ``config.resolve_for_problem`` builds them and passes them via
#: ``additional_tokens``.
BENCH_FAMILIES = {'fixed_trigonometric': fixed_trigonometric}
