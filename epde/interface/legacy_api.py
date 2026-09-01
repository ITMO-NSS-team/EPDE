"""Backward compatibility for the pre-``domain_refactor`` calling convention.

37 scripts under ``projects/`` still build a search the old way::

    search = EpdeSearch(use_solver=False, boundary=15, coordinate_tensors=(t,))
    search.fit(data=[x, y], variable_names=['u', 'v'], max_deriv_order=(1,),
               data_fun_pow=1, equation_terms_max_number=7, ...)

(``max_deriv_order`` and the two powers survive that move unchanged -- they
describe the pool, not the sample, so they are still ``fit`` arguments.)

The domain refactor replaced that with explicit ``createDomain`` /
``createTrajectory`` objects, which is a better model -- a search can now carry
several domains and several trajectories -- but it left every one of those
scripts raising ``TypeError`` on the first line. This module translates the old
form into the new one so they keep running, under ``DeprecationWarning``, until
they are migrated.

Nothing here is a second implementation: every legacy argument is mapped onto
the real ``createDomain`` / ``createTrajectory`` / ``create_pool`` call and then
forgotten. Arguments that the current pipeline genuinely no longer has
(``fourier_layers``, ``data_nn``, ``prune_domain``, ...) are accepted and
ignored with a warning naming them, which is what they already did -- silently.
"""

import warnings

from epde.globals import EPDEDeprecationWarning

__all__ = ['LEGACY_INIT_KEYS', 'LEGACY_DATA_KEYS', 'REMOVED_KEYS',
           'split_legacy', 'warn_legacy', 'reject_removed']


#: Old ``EpdeSearch.__init__`` arguments describing the domain, plus the domain
#: pruning cluster that ``set_domain_properties`` used to own.
LEGACY_INIT_KEYS = frozenset((
    'coordinate_tensors', 'boundary', 'dimensionality', 'define_domain',
    'function_form', 'prune_domain', 'pivotal_tensor_label', 'pruner',
    'threshold', 'division_fractions', 'rectangular', 'eq_search_iter',
))

#: Old ``create_pool`` / ``fit`` arguments describing the data. These moved to
#: ``createTrajectory`` or to the config, or no longer exist.
#:
#: ``max_deriv_order``/``data_fun_pow``/``deriv_fun_pow`` are deliberately NOT
#: here: they describe the token pool rather than a data sample, so they stayed
#: real ``create_pool``/``fit`` parameters and need no translation.
LEGACY_DATA_KEYS = frozenset((
    'variable_names', 'derivs', 'data_nn', 'fourier_layers',
    'method', 'method_kwargs',
    'deriv_method', 'deriv_method_kwargs', 'quiet', 'coordinate_tensors',
    'boundary', 'memory_for_cache', 'prune_domain', 'division_fractions',
    'ann_epochs_max', 'fourier_params',
))

#: Arguments that no longer exist and have no automatic translation, mapped to
#: the message explaining what to write instead. These raise rather than warn:
#: silently guessing an equivalent would change what the search optimizes.
REMOVED_KEYS = {
    'use_pic': (
        "use_pic has been removed. It was a bool standing in for one choice -- "
        "which objective occupies MOEA/D's second Pareto axis -- so say that "
        "directly: second_objective='instability' (what use_pic=True meant) or "
        "second_objective='complexity' (what use_pic=False meant)."),
    'use_default_strategy': (
        'use_default_strategy has been removed; it carried no information that '
        '``director is None`` did not already carry. Pass director=<your '
        'director> to supply your own strategy, or omit it to build one from '
        'the configuration.'),
    'fitness_cls': (
        'fitness_cls has been removed. Choose the discrepancy with '
        "discrepancy_metric=..., and the solver with use_solver=True / "
        'solver_backend=...'),
}

#: Accepted, then ignored -- the pipeline has no such stage any more. Warned
#: about by name so a script author can see the setting is doing nothing.
_INERT_KEYS = frozenset((
    'fourier_layers', 'fourier_params', 'data_nn', 'ann_epochs_max',
    'dimensionality', 'define_domain', 'prune_domain', 'pivotal_tensor_label',
    'pruner', 'threshold', 'division_fractions', 'rectangular', 'quiet',
    'eq_search_iter', 'memory_for_cache',
))


def reject_removed(kwargs: dict) -> None:
    """Raise a directed ``TypeError`` for arguments that were removed outright."""
    for key, message in REMOVED_KEYS.items():
        if key in kwargs:
            raise TypeError(message)


def split_legacy(kwargs: dict, keys) -> tuple:
    """Split ``kwargs`` into (recognised legacy arguments, everything else)."""
    legacy = {key: value for key, value in kwargs.items() if key in keys}
    rest = {key: value for key, value in kwargs.items() if key not in keys}
    return legacy, rest


def warn_legacy(where: str, legacy: dict, replacement: str,
                stacklevel: int = 3) -> None:
    """Emit one ``DeprecationWarning`` naming the arguments and the new form.

    ``stacklevel`` must land on the USER's line, not on an EPDE frame:
    DeprecationWarning is hidden by default everywhere except ``__main__``, so
    a warning attributed to ``interface.py`` is invisible to exactly the people
    who need to see it. 3 is right for a warning raised directly by the public
    method; pass 4 when there is a helper in between.

    The category is ``EPDEDeprecationWarning`` (a DeprecationWarning subclass,
    so ``pytest.warns(DeprecationWarning)`` and the usual filters still catch
    it) because ``globals.init_verbose`` installs a process-wide
    ``filterwarnings('ignore')`` unless ``show_warnings=True`` -- constructing
    a search would otherwise silence this whole channel.
    """
    if not legacy:
        return
    used = sorted(legacy)
    inert = sorted(set(used) & _INERT_KEYS)
    message = ('{0}: {1} {2} the pre-domain_refactor API. {3}'.format(
        where, ', '.join(used),
        'is' if len(used) == 1 else 'are', replacement))
    if inert:
        message += (' Note that {0} no longer affect the search at all and '
                    'are ignored.'.format(', '.join(inert)))
    warnings.warn(message, EPDEDeprecationWarning, stacklevel=stacklevel)
