"""Grouped, file-loadable search configuration for :class:`EpdeSearch`.

The pre-config interface spread its defaults across ``__init__``,
``set_preprocessor``, ``set_moeadd_params``, ``set_singleobjective_params``,
``createDomain``, ``createTrajectory``, ``create_pool``, ``fit`` and
``predict``, so one setting could be spelled differently at different entry
points and every script restated the defaults. This module gathers them into
one grouped declaration, with an optional user config and explicit kwargs on
top. The per-operator parameters that used to live in
``operators/utils/parameters/*.json`` are here too, as
:data:`MULTI_OBJECTIVE_OPERATORS` / :data:`SINGLE_OBJECTIVE_OPERATORS`.

Precedence, strictly::

    the dataclass defaults below  <  user config  <  explicit kwargs

**The dataclasses are the only place a default is written.** They used to
document values that were really supplied by a shipped JSON file, which meant
editing a field here changed nothing at all -- ``pinn_loss_mult`` was set to
0.0 in this module while every run kept reading 1e4 from the JSON, with the
whole unit suite green (the guard compared key *sets*, not values). The file
is gone; a caller who wants settings in a file passes one as ``config=``.

Groups are by *concern*, not by which method consumes them:

``domain``
    Geometry of the sampled region.
``preprocessing``
    How the data is smoothed and differentiated -- including
    ``max_deriv_order``, which is what ``preprocesser.run`` computes.
``search_space``
    What the search may build: token families and their power ranges, the
    term/factor budget, the admissibility guard. (The sparsity interval is
    NOT here -- it seeds one operator's metaparameter and belongs to that
    operator; see ``LASSOSparsity.initial_sparsity_interval``.)
``objectives``
    How many Pareto axes there are, and how a candidate is fitted and scored.
``solver``
    Everything the PDE solver needs, ``device`` included -- the GPU is only
    ever used there.
``evolution``
    The evolutionary engine.
``runtime``
    Process-level bookkeeping.

A kwarg always wins, whatever group its key lives in, so
``EpdeSearch(use_solver=True)`` enables the solver without the caller needing
to know that ``use_solver`` is filed under ``solver``. That flat-to-nested
bridge is :data:`KEY_GROUP`, derived from the group dataclasses at import
time so it cannot drift from them.

Not configurable here -- these need an object, a tensor or a callable, and are
passed at the call site: ``gfunction``, ``nds_method``, ``ndl_update_method``,
``early_stopping_callback``, ``director``, ``preprocessor_pipeline``,
``derivs``, ``pool``, ``population``, ``net``, and the data-bound token
families (see :data:`TOKEN_REGISTRY`).
"""

import copy
import json
import os
from dataclasses import MISSING, dataclass, field, fields, replace
from numbers import Integral
from typing import Any, Dict, Optional, Union

__all__ = ['UNSET', 'SearchConfig', 'load_search_config', 'KEY_GROUP',
           'collect_overrides', 'build_tokens', 'resolve_sparsity',
           'sparsity_settings', 'validate_sparsity_kwargs',
           'TOKEN_REGISTRY', 'SPARSITY_REGISTRY',
           'GROUP_CLASSES', 'METRIC_MENUS', 'METRIC_ALIASES',
           'MULTI_OBJECTIVE_OPERATORS', 'SINGLE_OBJECTIVE_OPERATORS',
           'default_device',
           'active_config', 'set_active_config', 'reset_active_config']


# ---------------------------------------------------------------------------
# The UNSET sentinel
# ---------------------------------------------------------------------------

class _UnsetType:
    """Marker for "the caller did not pass this argument".

    A plain ``None`` default cannot serve here: ``None`` is a *meaningful*
    value for several keys (``fourier_params`` and every objective metric,
    where it means "leave the process global alone"). An
    explicit ``discrepancy_metric=None`` must therefore beat a config file's
    ``"l2"``, which it can only do if "not passed" is a distinct third state.
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self):
        return 'UNSET'

    def __bool__(self):
        return False

    def __copy__(self):
        return self

    def __deepcopy__(self, memo):
        return self

    def __reduce__(self):
        return (_UnsetType, ())


UNSET = _UnsetType()


# ---------------------------------------------------------------------------
# Group dataclasses
# ---------------------------------------------------------------------------
# THE defaults. Written out as dataclasses so a typo in ``cfg.solver.devcie``
# fails at the attribute instead of silently reading a dict, so the available
# settings stay greppable, and -- since there is nowhere else a value can come
# from -- so that editing one here actually changes what a search does.
#
# Container-valued fields need ``default_factory``; a bare ``= {}`` is a shared
# mutable default. These were exactly the fields the deleted JSON was really
# supplying, because the dataclass could only write ``None`` for them.

@dataclass(frozen=True)
class DomainConfig:
    """Geometry of the sampled region.

    ``boundary_width`` is used twice inside ``createDomain`` -- for
    ``Domain.setBoundaries`` and for the ``BoundaryExclusion`` g-function width
    -- deliberately: they are one concept and must agree. ``'auto'`` (the
    default) excludes, per axis, exactly the samples the preprocessor's edge
    error reaches (``PreprocessingPipe.edge_reach``: the derivative order for
    finite differences, the polynomial window's one-sided band for 'poly',
    nothing on a periodic spectral axis, ...); it is resolved once, when the
    pool is first built (``create_pool`` / ``fit``), because only then are the
    preprocessor and ``max_deriv_order`` known.

    ``evaluation_stride`` is the opt-in row budget: the search scores every
    candidate on a regular sub-lattice of the interior box (every ``k``-th
    point per axis, an int strides every axis; see
    :func:`epde.structure.domain.resolve_evaluation_stride`) instead of on
    every interior point. Each axis is thinned only by its own stride, so time
    and space are never traded against each other. Derivatives are still
    computed on the full grid, so a kept point carries exactly the value it has
    without the budget. It does not work with the solver.
    The budget applies per trajectory, and the weights stay point values: a
    fit that stacks several trajectories' rows (the subset screen, the legacy
    LASSO and its refit) counts each by its kept rows, so trajectories thinned
    by different strides count differently than they would unthinned. Fits
    made per trajectory and then averaged do not.

    ``evaluation_form`` selects the rows the solver-free search fits and scores on.
    - ``'strong'`` (the default) uses pointwise values on the interior box.
    - ``'weak'`` uses window averages against compact test functions, after
      integration by parts (:mod:`epde.preprocessing.weak_form`). Each equation
      slot gets one window per trajectory, chosen once per fit from the data, and
      every candidate of the slot is fitted and scored on that window's rows.
      Weak rows are computed from the RAW fields, so the preprocessor's smoothing
      does not enter them. They work neither with the solver nor with
      ``evaluation_stride`` (weak picks its own lattice), and they do not work
      with ``instability_metric='vcoef'``, which reads pointwise fields.
    """
    boundary_width: Union[str, int, list, tuple] = 'auto'
    time_axis: int = 0
    evaluation_stride: Union[None, int, list, tuple] = None
    evaluation_form: str = 'strong'


@dataclass(frozen=True)
class PreprocessingConfig:
    """How the data is smoothed and differentiated.

    ``max_deriv_order`` lives here rather than in ``search_space``: it is what
    ``preprocesser.run`` actually computes.
    """
    default_preprocessor_type: str = 'poly'
    preprocessor_kwargs: dict = field(default_factory=dict)
    max_deriv_order: Union[int, list, tuple] = 1


@dataclass(frozen=True)
class SearchSpaceConfig:
    """What the search is allowed to build.

    ``data_fun_pow`` / ``deriv_fun_pow`` only set token-family power ranges;
    the preprocessor never sees them. ``tokens`` declares prepared token
    families; families needing tensors or callables are passed as objects
    instead.

    ``system_rps_mode`` decides how a system's right parts (targets) and
    supports are chosen (one of :data:`SYSTEM_RPS_MODES`), and
    ``system_rps_params`` holds that mode's own settings
    (:data:`SYSTEM_RPS_PARAMS` names them, with their defaults). A key the
    mode does not have is refused; the loaded configuration carries every
    setting of the mode, defaults filled in (:func:`resolve_rps_params`).

    * ``'repair'`` -- each equation selects its target on its own (best
      discrepancy over its derivative terms, support from the sparsity
      operator), then cross-equation conflicts are repaired by rerolling.
      No settings.
    * ``'assignment'`` -- the whole system is chosen at once. Every
      assignment of one own-derivative target per equation is enumerated; in
      each equation the terms equal to another equation's chosen target are
      removed; every subset of the remaining terms (and the constant) -- of
      at most ``selected_terms_max_number`` columns when that is set, and
      only the best of each size by least squares under ``screen_size`` --
      is fitted, and each equation keeps its non-dominated (discrepancy,
      instability) subsets. Systems already in the search history are
      skipped, the remaining variants are compared by dominance, and one
      non-dominated variant is taken (``variant_pick``). No sparsity
      threshold is involved: the keep-rule is replaced by the exhaustive
      search. Not available with the solver-based fitness.
    * ``'plain'`` -- no subset search at all: every candidate equation is
      fitted once, by weighted least squares, on exactly its own terms, and
      the evolution does the selection (add, drop and swap mutations). The
      target is a gene the candidate carries from generation to generation;
      no two equations share one, and the terms equal to another equation's
      target are removed, as in ``'assignment'``. The constant is not a gene:
      each candidate is fitted without it and with it, the version that
      dominates the other on (discrepancy, instability) is kept, and when
      neither does both enter the population. ``equation_terms_max_number``
      alone bounds it, as in every mode: an equation keeps every term it
      holds, at most ``equation_terms_max_number - 1`` besides its target, the
      constant counting 1. With no subset search the mode has no settings.
      Not available with the solver-based fitness.

    ``equation_terms_max_number`` bounds an equation's STRUCTURE in every
    mode: the terms it holds, the target included -- the library
    ``'assignment'`` mode chooses a right part from, the whole candidate in
    ``'plain'`` mode. A new equation draws its number of terms uniformly from
    2 to that bound (``Equation.__init__``).

    ``'assignment'`` mode's settings (``system_rps_params``):

    * ``selected_terms_max_number`` bounds what the right part KEEPS: at most
      that many columns of the right-hand side, each term counting 1 whatever
      its factors and the constant counting 1, the target not counted.
      ``None`` (the default) keeps every subset. The subset search then
      scores only the subsets within the cap, so a structure can hold more
      terms than the ``2**12``-subset ceiling would otherwise allow
      (:func:`assignment_subset_count`).
    * ``screen_size`` screens those subsets: of each size, only the
      ``screen_size`` best by weighted least squares are scored with the
      objectives (``epde.operators.common.subset_search``; every subset is
      still visited, at a rank-one update each, up to
      :data:`ASSIGNMENT_MAX_VISITED`). ``None`` (the default) scores every
      subset within the cap. It needs the cap, and gives up the guarantee
      that the Pareto axes see every subset: a subset with a worse
      least-squares fit but a better WAPE or instability can be screened
      out. The ranking pools an equation's trajectories, and it comes before
      the candidate rules (``rps_amplification_cap``, supports reducible by a
      factor shared with the target or by a common power, exactly
      rank-deficient supports): the subsets those rules refuse keep their
      slots, so a narrow screen can leave fewer candidates -- or none, and an
      equation without a candidate for every target leaves its system
      degenerate.
    * ``variant_pick`` decides which of several non-dominated variants a
      system takes (one of :data:`VARIANT_PICKS`): ``'random'`` (the default)
      draws one uniformly; ``'sector'`` takes the one closest, by PBI in each
      equation's own plane, to the weight direction of the MOEA/DD sector the
      offspring were generated for (random before any sector exists, i.e. for
      the initial population).
    """
    data_fun_pow: int = 1
    deriv_fun_pow: int = 1
    equation_terms_max_number: int = 6
    equation_factors_max_number: Union[int, dict] = 1
    rps_amplification_cap: float = 100.0
    tokens: tuple = ()
    system_rps_mode: str = 'repair'
    system_rps_params: dict = field(default_factory=dict)


#: How a system's right parts are chosen; see
#: :attr:`SearchSpaceConfig.system_rps_mode`.
SYSTEM_RPS_MODES = ('repair', 'assignment', 'plain')
#: Each mode's settings (``search_space.system_rps_params``) and their
#: defaults; see :class:`SearchSpaceConfig`.
SYSTEM_RPS_PARAMS = {
    'repair': {},
    'assignment': {'selected_terms_max_number': None, 'screen_size': None,
                   'variant_pick': 'random'},
    'plain': {},
}
assert tuple(SYSTEM_RPS_PARAMS) == SYSTEM_RPS_MODES
#: Which non-dominated variant a system takes in ``'assignment'`` mode
#: (its ``variant_pick`` setting).
VARIANT_PICKS = ('random', 'sector')
#: The most subsets ``'assignment'`` mode scores per equation, target and
#: removed set: every non-empty subset of 12 columns, as many as the uncapped
#: search has always scored at most. It bounds the NUMBER of subsets only.
#: Under ``instability_metric='dyad'`` memory also grows with the slab
#: Grams of all the columns -- the slab count, finer under a small
#: ``selected_terms_max_number``, times the column count squared --
#: so a large ``equation_terms_max_number`` with a cap of 1 or 2 on a long
#: record can pass this count and still need gigabytes.
ASSIGNMENT_MAX_SUBSETS = (1 << 12) - 1
#: The most subsets ``screen_size``'s exact search VISITS per
#: equation, target and removed set (every subset of at most
#: ``selected_terms_max_number`` columns, at a rank-one update each):
#: about ten seconds at the ~1e7 subsets per second measured on Navier-Stokes,
#: e.g. 60 columns with at most 6 kept (5.6e7). Above it the screen is refused.
ASSIGNMENT_MAX_VISITED = 10 ** 8


def assignment_subset_count(n_cols: int, selected_max=None, screen_size=None) -> int:
    """How many subsets ``'assignment'`` mode scores for an equation of
    ``n_cols`` columns (the constant among them): every non-empty subset of at
    most ``selected_max`` columns (every one when it is None), or under
    ``screen_size`` at most that many of each size."""
    from math import comb
    top = n_cols if selected_max is None else min(int(selected_max), n_cols)
    if screen_size is None:
        return sum(comb(n_cols, k) for k in range(1, top + 1))
    return sum(min(comb(n_cols, k), int(screen_size)) for k in range(1, top + 1))


def _positive_int_or_none(value) -> bool:
    return value is None or (isinstance(value, Integral) and not isinstance(value, bool)
                             and value >= 1)


def check_evaluation_budget(stride, use_solver) -> None:
    """The row budget's refusals (``domain.evaluation_stride``): positive
    integers, and never with the solver -- a solution is defined on the whole
    grid, not on a sub-lattice of it. Checks that need the grid run when the
    domain is built (:func:`epde.structure.domain.resolve_evaluation_stride`)."""
    entries = stride if isinstance(stride, (list, tuple)) else (stride,)
    if (isinstance(stride, (list, tuple)) and not stride) \
            or not all(_positive_int_or_none(k) for k in entries) \
            or (isinstance(stride, (list, tuple)) and any(k is None for k in entries)):
        raise ValueError('domain.evaluation_stride must be a positive integer, a list of '
                         'positive integers (one per axis) or None, instead got {0!r}.'.format(stride))
    if use_solver and stride is not None:
        raise ValueError('domain.evaluation_stride is not supported with solver.use_solver=True: '
                         'the solver needs the whole grid, the row budget keeps a sub-lattice of it.')


#: ``domain.evaluation_form`` values.
EVALUATION_FORMS = ('strong', 'weak')


def check_evaluation_form(form, stride, use_solver, instability_metric) -> None:
    """``domain.evaluation_form`` refusals.

    - It must be one of :data:`EVALUATION_FORMS`.
    - ``'weak'`` refuses the solver: a solution is compared with the data
      pointwise, not window by window.
    - ``'weak'`` refuses ``evaluation_stride``: the weak window sets its own
      row lattice.
    - ``'weak'`` refuses ``instability_metric='vcoef'``: its basis is resolved
      from the pointwise fields of the interior box."""
    if form not in EVALUATION_FORMS:
        raise ValueError('domain.evaluation_form must be one of {0}, instead got {1!r}.'.format(
            list(EVALUATION_FORMS), form))
    if form != 'weak':
        return
    if use_solver:
        raise ValueError("domain.evaluation_form='weak' is not supported with solver.use_solver=True: "
                         'the solver compares its solution with the data pointwise.')
    if stride is not None:
        raise ValueError("domain.evaluation_form='weak' is not supported with domain.evaluation_stride: "
                         'the weak window sets its own row lattice.')
    if instability_metric == 'vcoef':
        raise ValueError("domain.evaluation_form='weak' is not supported with "
                         "objectives.instability_metric='vcoef': its basis is resolved from the "
                         'pointwise fields.')


def plain_kept_terms_cap(terms) -> int:
    """``'plain'`` mode's cap on what an equation keeps besides its target --
    its other terms plus the constant, each counting 1: the ``terms - 1``
    other terms a structure of ``equation_terms_max_number`` can hold. Plain
    mode searches no subsets, so no subset setting enters it."""
    return int(terms) - 1


def resolve_rps_params(rps_mode, params) -> dict:
    """``search_space.system_rps_params`` for ``rps_mode``: every setting the
    mode has (:data:`SYSTEM_RPS_PARAMS`), the given values over the defaults.
    A key the mode does not have is refused, naming the modes that have it."""
    if rps_mode not in SYSTEM_RPS_PARAMS:
        raise ValueError(
            'search_space.system_rps_mode must be one of {0}, instead got '
            '{1!r}.'.format(list(SYSTEM_RPS_MODES), rps_mode))
    if params is None:
        params = {}
    if not isinstance(params, dict):
        raise ValueError('search_space.system_rps_params must be a mapping, instead got '
                         '{0!r}.'.format(params))
    allowed = SYSTEM_RPS_PARAMS[rps_mode]
    unknown = sorted(set(params) - set(allowed))
    if unknown:
        owners = {key: [mode for mode, settings in SYSTEM_RPS_PARAMS.items() if key in settings]
                  for key in unknown}
        raise ValueError(
            'Unknown search_space.system_rps_params {0} for system_rps_mode={1!r}; it accepts '
            '{2}.{3}'.format(
                unknown, rps_mode, sorted(allowed) or 'no settings',
                ''.join(' {0} is a setting of system_rps_mode {1}.'.format(
                    key, ' / '.join(repr(mode) for mode in modes))
                    for key, modes in owners.items() if modes)))
    resolved = dict(allowed)
    resolved.update(params)
    return resolved


def rps_param(search_space, key):
    """The ``key`` setting of ``search_space``'s mode: its
    ``system_rps_params`` value, else the mode's default; None when the mode
    has no such setting (``'plain'`` mode has no selected-terms cap)."""
    defaults = SYSTEM_RPS_PARAMS.get(search_space.system_rps_mode, {})
    return search_space.system_rps_params.get(key, defaults.get(key))


def check_rps_params(rps_mode, terms, params) -> None:
    """The mode settings' refusals, shared by the config load and ``fit``
    (``params`` resolved by :func:`resolve_rps_params`). ``'assignment'``
    mode: the selected-terms cap and the screen size are positive integers or
    None, the screen needs the cap, ``variant_pick`` is one of
    :data:`VARIANT_PICKS`, and the subset search scores at most
    :data:`ASSIGNMENT_MAX_SUBSETS` subsets of an equation's ``terms`` columns
    (``terms - 1`` features and the constant)."""
    if rps_mode != 'assignment':
        return
    selected, screen = params['selected_terms_max_number'], params['screen_size']
    for key, value in (('selected_terms_max_number', selected), ('screen_size', screen)):
        if not _positive_int_or_none(value):
            raise ValueError('search_space.system_rps_params[{0!r}] must be a positive integer '
                             'or None, instead got {1!r}.'.format(key, value))
    if params['variant_pick'] not in VARIANT_PICKS:
        raise ValueError(
            "search_space.system_rps_params['variant_pick'] must be one of {0}, instead got "
            '{1!r}.'.format(list(VARIANT_PICKS), params['variant_pick']))
    if screen is not None and selected is None:
        raise ValueError(
            "search_space.system_rps_params['screen_size'] ranks the subsets of each size up "
            "to 'selected_terms_max_number'; set that cap too.")
    if isinstance(terms, bool) or not isinstance(terms, Integral):
        return
    if selected is None:
        if terms > 12:
            raise ValueError(
                "search_space.system_rps_mode='assignment' enumerates every subset "
                'of an equation\'s terms; equation_terms_max_number must be at most '
                '12, instead got {0}.'.format(terms))
        return
    if screen is not None:
        visited = assignment_subset_count(terms, selected)
        if visited > ASSIGNMENT_MAX_VISITED:
            raise ValueError(
                "search_space.system_rps_params['screen_size'] visits every subset of at most "
                "{0} of an equation's {1} columns: {2:.3g} subsets, more than {3:.3g}. Lower "
                "equation_terms_max_number or 'selected_terms_max_number'.".format(
                    selected, terms, visited, ASSIGNMENT_MAX_VISITED))
    count = assignment_subset_count(terms, selected, screen)
    if count > ASSIGNMENT_MAX_SUBSETS:
        raise ValueError(
            "search_space.system_rps_mode='assignment' scores {0} subsets of at most "
            '{1} of an equation\'s {2} columns{3}: more than {4}. Lower '
            "equation_terms_max_number or 'selected_terms_max_number'{5}.".format(
                count, selected, terms,
                '' if screen is None else ' ({0} of each size)'.format(screen),
                ASSIGNMENT_MAX_SUBSETS,
                ", or set 'screen_size'" if screen is None else
                ", or lower 'screen_size'"))


#: Which Gram the sparsity keep-rule needs, per instability estimator. The
#: keep-rule's L1 threshold and the Instability Pareto axis are ONE statistic
#: (``sparsity.instability_scores``, which ``PhysicsInformedLasso`` reaches
#: through ``_keep_rule_scores``), so what the threshold needs is whatever that
#: estimator is computed from:
#:   'vcoef' -> VaryingCoefSetup   (its ``score``; also caches _cached_vc_score)
#:   'cv'    -> GramSetup, 'axis'  (its window stack; caches _cached_sw_weights)
#:   everything else -> NO Gram at all: 'chi2', 'chi2_centered', 'het',
#:                      'het_raw', 'tile', 'dyad' and 'survival' score
#:                      straight from the active columns.
_GRAM_BY_INSTABILITY = {'vcoef': 'vcoef', 'cv': 'axis'}


@dataclass(frozen=True)
class ObjectivesConfig:
    """How many Pareto axes, and how a candidate is fitted and scored.

    ``second_objective`` replaces the removed ``use_pic`` flag
    (``use_pic=True`` meant ``'instability'``). A ``None`` value means "do not
    call the setter, leave whatever is already in ``epde.globals``".
    ``sparsity_kwargs`` configures the operator named by ``sparsity_cls`` --
    e.g. ``{'initial_sparsity_interval': [1e-4, 2.5]}`` for ``'lasso'``, the
    range its alpha metaparameter is seeded from. The Gram construction is NOT
    a field here: it follows ``instability_metric``, see :attr:`gram_mode`.
    """
    multiobjective_mode: bool = True
    discrepancy_metric: Optional[str] = 'wape'
    second_objective: Optional[str] = 'instability'
    complexity_metric: Optional[str] = 'factors'
    instability_metric: Optional[str] = 'dyad'
    single_objective_metric: Optional[str] = 'discrepancy'
    anchor_on_residual: bool = False
    sparsity_cls: Any = 'vwsr'
    sparsity_kwargs: dict = field(default_factory=dict)

    @property
    def gram_mode(self):
        """Which Gram the sparsity keep-rule builds: ``'vcoef'``, ``'axis'`` or
        ``None`` when the instability estimator needs neither.

        DERIVED, never configured: it follows :attr:`instability_metric`. An
        independent knob could only ever be set to agree with the instability
        estimator or to disagree with it, and disagreement has no upside -- the
        threshold would then prune by a statistic the Pareto axis does not
        score. Being a property rather than a field, it is not a dataclass
        field, so it never appears in :data:`KEY_GROUP`, ``as_dict()`` or the
        JSON: it cannot be set from a config or a kwarg.
        """
        return _GRAM_BY_INSTABILITY.get(self.instability_metric)


def default_device() -> str:
    """``'cuda'`` when a working GPU is visible, otherwise ``'cpu'``.

    Probed once, when this module is imported (``_DEFAULTS`` calls every
    ``default_factory`` there), so a search does not re-ask the driver on each
    construction. ``device='cpu'`` remains a plain kwarg for a caller who wants
    the CPU on a GPU machine -- for a bit-identical A/B, say, where the two
    backends do not produce the same rounding.

    A torch that cannot be imported, or a GPU that is present but unusable
    (busy, wrong driver), reports unavailable and lands on the CPU. Requesting
    cuda when it is not there is not silent either: ``_resolved_device`` warns
    and falls back.
    """
    try:
        import torch
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    except Exception:                                          # noqa: BLE001
        return 'cpu'


def _default_deepxde_config() -> dict:
    # ``train_frac``: share of the time levels the observation term sees; the
    # rest is where the candidate is scored. ``lbfgs_maxiter``: L-BFGS
    # iterations after Adam (0 = off). No boundary/initial point counts: the
    # solve has no IC/BC terms, the observation term covers them.
    #
    # EVERY VALUE HERE IS MEASURED, not inherited. Gate = min over 3 seeds of
    # ln(RMSE_wrong / RMSE_true), scored on the held-out tail, one fresh process
    # per solve (duplicate-config controls came back bit-identical, so the
    # replication noise floor is zero). It must be > 0 on BOTH Allen-Cahn and
    # forced Duffing: every single-system conclusion measured here was later
    # overturned by the second system.
    #
    #   lbfgs_maxiter  THE critical one. With no second-order phase the fitness
    #                  does not merely weaken, it INVERTS: Adam-only scores AC
    #                  at -0.131/-0.011/-0.096, i.e. the WRONG equation wins on
    #                  every seed. No Adam budget rescues it (tested to 20k with
    #                  cosine decay). 2000 L-BFGS iterations give +0.474; 5000
    #                  give +0.826, at 2.7x the cost (both at the old lr=1e-3).
    #                  RE-MEASURED Sep 18 2026 at the shipped lr=1e-4, after the
    #                  cap was made exact (it used to round up to the next
    #                  1000-iteration chunk):
    #                      cap   1000   2000   3000   4000+
    #                      AC   -0.011 +0.650 +0.816 +0.829
    #                      Duff -0.299 +4.277 +4.277 +4.277
    #                  The shipped 2000 is UNCHANGED by that fix on both systems
    #                  -- 2000 is an exact multiple of the chunk, so rounding up
    #                  and stopping exactly land on the same iterate; the live
    #                  12-solve re-run reproduced every stored RMSE. Raising it
    #                  is the one lever left on AC: 3000 buys +0.166 for 1.50x
    #                  the L-BFGS iterations and 4000 buys +0.179 for 1.77x,
    #                  after which AC converges on its own (~3300-3600) and the
    #                  gate saturates. Duffing pays NOTHING for a higher cap --
    #                  it converges at ~1481 iterations, below the cap already.
    #                  Left at 2000 pending a decision on that cost.
    #   loss_weight_mode  'gradient' (1/||grad L_i|| at init, frozen) is the only
    #                  mode whose usable lr window is two decades wide on both
    #                  systems. 'variance' needs lr=1e-3 exactly and collapses at
    #                  1e-4 (-0.810/-3.025); 'gradient_per_step' peaks higher on
    #                  Duffing (+5.683) but fails at lr=1e-4 (-0.028), so its
    #                  window is one rung. Basin width beats peak height for a
    #                  default that must survive unseen systems.
    #   lr             1e-4 beats the old 1e-3 on both systems (+0.650 vs +0.474
    #                  on AC, +4.277 vs +3.485 on Duffing) at identical cost, and
    #                  1e-5 is worse on both, so this is an interior optimum and
    #                  not the edge of the sampled range.
    #   net/num_domain [95,100,95] with 1000 points was never measured through
    #                  the fitness host; [64]*4 with 4000 is what every reported
    #                  number used.
    #
    # Cost: ~120 s/solve against the previous ~20 s. That is the price of a
    # fitness that ranks the true equation first instead of last.
    return {'net': [64, 64, 64, 64], 'activation': 'tanh', 'optimizer': 'adam',
            'lr': 1e-4, 'num_domain': 4000, 'num_test': 500, 'epochs': 2000,
            'train_frac': 0.8, 'lbfgs_maxiter': 2000,
            'loss_weight_mode': 'gradient'}


def _default_basis_config() -> dict:
    """Fixed-basis backend (``solver_backend='basis'``, no network, no learning
    rate; see ``epde.integrate.basis_integration``). Stage-1 values: every
    entry is a cell of the solver-gate sweep, to be revisited on its results.
    ``start='zero'`` and ``precondition='jacobi'`` are the two measured as
    load-bearing (a min-norm data start extrapolates wildly into the tail,
    start loss 4e42 on Duffing; without the column scaling Duffing's gate is
    +0.19 against +5.87)."""
    return {'train_frac': 0.8, 'energy': 0.99999, 'periodic': 'auto',
            'periodic_modes': 'nyquist', 'nyquist_factor': 1.0, 'time_factor': 2.0,
            'space_factor': 2.0, 'collocation_factor': 2, 'phys_weight_scale': 1.0,
            'pde_loss': 'mse', 'sinv_floor_rel': 0.0, 'start': 'zero',
            'precondition': 'jacobi', 'refine': 'spectral', 'refine_tol': 1e-10,
            'refine_min_gain': 2.0,
            'max_refinements': 3, 'lbfgs_maxiter': 4000, 'history_size': 100,
            'gtol': 1e-8, 'ftol': 0.0, 'threads': None}


@dataclass(frozen=True)
class SolverConfig:
    """Everything the PDE solver needs.

    ``device`` lives here because the GPU is only ever used with the solver --
    the caches are cpu/numpy by construction. It defaults to the GPU when one
    is available (see :func:`default_device`); pass ``device='cpu'`` to force
    the CPU. ``pinn_loss_mult`` /
    ``error_metric`` / ``deepxde_config`` are also parameters of the
    ``SolverBasedFitness`` operator; they are declared HERE and injected into
    the operator mapping by :func:`_resolve_operators`, so there is one
    declaration rather than two that can disagree.
    """
    use_solver: bool = False
    solver_backend: str = 'autograd'
    device: str = field(default_factory=default_device)
    # 0.0, not the old 1e4: at 1e4 the PINN's own training loss was 99.4-100%
    # of the discrepancy objective (measured over 30 Allen-Cahn candidates),
    # so the Pareto axis ranked candidates by how EASY they were for the
    # solver, not by how well they explained the data -- the best-scoring
    # candidate of a run was a form with wape 1.001. The data term now stands
    # alone (and is relative, see Discrepancy._compute_solver_l2); a diverged
    # solve is already penalised by the bad solution it produces, and a NaN
    # loss still short-circuits to 2 * LOSS_NAN_VAL. Raise this only to fuse
    # the two deliberately.
    pinn_loss_mult: float = 0.0
    error_metric: str = 'rmse'
    deepxde_config: dict = field(default_factory=_default_deepxde_config)
    basis_config: dict = field(default_factory=_default_basis_config)
    mode: str = 'NN'
    use_cache: bool = False
    use_fourier: bool = False
    fourier_params: Optional[dict] = None
    use_adaptive_lambdas: bool = False
    compiling_params: dict = field(default_factory=dict)
    optimizer_params: dict = field(default_factory=dict)
    cache_params: dict = field(default_factory=dict)
    early_stopping_params: dict = field(default_factory=dict)
    plotting_params: dict = field(default_factory=dict)
    training_params: dict = field(default_factory=dict)


def _default_director_params() -> dict:
    return {'variation_params': {}, 'mutation_params': {},
            'sorter_params': {}, 'pareto_combiner_params': {},
            'pareto_updater_params': {}}


@dataclass(frozen=True)
class EvolutionConfig:
    """The evolutionary engine.

    ``population_size`` takes an int, or ``'auto'`` (the default, in either
    mode): it seeds as many distinct initial systems as the token pool yields
    -- up to :data:`AUTO_POPULATION_CEILING`, stopping at the first slot where
    ``uniqueness_attempt_limit`` consecutive draws find no new structure -- but
    never fewer than :data:`AUTO_POPULATION_FLOOR` (below it the slots are
    filled as with an explicit size). It is not derived from the size of the
    search space.

    ``training_epochs`` is the most epochs a fit runs (an int >= 0), and
    ``early_stopping_patience`` stops it earlier: once the set of structures on
    the first Pareto level (NSGA-II's archive; the best individual's structure
    in single-objective mode) has not changed for that many epochs in a row.
    ``None`` runs every epoch. Either engine, either mode, any limit.

    ``operators`` is the per-operator parameter mapping. As DECLARED it holds
    only what a caller overrode; :func:`load_search_config` resolves it into
    the full mapping -- :data:`MULTI_OBJECTIVE_OPERATORS` (or the
    single-objective set) under those overrides, with the cross-referenced
    keys injected from the group that owns them.

    ``niching`` decides how a multi-equation system survives and mates (one of
    :data:`NICHING_MODES`):

    * ``'system'`` -- MOEA/DD as published: one 2-D weight direction, repeated
      over the equations, places and ranks the whole system.
    * ``'per_equation'`` -- each equation has its own (discrepancy,
      instability) plane. Survival protects every system that holds a
      structure on some equation's own front (one holder per structure: the
      same structure is the same equation); the victim is taken from the
      unprotected systems, the one whose best per-equation non-domination rank
      is the worst (ranks are compared, errors are never added up), or, when
      every system is protected, the one on the fewest equation fronts, ties
      by crowding within its own plane. Mating serves equation ``k mod n`` at
      weight vector ``k``, in that equation's own plane. A single-equation
      system keeps ``'system'``.

    ``complementary_mating_prob`` is the probability that the second parent is
    drawn from the systems holding some equation's front where the first
    parent does not (e.g. an exact-v system for an exact-u one), so crossover
    can combine them. ``0`` leaves mating untouched.

    ``engine`` is the multi-objective algorithm (one of :data:`ENGINES`):
    ``'moeadd'`` (MOEA/DD, the default) or ``'nsga2'`` -- NSGA-II: binary
    tournaments on (level, crowding), then the parents and offspring are
    sorted and cut back to the population size, crowding measured in each
    equation's own plane. Its archive keeps every system no archived system
    dominates and is sorted into the result after the fit, so a system found
    once is never crowded out (for one equation the first Pareto level is
    exactly the archive). One of its epochs runs as many offspring as a
    MOEA/DD epoch does for the same population, so ``training_epochs`` means
    the same budget. ``'assignment'`` mode's ``variant_pick='sector'``
    belongs to MOEA/DD's weight vectors and is refused with it. ``niching='per_equation'`` has an NSGA-II
    form: the truncation keeps the systems holding a structure on some
    equation's own front first (one holder per structure; when they alone
    overflow, those on more fronts, then the less crowded), and the tournament
    compares the systems' per-equation plane ranks. ``complementary_mating_prob``
    forces one complementary pair per generation, as in MOEA/DD.

    Whatever the engine, the initial population keeps at most half its
    members on the first Pareto level: a population that ends with more
    there grows past ``population_size`` (an explicit one too) until it does
    not, or until the pool yields no further system.
    """
    population_size: Union[str, int] = 'auto'
    training_epochs: int = 100
    early_stopping_patience: Optional[int] = 5
    neighbors_number: int = 3
    PBI_penalty: float = 5.0
    subregion_mating_limitation: float = 0.9
    solution_params: dict = field(default_factory=dict)
    director_params: dict = field(default_factory=_default_director_params)
    operators: dict = field(default_factory=dict)
    niching: str = 'system'
    complementary_mating_prob: float = 0.0
    engine: str = 'moeadd'


#: How a multi-equation system survives and mates; see
#: :attr:`EvolutionConfig.niching`.
NICHING_MODES = ('system', 'per_equation')
#: The multi-objective algorithm; see :attr:`EvolutionConfig.engine`.
ENGINES = ('moeadd', 'nsga2')
#: ``population_size='auto'``: the fewest systems seeded (raised to twice the
#: equation count under ``niching='per_equation'``, where weight vector ``k``
#: serves equation ``k mod n``, and above ``neighbors_number``) and the most.
AUTO_POPULATION_FLOOR = 8
AUTO_POPULATION_CEILING = 32


def check_auto_settings(boundary_width, population_size, training_epochs,
                        early_stopping_patience=None) -> None:
    """Refuse values of the settings that take ``'auto'`` other than ``'auto'``
    or what each accepts besides: a non-negative int (or one per axis) for
    ``boundary_width`` and an int >= 2 for ``population_size``; and the budget,
    an int >= 0 for ``training_epochs`` and an int >= 1 or None for
    ``early_stopping_patience``."""
    def integral(value, low):
        return isinstance(value, Integral) and not isinstance(value, bool) and value >= low

    if not (boundary_width == 'auto' if isinstance(boundary_width, str) else
            integral(boundary_width, 0) or (isinstance(boundary_width, (list, tuple))
                                            and all(integral(b, 0) for b in boundary_width))):
        raise ValueError(
            "domain.boundary_width must be 'auto', a non-negative int or one per "
            'axis, instead got {0!r}.'.format(boundary_width))
    if not (population_size == 'auto' if isinstance(population_size, str)
            else integral(population_size, 2)):
        raise ValueError(
            "evolution.population_size must be 'auto' or an int >= 2, instead got "
            '{0!r}.'.format(population_size))
    if not integral(training_epochs, 0):
        raise ValueError(
            'evolution.training_epochs must be an int >= 0 -- the most epochs; '
            'evolution.early_stopping_patience stops earlier -- instead got '
            '{0!r}.'.format(training_epochs))
    if not (early_stopping_patience is None or integral(early_stopping_patience, 1)):
        raise ValueError(
            'evolution.early_stopping_patience must be an int >= 1 or None, instead '
            'got {0!r}.'.format(early_stopping_patience))


@dataclass(frozen=True)
class RuntimeConfig:
    """Process-level bookkeeping.

    ``free_tensor_cache_after_fit`` releases the evaluated-term cache when
    ``fit()`` returns -- it is the bulk of the memory and is rebuildable. The
    grid, sample and initial-data caches are kept, because ``equations()``,
    ``predict()``, ``solver_forms()`` and ``visualize_solutions()`` all read
    them after fit. Use ``EpdeSearch.close()``, or the context-manager form,
    to release those too.
    """
    memory_for_cache: Union[int, float] = 15
    verbose_params: dict = field(
        default_factory=lambda: {'show_iter_idx': True})
    free_tensor_cache_after_fit: bool = True


# ---------------------------------------------------------------------------
# Operator parameters
# ---------------------------------------------------------------------------
# What ``operators/utils/parameters/default_parameters_*.json`` used to hold.
# ``add_base_param_to_operator`` reads the block for an operator's ``key`` and
# takes exactly the parameters named there, so a block declares BOTH which
# parameters an operator has and what they default to.
#
# Six of those parameters are also search settings. They keep their place in
# the block -- otherwise the operator would not receive them at all -- but
# their VALUE is written once, in the group that owns it, and referenced here
# with :class:`FromConfig`. That is what stops a second declaration from
# drifting away from the first, which is how ``pinn_loss_mult`` came to be 0.0
# in one file and 1e4 in two others.

@dataclass(frozen=True)
class FromConfig:
    """Placeholder for an operator parameter owned by a config group."""
    group: str
    key: str

    def resolve(self, resolved: dict):
        return resolved[self.group][self.key]


MULTI_OBJECTIVE_OPERATORS = {
    'MOEADDSelection': {'delta': FromConfig('evolution',
                                            'subregion_mating_limitation'),
                        'parents_fraction': 0.2},
    'PopulationUpdater': {'PBI_penalty': FromConfig('evolution',
                                                    'PBI_penalty')},
    # evolution.engine='nsga2': every member enters a binary tournament, so as
    # many parents are drawn as the population holds.
    'NSGASelection': {'parents_fraction': 1.0},
    'AppendUpdater': {},
    'SortingBasedNeighborSelector': {
        'number_of_neighbors': FromConfig('evolution', 'neighbors_number')},
    'ParetoLevelUpdater': {'mutation_attempt_limit': 3,
                           'offspring_attempt_limit': 3},
    'InitialParetoLevelSorting': {'uniqueness_attempt_limit': 100},
    'SolverFreeFitness': {'penalty_coeff': 0.2},
    'SolverBasedFitness': {
        'penalty_coeff': 0.2,
        'pinn_loss_mult': FromConfig('solver', 'pinn_loss_mult'),
        'error_metric': FromConfig('solver', 'error_metric'),
        'deepxde_config': FromConfig('solver', 'deepxde_config'),
        'basis_config': FromConfig('solver', 'basis_config'),
    },
    # Aliases kept ONLY so the functional harness (tests/functional) can fetch
    # params by the historical operator names; the production search builds
    # the SolverFreeFitness / SolverBasedFitness hosts above.
    'DiscrepancyBasedFitnessWithCV': {'penalty_coeff': 0.2},
    'PIC': {'penalty_coeff': 0.2,
            'pinn_loss_mult': FromConfig('solver', 'pinn_loss_mult')},
    'DeepXDEBasedFitness': {
        'penalty_coeff': 0.2,
        'error_metric': FromConfig('solver', 'error_metric'),
        'deepxde_config': FromConfig('solver', 'deepxde_config'),
    },
    'ParetoLevelsCrossover': {},
    'ChromosomeCrossover': {'equation_exchange_prob': 0.1},
    'MetaparamerCrossover': {'metaparam_proportion': 0.05},
    'EquationCrossover': {'crossover_probability': 1},
    'TermCrossover': {'crossover_probability': 0.3},
    'TermParamCrossover': {'term_param_proportion': 0.4},
    'SystemMutation': {'indiv_mutation_prob': 1},
    'EquationMutation': {'r_mutation': 0.6, 'n_added_terms': 12,
                         'term_addition_prob': 0.5},
    'MetaparameterMutation': {'mean': 0.0, 'std': 1.0},
    'TermMutation': {},
    'TermParameterMutation': {'r_param_mutation': 0.2, 'multiplier': 0.1,
                              'strict_restrictions': 1},
    'LinRegCoeffCalc': {},
    'LASSOBasedSparsity': {},
}

SINGLE_OBJECTIVE_OPERATORS = {
    'RouletteWheelSelection': {'parents_fraction': 0.4},
    'SizeRestriction': {},
    'FractionElitism': {
        'number_of_neighbors': FromConfig('evolution', 'neighbors_number')},
    'ParetoLevelUpdater': {'attempt_limit': 5},
    'InitialParetoLevelSorting': {},
    'SolverFreeFitness': {'penalty_coeff': 0.2},
    'PopulationLevelCrossover': {},
    'ChromosomeCrossover': {},
    'EquationCrossover': {},
    'TermCrossover': {'crossover_probability': 0.3},
    'TermParamCrossover': {'term_param_proportion': 0.4},
    'SystemMutation': {'indiv_mutation_prob': 0.7},
    'EquationMutation': {'r_mutation': 0.6},
    'MetaparameterMutation': {'mean': 0.0, 'std': 1e-4},
    'TermMutation': {},
    'TermParameterMutation': {'r_param_mutation': 0.2, 'multiplier': 0.1,
                              'strict_restrictions': 1},
    'LinRegCoeffCalc': {},
    'LASSOBasedSparsity': {},
}


def _resolve_operators(resolved: dict) -> dict:
    """The full per-operator mapping for this configuration.

    The mode's base blocks, under whatever the caller put in
    ``evolution.operators``, with every :class:`FromConfig` replaced by the
    value from the group that owns it. A caller override wins over the
    reference, so ``operators={'SolverBasedFitness': {'pinn_loss_mult': 5}}``
    still works and is the more specific statement.
    """
    base = (MULTI_OBJECTIVE_OPERATORS
            if resolved['objectives']['multiobjective_mode']
            else SINGLE_OBJECTIVE_OPERATORS)
    overrides = resolved['evolution'].get('operators') or {}
    unknown = set(overrides) - set(base)
    if unknown:
        raise ValueError(
            'operators={0}: no such operator in the {1}-objective set. '
            'Known operators: {2}.'.format(
                sorted(unknown),
                'multi' if resolved['objectives']['multiobjective_mode'] else 'single',
                ', '.join(sorted(base))))

    out = {}
    for name, params in base.items():
        block = dict(params)
        block.update(overrides.get(name, {}))
        out[name] = {key: value.resolve(resolved)
                     if isinstance(value, FromConfig) else value
                     for key, value in block.items()}
    return out


GROUP_CLASSES = {
    'domain': DomainConfig,
    'preprocessing': PreprocessingConfig,
    'search_space': SearchSpaceConfig,
    'objectives': ObjectivesConfig,
    'solver': SolverConfig,
    'evolution': EvolutionConfig,
    'runtime': RuntimeConfig,
}

@dataclass(frozen=True)
class SearchConfig:
    """The resolved configuration of one :class:`EpdeSearch` instance.

    Resolved per instance and never shared: ``pic_test_cases.py`` builds many
    searches in one process, and the ``EvolutionaryParams`` singleton's
    reset-before-use ritual is exactly the hazard this avoids.
    """

    domain: DomainConfig
    preprocessing: PreprocessingConfig
    search_space: SearchSpaceConfig
    objectives: ObjectivesConfig
    solver: SolverConfig
    evolution: EvolutionConfig
    runtime: RuntimeConfig

    def as_dict(self) -> dict:
        """Nested plain-dict view, for logging and round-trip tests."""
        return {group: {f.name: getattr(getattr(self, group), f.name)
                        for f in fields(cls)}
                for group, cls in GROUP_CLASSES.items()}

    def with_overrides(self, **overrides) -> 'SearchConfig':
        """A copy with ``overrides`` (flat kwarg names) applied.

        Used by the data-path methods, whose config-backed parameters default
        to :data:`UNSET` and fall back to the stored config.
        """
        grouped = _group_overrides(collect_overrides(**overrides))
        return replace(self, **{
            group: replace(getattr(self, group), **values)
            for group, values in grouped.items()})


# Keys whose value is a dict that merges one level deep at the file layer.
# The kwarg layer always replaces wholesale -- see ``load_search_config``.
_MERGED_DICT_KEYS = frozenset((
    'preprocessor_kwargs', 'verbose_params', 'director_params', 'operators',
    'sparsity_kwargs', 'system_rps_params',
    'deepxde_config', 'basis_config', 'solution_params', 'compiling_params',
    'optimizer_params', 'cache_params', 'early_stopping_params',
    'plotting_params', 'training_params',
))

# ---------------------------------------------------------------------------
# Registries
# ---------------------------------------------------------------------------

#: Sparsity operator by name. Imported lazily -- ``operators.common.sparsity``
#: pulls in sklearn and the whole stability stack.
SPARSITY_REGISTRY = ('vwsr', 'lasso')

#: Prepared token families that are fully declarative, i.e. whose constructors
#: take only scalars and number lists (e.g. ``freq_values``) and so survive a
#: JSON round trip. Families needing a
#: tensor or a callable (``CustomTokens``, ``CacheStoredTokens``,
#: ``ExternalDerivativesTokens``, ``ControlVarTokens``,
#: ``ArbitraryDataFunction``, ``DerivSignFunction``) cannot be declared and are
#: passed as objects via ``additional_tokens=``.
TOKEN_REGISTRY = {
    'trigonometric': 'TrigonometricTokens',
    'phased_sine_1d': 'PhasedSine1DTokens',
    'grid': 'GridTokens',
    'data_polynomials': 'DataPolynomials',
    'data_sign': 'DataSign',
    'constant': 'ConstantToken',
    'velocity_heq': 'VelocityHEQTokens',
    'logfun': 'LogfunTokens',
}


def sparsity_settings(sparsity_cls) -> tuple:
    """The attribute names ``sparsity_kwargs`` may set on ``sparsity_cls``.

    Public, non-callable class attributes, minus ``key`` (the operator's
    registry name, which the parameter loader matches on) and minus
    ``fits_physical_scale``. Today that is ``initial_sparsity_interval`` for
    every shipped operator; a custom sparsity class becomes configurable simply
    by declaring an attribute.

    ``fits_physical_scale`` is excluded because it is a DECLARATION, not a
    setting: it states whether this estimator's coefficients are already on the
    physical scale, which ``LinRegBasedCoeffsEquation`` uses to decide between
    refitting and promoting. Overriding it from ``sparsity_kwargs`` would run
    the legacy min-max refit over VWSR's physical coefficients, or skip it
    where LASSO needs it -- silently wrong coefficients either way.
    """
    if sparsity_cls is None:
        return ()
    _NOT_SETTINGS = ('key', 'fits_physical_scale')
    return tuple(sorted(
        name for name in dir(sparsity_cls)
        if not name.startswith('_') and name not in _NOT_SETTINGS
        and not callable(getattr(sparsity_cls, name, None))))


def validate_sparsity_kwargs(sparsity_cls, kwargs: dict) -> dict:
    """Reject settings the chosen sparsity operator does not have.

    Fail loud rather than ``setattr`` anything: a typo'd
    ``initial_sparsity_intervals`` would otherwise sit on the operator doing
    nothing, which is exactly the failure mode this config layer exists to
    remove.
    """
    if not kwargs:
        return {}
    allowed = sparsity_settings(sparsity_cls)
    unknown = [key for key in kwargs if key not in allowed]
    if unknown:
        raise ValueError(
            'Unknown sparsity_kwargs {0} for {1}; it accepts {2}.'.format(
                sorted(unknown),
                getattr(sparsity_cls, '__name__', sparsity_cls),
                list(allowed) or 'no settings'))
    resolved = dict(kwargs)
    interval = resolved.get('initial_sparsity_interval')
    if interval is not None:                    # JSON has no tuple type
        resolved['initial_sparsity_interval'] = tuple(interval)
    return resolved


def resolve_sparsity(value):
    """Map ``'vwsr'`` / ``'lasso'`` to the operator class.

    A class or ``None`` passes through untouched, so a caller can still hand in
    their own operator.
    """
    if value is None or not isinstance(value, str):
        return value
    key = value.lower()
    if key not in SPARSITY_REGISTRY:
        raise ValueError(
            'Unknown sparsity {0!r}; expected one of {1} or an operator '
            'class.'.format(value, list(SPARSITY_REGISTRY)))
    from epde.operators.common.sparsity import LASSOSparsity, VWSRSparsity
    return {'vwsr': VWSRSparsity, 'lasso': LASSOSparsity}[key]


def build_tokens(specs):
    """Instantiate the declarative token families named in ``specs``.

    Each spec is a dict with a ``family`` key naming an entry of
    :data:`TOKEN_REGISTRY`; the remaining keys are forwarded to that family's
    constructor verbatim.
    """
    import epde.interface.prepared_tokens as prepared

    built = []
    for spec in specs or ():
        if not isinstance(spec, dict):
            raise TypeError(
                'Each entry of search_space.tokens must be a dict with a '
                "'family' key, instead got {0!r}.".format(spec))
        spec = dict(spec)
        try:
            family = spec.pop('family')
        except KeyError:
            raise ValueError(
                "Token spec {0!r} has no 'family' key; expected one of "
                '{1}.'.format(spec, sorted(TOKEN_REGISTRY)))
        if family not in TOKEN_REGISTRY:
            raise ValueError(
                'Unknown token family {0!r}; expected one of {1}. Families '
                'that need tensors or callables (CacheStoredTokens, '
                'CustomTokens, ExternalDerivativesTokens, ControlVarTokens) '
                'cannot be declared in a config -- pass them as objects via '
                'additional_tokens.'.format(family, sorted(TOKEN_REGISTRY)))
        cls = getattr(prepared, TOKEN_REGISTRY[family])
        try:
            built.append(cls(**spec))
        except (TypeError, ValueError) as exc:
            raise ValueError(
                'Could not build token family {0!r} from {1!r}: {2}'.format(
                    family, spec, exc)) from exc
    return built


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def _read_default_config() -> dict:
    """The shipped defaults, read off the dataclasses themselves.

    There is no default file to disagree with: a field's declared default IS
    the default. ``default_factory`` is called per group so the mutable ones
    are fresh, and ``load_search_config`` deep-copies again, so no two
    ``SearchConfig`` objects can share a sub-dict.
    """
    defaults = {}
    for group, cls in GROUP_CLASSES.items():
        values = {}
        for spec in fields(cls):
            if spec.default is not MISSING:
                values[spec.name] = spec.default
            elif spec.default_factory is not MISSING:   # noqa: B009
                values[spec.name] = spec.default_factory()
            else:
                raise TypeError(
                    '{0}.{1} declares no default; every search setting must '
                    'have one here, since this module is the only place a '
                    'default is written.'.format(cls.__name__, spec.name))
        defaults[group] = values
    return defaults


_DEFAULTS = _read_default_config()


def _build_key_group() -> Dict[str, str]:
    """Flat kwarg name -> group name, derived from the shipped defaults."""
    mapping = {}
    for group, values in _DEFAULTS.items():
        for key in values:
            if key in mapping:
                raise RuntimeError(
                    'Config key {0!r} is declared in both {1!r} and {2!r}; '
                    'flat kwargs would be ambiguous.'.format(
                        key, mapping[key], group))
            mapping[key] = group
    return mapping


#: The flat-kwarg to group bridge. Kwargs stay flat for backward
#: compatibility (the shim and any code that splats kwarg names as data need
#: stable names); the file is nested for readability.
KEY_GROUP = _build_key_group()


def collect_overrides(**kwargs) -> dict:
    """Drop every :data:`UNSET` argument, keeping explicit ``None``.

    Callers build their override dict by passing every config-scope parameter
    through this, so a sentinel can never reach the loader.
    """
    return {key: value for key, value in kwargs.items() if value is not UNSET}


def _coerce_user_config(user_config) -> dict:
    if user_config is None:
        return {}
    if isinstance(user_config, dict):
        return copy.deepcopy(user_config)

    path = os.fspath(user_config)
    if not os.path.exists(path):
        raise FileNotFoundError('Search config file not found: {0}'.format(path))
    with open(path, encoding='utf-8') as handle:
        if path.lower().endswith(('.yaml', '.yml')):
            try:
                import yaml
            except ImportError:
                raise ImportError(
                    'Reading {0} needs PyYAML; install it or use a JSON '
                    'config.'.format(path))
            loaded = yaml.safe_load(handle)
        else:
            loaded = json.load(handle)
    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        raise ValueError(
            'Search config {0} must contain a mapping of groups, instead got '
            '{1}.'.format(path, type(loaded).__name__))
    return loaded


def _validate_group(group: str) -> None:
    if group not in GROUP_CLASSES:
        raise ValueError(
            'Unknown config group {0!r}; expected one of {1}.'.format(
                group, sorted(GROUP_CLASSES)))


def _validate_key(group: str, key: str) -> None:
    if key not in _DEFAULTS[group]:
        raise ValueError(
            'Unknown key {0!r} in config group {1!r}; that group accepts '
            '{2}.'.format(key, group, sorted(_DEFAULTS[group])))


def _apply_file_layer(resolved: dict, layer: dict) -> None:
    """Overlay a config file / dict, merging dict-valued keys one level deep."""
    for group, values in layer.items():
        if group.startswith('_'):
            continue
        _validate_group(group)
        if not isinstance(values, dict):
            raise ValueError(
                'Config group {0!r} must be a mapping, instead got {1}.'.format(
                    group, type(values).__name__))
        for key, value in values.items():
            if key.startswith('_'):
                continue
            _validate_key(group, key)
            current = resolved[group].get(key)
            if key in _MERGED_DICT_KEYS and isinstance(current, dict) \
                    and isinstance(value, dict):
                merged = dict(current)
                merged.update(value)
                resolved[group][key] = merged
            else:
                resolved[group][key] = value


def _group_overrides(overrides: dict) -> Dict[str, dict]:
    """Bucket flat kwargs by group, rejecting names that belong to no group."""
    grouped: Dict[str, dict] = {}
    unknown = []
    for key, value in overrides.items():
        group = KEY_GROUP.get(key)
        if group is None:
            unknown.append(key)
            continue
        grouped.setdefault(group, {})[key] = value
    if unknown:
        raise ValueError(
            'Unknown search-config parameter(s) {0}. Valid parameters, by '
            'group: {1}.'.format(
                sorted(unknown),
                {g: sorted(v) for g, v in sorted(_DEFAULTS.items())}))
    return grouped


#: The valid values of each objective setting, and the default a ``null``
#: resolves to. These used to live in ``epde.globals`` beside a mutable module
#: scalar and a ``set_*`` validator, i.e. every menu was declared twice -- once
#: here as a JSON value and once there as a Python tuple. The loader is now the
#: only validator.
#:
#: The instability menu is documented at length on ``Instability.compute``; in
#: brief: 'chi2' (Nyblom-Hansen score path, no refits),
#: 'chi2_centered' (the same on weighted-centred columns), 'vcoef'
#: (varying-coefficient NC/gamma_0^2), 'cv' (axis-aligned sliding window),
#: 'survival' (refits on iid resamples of fixed time slabs), 'tile' (one refit
#: per time slab along storage axis 0), 'dyad' (tile's statistic averaged
#: over every dyadic partition of every grid axis in turn, then over the
#: axes; parameter-free; the default), 'het' (per-block refits,
#: calibrated heterogeneity bounded to [0, 1]) and
#: 'het_raw' (the same heterogeneity, unbounded tau^2 / theta_bar^2).
METRIC_MENUS = {
    'discrepancy_metric': ('wape', 'l2', 'l2_relative', 'scale_invariant'),
    'complexity_metric': ('factors', 'terms'),
    'instability_metric': ('vcoef', 'cv', 'survival', 'tile', 'het', 'chi2',
                           'chi2_centered', 'het_raw', 'dyad'),
    'single_objective_metric': ('discrepancy', 'instability'),
    'second_objective': ('instability', 'complexity'),
}

#: Accepted spellings, normalised to the canonical name at LOAD time -- where
#: the ``set_*`` functions used to normalise them. ``Instability.compute`` and
#: ``Discrepancy._compute_*`` dispatch on canonical names, so an un-normalised
#: alias would silently fall through to the wrong branch.
METRIC_ALIASES = {
    'discrepancy_metric': {'l2_scaled': 'l2_relative', 'l2_rel': 'l2_relative',
                           'residual': 'l2_relative',
                           'scale_inv': 'scale_invariant',
                           'sinv': 'scale_invariant',
                           'cancellation': 'scale_invariant'},
    'instability_metric': {'chi': 'chi2'},
}


def _normalise_metrics(resolved: dict) -> None:
    """Apply the alias map, then resolve ``None`` to the shipped default.

    ``null`` used to mean "do not call the setter, leave whatever is already in
    ``epde.globals``". With the settings no longer living in mutable module
    scalars there is nothing to leave alone, so it now means "use the built-in
    default" -- which is what the globals' own ``None`` resolved to anyway, so
    the effective value is unchanged. Consumers therefore never see ``None``
    and need no fallback of their own.
    """
    objectives = resolved['objectives']
    for key, menu in METRIC_MENUS.items():
        value = objectives[key]
        value = METRIC_ALIASES.get(key, {}).get(value, value)
        if value is None:
            value = _DEFAULTS['objectives'][key]
            value = METRIC_ALIASES.get(key, {}).get(value, value)
        if value not in menu:
            aliases = tuple(METRIC_ALIASES.get(key, ()))
            raise ValueError(
                'objectives.{0} must be one of {1}{2}, instead got {3!r}.'.format(
                    key, list(menu),
                    ' (or aliases {0})'.format(list(aliases)) if aliases else '',
                    objectives[key]))
        objectives[key] = value


def _check_types(resolved: dict) -> None:
    bool_keys = (('objectives', 'multiobjective_mode'),
                 ('objectives', 'anchor_on_residual'),
                 ('solver', 'use_solver'), ('solver', 'use_cache'),
                 ('solver', 'use_fourier'), ('solver', 'use_adaptive_lambdas'),
                 ('runtime', 'free_tensor_cache_after_fit'))
    for group, key in bool_keys:
        value = resolved[group][key]
        if not isinstance(value, bool):
            raise ValueError(
                '{0}.{1} must be a bool, instead got {2!r}.'.format(
                    group, key, value))

    backend = resolved['solver']['solver_backend']
    if backend not in ('autograd', 'deepxde', 'basis'):
        raise ValueError(
            "solver.solver_backend must be 'autograd', 'deepxde' or 'basis', instead "
            'got {0!r}.'.format(backend))

    if resolved['solver']['use_solver'] \
            and not resolved['objectives']['multiobjective_mode']:
        # The single-objective director ignores ``use_solver`` entirely
        # (BaselineDirector.use_baseline takes only ``params``), so this
        # combination silently yields a solver-free strategy.
        raise ValueError(
            'solver.use_solver=True is not supported with '
            'objectives.multiobjective_mode=False: the single-objective '
            'strategy has no solver-based fitness, so the solver would be '
            'silently ignored.')

    space = resolved['search_space']
    rps_mode = space['system_rps_mode']
    # The mode's settings, defaults filled in; a key of another mode raises.
    space['system_rps_params'] = resolve_rps_params(rps_mode, space['system_rps_params'])
    pick = space['system_rps_params'].get('variant_pick')
    niching = resolved['evolution']['niching']
    if niching not in NICHING_MODES:
        raise ValueError(
            'evolution.niching must be one of {0}, instead got {1!r}.'.format(
                list(NICHING_MODES), niching))
    mating = resolved['evolution']['complementary_mating_prob']
    if isinstance(mating, bool) or not isinstance(mating, (int, float)) \
            or not 0.0 <= mating <= 1.0:
        raise ValueError(
            'evolution.complementary_mating_prob must be a number in [0, 1], '
            'instead got {0!r}.'.format(mating))
    if rps_mode in ('assignment', 'plain'):
        if resolved['solver']['use_solver']:
            # The variants are compared on per-equation objectives scored
            # without a solve; the solver-based fitness scores the system jointly.
            raise ValueError(
                "search_space.system_rps_mode={0!r} is not supported with "
                'solver.use_solver=True: its variants are compared on '
                'solver-free per-equation objectives.'.format(rps_mode))
        objectives = resolved['objectives']
        if not objectives['multiobjective_mode'] \
                or objectives['second_objective'] != 'instability':
            # Each equation's subsets are screened on (discrepancy, instability)
            # -- the run's own Pareto axes only in this configuration. With one
            # objective the densest fit would always win.
            raise ValueError(
                "search_space.system_rps_mode={0!r} requires "
                "objectives.multiobjective_mode=True and "
                "objectives.second_objective='instability'.".format(rps_mode))
    engine = resolved['evolution']['engine']
    if engine not in ENGINES:
        raise ValueError(
            'evolution.engine must be one of {0}, instead got {1!r}.'.format(
                list(ENGINES), engine))
    if engine == 'nsga2':
        if not resolved['objectives']['multiobjective_mode']:
            raise ValueError(
                "evolution.engine='nsga2' requires objectives.multiobjective_mode=True.")
        # The sector pick lives inside MOEA/DD's weight vectors. Per-equation
        # niching and complementary mating have NSGA-II forms (the truncation
        # and the tournament, epde.optimizers.nsga).
        if pick == 'sector':
            raise ValueError(
                "search_space.system_rps_params['variant_pick']='sector' is a MOEA/DD "
                "setting, not supported with evolution.engine='nsga2'.")
    check_rps_params(rps_mode, space['equation_terms_max_number'], space['system_rps_params'])
    check_evaluation_budget(resolved['domain']['evaluation_stride'],
                            resolved['solver']['use_solver'])
    check_evaluation_form(resolved['domain']['evaluation_form'],
                          resolved['domain']['evaluation_stride'],
                          resolved['solver']['use_solver'],
                          resolved['objectives']['instability_metric'])
    check_auto_settings(resolved['domain']['boundary_width'],
                        resolved['evolution']['population_size'],
                        resolved['evolution']['training_epochs'],
                        resolved['evolution']['early_stopping_patience'])


def check_plain_sparsity(resolved: dict) -> None:
    """``'plain'`` mode keeps the weighted least-squares fit of a candidate's
    own terms. Refused with a sparsity class that does not declare its weights
    physical (``fits_physical_scale``; e.g. ``'lasso'``), since the
    coefficient step would then refit them."""
    if resolved['search_space']['system_rps_mode'] != 'plain':
        return
    sparsity_cls = resolved['objectives']['sparsity_cls']
    name = getattr(sparsity_cls, '__name__', sparsity_cls)
    if not getattr(sparsity_cls, 'fits_physical_scale', False):
        raise ValueError(
            "search_space.system_rps_mode='plain' keeps the least-squares fit of each "
            'candidate; objectives.sparsity_cls={0!r} would have it refitted. Use a sparsity '
            "class whose weights are physical (the default 'vwsr').".format(name))


def load_search_config(user_config=None, overrides=None) -> SearchConfig:
    """Resolve the shipped defaults, a user config and explicit kwargs.

    Args:
        user_config: path to a JSON/YAML file, a nested ``{group: {key: value}}``
            dict, or ``None`` for the shipped defaults alone.
        overrides: flat ``{kwarg_name: value}`` mapping, already stripped of
            :data:`UNSET` by :func:`collect_overrides`. These win over both
            other layers.

    Returns:
        A frozen :class:`SearchConfig`.

    Merge policy, which is asymmetric on purpose. Dict-valued keys
    (``verbose_params``, ``director_params``, ``preprocessor_kwargs``,
    ``operators``, ``deepxde_config``, and ``predict``'s six ``*_params``)
    merge one level deep when they come from a config *file*, so a file can
    set one sub-key without restating the rest. A *kwarg* replaces the whole
    dict, because that is what passing ``verbose_params={...}`` has always
    done and scripts depend on it.
    """
    resolved = copy.deepcopy(_DEFAULTS)

    file_layer = _coerce_user_config(user_config)
    _apply_file_layer(resolved, file_layer)

    overrides = overrides or {}
    grouped = _group_overrides(overrides)
    for group, values in grouped.items():
        resolved[group].update(values)

    _normalise_metrics(resolved)
    _check_types(resolved)
    # After every layer, so a kwarg or a config file can move both a setting
    # and the operator parameter that references it in one step.
    resolved['evolution']['operators'] = _resolve_operators(resolved)
    resolved['objectives']['sparsity_cls'] = resolve_sparsity(
        resolved['objectives']['sparsity_cls'])
    resolved['objectives']['sparsity_kwargs'] = validate_sparsity_kwargs(
        resolved['objectives']['sparsity_cls'],
        resolved['objectives']['sparsity_kwargs'])
    check_plain_sparsity(resolved)

    config = SearchConfig(**{group: cls(**resolved[group])
                             for group, cls in GROUP_CLASSES.items()})

    for group in GROUP_CLASSES:
        for field in fields(GROUP_CLASSES[group]):
            if getattr(getattr(config, group), field.name) is UNSET:
                raise AssertionError(
                    'UNSET leaked into {0}.{1}'.format(group, field.name))
    return config


# ---------------------------------------------------------------------------
# The active configuration
# ---------------------------------------------------------------------------
# One resolved config per process, published by ``EpdeSearch.__init__`` and
# again by ``EpdeSearch.fit`` (with that fit's search-space arguments applied).
# This is where the objective settings used to be: seven mutable scalars in
# ``epde.globals``, each with its own setter, its own validator and its own
# default. Consumers now read the resolved config directly, so a setting is
# declared once (the JSON), validated once (the loader) and read one way.
#
# It remains process-level -- the last search constructed or fitted wins --
# which is the same trade-off as before. What changed is that it is a single
# immutable object rather than twelve independently mutable names, so the
# settings cannot drift out of step with each other.

_ACTIVE = None


def active_config() -> SearchConfig:
    """The configuration the current search is running under.

    Falls back to the shipped defaults when no :class:`EpdeSearch` has been
    built, so an operator driven directly -- unit tests, offline analysis
    tools -- still resolves every setting from the config file rather than from
    a Python-level duplicate of it.
    """
    global _ACTIVE
    if _ACTIVE is None:
        _ACTIVE = load_search_config()
    return _ACTIVE


def set_active_config(config: SearchConfig) -> None:
    """Publish ``config`` as the one the operators should read.

    Called by ``EpdeSearch.__init__`` before the director is assembled: the
    strategy resolves the second Pareto axis at assembly time, and the fillers
    it builds are fixed from then on. ``EpdeSearch.fit`` publishes again, with
    its own term/factor budget and pool structure applied, because operators
    read those during the search (``survival._reference_width``).
    """
    global _ACTIVE
    if not isinstance(config, SearchConfig):
        raise TypeError(
            'set_active_config expects a SearchConfig, instead got '
            '{0}.'.format(type(config).__name__))
    _ACTIVE = config


def reset_active_config() -> None:
    """Forget the active config; the next read re-resolves the defaults."""
    global _ACTIVE
    _ACTIVE = None
