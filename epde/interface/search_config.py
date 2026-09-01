"""Grouped, file-loadable search configuration for :class:`EpdeSearch`.

The pre-config interface spread its defaults across ``__init__``,
``set_preprocessor``, ``set_moeadd_params``, ``set_singleobjective_params``,
``createDomain``, ``createTrajectory``, ``create_pool``, ``fit`` and
``predict``, so one setting could be spelled differently at different entry
points and every script restated the defaults. This module lifts the
``operators/utils/parameters`` JSON pattern to the search level: one grouped
default file, an optional user config, and explicit kwargs on top.

Precedence, strictly::

    parameters/default_search_config.json  <  user config  <  explicit kwargs

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
bridge is :data:`KEY_GROUP`, derived from the JSON at import time so it cannot
drift from the shipped defaults.

Not configurable here -- these need an object, a tensor or a callable, and are
passed at the call site: ``gfunction``, ``nds_method``, ``ndl_update_method``,
``early_stopping_callback``, ``director``, ``preprocessor_pipeline``,
``derivs``, ``pool``, ``population``, ``net``, and the data-bound token
families (see :data:`TOKEN_REGISTRY`).
"""

import copy
import json
import os
from dataclasses import dataclass, fields, replace
from typing import Any, Dict, Optional, Union

__all__ = ['UNSET', 'SearchConfig', 'load_search_config', 'KEY_GROUP',
           'collect_overrides', 'build_tokens', 'resolve_sparsity',
           'sparsity_settings', 'validate_sparsity_kwargs',
           'DEFAULT_CONFIG_PATH', 'TOKEN_REGISTRY', 'SPARSITY_REGISTRY',
           'GROUP_CLASSES', 'METRIC_MENUS', 'METRIC_ALIASES',
           'active_config', 'set_active_config', 'reset_active_config']


DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'parameters',
                                   'default_search_config.json')


# ---------------------------------------------------------------------------
# The UNSET sentinel
# ---------------------------------------------------------------------------

class _UnsetType:
    """Marker for "the caller did not pass this argument".

    A plain ``None`` default cannot serve here: ``None`` is a *meaningful*
    value for several keys (``fourier_params``, ``params_filename``, and every
    objective metric, where it means "leave the process global alone"). An
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
# Written out rather than generated from the JSON, so that a typo in
# ``cfg.solver.devcie`` fails at the attribute instead of silently reading a
# dict, and so the available settings stay greppable. The defaults below are
# never actually used -- every field is filled from the JSON -- but they
# document the shipped values inline. ``test_search_config.py`` pins the two
# against each other so they cannot drift.

@dataclass(frozen=True)
class DomainConfig:
    boundary_width: Union[int, list, tuple] = 5
    time_axis: int = 0


@dataclass(frozen=True)
class PreprocessingConfig:
    default_preprocessor_type: str = 'poly'
    preprocessor_kwargs: dict = None
    max_deriv_order: Union[int, list, tuple] = 1


@dataclass(frozen=True)
class SearchSpaceConfig:
    data_fun_pow: int = 1
    deriv_fun_pow: int = 1
    equation_terms_max_number: int = 6
    equation_factors_max_number: Union[int, dict] = 1
    rps_amplification_cap: float = 100.0
    tokens: tuple = ()


#: Which Gram the sparsity keep-rule needs, per instability estimator. The
#: keep-rule's L1 threshold and the Instability Pareto axis are ONE statistic
#: (``PhysicsInformedLasso._active_scores``), so what the threshold needs is
#: whatever that estimator is computed from:
#:   'vcoef' -> VaryingCoefSetup   (its ``score``; also caches _cached_vc_score)
#:   'cv'    -> GramSetup, 'axis'  (its window stack; caches _cached_sw_weights)
#:   everything else -> NO Gram at all: 'chi2', 'het', 'tile' and 'survival'
#:                      score straight from the active columns.
_GRAM_BY_INSTABILITY = {'vcoef': 'vcoef', 'cv': 'axis'}


@dataclass(frozen=True)
class ObjectivesConfig:
    multiobjective_mode: bool = True
    discrepancy_metric: Optional[str] = 'wape'
    second_objective: Optional[str] = 'instability'
    complexity_metric: Optional[str] = 'factors'
    instability_metric: Optional[str] = 'chi2'
    single_objective_metric: Optional[str] = 'discrepancy'
    anchor_on_residual: bool = False
    sparsity_cls: Any = 'vwsr'
    sparsity_kwargs: dict = None

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


@dataclass(frozen=True)
class SolverConfig:
    use_solver: bool = False
    solver_backend: str = 'autograd'
    device: str = 'cpu'
    pinn_loss_mult: float = 1e4
    error_metric: str = 'rmse'
    deepxde_config: dict = None
    mode: str = 'NN'
    use_cache: bool = False
    use_fourier: bool = False
    fourier_params: Optional[dict] = None
    use_adaptive_lambdas: bool = False
    compiling_params: dict = None
    optimizer_params: dict = None
    cache_params: dict = None
    early_stopping_params: dict = None
    plotting_params: dict = None
    training_params: dict = None


@dataclass(frozen=True)
class EvolutionConfig:
    population_size: int = 6
    training_epochs: int = 100
    neighbors_number: int = 3
    PBI_penalty: float = 5.0
    subregion_mating_limitation: float = 0.9
    solution_params: dict = None
    director_params: dict = None
    operators: dict = None


@dataclass(frozen=True)
class RuntimeConfig:
    memory_for_cache: Union[int, float] = 15
    verbose_params: dict = None
    params_filename: Optional[str] = None
    free_tensor_cache_after_fit: bool = True


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
    'sparsity_kwargs',
    'deepxde_config', 'solution_params', 'compiling_params',
    'optimizer_params', 'cache_params', 'early_stopping_params',
    'plotting_params', 'training_params',
))

# Single-objective mode used to carry its own defaults
# (``set_singleobjective_params(population_size=4, training_epochs=50)`` versus
# the multiobjective ``6`` / ``100``). The JSON can hold only one value, so
# these are substituted when the mode is single-objective AND the caller left
# the key alone -- otherwise flipping ``multiobjective_mode`` would silently
# change the population size.
_SINGLE_OBJECTIVE_DEFAULTS = {'population_size': 4, 'training_epochs': 50}

# ---------------------------------------------------------------------------
# Registries
# ---------------------------------------------------------------------------

#: Sparsity operator by name. Imported lazily -- ``operators.common.sparsity``
#: pulls in sklearn and the whole stability stack.
SPARSITY_REGISTRY = ('vwsr', 'lasso', 'knee')

#: Prepared token families that are fully declarative, i.e. whose constructors
#: take only scalars and so survive a JSON round trip. Families needing a
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
    registry name, which the parameter loader matches on). Today that is
    ``initial_sparsity_interval`` for both shipped operators; a custom
    sparsity class becomes configurable simply by declaring an attribute.
    """
    if sparsity_cls is None:
        return ()
    return tuple(sorted(
        name for name in dir(sparsity_cls)
        if not name.startswith('_') and name != 'key'
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
    """Map ``'vwsr'`` / ``'lasso'`` / ``'knee'`` to the operator class.

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
    from epde.operators.common.subset_selection import KneeSparsity
    return {'vwsr': VWSRSparsity, 'lasso': LASSOSparsity,
            'knee': KneeSparsity}[key]


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
        except TypeError as exc:
            raise ValueError(
                'Could not build token family {0!r} from {1!r}: {2}'.format(
                    family, spec, exc))
    return built


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def _read_default_config() -> dict:
    # ``_``-prefixed names are comments, at both levels: JSON has no comment
    # syntax, and the shipped file explains its groups and the odd key inline.
    # A user config is stripped the same way in ``_apply_file_layer``.
    with open(DEFAULT_CONFIG_PATH, encoding='utf-8') as handle:
        raw = json.load(handle)
    return {group: {key: value for key, value in values.items()
                    if not key.startswith('_')}
            for group, values in raw.items() if not group.startswith('_')}


# Parsed once; every resolution deep-copies it, so no two ``SearchConfig``
# objects can share a mutable sub-dict.
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
#: brief: 'chi2' (default, Nyblom-Hansen score path, no refits), 'vcoef'
#: (varying-coefficient NC/gamma_0^2), 'cv' (axis-aligned sliding window),
#: 'survival' / 'tile' / 'het' (resampling-based).
METRIC_MENUS = {
    'discrepancy_metric': ('wape', 'l2', 'l2_relative', 'scale_invariant'),
    'complexity_metric': ('factors', 'terms'),
    'instability_metric': ('vcoef', 'cv', 'survival', 'tile', 'het', 'chi2'),
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
    if backend not in ('autograd', 'deepxde'):
        raise ValueError(
            "solver.solver_backend must be 'autograd' or 'deepxde', instead "
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

    # Restore the single-objective population defaults when the mode is
    # single-objective and the caller said nothing about them.
    if not resolved['objectives']['multiobjective_mode']:
        file_keys = set()
        for group_values in file_layer.values():
            if isinstance(group_values, dict):
                file_keys.update(group_values)
        for key, value in _SINGLE_OBJECTIVE_DEFAULTS.items():
            if key not in overrides and key not in file_keys:
                resolved['evolution'][key] = value

    _normalise_metrics(resolved)
    _check_types(resolved)
    resolved['objectives']['sparsity_cls'] = resolve_sparsity(
        resolved['objectives']['sparsity_cls'])
    resolved['objectives']['sparsity_kwargs'] = validate_sparsity_kwargs(
        resolved['objectives']['sparsity_cls'],
        resolved['objectives']['sparsity_kwargs'])

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
# One resolved config per process, published by ``EpdeSearch.__init__``. This
# is where the objective settings used to be: seven mutable scalars in
# ``epde.globals``, each with its own setter, its own validator and its own
# default. Consumers now read the resolved config directly, so a setting is
# declared once (the JSON), validated once (the loader) and read one way.
#
# It remains process-level -- the last search constructed wins -- which is the
# same trade-off as before. What changed is that it is a single immutable
# object rather than twelve independently mutable names, so the settings cannot
# drift out of step with each other.

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
    it builds are fixed from then on.
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
