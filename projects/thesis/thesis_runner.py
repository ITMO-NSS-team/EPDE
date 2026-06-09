"""
Shared runner module for the thesis Section 4.5 within-platform comparison.

This module is the single source of truth for:
    * the 8-cell pipeline table (``_PIPELINE_SETTINGS`` -> ``pipeline_settings``)
    * the ``SystemCfg`` dataclass consumed by ``build_search`` / ``run_one``
    * ``run_smoke`` (batched per-rep JSON dumps with resume-on-restart)
    * ``load_config`` for parsing per-system YAML configs

Per-system configuration lives at:

    projects/thesis/configs/<name>.yaml       (declarative: truth equations,
                                               output dir, data_fun_pow, ...)
    projects/thesis/adapters/<name>.py        (Python: load_data(),
                                               build_extra_tokens(coords, dim))

Pipeline selection (``legacy`` vs ``new`` is the main thesis comparison; the
six ablation cells off the 000/111 diagonal cover the 2x2x2 factorial):

    LEGACY  -> L2Fitness   + LASSOSparsity + use_pic=False
    NEW     -> L2LRFitness + VWSRSparsity  + use_pic=True
"""

from __future__ import annotations

import importlib
import json
import os
import sys
import time
import traceback
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Optional

import numpy as np
import torch

# Make sure the EPDE package is importable when running this module's CLI
# entries directly (``python projects/thesis/run.py lv``).
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, '..', '..'))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from epde.interface.interface import EpdeSearch  # noqa: E402
from epde.operators.common.fitness import L2Fitness, L2LRFitness  # noqa: E402
from epde.operators.common.sparsity import LASSOSparsity, VWSRSparsity  # noqa: E402
from epde import GridTokens, TrigonometricTokens  # noqa: E402


CONFIGS_DIR = os.path.join(_THIS_DIR, 'configs')
ADAPTERS_DIR = os.path.join(_THIS_DIR, 'adapters')
EXPERIMENTS_DIR = os.path.join(_THIS_DIR, 'experiments')
RESULTS_DIR = os.path.join(_THIS_DIR, 'results')
DEFAULTS_PATH = os.path.join(CONFIGS_DIR, 'defaults.yaml')

# Keys consumed from per-system YAMLs by the deep-merge into hparams.
# Any other top-level key in a per-system YAML (``name``,
# ``truth_equations``, ``adapter``, ``outdir``) is consumed by
# load_config directly and never reaches hparams.
HPARAM_KEYS = ('search', 'preprocessor', 'moeadd', 'grid_tokens',
               'additional_tokens', 'fit')

# Token classes constructible from YAML (kwargs dict + dimensionality
# injected at build time). CustomTokens stays out of the registry
# because its evaluator is a Python callable that doesn't round-trip
# through YAML -- such tokens live in an adapter's build_extra_tokens.
_TOKEN_REGISTRY = {
    'TrigonometricTokens': TrigonometricTokens,
}


# Full 2x2x2 ablation table for the three thesis-NEW contributions:
# (1) WAPE fitness      -> L2LRFitness   vs LEGACY L2Fitness
# (2) Instability obj   -> use_pic=True swaps MOEA/D's 2nd objective
#                          (equation_terms_stability) vs LEGACY
#                          (equation_complexity_by_factors)
# (3) Novel regularizer -> VWSRSparsity (PhysicsInformedLasso, CV-weighted)
#                          vs LEGACY LASSOSparsity (sklearn.Lasso)
_PIPELINE_SETTINGS = {
    'legacy':      {'fitness_cls': L2Fitness,   'sparsity_cls': LASSOSparsity, 'use_pic': False},
    'wape':        {'fitness_cls': L2LRFitness, 'sparsity_cls': LASSOSparsity, 'use_pic': False},
    'instab':      {'fitness_cls': L2Fitness,   'sparsity_cls': LASSOSparsity, 'use_pic': True},
    'reg':         {'fitness_cls': L2Fitness,   'sparsity_cls': VWSRSparsity,  'use_pic': False},
    'wape_instab': {'fitness_cls': L2LRFitness, 'sparsity_cls': LASSOSparsity, 'use_pic': True},
    'wape_reg':    {'fitness_cls': L2LRFitness, 'sparsity_cls': VWSRSparsity,  'use_pic': False},
    'instab_reg':  {'fitness_cls': L2Fitness,   'sparsity_cls': VWSRSparsity,  'use_pic': True},
    'new':         {'fitness_cls': L2LRFitness, 'sparsity_cls': VWSRSparsity,  'use_pic': True},
}

# Default pipelines for the main Section 4.5 comparison.
PIPELINES = ('legacy', 'new')

# Off-diagonal cells of the 2x2x2 factorial -- pass to ``run_smoke`` as the
# ``pipelines`` argument from the ablation entry point. Excludes
# ``legacy`` and ``new`` since their reps live in the default results tree
# (the 000 and 111 corners of the cube).
ABLATION_PIPELINES = (
    'wape', 'instab', 'reg',
    'wape_instab', 'wape_reg', 'instab_reg',
)


def pipeline_settings(pipeline: str) -> dict:
    """Return ``EpdeSearch`` kwargs for a single pipeline label.

    Recognises the original two labels (``legacy``, ``new``) and the six
    off-diagonal ablation labels. Forward the returned dict directly to
    :class:`EpdeSearch` (``use_pic``, ``fitness_cls``, ``sparsity_cls``).
    """
    try:
        return dict(_PIPELINE_SETTINGS[pipeline])
    except KeyError:
        raise ValueError(
            f"Unknown pipeline {pipeline!r}; expected one of {tuple(_PIPELINE_SETTINGS)}"
        )


@dataclass
class SystemCfg:
    """Per-system configuration consumed by :func:`run_one`.

    name: short system identifier used in output filenames.
    truth_tokens: canonical token set encoding the ground-truth equations.
    outdir: directory to write per-rep JSON results into.
    load_data: callable returning ``(coordinate_tensors, data_list,
        variable_names, dimensionality)``. ``dimensionality`` is ``0`` for
        ODE systems and the number of spatial axes for PDE systems.
    build_extra_tokens: optional callable returning **truth-specific**
        EPDE tokens beyond what ``hparams['additional_tokens']`` already
        provides. Signature ``(coords, dim) -> list``. Default: returns
        ``[]``. Used by adapters whose token-list contains Python
        callables (e.g. ``CustomTokens`` evaluators) that can't live in
        YAML.
    hparams: nested dict of every hyperparameter EpdeSearch /
        set_preprocessor / set_moeadd_params / search.fit consume.
        Populated by :func:`load_config` via deep-merge of
        ``configs/defaults.yaml`` and the per-system YAML. Layout:
        ``{'search': {...}, 'preprocessor': {...}, 'moeadd': {...},
        'grid_tokens': {...}, 'additional_tokens': [...], 'fit': {...}}``.
    """

    name: str
    truth_tokens: frozenset
    outdir: str
    load_data: Callable[[], tuple]
    build_extra_tokens: Callable[[Any, int], list] = field(
        default_factory=lambda: (lambda coords, dim: [])
    )
    hparams: dict = field(default_factory=dict)
    # Optional alternative canonical truth systems (e.g. similarity-solution
    # identities that EPDE may legitimately discover alongside the PDE).
    # ``truth_tokens`` is always the first alternative; additional forms
    # populate ``truth_alternatives`` from the per-system YAML's
    # ``truth_alternatives`` list. ``hamming_best`` /
    # ``structural_success_any`` from thesis_metrics walk all of them.
    truth_alternatives: tuple = field(default_factory=tuple)


def _load_yaml(path: str) -> dict:
    """Read a YAML file into a dict, returning ``{}`` for an empty file."""
    import yaml
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f) or {}


def _deep_merge(base: dict, overrides: dict) -> dict:
    """Recursively merge ``overrides`` on top of ``base``.

    Nested dicts merge key-by-key (overrides win on conflict). Lists are
    REPLACED, not concatenated -- so a per-system YAML overriding
    ``eq_sparsity_interval: [1.0e-3, 1.0]`` wins outright, and
    ``additional_tokens: []`` disables the defaults' YAML tokens (the
    adapter's build_extra_tokens still runs).
    """
    result = dict(base)
    for key, override_value in overrides.items():
        base_value = result.get(key)
        if isinstance(base_value, dict) and isinstance(override_value, dict):
            result[key] = _deep_merge(base_value, override_value)
        else:
            result[key] = override_value
    return result


def load_config(name_or_path: str) -> SystemCfg:
    """Resolve a YAML config into a fully-populated :class:`SystemCfg`.

    ``name_or_path`` may be a bare system name (e.g. ``"lv"`` -- looked up
    as ``configs/lv.yaml``) or an explicit path to a YAML file.

    The hparams blocks (``search``, ``preprocessor``, ``moeadd``,
    ``grid_tokens``, ``additional_tokens``, ``fit``) are deep-merged
    from ``configs/defaults.yaml`` with the per-system YAML on top. Any
    other top-level key (``name``, ``truth_equations``, ``adapter``,
    ``outdir``) is consumed directly by this loader.

    The adapter module is imported via ``importlib`` from
    ``adapters/<adapter>.py``; it must export ``load_data`` and may
    optionally export ``build_extra_tokens``.
    """
    yaml_path = (
        name_or_path
        if os.path.sep in name_or_path or name_or_path.endswith('.yaml')
        else os.path.join(CONFIGS_DIR, f'{name_or_path}.yaml')
    )
    yaml_path = os.path.abspath(yaml_path)
    if not os.path.exists(yaml_path):
        raise FileNotFoundError(f"config not found: {yaml_path}")

    defaults_dict = _load_yaml(DEFAULTS_PATH) if os.path.exists(DEFAULTS_PATH) else {}
    system_dict = _load_yaml(yaml_path)

    name = system_dict.get('name')
    if not name:
        raise ValueError(f"{yaml_path}: 'name' is required")

    from thesis_metrics import canonical_tokens
    truth_equations = system_dict.get('truth_equations') or []
    truth_tokens = canonical_tokens(truth_equations)
    # Optional alternative canonical truth forms. Each entry is a list
    # of equation strings (same shape as ``truth_equations``); each
    # canonicalises to a tuple of frozensets that the metric helpers
    # compare against in addition to the primary truth.
    alt_lists = system_dict.get('truth_alternatives') or []
    truth_alternatives = tuple(
        canonical_tokens(eq_list) for eq_list in alt_lists if eq_list
    )

    adapter_name = system_dict.get('adapter', name)
    if _THIS_DIR not in sys.path:
        sys.path.insert(0, _THIS_DIR)
    adapter_mod = importlib.import_module(f'adapters.{adapter_name}')

    if not hasattr(adapter_mod, 'load_data'):
        raise AttributeError(
            f"adapter {adapter_name!r} must export load_data() -> "
            "(coords, data, variable_names, dim)"
        )

    outdir_rel = system_dict.get('outdir', name)
    outdir = (
        outdir_rel
        if os.path.isabs(outdir_rel)
        else os.path.abspath(os.path.join(RESULTS_DIR, outdir_rel))
    )

    overrides = {k: v for k, v in system_dict.items() if k in HPARAM_KEYS}
    hparams = _deep_merge(defaults_dict, overrides)

    kwargs: dict = dict(
        name=name,
        truth_tokens=truth_tokens,
        truth_alternatives=truth_alternatives,
        outdir=outdir,
        load_data=adapter_mod.load_data,
        hparams=hparams,
    )
    if hasattr(adapter_mod, 'build_extra_tokens'):
        kwargs['build_extra_tokens'] = adapter_mod.build_extra_tokens
    return SystemCfg(**kwargs)


def _boundary_for(coords) -> Any:
    """Return ``10%``-of-axis boundary for the supplied EPDE coordinate tensors.

    ODE problems pass a single 1-D array via ``(t,)``; the returned
    boundary is a scalar ``len(t) // 10``. PDE problems pass a meshgrid
    tuple where every array has the same multidimensional shape; the
    returned boundary is a per-axis tuple of ``axis_size // 10``.
    """
    sample = np.asarray(coords[0])
    if sample.ndim <= 1:
        return max(1, len(sample) // 10)
    return tuple(max(1, n // 10) for n in sample.shape)


def _truth_alts(cfg: 'SystemCfg') -> tuple:
    """Return the full tuple of canonical truth alternatives for ``cfg``.

    The primary ``truth_tokens`` is always the first element; any
    ``truth_alternatives`` declared in the YAML follow. Used everywhere
    the metric is computed so EPDE is credited for discovering any
    declared valid form (e.g. burgers_inviscid's similarity-solution
    identity in addition to the PDE form).
    """
    return (cfg.truth_tokens,) + tuple(cfg.truth_alternatives)


def _build_truth_match_callback(cfg: 'SystemCfg') -> Callable:
    """Return a per-epoch callback that stops MOEA/D once any Pareto-0
    candidate canonically matches one of ``cfg``'s truth alternatives.
    """
    from thesis_metrics import canonical_tokens, structural_success_any
    truth_alts = _truth_alts(cfg)

    def _cb(snapshot, epoch_idx):
        for entry in snapshot:
            text = entry.get('text_form', '') if isinstance(entry, dict) else str(entry)
            lines = [line for line in text.split('\n') if line.strip()]
            try:
                canon = canonical_tokens(lines)
            except Exception:
                continue
            if structural_success_any(canon, truth_alts):
                return True
        return False

    return _cb


def _build_token_pool(cfg: 'SystemCfg', coords, dim: int) -> list:
    """Assemble the system's full token list.

    Order: GridTokens (auto-derived labels) + YAML-declared
    additional_tokens (from cfg.hparams) + adapter's truth-specific
    build_extra_tokens. ``dimensionality=dim`` is injected at
    construction time so the same YAML spec works for ODE (dim=0) and
    every PDE dim>=1 without per-system overrides.
    """
    gt = cfg.hparams['grid_tokens']
    # Single grid-token family: one label ``x``, with ``dim`` as the
    # inner parameter discriminating axes. Previously this registered
    # one token per axis (``x_0``, ``x_1``, ...) AND also a ``dim``
    # parameter -- redundant, and the metric's ``_GRID_COORD_NAME_RE``
    # collapse was a workaround. ``unique_token_type=True`` on the
    # family already caps grid factors at one per term, so collapsing
    # the labels doesn't change observable behaviour.
    grid_labels = ['x']
    pool: list = [GridTokens(grid_labels, dimensionality=dim, max_power=gt['max_power'])]

    for spec in cfg.hparams.get('additional_tokens') or []:
        type_name = spec['type']
        cls = _TOKEN_REGISTRY.get(type_name)
        if cls is None:
            raise ValueError(
                f"Unknown additional_tokens type {type_name!r}; "
                f"expected one of {tuple(_TOKEN_REGISTRY)}"
            )
        kwargs = dict(spec.get('kwargs') or {})
        # YAML lists -> tuples where the EPDE class expects a tuple.
        if 'freq' in kwargs and isinstance(kwargs['freq'], list):
            kwargs['freq'] = tuple(kwargs['freq'])
        kwargs.setdefault('dimensionality', dim)
        pool.append(cls(**kwargs))

    pool.extend(list(cfg.build_extra_tokens(coords, dim)))
    return pool


def _construct_search(cfg: 'SystemCfg', coords, pipeline_kwargs: dict) -> EpdeSearch:
    """Instantiate EpdeSearch from cfg.hparams['search'] + pipeline_kwargs."""
    sh = cfg.hparams['search']
    return EpdeSearch(
        use_solver=sh['use_solver'],
        multiobjective_mode=sh['multiobjective_mode'],
        boundary=_boundary_for(coords),
        coordinate_tensors=coords,
        verbose_params=sh['verbose'],
        device=sh['device'],
        **pipeline_kwargs,
    )


def _configure_preprocessor(search: EpdeSearch, cfg: 'SystemCfg') -> None:
    pp = cfg.hparams['preprocessor']
    search.set_preprocessor(default_preprocessor_type=pp['type'],
                            preprocessor_kwargs=pp.get('kwargs') or {})


def _configure_moeadd(search: EpdeSearch, cfg: 'SystemCfg',
                       variable_names: list) -> None:
    mo = cfg.hparams['moeadd']
    early_stop_cb = _build_truth_match_callback(cfg) if mo.get('early_stop_on_truth') else None
    population_size = int(mo['population_size'])
    # Coupled systems (multi-equation: lv, lorenz, ns) live in a joint
    # (discrepancy, complexity)^k objective space where k is the number
    # of equations. The default population_size=16 weight vectors
    # produces a thin Pareto front (LV reps often return 1-11 solutions,
    # Lorenz ~14-16). Double the weight population to 32 for coupled
    # systems so the front actually covers the per-equation trade-offs.
    if len(variable_names) > 1:
        population_size = max(population_size, 32)
    search.set_moeadd_params(
        population_size=population_size,
        training_epochs=mo['training_epochs'],
        early_stopping_callback=early_stop_cb,
    )


def _run_fit(search: EpdeSearch, cfg: 'SystemCfg', data, variable_names,
             dim: int, additional_tokens: list) -> None:
    f = cfg.hparams['fit']
    max_deriv_order = f.get('max_deriv_order')
    if max_deriv_order is None:
        # ODE (dim=0) -> (2,); 1+1D PDE -> (2, 4); 2+1D PDE -> (2, 4, 4).
        max_deriv_order = (2,) + (4,) * dim
    else:
        max_deriv_order = tuple(max_deriv_order)
    fma = f['equation_factors_max_number']
    search.fit(
        data=data,
        variable_names=variable_names,
        max_deriv_order=max_deriv_order,
        derivs=None,
        equation_terms_max_number=f['equation_terms_max_number'],
        data_fun_pow=f['data_fun_pow'],
        deriv_fun_pow=f['deriv_fun_pow'],
        additional_tokens=additional_tokens,
        equation_factors_max_number={
            'factors_num': fma['factors_num'],
            'probas': fma['probas'],
        },
        eq_sparsity_interval=tuple(f['eq_sparsity_interval']),
        fourier_layers=f['fourier_layers'],
    )


def build_search(cfg: 'SystemCfg', pipeline_kwargs: dict) -> EpdeSearch:
    """Universal EPDE search builder.

    Thin orchestrator: delegates token-pool assembly, EpdeSearch
    construction, preprocessor / MOEA/D configuration, and the ``.fit``
    call to dedicated helpers. Every hyperparameter lives in
    ``cfg.hparams`` (loaded from ``configs/defaults.yaml`` + per-system
    overrides); only the pipeline_kwargs (``use_pic``, ``fitness_cls``,
    ``sparsity_cls``) are passed in directly so the LEGACY vs NEW
    selection can vary per-rep without touching the YAML.
    """
    coords, data, variable_names, dim = cfg.load_data()
    additional_tokens = _build_token_pool(cfg, coords, dim)
    search = _construct_search(cfg, coords, pipeline_kwargs)
    _configure_preprocessor(search, cfg)
    _configure_moeadd(search, cfg, variable_names)
    _run_fit(search, cfg, data, variable_names, dim, additional_tokens)
    return search


def _set_seeds(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _tokens_to_json(tokens) -> list:
    """Recursively convert a canonical token structure into JSON-friendly lists.

    The canonical structure (per :func:`thesis_metrics.canonical_tokens`)
    is a tuple of frozensets, where each frozenset is the **flat term
    set** of one equation (target-side-independent: target and rhs
    merged). Each term is a frozenset of factors; each factor is a
    ``(name, frozenset_of_param_items)`` tuple.

    Output shape: ``[ [ [ [name, [[k,v], ...]], ... ], ... ], ... ]``
    (equations → terms → factors → factor as ``[name, sorted_params]``).
    """
    def factor(f):
        name, params = f
        return [name, sorted(([k, v] for k, v in params), key=lambda p: p[0])]

    def term(t):
        return sorted([factor(f) for f in t], key=lambda f: (f[0], repr(f[1])))

    def equation(eq):
        return sorted([term(t) for t in eq], key=lambda t: repr(t))

    return sorted([equation(eq) for eq in tokens], key=lambda x: repr(x))


def _discovery_epochs(final_token_sets, pareto_history) -> list:
    """For each canonical token set in ``final_token_sets``, return the
    first epoch index (0-based) in ``pareto_history`` whose Pareto-0
    snapshot contains a solution with the same canonical structure.
    """
    from thesis_metrics import canonical_tokens

    snapshot_canon = []
    for epoch_snapshot in pareto_history:
        per_epoch = []
        for sol_record in epoch_snapshot:
            text = sol_record.get('text_form', '') if isinstance(sol_record, dict) else str(sol_record)
            lines = [line for line in text.split('\n') if line.strip()]
            per_epoch.append(canonical_tokens(lines))
        snapshot_canon.append(per_epoch)

    epochs = []
    for target in final_token_sets:
        first = None
        for epoch_idx, epoch_canon in enumerate(snapshot_canon):
            if any(c == target for c in epoch_canon):
                first = epoch_idx
                break
        epochs.append(first)
    return epochs


def _extract_discovered(search: EpdeSearch) -> list:
    """Return all solutions from the non-dominated Pareto level."""
    eqs = search.equations(only_print=False, only_str=True, num=1)
    if not eqs:
        return []
    if isinstance(eqs[0], list):
        level0_solutions = eqs[0]
    else:
        level0_solutions = eqs

    out = []
    for solution in level0_solutions:
        if not isinstance(solution, str):
            solution = str(solution)
        out.append([line for line in solution.split('\n') if line.strip()])
    return out


def _extract_objectives(search: EpdeSearch) -> list:
    """Return per-solution objective vectors aligned with ``_extract_discovered``."""
    try:
        level0 = search.optimizer.pareto_levels.levels[0]
    except Exception:
        return []
    out = []
    for sol in level0:
        try:
            obj = sol.obj_fun.tolist() if hasattr(sol.obj_fun, 'tolist') else list(sol.obj_fun)
        except Exception:
            obj = None
        out.append(obj)
    return out


def _extract_candidate_history(search: EpdeSearch) -> list:
    """Return per-epoch all-candidate objective vectors.

    Source: ``MOEADDOptimizer._hist`` (populated at
    ``epde/optimizers/moeadd/moeadd.py:671`` via
    ``pareto_levels.get_stats()``). Each entry is an ndarray of shape
    ``(n_candidates_at_epoch, n_objectives)`` -- here returned as
    JSON-safe nested lists. Captures ALL candidates across ALL Pareto
    levels, not just front 0, so density plots can show full search
    coverage. Returns ``[]`` if the optimizer didn't track history.
    """
    try:
        raw_hist = getattr(search.optimizer, '_hist', []) or []
    except Exception:
        return []
    out = []
    for snap in raw_hist:
        try:
            out.append(np.asarray(snap, dtype=float).tolist())
        except Exception:
            out.append([])
    return out


def run_one(system_cfg: SystemCfg, pipeline: str, seed: int) -> dict:
    """Run a single (system, pipeline, seed) repetition.

    Returns a dict suitable for JSON serialization. Exceptions are caught
    and recorded as ``error`` and ``traceback`` fields so a failing rep
    does not kill the batch.
    """
    from thesis_metrics import (
        canonical_tokens, hamming_best, structural_success_any,
    )

    pipeline_kwargs = pipeline_settings(pipeline)
    truth_alts = _truth_alts(system_cfg)
    _set_seeds(seed)

    record: dict = {
        'system': system_cfg.name,
        'pipeline': pipeline,
        'seed': seed,
        'pipeline_kwargs': {
            'use_pic': pipeline_kwargs['use_pic'],
            'fitness_cls': pipeline_kwargs['fitness_cls'].__name__,
            'sparsity_cls': pipeline_kwargs['sparsity_cls'].__name__,
        },
    }

    t0 = time.time()
    try:
        search = build_search(system_cfg, pipeline_kwargs)
        elapsed = time.time() - t0
        solutions_text = _extract_discovered(search)
        objectives_per_solution = _extract_objectives(search)
        per_solution_tokens = [canonical_tokens(sol) for sol in solutions_text]
        pareto_history = list(getattr(search, 'pareto_history', []))
        candidate_history = _extract_candidate_history(search)
        if per_solution_tokens:
            hammings = [hamming_best(c, truth_alts) for c in per_solution_tokens]
            best_idx = int(min(range(len(hammings)), key=lambda i: hammings[i]))
            discovery_epochs = _discovery_epochs(per_solution_tokens, pareto_history)
            best_objectives = (
                objectives_per_solution[best_idx]
                if best_idx < len(objectives_per_solution) else None
            )
            record.update({
                'runtime_sec': elapsed,
                'n_pareto_solutions': len(per_solution_tokens),
                'discovered_text_per_solution': solutions_text,
                'discovered_text': solutions_text[best_idx],
                'discovered_tokens_per_solution': [_tokens_to_json(c) for c in per_solution_tokens],
                'discovered_tokens': _tokens_to_json(per_solution_tokens[best_idx]),
                'truth_tokens': _tokens_to_json(system_cfg.truth_tokens),
                'hamming_per_solution': hammings,
                'hamming': hammings[best_idx],
                'discovery_epoch_per_solution': discovery_epochs,
                'discovery_epoch': discovery_epochs[best_idx],
                'n_epochs': len(pareto_history),
                'objectives_per_solution': objectives_per_solution,
                'objectives': best_objectives,
                'structural_success': any(
                    structural_success_any(c, truth_alts) for c in per_solution_tokens
                ),
                # Sidecar payload: stripped by run_smoke before the rep
                # JSON is written; persisted to <rep>.history.json
                # alongside the rep file. Underscore prefix marks this as
                # consumer-private (run_smoke pops it; the small metrics
                # JSON never carries the bulky array).
                '_candidate_history': candidate_history,
            })
        else:
            record.update({
                'runtime_sec': elapsed,
                'n_pareto_solutions': 0,
                'discovered_text_per_solution': [],
                'discovered_text': [],
                'discovered_tokens_per_solution': [],
                'discovered_tokens': [],
                'truth_tokens': _tokens_to_json(system_cfg.truth_tokens),
                'hamming_per_solution': [],
                'hamming': None,
                'objectives_per_solution': [],
                'objectives': None,
                'structural_success': False,
                '_candidate_history': candidate_history,
            })
    except Exception as exc:  # pragma: no cover - smoke-time diagnostic
        record.update({
            'runtime_sec': time.time() - t0,
            'error': repr(exc),
            'traceback': traceback.format_exc(),
            'n_pareto_solutions': 0,
            'discovered_text_per_solution': [],
            'discovered_text': [],
            'discovered_tokens': [],
            'hamming': None,
            'objectives_per_solution': [],
            'objectives': None,
            'structural_success': False,
            '_candidate_history': [],
        })
    return record


def _resolve_out_root(system_cfg: SystemCfg, outdir: Optional[str]) -> str:
    """Resolve the final output directory for a batch.

    Default (``outdir is None``)    -> ``system_cfg.outdir`` (typically
                                       ``projects/thesis/results/<name>``).
    Absolute path                   -> used as-is.
    Bare tag (e.g. ``ablation_v2``) -> ``results/<tag>/<name>``, so a
                                       tagged sweep across all systems
                                       stays grouped under one folder
                                       (``results/<tag>/lv``, .../lorenz, ...)
                                       and the aggregator can scan a tag in
                                       one glob.
    """
    if outdir is None:
        return system_cfg.outdir
    if os.path.isabs(outdir):
        return outdir
    return os.path.join(RESULTS_DIR, outdir, system_cfg.name)


def run_smoke(
    system_cfg: SystemCfg,
    reps: int = 3,
    pipelines: Iterable[str] = PIPELINES,
    seed_base: int = 0,
    resume: bool = True,
    outdir: Optional[str] = None,
) -> None:
    """Run ``reps`` × len(pipelines) repetitions and write JSON per rep.

    With ``resume=True`` (default) any ``(pipeline, rep)`` whose target JSON
    already exists and parses as JSON is skipped. Pass ``resume=False`` to
    overwrite. See :func:`_resolve_out_root` for ``outdir`` semantics.
    """
    out_root = _resolve_out_root(system_cfg, outdir)
    os.makedirs(out_root, exist_ok=True)
    from epde import globals as _gv
    for pipeline in pipelines:
        for rep in range(reps):
            seed = seed_base + rep
            # Per-rep noise seed for the ``--noise-level`` injection
            # path (no-op when noise is disabled).
            _gv.noise_seed = seed
            out_path = os.path.join(out_root, f"{pipeline}_rep{rep:02d}.json")
            if resume and os.path.exists(out_path):
                try:
                    with open(out_path, 'r', encoding='utf-8') as fh:
                        json.load(fh)
                    print(f"\n========== {system_cfg.name} / {pipeline} / rep {rep} -- "
                          f"skipping (resume; {out_path} exists) ==========")
                    continue
                except (json.JSONDecodeError, OSError) as exc:
                    print(f"[resume] {out_path} unreadable ({exc!r}); re-running rep")
            print(f"\n========== {system_cfg.name} / {pipeline} / rep {rep} (seed={seed}) ==========")
            record = run_one(system_cfg, pipeline, seed)
            # Strip the bulky per-epoch candidate history into a sidecar
            # file so the rep JSON stays light. See _extract_candidate_history.
            candidate_history = record.pop('_candidate_history', [])
            if candidate_history:
                history_basename = f"{pipeline}_rep{rep:02d}.history.json"
                history_path = os.path.join(out_root, history_basename)
                first_snap = next((s for s in candidate_history if s), None)
                n_obj = len(first_snap[0]) if first_snap else 0
                with open(history_path, 'w', encoding='utf-8') as fh:
                    json.dump({
                        'system': system_cfg.name,
                        'pipeline': pipeline,
                        'seed': seed,
                        'n_epochs': len(candidate_history),
                        'n_objectives': n_obj,
                        'candidate_history': candidate_history,
                    }, fh)
                record['history_path'] = history_basename
            with open(out_path, 'w', encoding='utf-8') as fh:
                json.dump(record, fh, indent=2, default=str)
            status = 'OK' if 'error' not in record else 'FAIL'
            ham = record.get('hamming')
            epoch = record.get('discovery_epoch')
            n_ep = record.get('n_epochs')
            epoch_str = f"epoch={epoch}/{n_ep}" if epoch is not None else "epoch=?"
            print(f"  -> {status}  hamming={ham}  {epoch_str}  time={record.get('runtime_sec', 0.0):.1f}s")
            print(f"  -> saved {out_path}")
