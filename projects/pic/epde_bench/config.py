"""Run configuration: YAML layers merged into one dict.

Layers, later ones win::

    configs/_base.yaml          the common protocol (same token pool for all)
    configs/<dataset>.yaml      what this data set needs on top of it
    configs/variants.yaml[v]    the method variant being compared
    --set key.path=value        ad-hoc overrides from the command line

Dicts merge key by key; lists and scalars replace. The ``search`` section is
handed to ``EpdeSearch(config=...)`` as is, so its keys are exactly EPDE's
config groups (``domain``, ``preprocessing``, ``search_space``,
``objectives``, ``solver``, ``evolution``, ``runtime``; see
``epde/interface/search_config.py``). Three values are filled in per data
set by :func:`resolve_for_problem`:

* ``domain.boundary_width: auto`` -> 10 % of every axis;
* ``preprocessing.max_deriv_order: auto`` -> 2 in time, 4 in each space axis;
* a token spec without ``dimensionality`` gets the problem's.
"""

import copy
from pathlib import Path
from typing import Iterable, Optional

import yaml

from .paths import CONFIG_DIR

BASE = '_base'
VARIANTS_FILE = 'variants.yaml'


def deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    # Builder arguments belong to one preprocessor and cannot cross method changes.
    if ('default_preprocessor_type' in (override or {})
            and override['default_preprocessor_type'] != base.get('default_preprocessor_type')):
        out['preprocessor_kwargs'] = {}
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def read_yaml(path: Path) -> dict:
    with open(path, encoding='utf-8') as f:
        return yaml.safe_load(f) or {}


def variants() -> dict:
    return read_yaml(CONFIG_DIR / VARIANTS_FILE)


def parse_set(assignments: Iterable[str]) -> dict:
    """``['search.evolution.training_epochs=10']`` -> nested dict. Values are
    parsed as YAML, so numbers, lists and booleans keep their type."""
    out: dict = {}
    for item in assignments or ():
        if '=' not in item:
            raise ValueError(f'--set expects key.path=value, got {item!r}')
        path, raw = item.split('=', 1)
        node = out
        keys = path.strip().split('.')
        for key in keys[:-1]:
            node = node.setdefault(key, {})
        node[keys[-1]] = yaml.safe_load(raw)
    return out


def load_config(dataset: str, variant: str = 'default', overrides: Optional[dict] = None,
                config_path: Optional[Path] = None) -> dict:
    """The merged configuration of one (data set, variant) pair.

    ``config_path`` replaces ``configs/<dataset>.yaml`` (for experiments
    with a hand-written file)."""
    cfg = read_yaml(CONFIG_DIR / f'{BASE}.yaml')
    own = Path(config_path) if config_path else CONFIG_DIR / f'{dataset}.yaml'
    if config_path is not None and not own.is_file():
        raise FileNotFoundError(f'Explicit configuration file does not exist: {own}')
    if own.exists():
        cfg = deep_merge(cfg, read_yaml(own))
    all_variants = variants()
    if variant not in all_variants:
        raise KeyError(f'Unknown variant {variant!r}; defined in configs/{VARIANTS_FILE}: '
                       f'{", ".join(all_variants)}')
    cfg = deep_merge(cfg, all_variants[variant] or {})
    cfg = deep_merge(cfg, overrides or {})
    cfg['dataset'] = dataset
    cfg['variant'] = variant
    cfg.setdefault('method', 'epde')
    return cfg


def resolve_for_problem(cfg: dict, problem) -> dict:
    """Replace the ``auto`` placeholders with values for ``problem`` and
    return the ``search`` section ready for ``EpdeSearch(config=...)``."""
    search = copy.deepcopy(cfg.get('search', {}))
    shape = problem.shape
    domain = search.setdefault('domain', {})
    if domain.get('boundary_width', 'auto') == 'auto':
        widths = [max(1, n // 10) for n in shape]
        domain['boundary_width'] = widths[0] if len(widths) == 1 else widths
    prep = search.setdefault('preprocessing', {})
    if prep.get('max_deriv_order', 'auto') == 'auto':
        prep['max_deriv_order'] = [2] + [4] * problem.dim
    space = search.setdefault('search_space', {})
    for spec in space.get('tokens', []) or []:
        spec.setdefault('dimensionality', problem.dim)
        if isinstance(spec.get('freq'), list):
            spec['freq'] = tuple(spec['freq'])
    return search


def split_bench_tokens(search: dict):
    """Build every declared token family here and hand EPDE a config with an
    empty ``search_space.tokens``. Returns ``(search, families)``.

    Two reasons. (1) Families only this package knows (``tokens.BENCH_FAMILIES``)
    cannot go through EPDE's registry. (2) Families declared
    in ``search_space.tokens`` end up in the pool TWICE when ``fit()`` is used:
    ``fit`` merges them into ``additional_tokens`` and ``create_pool`` merges
    them again (both call ``_resolve_token_families``), which duplicates e.g.
    the grid family. Passing everything as ``additional_tokens`` avoids that.
    """
    from epde.interface.search_config import build_tokens
    from .tokens import BENCH_FAMILIES
    search = copy.deepcopy(search)
    space = search.get('search_space', {})
    families = []
    for spec in space.get('tokens', []) or []:
        kwargs = {k: v for k, v in spec.items() if k != 'family'}
        if spec.get('family') in BENCH_FAMILIES:
            families.append(BENCH_FAMILIES[spec['family']](**kwargs))
        else:
            families.extend(build_tokens([spec]))
    if 'tokens' in space:
        space['tokens'] = []
    return search, families
