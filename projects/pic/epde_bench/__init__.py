"""Benchmark and tooling for the data sets in ``projects/pic/data``.

Public entry points::

    from epde_bench import datasets, load_config, resolve_for_problem, discover, run_one
    problem = datasets.load('ac')            # data + grids + known truth
    cfg = load_config('ac')                  # merged YAML configuration
    search, front, objectives, seconds = discover(problem, resolve_for_problem(cfg, problem))

``bench.py`` (one level up) is the command line over the same functions.

Submodules are imported lazily, so ``from epde_bench import env`` does not
load numpy: ``env.pin_blas_threads`` must run before numpy is imported.
"""

import importlib
import sys
from pathlib import Path

# Commands and modules belong to the EPDE checkout containing this package.
_CHECKOUT = Path(__file__).resolve().parents[3]
if (_CHECKOUT / 'epde').is_dir():
    sys.path.insert(0, str(_CHECKOUT))

_LAZY = {'datasets': ('.datasets', None), 'metrics': ('.metrics', None),
         'load_config': ('.config', 'load_config'),
         'resolve_for_problem': ('.config', 'resolve_for_problem'),
         'discover': ('.runner', 'discover'), 'run_one': ('.runner', 'run_one')}

__all__ = list(_LAZY)


def __getattr__(name):
    if name not in _LAZY:
        raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
    module_name, attr = _LAZY[name]
    module = importlib.import_module(module_name, __name__)
    return module if attr is None else getattr(module, attr)
