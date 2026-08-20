"""Shared scaffolding for the EPDE-vs-GOLEM comparison experiments."""

import os
import sys
import time
import json
import random
import contextlib

import numpy as np

#: Repository root: benchmarks/golem/_common.py -> benchmarks -> <repo>.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

#: The scenarios and their reference data are EPDE's own functional-test suite.
DATA_DIR = os.path.join(REPO_ROOT, 'tests', 'functional', 'scenarios')

#: Where the benchmark writes its raw records and figures.
RESULTS_DIR = os.path.join(REPO_ROOT, 'benchmarks', 'golem', 'results')


def check_hash_seed():
    """Warn when PYTHONHASHSEED is unset.

    EPDE iterates over ``set`` objects keyed by strings (the token pool's
    variable set, the duplicate-systems history), so its search path depends on
    Python's per-process string-hash salt. Without ``PYTHONHASHSEED`` fixed,
    two runs with the same explicit numpy/torch seed still diverge -- runs are
    not reproducible, and A/B comparisons pick up the salt as noise.
    """
    if os.environ.get('PYTHONHASHSEED') is None:
        print('WARNING: PYTHONHASHSEED is not set -- runs will not be '
              'reproducible. Re-run with PYTHONHASHSEED=0.', file=sys.stderr)


def set_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except Exception:
        pass


class Timer:
    def __enter__(self):
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.elapsed = time.perf_counter() - self.t0
        return False


@contextlib.contextmanager
def suppressed_stdout(enabled=True):
    if not enabled:
        yield
        return
    import io
    buf = io.StringIO()
    old = sys.stdout
    sys.stdout = buf
    try:
        yield buf
    finally:
        sys.stdout = old
