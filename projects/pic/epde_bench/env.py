"""Process environment for reproducible, cross-platform EPDE runs.

Everything here must work the same on Windows, Linux and macOS:

* BLAS threading is pinned to one thread *before* numpy is imported. EPDE's
  inner loop is many small least-squares solves, where a multi-threaded BLAS
  only adds contention; one thread per process plus process-level parallelism
  over seeds is much faster (measured on the lab PC: 12 parallel seeds with
  ``OPENBLAS_NUM_THREADS=1`` vs. one 12-thread seed).
* The device is resolved, never hard-coded. The scripts in ``data/`` used to
  say ``device='cuda'`` and ``pickle.load(...).cuda()``, which fails outright
  on a machine without CUDA. EPDE only uses the GPU inside the PDE solver, so
  ``cpu`` is always a valid choice.
* Every source of randomness is seeded from one integer.
"""

import os
import platform
import random
import subprocess
import sys
from pathlib import Path

_THREAD_VARS = ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS')


def pin_blas_threads(n: int = 1) -> None:
    """Limit BLAS/OpenMP to ``n`` threads. Effective only before numpy loads,
    so call it first thing in an entry point (``bench.py`` does)."""
    for var in _THREAD_VARS:
        os.environ.setdefault(var, str(n))


def resolve_device(requested: str = 'auto') -> str:
    """``'auto'`` -> ``'cuda'`` when a working GPU is visible, else ``'cpu'``.

    Apple ``mps`` is deliberately not offered: EPDE's solver code paths are
    written for cuda/cpu tensors.
    """
    if requested not in ('auto', None):
        if requested == 'cuda' and not cuda_available():
            print('[epde_bench] cuda requested but not available, using cpu',
                  file=sys.stderr)
            return 'cpu'
        return requested
    return 'cuda' if cuda_available() else 'cpu'


def cuda_available() -> bool:
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:                                          # noqa: BLE001
        return False


def seed_everything(seed: int) -> None:
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:                                          # noqa: BLE001
        pass


def git_commit(repo: Path) -> str:
    """Short commit hash of ``repo`` (with ``-dirty`` when it has local
    changes), or ``'unknown'`` when git is not available."""
    try:
        head = subprocess.run(['git', 'rev-parse', '--short', 'HEAD'], cwd=repo,
                              capture_output=True, text=True, timeout=10)
        if head.returncode != 0:
            return 'unknown'
        dirty = subprocess.run(['git', 'status', '--porcelain', '--untracked-files=no'],
                               cwd=repo, capture_output=True, text=True, timeout=10)
        return head.stdout.strip() + ('-dirty' if dirty.stdout.strip() else '')
    except Exception:                                          # noqa: BLE001
        return 'unknown'


def environment_info(repo: Path, probe_cuda: bool = True) -> dict:
    """What a result needs to be reproducible: platform, versions, commit.

    Versions are read from package metadata, not by importing the packages
    (importing torch alone costs ~600 MB on Windows)."""
    from importlib import metadata
    info = {'platform': platform.platform(), 'python': platform.python_version(),
            'machine': platform.machine(), 'cpu_count': os.cpu_count(),
            'epde_commit': git_commit(repo),
            'blas_threads': os.environ.get('OPENBLAS_NUM_THREADS')}
    for package in ('numpy', 'scipy', 'torch', 'pysindy'):
        try:
            info[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            info[package] = None
    if probe_cuda:
        info['cuda'] = cuda_available()
    return info
