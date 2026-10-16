"""Filesystem layout. All paths are built with pathlib from this file's
location, so nothing depends on the current working directory or on the
path separator of the OS (the old scripts opened e.g.
``'kuramoto_sivishinky.mat'`` relative to the CWD and ``trial + '//...'``)."""

import os
from pathlib import Path

PIC_DIR = Path(__file__).resolve().parent.parent          # projects/pic
REPO_ROOT = PIC_DIR.parent.parent                          # EPDE checkout
DATA_DIR = PIC_DIR / 'data'
CONFIG_DIR = PIC_DIR / 'configs'
NOTEBOOK_DIR = PIC_DIR / 'notebooks'

#: Where campaigns write by default. Override with ``EPDE_BENCH_RESULTS``
#: (e.g. to keep large result trees outside the repository).
RESULTS_DIR = Path(os.environ.get('EPDE_BENCH_RESULTS', PIC_DIR / 'results'))


def data_path(*parts: str) -> Path:
    """``data_path('ac', 'ac_data.npy')`` -> ``projects/pic/data/ac/ac_data.npy``,
    with a clear error instead of numpy's when the file is missing."""
    path = DATA_DIR.joinpath(*parts)
    if not path.exists():
        raise FileNotFoundError(
            f'{path} is missing. Data files live in projects/pic/data; '
            'see projects/pic/DATASETS.md for where each one comes from.')
    return path
