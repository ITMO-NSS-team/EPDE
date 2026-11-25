from pathlib import Path
import os
import sys

ROOT = Path(__file__).resolve().parent
NOISE_SUITE = ROOT.parent / 'epoch_noise_families_2026-11-23'
sys.path.insert(0, str(NOISE_SUITE))
from cpu_budget import restrict
import psutil

DATASETS = ('ode', 'vdp', 'duffing', 'lv')
FRESH = (924801, 924802, 924803)


def limit_threads():
    restrict(psutil.Process())
    for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[name] = '1'


def fresh_directory(name, seed, family):
    return ROOT / 'results' / name / str(seed) / family
