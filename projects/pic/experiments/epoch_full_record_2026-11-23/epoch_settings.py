from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent/'structural_holdout_2026-11-23'))
from experiment import live, np, trial, whole_record

SEEDS = tuple(trial.CONFIRMATION[:3])
NOISE = (0, 5)
COUPLED = {'lv', 'lorenz', 'ns'}
OPTIONS = {'lorenz': dict(i0=0, n=None, step=1), 'ns': dict(subset='full')}


def directory(name, seed, noise):
    return ROOT/'results'/name/str(seed)/f'noise{noise}'


def epochs(name):
    return 150 if name in COUPLED else 50
