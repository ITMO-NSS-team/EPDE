from functools import partial
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parent
BASELINE = ROOT.parent / 'epoch_full_record_2026-11-23'
sys.path.insert(0, str(BASELINE))
from epoch_settings import OPTIONS, SEEDS, directory as baseline_directory, epochs, live, trial, whole_record
from measurements import FAMILIES, LEVEL, observe


def directory(name, seed, family):
    return ROOT / 'results' / name / str(seed) / f'{family}_{LEVEL}pct'


def measurement(family):
    return partial(observe, family=family)


def jobs():
    names = ['ode_simple', *[name for name in live.DATASETS if name != 'ode_simple']]
    return [(name, seed, family) for seed in SEEDS for name in names for family in FAMILIES]


def read_json(path):
    return json.loads(path.read_text(encoding='utf8'))


def atomic_text(path, text):
    temp = path.with_suffix(path.suffix + '.pending')
    temp.write_text(text, encoding='utf8')
    temp.replace(path)
